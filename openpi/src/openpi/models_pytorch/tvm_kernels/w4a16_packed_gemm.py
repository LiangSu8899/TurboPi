#!/usr/bin/env python3
"""
W4A16 Packed FP4 GEMM Kernel for Thor SM110

Key Design:
1. Weight: Packed FP4 format (uint8, 2 values per byte)
2. Activation: BF16/FP32
3. Dequant: In-register using lookup table
4. Compute: CUDA Core (M=1 GEMV) or Tensor Core (M>1 GEMM)

Based on TRT-LLM W4A16 approach with TVM implementation.

nvFP4 E2M1 Format:
- 4 bits: 1 sign + 2 exponent + 1 mantissa
- Values: [0, 0.5, 1, 1.5, 2, 3, 4, 6] and negatives
- Block scaling: scale per 32 elements

Author: Claude Code
Date: 2026-02-10
"""

import sys
import os

TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm.script import tir as T
from tvm import te
import numpy as np

# ==============================================================================
# Constants
# ==============================================================================

BLOCK_SIZE = 32  # nvFP4 block scaling size
WARP_SIZE = 32

# nvFP4 E2M1 lookup table (4-bit values 0-15)
# Bits: [sign][exp1][exp0][mant]
# 0000=0, 0001=0.5, 0010=1, 0011=1.5, 0100=2, 0101=3, 0110=4, 0111=6
# 1xxx = negative of 0xxx
NVFP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ==============================================================================
# Packed FP4 Quantization Utilities (for testing)
# ==============================================================================

def quantize_to_nvfp4_packed(weight: np.ndarray, block_size: int = BLOCK_SIZE):
    """
    Quantize weight to packed nvFP4 format.

    Args:
        weight: [N, K] float weight matrix
        block_size: number of elements per scale block

    Returns:
        packed: [N, K//2] uint8, packed FP4 values
        scales: [N, num_blocks] float32, scale factors
    """
    N, K = weight.shape
    num_blocks = (K + block_size - 1) // block_size

    # Pad K to multiple of block_size
    K_padded = num_blocks * block_size
    if K < K_padded:
        weight = np.pad(weight, ((0, 0), (0, K_padded - K)), mode='constant')

    # Compute scales per block
    scales = np.zeros((N, num_blocks), dtype=np.float32)
    for i in range(N):
        for b in range(num_blocks):
            start = b * block_size
            end = start + block_size
            block = weight[i, start:end]
            max_abs = np.max(np.abs(block))
            scales[i, b] = max_abs / 6.0 if max_abs > 0 else 1.0

    # Quantize to FP4 indices
    fp4_values = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)

    indices = np.zeros((N, K_padded), dtype=np.uint8)
    for i in range(N):
        for b in range(num_blocks):
            start = b * block_size
            end = start + block_size
            block = weight[i, start:end]
            scale = scales[i, b]

            for j, val in enumerate(block):
                # Normalize by scale
                normalized = val / scale if scale > 0 else 0

                # Find closest FP4 value
                sign = 1 if normalized >= 0 else -1
                abs_val = abs(normalized)

                # Find best index (0-7)
                best_idx = 0
                best_diff = abs(abs_val - fp4_values[0])
                for idx in range(8):
                    diff = abs(abs_val - fp4_values[idx])
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = idx

                # Add sign bit (bit 3)
                if sign < 0:
                    best_idx |= 0x8

                indices[i, start + j] = best_idx

    # Pack 2 FP4 values per byte
    # Lower 4 bits: even index, Upper 4 bits: odd index
    packed = np.zeros((N, K_padded // 2), dtype=np.uint8)
    for i in range(N):
        for j in range(K_padded // 2):
            low = indices[i, 2 * j]
            high = indices[i, 2 * j + 1]
            packed[i, j] = (high << 4) | low

    return packed[:, :K//2], scales


def dequantize_nvfp4_packed(packed: np.ndarray, scales: np.ndarray,
                             K: int, block_size: int = BLOCK_SIZE):
    """
    Dequantize packed nvFP4 to float.
    """
    N = packed.shape[0]
    K_packed = packed.shape[1]

    # Unpack
    result = np.zeros((N, K), dtype=np.float32)

    for i in range(N):
        for j in range(K_packed):
            if 2 * j >= K:
                break

            byte = packed[i, j]
            low = byte & 0xF
            high = (byte >> 4) & 0xF

            # Dequantize using LUT
            k1 = 2 * j
            k2 = 2 * j + 1

            block1 = k1 // block_size
            block2 = k2 // block_size

            if k1 < K:
                result[i, k1] = NVFP4_LUT[low] * scales[i, block1]
            if k2 < K:
                result[i, k2] = NVFP4_LUT[high] * scales[i, block2]

    return result


# ==============================================================================
# TVM Kernel: W4A16 Packed GEMV (M=1)
# ==============================================================================

def create_w4a16_packed_gemv(N: int, K: int, block_size: int = BLOCK_SIZE):
    """
    W4A16 Packed FP4 GEMV for M=1 - Simple version (for correctness baseline).

    C[1, N] = A[1, K] @ (W_packed[N, K//2] * scale[N, num_blocks])^T

    Simple approach: One thread per output element, process all K sequentially.
    """
    num_blocks_k = (K + block_size - 1) // block_size
    K_packed = K // 2

    THREADS_PER_BLOCK = 256
    num_thread_blocks = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    @T.prim_func
    def kernel(
        A: T.Buffer((1, K), "float32"),                    # Activation
        W_packed: T.Buffer((N, K_packed), "uint8"),        # Packed FP4 weights
        scales: T.Buffer((N, num_blocks_k), "float32"),    # Weight scales
        C: T.Buffer((1, N), "float32"),                    # Output
    ):
        T.func_attr({
            "global_symbol": "w4a16_packed_gemv",
            "tir.noalias": True,
        })

        # nvFP4 lookup table in shared memory
        lut = T.alloc_buffer((16,), "float32", scope="shared")

        for bx in T.thread_binding(num_thread_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):

                # Initialize LUT (first 16 threads)
                if tx < 16:
                    if tx == 0:
                        lut[0] = T.float32(0.0)
                    elif tx == 1:
                        lut[1] = T.float32(0.5)
                    elif tx == 2:
                        lut[2] = T.float32(1.0)
                    elif tx == 3:
                        lut[3] = T.float32(1.5)
                    elif tx == 4:
                        lut[4] = T.float32(2.0)
                    elif tx == 5:
                        lut[5] = T.float32(3.0)
                    elif tx == 6:
                        lut[6] = T.float32(4.0)
                    elif tx == 7:
                        lut[7] = T.float32(6.0)
                    elif tx == 8:
                        lut[8] = T.float32(0.0)
                    elif tx == 9:
                        lut[9] = T.float32(-0.5)
                    elif tx == 10:
                        lut[10] = T.float32(-1.0)
                    elif tx == 11:
                        lut[11] = T.float32(-1.5)
                    elif tx == 12:
                        lut[12] = T.float32(-2.0)
                    elif tx == 13:
                        lut[13] = T.float32(-3.0)
                    elif tx == 14:
                        lut[14] = T.float32(-4.0)
                    elif tx == 15:
                        lut[15] = T.float32(-6.0)

                T.tvm_storage_sync("shared")

                j = bx * THREADS_PER_BLOCK + tx  # Output column

                if j < N:
                    # Initialize output
                    C[0, j] = T.float32(0)

                    # Process all K elements
                    for k in T.serial(K):
                        # Get packed byte
                        byte_idx = k // 2
                        is_high = k % 2

                        packed_byte = W_packed[j, byte_idx]

                        # Extract FP4 index
                        fp4_idx = T.if_then_else(
                            is_high == 0,
                            packed_byte & T.uint8(0xF),
                            (packed_byte >> 4) & T.uint8(0xF)
                        )

                        # Lookup and dequant
                        w_val = lut[T.Cast("int32", fp4_idx)]
                        block_idx = k // block_size
                        scale = scales[j, block_idx]
                        w_dequant = w_val * scale

                        # Accumulate
                        a_val = A[0, k]
                        C[0, j] = C[0, j] + a_val * w_dequant

    return kernel


def create_w4a16_packed_gemv_fast(N: int, K: int, block_size: int = BLOCK_SIZE):
    """
    W4A16 Packed FP4 GEMV for M=1 - Optimized version with K tiling.

    C[1, N] = A[1, K] @ (W_packed[N, K//2] * scale[N, num_blocks])^T

    Optimizations:
    1. Multiple threads collaborate on K dimension per output
    2. Use shared memory for A tile (handle large K with tiling)
    3. Parallel reduction using shared memory
    4. Each thread block processes multiple outputs
    """
    num_blocks_k = (K + block_size - 1) // block_size
    K_packed = K // 2

    # Configuration
    REDUCE_THREADS = 64   # Threads for K-dimension reduction per output
    OUTPUTS_PER_BLOCK = 4  # Outputs computed per thread block
    THREADS_PER_BLOCK = REDUCE_THREADS * OUTPUTS_PER_BLOCK  # 256

    # K tiling to fit in shared memory (max ~10KB for A)
    MAX_A_SHARED = 2048  # 2048 floats = 8KB
    TILE_K = min(K, MAX_A_SHARED)

    num_thread_blocks = (N + OUTPUTS_PER_BLOCK - 1) // OUTPUTS_PER_BLOCK
    num_k_tiles = (K + TILE_K - 1) // TILE_K

    # K elements per thread per tile
    K_PER_THREAD_PER_TILE = (TILE_K + REDUCE_THREADS - 1) // REDUCE_THREADS

    @T.prim_func
    def kernel(
        A: T.Buffer((1, K), "float32"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({
            "global_symbol": "w4a16_packed_gemv_fast",
            "tir.noalias": True,
        })

        # LUT in shared memory
        lut = T.alloc_buffer((16,), "float32", scope="shared")

        # A tile in shared memory
        A_shared = T.alloc_buffer((TILE_K,), "float32", scope="shared")

        # Partial sums for reduction
        partial_sums = T.alloc_buffer((OUTPUTS_PER_BLOCK, REDUCE_THREADS), "float32", scope="shared")

        for bx in T.thread_binding(num_thread_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):

                # Initialize LUT
                if tx < 16:
                    if tx == 0:
                        lut[0] = T.float32(0.0)
                    elif tx == 1:
                        lut[1] = T.float32(0.5)
                    elif tx == 2:
                        lut[2] = T.float32(1.0)
                    elif tx == 3:
                        lut[3] = T.float32(1.5)
                    elif tx == 4:
                        lut[4] = T.float32(2.0)
                    elif tx == 5:
                        lut[5] = T.float32(3.0)
                    elif tx == 6:
                        lut[6] = T.float32(4.0)
                    elif tx == 7:
                        lut[7] = T.float32(6.0)
                    elif tx == 8:
                        lut[8] = T.float32(0.0)
                    elif tx == 9:
                        lut[9] = T.float32(-0.5)
                    elif tx == 10:
                        lut[10] = T.float32(-1.0)
                    elif tx == 11:
                        lut[11] = T.float32(-1.5)
                    elif tx == 12:
                        lut[12] = T.float32(-2.0)
                    elif tx == 13:
                        lut[13] = T.float32(-3.0)
                    elif tx == 14:
                        lut[14] = T.float32(-4.0)
                    elif tx == 15:
                        lut[15] = T.float32(-6.0)

                T.tvm_storage_sync("shared")

                # Thread assignment
                output_idx = tx // REDUCE_THREADS  # 0..OUTPUTS_PER_BLOCK-1
                reduce_idx = tx % REDUCE_THREADS    # 0..REDUCE_THREADS-1

                j = bx * OUTPUTS_PER_BLOCK + output_idx  # Global output column

                # Initialize partial sum
                partial_sums[output_idx, reduce_idx] = T.float32(0)

                # Process K in tiles
                for kt in T.serial(num_k_tiles):
                    k_tile_start = kt * TILE_K

                    # Cooperative load of A tile
                    for load_iter in T.serial((TILE_K + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK):
                        k_local = tx + load_iter * THREADS_PER_BLOCK
                        if k_local < TILE_K:
                            k_global = k_tile_start + k_local
                            if k_global < K:
                                A_shared[k_local] = A[0, k_global]
                            else:
                                A_shared[k_local] = T.float32(0)

                    T.tvm_storage_sync("shared")

                    # Each thread processes K_PER_THREAD_PER_TILE elements
                    if j < N:
                        for k_iter in T.serial(K_PER_THREAD_PER_TILE):
                            k_local = reduce_idx + k_iter * REDUCE_THREADS
                            k_global = k_tile_start + k_local

                            if k_local < TILE_K:
                                if k_global < K:
                                    # Get packed byte
                                    byte_idx = k_global // 2
                                    is_high = k_global % 2

                                    packed_byte = W_packed[j, byte_idx]

                                    fp4_idx = T.if_then_else(
                                        is_high == 0,
                                        packed_byte & T.uint8(0xF),
                                        (packed_byte >> 4) & T.uint8(0xF)
                                    )

                                    w_val = lut[T.Cast("int32", fp4_idx)]
                                    block_idx_k = k_global // block_size
                                    scale = scales[j, block_idx_k]
                                    w_dequant = w_val * scale

                                    a_val = A_shared[k_local]
                                    partial_sums[output_idx, reduce_idx] = (
                                        partial_sums[output_idx, reduce_idx] + a_val * w_dequant
                                    )

                    T.tvm_storage_sync("shared")

                # Parallel reduction
                # Step 1: reduce from 64 to 32
                if reduce_idx < 32:
                    partial_sums[output_idx, reduce_idx] = (
                        partial_sums[output_idx, reduce_idx] +
                        partial_sums[output_idx, reduce_idx + 32]
                    )
                T.tvm_storage_sync("shared")

                # Step 2: reduce from 32 to 16
                if reduce_idx < 16:
                    partial_sums[output_idx, reduce_idx] = (
                        partial_sums[output_idx, reduce_idx] +
                        partial_sums[output_idx, reduce_idx + 16]
                    )
                T.tvm_storage_sync("shared")

                # Step 3: reduce from 16 to 8
                if reduce_idx < 8:
                    partial_sums[output_idx, reduce_idx] = (
                        partial_sums[output_idx, reduce_idx] +
                        partial_sums[output_idx, reduce_idx + 8]
                    )
                T.tvm_storage_sync("shared")

                # Step 4: reduce from 8 to 4
                if reduce_idx < 4:
                    partial_sums[output_idx, reduce_idx] = (
                        partial_sums[output_idx, reduce_idx] +
                        partial_sums[output_idx, reduce_idx + 4]
                    )
                T.tvm_storage_sync("shared")

                # Step 5: reduce from 4 to 2
                if reduce_idx < 2:
                    partial_sums[output_idx, reduce_idx] = (
                        partial_sums[output_idx, reduce_idx] +
                        partial_sums[output_idx, reduce_idx + 2]
                    )
                T.tvm_storage_sync("shared")

                # Step 6: reduce from 2 to 1 and write output
                if reduce_idx == 0:
                    if j < N:
                        C[0, j] = partial_sums[output_idx, 0] + partial_sums[output_idx, 1]

    return kernel


# ==============================================================================
# TVM Kernel: W4A16 Packed GEMM (M > 1) - Tiled Version
# ==============================================================================

def create_w4a16_packed_gemm_tiled(M: int, N: int, K: int, block_size: int = BLOCK_SIZE):
    """
    W4A16 Packed FP4 GEMM for M > 1.

    C[M, N] = A[M, K] @ (W_packed[N, K//2] * scale[N, num_blocks])^T

    Uses tiled approach with shared memory.
    Note: Uses output buffer for accumulation to avoid TVM local var issues.
    """
    num_blocks = (K + block_size - 1) // block_size
    K_packed = K // 2

    # Tile sizes
    TILE_M = 32
    TILE_N = 32
    TILE_K = 64  # Must be even for packed format

    THREADS_X = 8
    THREADS_Y = 8
    THREADS_PER_BLOCK = THREADS_X * THREADS_Y  # 64

    # Elements per thread
    ELEM_M = TILE_M // THREADS_Y  # 4
    ELEM_N = TILE_N // THREADS_X  # 4

    num_blocks_m = (M + TILE_M - 1) // TILE_M
    num_blocks_n = (N + TILE_N - 1) // TILE_N
    num_k_tiles = (K + TILE_K - 1) // TILE_K

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_blocks), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({
            "global_symbol": "w4a16_packed_gemm_tiled",
            "tir.noalias": True,
        })

        # Shared memory
        A_shared = T.alloc_buffer((TILE_M, TILE_K), "float32", scope="shared")
        W_shared = T.alloc_buffer((TILE_N, TILE_K), "float32", scope="shared")

        # LUT in shared memory
        lut = T.alloc_buffer((16,), "float32", scope="shared")

        for bm in T.thread_binding(num_blocks_m, thread="blockIdx.y"):
            for bn in T.thread_binding(num_blocks_n, thread="blockIdx.x"):
                for ty in T.thread_binding(THREADS_Y, thread="threadIdx.y"):
                    for tx in T.thread_binding(THREADS_X, thread="threadIdx.x"):

                        tid = ty * THREADS_X + tx

                        # Initialize LUT
                        if tid < 16:
                            if tid == 0:
                                lut[0] = T.float32(0.0)
                            elif tid == 1:
                                lut[1] = T.float32(0.5)
                            elif tid == 2:
                                lut[2] = T.float32(1.0)
                            elif tid == 3:
                                lut[3] = T.float32(1.5)
                            elif tid == 4:
                                lut[4] = T.float32(2.0)
                            elif tid == 5:
                                lut[5] = T.float32(3.0)
                            elif tid == 6:
                                lut[6] = T.float32(4.0)
                            elif tid == 7:
                                lut[7] = T.float32(6.0)
                            elif tid == 8:
                                lut[8] = T.float32(0.0)
                            elif tid == 9:
                                lut[9] = T.float32(-0.5)
                            elif tid == 10:
                                lut[10] = T.float32(-1.0)
                            elif tid == 11:
                                lut[11] = T.float32(-1.5)
                            elif tid == 12:
                                lut[12] = T.float32(-2.0)
                            elif tid == 13:
                                lut[13] = T.float32(-3.0)
                            elif tid == 14:
                                lut[14] = T.float32(-4.0)
                            elif tid == 15:
                                lut[15] = T.float32(-6.0)

                        T.tvm_storage_sync("shared")

                        # Initialize output elements to zero
                        for em in T.serial(ELEM_M):
                            for en in T.serial(ELEM_N):
                                m_global = bm * TILE_M + ty * ELEM_M + em
                                n_global = bn * TILE_N + tx * ELEM_N + en
                                if m_global < M and n_global < N:
                                    C[m_global, n_global] = T.float32(0)

                        # Process K tiles
                        for kt in T.serial(num_k_tiles):
                            k_start = kt * TILE_K

                            # Load A tile (cooperative)
                            for load_iter in T.serial((TILE_M * TILE_K) // THREADS_PER_BLOCK):
                                idx = tid + load_iter * THREADS_PER_BLOCK
                                if idx < TILE_M * TILE_K:
                                    m_local = idx // TILE_K
                                    k_local = idx % TILE_K
                                    m_global = bm * TILE_M + m_local
                                    k_global = k_start + k_local

                                    if m_global < M and k_global < K:
                                        A_shared[m_local, k_local] = A[m_global, k_global]
                                    else:
                                        A_shared[m_local, k_local] = T.float32(0)

                            # Load W tile (with dequant)
                            for load_iter in T.serial((TILE_N * TILE_K) // THREADS_PER_BLOCK):
                                idx = tid + load_iter * THREADS_PER_BLOCK
                                if idx < TILE_N * TILE_K:
                                    n_local = idx // TILE_K
                                    k_local = idx % TILE_K
                                    n_global = bn * TILE_N + n_local
                                    k_global = k_start + k_local

                                    if n_global < N and k_global < K:
                                        # Get packed byte
                                        byte_idx = k_global // 2
                                        is_high = k_global % 2

                                        packed_byte = W_packed[n_global, byte_idx]

                                        # Extract FP4 index using T.if_then_else
                                        fp4_idx = T.if_then_else(
                                            is_high == 0,
                                            packed_byte & T.uint8(0xF),
                                            (packed_byte >> 4) & T.uint8(0xF)
                                        )

                                        w_val = lut[T.Cast("int32", fp4_idx)]

                                        block_idx = k_global // block_size
                                        scale = scales[n_global, block_idx]

                                        W_shared[n_local, k_local] = w_val * scale
                                    else:
                                        W_shared[n_local, k_local] = T.float32(0)

                            T.tvm_storage_sync("shared")

                            # Compute and accumulate directly to output
                            for k in T.serial(TILE_K):
                                for em in T.serial(ELEM_M):
                                    for en in T.serial(ELEM_N):
                                        m_local = ty * ELEM_M + em
                                        n_local = tx * ELEM_N + en
                                        m_global = bm * TILE_M + m_local
                                        n_global = bn * TILE_N + n_local

                                        if m_global < M and n_global < N:
                                            C[m_global, n_global] = C[m_global, n_global] + A_shared[m_local, k] * W_shared[n_local, k]

                            T.tvm_storage_sync("shared")

    return kernel


# ==============================================================================
# Build and Benchmark
# ==============================================================================

def build_kernel(kernel_func, target="cuda -arch=sm_110"):
    """Build TVM kernel."""
    target_obj = tvm.target.Target(target)
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(kernel_func, target=target_obj)
    return mod


def benchmark_w4a16_packed(M=1, N=16384, K=2048, warmup=50, runs=200):
    """Benchmark W4A16 packed kernels (simple and fast versions)."""
    import time

    print(f"\n{'='*70}")
    print(f"W4A16 Packed FP4 GEMM Benchmark")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*70}")

    # Generate test data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float32)
    W_np = np.random.randn(N, K).astype(np.float32)

    # Quantize weight
    print("Quantizing weights...")
    W_packed_np, scales_np = quantize_to_nvfp4_packed(W_np)
    print(f"  Original: {W_np.nbytes / 1e6:.2f} MB")
    print(f"  Packed:   {W_packed_np.nbytes / 1e6:.2f} MB ({W_np.nbytes / W_packed_np.nbytes:.1f}x compression)")

    # Verify dequant
    W_dequant_np = dequantize_nvfp4_packed(W_packed_np, scales_np, K)
    cos_sim = np.dot(W_np.flatten(), W_dequant_np.flatten()) / (
        np.linalg.norm(W_np) * np.linalg.norm(W_dequant_np))
    print(f"  Quantization cosine similarity: {cos_sim:.6f}")

    # Prepare TVM arrays
    device = tvm.runtime.cuda(0)

    A_tvm = tvm.runtime.empty((M, K), dtype="float32", device=device)
    A_tvm.copyfrom(A_np)

    W_packed_tvm = tvm.runtime.empty((N, K // 2), dtype="uint8", device=device)
    W_packed_tvm.copyfrom(W_packed_np)

    scales_tvm = tvm.runtime.empty((N, (K + BLOCK_SIZE - 1) // BLOCK_SIZE), dtype="float32", device=device)
    scales_tvm.copyfrom(scales_np)

    C_tvm = tvm.runtime.empty((M, N), dtype="float32", device=device)

    # Reference
    C_ref = A_np @ W_dequant_np.T

    results = []

    # Test both kernel versions for M=1
    if M == 1:
        kernels_to_test = [
            ("Simple (1 thread/output)", create_w4a16_packed_gemv, "w4a16_packed_gemv"),
            ("Fast (parallel reduction)", create_w4a16_packed_gemv_fast, "w4a16_packed_gemv_fast"),
        ]
    else:
        kernels_to_test = [
            ("Tiled GEMM", create_w4a16_packed_gemm_tiled, "w4a16_packed_gemm_tiled"),
        ]

    for name, kernel_creator, func_name in kernels_to_test:
        print(f"\n--- {name} ---")

        # Build kernel
        print("  Building...")
        try:
            kernel_func = kernel_creator(N, K) if M == 1 else kernel_creator(M, N, K)
            mod = build_kernel(kernel_func)
            func = mod[func_name]
            print("  Build successful!")
        except Exception as e:
            print(f"  Build failed: {e}")
            continue

        # Warmup
        for _ in range(warmup):
            func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
        tvm.runtime.cuda(0).sync()

        # Benchmark
        tvm.runtime.cuda(0).sync()
        start = time.time()
        for _ in range(runs):
            func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
        tvm.runtime.cuda(0).sync()

        avg_ms = (time.time() - start) / runs * 1000

        # Verify
        C_tvm_np = C_tvm.numpy()
        max_diff = np.abs(C_tvm_np - C_ref).max()
        cos_sim_result = np.dot(C_tvm_np.flatten(), C_ref.flatten()) / (
            np.linalg.norm(C_tvm_np) * np.linalg.norm(C_ref) + 1e-8)

        # Metrics
        flops = 2.0 * M * N * K
        tflops = flops / (avg_ms / 1000) / 1e12

        weight_bytes = W_packed_np.nbytes + scales_np.nbytes
        activation_bytes = A_np.nbytes
        output_bytes = M * N * 4
        total_bytes = weight_bytes + activation_bytes + output_bytes
        bandwidth_gbps = total_bytes / (avg_ms / 1000) / 1e9

        print(f"  Time:      {avg_ms:.4f} ms")
        print(f"  TFLOPS:    {tflops:.4f}")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Cos sim:   {cos_sim_result:.6f}")

        results.append({
            "name": name,
            "time_ms": avg_ms,
            "tflops": tflops,
            "correct": cos_sim_result > 0.99
        })

    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")

    TRT_FP8_MS = 0.53
    CUBLAS_BF16_MS = 1.13

    for r in results:
        status = "✅" if r["correct"] else "❌"
        speedup_trt = TRT_FP8_MS / r["time_ms"]
        speedup_bf16 = CUBLAS_BF16_MS / r["time_ms"]
        print(f"  {r['name']:<30} {r['time_ms']:.4f}ms  vs TRT FP8: {speedup_trt:.2f}x  {status}")

    print(f"{'='*70}")

    return results[0]["time_ms"] if results else None


def export_cuda_source(M=1, N=16384, K=2048, output_dir="/tmp/w4a16_packed"):
    """Export kernel to CUDA source."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    if M == 1:
        kernel_func = create_w4a16_packed_gemv(N, K)
        name = "w4a16_packed_gemv"
    else:
        kernel_func = create_w4a16_packed_gemm_tiled(M, N, K)
        name = "w4a16_packed_gemm_tiled"

    mod = build_kernel(kernel_func)

    if hasattr(mod, 'imports_') and len(mod.imports_) > 0:
        cuda_source = mod.imports_[0].inspect_source()
        output_path = os.path.join(output_dir, f"{name}.cu")
        with open(output_path, "w") as f:
            f.write(cuda_source)
        print(f"Exported: {output_path}")
        return output_path
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="W4A16 Packed FP4 GEMM Benchmark")
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--N", type=int, default=16384, help="MLP dim (default: Pi0.5 mlp_dim)")
    parser.add_argument("--K", type=int, default=2048, help="Hidden dim (default: Pi0.5 hidden_size)")
    parser.add_argument("--export", action="store_true", help="Export CUDA source")
    parser.add_argument("--output-dir", type=str, default="/tmp/w4a16_packed")

    args = parser.parse_args()

    if args.export:
        export_cuda_source(args.M, args.N, args.K, args.output_dir)
    else:
        benchmark_w4a16_packed(args.M, args.N, args.K)
