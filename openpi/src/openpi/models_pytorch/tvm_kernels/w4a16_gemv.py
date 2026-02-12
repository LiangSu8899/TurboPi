"""
TVM TensorIR: W4A16 GEMV Kernel

Fast GEMV kernel for packed nvFP4 weights with FP32 activation.
Optimized for Thor SM110 with K-dimension tiling and parallel reduction.

Performance on Thor SM110 (batch=1):
- gate/up_proj (N=16384, K=2048): 0.224ms (2.37x vs TRT FP8)
- down_proj (N=2048, K=16384): 0.202ms (2.62x vs TRT FP8)

Author: Claude Code
Date: 2026-02-10
"""

import tvm
from tvm import te, tir
from tvm.script import tir as T
import numpy as np
from typing import Tuple, Optional, Dict, Any
import time

# Constants
BLOCK_SIZE = 32  # nvFP4 scaling block size

# nvFP4 E2M1 lookup table
NVFP4_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # positive
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0  # negative
], dtype=np.float32)

# Module cache
_module_cache: Dict[str, Any] = {}


def quantize_to_nvfp4_packed(weight: np.ndarray, block_size: int = BLOCK_SIZE):
    """Quantize weight to packed nvFP4 format."""
    N, K = weight.shape
    num_blocks = (K + block_size - 1) // block_size

    # Compute scales per block
    scales = np.zeros((N, num_blocks), dtype=np.float32)
    for n in range(N):
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, K)
            block_max = np.abs(weight[n, start:end]).max()
            scales[n, b] = block_max / 6.0 if block_max > 0 else 1.0

    # Quantize to nvFP4 indices
    W_quant = np.zeros((N, K), dtype=np.int32)
    for n in range(N):
        for k in range(K):
            block_idx = k // block_size
            scaled_val = weight[n, k] / scales[n, block_idx]

            # Find closest nvFP4 value
            best_idx = 0
            best_diff = abs(scaled_val - NVFP4_LUT[0])
            for i in range(1, 16):
                diff = abs(scaled_val - NVFP4_LUT[i])
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            W_quant[n, k] = best_idx

    # Pack to uint8 (2 FP4 per byte)
    K_packed = K // 2
    W_packed = np.zeros((N, K_packed), dtype=np.uint8)
    for n in range(N):
        for k in range(0, K, 2):
            low = W_quant[n, k] & 0xF
            high = W_quant[n, k + 1] & 0xF
            W_packed[n, k // 2] = low | (high << 4)

    return W_packed, scales


# ==============================================================================
# W4A16 GEMV Fast Kernel
# ==============================================================================

def create_w4a16_gemv_fast(N: int, K: int, block_size: int = BLOCK_SIZE):
    """
    Create optimized W4A16 GEMV kernel with K-dimension tiling and parallel reduction.

    Computes: C[1, N] = A[1, K] @ W[N, K].T

    Where W is packed as uint8 (2 FP4 per byte) with per-block scales.

    Args:
        N: Output dimension
        K: Input dimension (reduction dimension)
        block_size: nvFP4 scaling block size

    Returns:
        TVM PrimFunc
    """
    num_blocks_k = (K + block_size - 1) // block_size
    K_packed = K // 2

    # Thread configuration
    REDUCE_THREADS = 64   # Threads for K-dimension reduction per output
    OUTPUTS_PER_BLOCK = 4 # Outputs computed per thread block
    THREADS_PER_BLOCK = REDUCE_THREADS * OUTPUTS_PER_BLOCK  # 256

    # K tiling
    TILE_K = min(K, 2048)  # 2048 floats = 8KB shared memory
    num_thread_blocks = (N + OUTPUTS_PER_BLOCK - 1) // OUTPUTS_PER_BLOCK
    num_k_tiles = (K + TILE_K - 1) // TILE_K
    K_PER_THREAD_PER_TILE = (TILE_K + REDUCE_THREADS - 1) // REDUCE_THREADS

    @T.prim_func
    def kernel(
        A: T.Buffer((1, K), "float32"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({
            "global_symbol": "w4a16_gemv_fast",
            "tir.noalias": True,
        })

        # Shared memory
        lut = T.alloc_buffer((16,), "float32", scope="shared")
        A_shared = T.alloc_buffer((TILE_K,), "float32", scope="shared")
        partial_sums = T.alloc_buffer((OUTPUTS_PER_BLOCK, REDUCE_THREADS), "float32", scope="shared")

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

                # Thread assignment
                output_idx = tx // REDUCE_THREADS  # 0..OUTPUTS_PER_BLOCK-1
                reduce_idx = tx % REDUCE_THREADS   # 0..REDUCE_THREADS-1
                j = bx * OUTPUTS_PER_BLOCK + output_idx  # Global output column

                # Initialize partial sum in register
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

                                    # Extract FP4 index
                                    fp4_idx = T.if_then_else(
                                        is_high == 0,
                                        packed_byte & T.uint8(0xF),
                                        (packed_byte >> 4) & T.uint8(0xF)
                                    )

                                    # Lookup and dequant
                                    w_val = lut[T.Cast("int32", fp4_idx)]
                                    block_idx = k_global // block_size
                                    scale = scales[j, block_idx]
                                    w_dequant = w_val * scale

                                    # Accumulate
                                    a_val = A_shared[k_local]
                                    partial_sums[output_idx, reduce_idx] = (
                                        partial_sums[output_idx, reduce_idx] + a_val * w_dequant
                                    )

                    T.tvm_storage_sync("shared")

                # Parallel reduction (6 steps for 64 threads)
                if reduce_idx < 32:
                    partial_sums[output_idx, reduce_idx] = (
                        partial_sums[output_idx, reduce_idx] +
                        partial_sums[output_idx, reduce_idx + 32]
                    )
                T.tvm_storage_sync("shared")

                if reduce_idx < 16:
                    partial_sums[output_idx, reduce_idx] = (
                        partial_sums[output_idx, reduce_idx] +
                        partial_sums[output_idx, reduce_idx + 16]
                    )
                T.tvm_storage_sync("shared")

                if reduce_idx < 8:
                    partial_sums[output_idx, reduce_idx] = (
                        partial_sums[output_idx, reduce_idx] +
                        partial_sums[output_idx, reduce_idx + 8]
                    )
                T.tvm_storage_sync("shared")

                if reduce_idx < 4:
                    partial_sums[output_idx, reduce_idx] = (
                        partial_sums[output_idx, reduce_idx] +
                        partial_sums[output_idx, reduce_idx + 4]
                    )
                T.tvm_storage_sync("shared")

                if reduce_idx < 2:
                    partial_sums[output_idx, reduce_idx] = (
                        partial_sums[output_idx, reduce_idx] +
                        partial_sums[output_idx, reduce_idx + 2]
                    )
                T.tvm_storage_sync("shared")

                # Final write (reduce_idx == 0)
                if reduce_idx == 0:
                    if j < N:
                        C[0, j] = (partial_sums[output_idx, 0] +
                                   partial_sums[output_idx, 1])

    return kernel


# ==============================================================================
# W4A16 GEMV Simple Kernel (for verification)
# ==============================================================================

def create_w4a16_gemv_simple(N: int, K: int, block_size: int = BLOCK_SIZE):
    """
    Create simple W4A16 GEMV kernel (1 thread per output).

    Used for verification. Slower but simpler.
    """
    num_blocks_k = (K + block_size - 1) // block_size
    K_packed = K // 2

    THREADS_PER_BLOCK = 256
    num_thread_blocks = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    @T.prim_func
    def kernel(
        A: T.Buffer((1, K), "float32"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({
            "global_symbol": "w4a16_gemv_simple",
            "tir.noalias": True,
        })

        lut = T.alloc_buffer((16,), "float32", scope="shared")

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

                j = bx * THREADS_PER_BLOCK + tx

                if j < N:
                    acc = T.float32(0)

                    for k in T.serial(K):
                        byte_idx = k // 2
                        is_high = k % 2

                        packed_byte = W_packed[j, byte_idx]

                        fp4_idx = T.if_then_else(
                            is_high == 0,
                            packed_byte & T.uint8(0xF),
                            (packed_byte >> 4) & T.uint8(0xF)
                        )

                        w_val = lut[T.Cast("int32", fp4_idx)]
                        block_idx = k // block_size
                        scale = scales[j, block_idx]
                        w_dequant = w_val * scale

                        acc = acc + A[0, k] * w_dequant

                    C[0, j] = acc

    return kernel


# ==============================================================================
# Kernel Build and Cache
# ==============================================================================

def build_kernel(kernel_func, target="cuda -arch=sm_110"):
    """Build TVM kernel."""
    target_obj = tvm.target.Target(target)
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(kernel_func, target=target_obj)
    return mod


def get_gemv_kernel(N: int, K: int, fast: bool = True, target: str = "cuda -arch=sm_110"):
    """
    Get cached GEMV kernel for given dimensions.

    Args:
        N: Output dimension
        K: Input dimension
        fast: Use fast (parallel reduction) or simple (single thread) kernel
        target: TVM target string

    Returns:
        Compiled TVM function
    """
    cache_key = f"gemv_{N}_{K}_{fast}"

    if cache_key in _module_cache:
        return _module_cache[cache_key]

    if fast:
        kernel_func = create_w4a16_gemv_fast(N, K)
        func_name = "w4a16_gemv_fast"
    else:
        kernel_func = create_w4a16_gemv_simple(N, K)
        func_name = "w4a16_gemv_simple"

    mod = build_kernel(kernel_func, target)
    func = mod[func_name]

    _module_cache[cache_key] = func
    return func


# ==============================================================================
# High-level API
# ==============================================================================

def w4a16_gemv(
    A: np.ndarray,
    W_packed: np.ndarray,
    scales: np.ndarray,
    N: int,
    K: int,
    fast: bool = True,
) -> np.ndarray:
    """
    Execute W4A16 GEMV using TVM kernel.

    Args:
        A: [1, K] input activation (float32)
        W_packed: [N, K//2] packed FP4 weights (uint8)
        scales: [N, num_blocks] weight scales (float32)
        N: Output dimension
        K: Input dimension
        fast: Use fast kernel

    Returns:
        [1, N] output (float32)
    """
    func = get_gemv_kernel(N, K, fast)
    device = tvm.runtime.cuda(0)

    # Prepare TVM arrays
    A_tvm = tvm.runtime.empty((1, K), dtype="float32", device=device)
    A_tvm.copyfrom(A)

    W_packed_tvm = tvm.runtime.empty((N, K // 2), dtype="uint8", device=device)
    W_packed_tvm.copyfrom(W_packed)

    scales_tvm = tvm.runtime.empty(scales.shape, dtype="float32", device=device)
    scales_tvm.copyfrom(scales)

    out_tvm = tvm.runtime.empty((1, N), dtype="float32", device=device)

    # Execute
    func(A_tvm, W_packed_tvm, scales_tvm, out_tvm)
    device.sync()

    return out_tvm.numpy()


# ==============================================================================
# Benchmark
# ==============================================================================

def benchmark_gemv(N=16384, K=2048, warmup=50, runs=200):
    """Benchmark W4A16 GEMV kernel."""
    print(f"\n{'='*70}")
    print(f"W4A16 GEMV Benchmark")
    print(f"Shape: M=1, N={N}, K={K}")
    print(f"{'='*70}")

    # Generate test data
    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float32)
    W_np = np.random.randn(N, K).astype(np.float32) * 0.1

    # Quantize
    W_packed, scales = quantize_to_nvfp4_packed(W_np)

    num_blocks_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    print(f"\nWeight memory:")
    print(f"  Original (BF16): {N * K * 2 / 1e6:.2f} MB")
    print(f"  Packed (FP4):    {W_packed.nbytes / 1e6:.2f} MB ({N * K * 2 / W_packed.nbytes:.1f}x compression)")

    # Reference computation
    def dequant_and_compute(A, W_packed, scales, K):
        N_out = W_packed.shape[0]
        W_dequant = np.zeros((N_out, K), dtype=np.float32)
        for n in range(N_out):
            for k in range(K):
                byte_idx = k // 2
                is_high = k % 2
                packed = W_packed[n, byte_idx]
                fp4_idx = ((packed >> 4) & 0xF) if is_high else (packed & 0xF)
                w_val = NVFP4_LUT[fp4_idx]
                block_idx = k // BLOCK_SIZE
                W_dequant[n, k] = w_val * scales[n, block_idx]
        return A @ W_dequant.T

    out_ref = dequant_and_compute(A_np, W_packed, scales, K)

    # Prepare TVM arrays
    device = tvm.runtime.cuda(0)

    A_tvm = tvm.runtime.empty((1, K), dtype="float32", device=device)
    A_tvm.copyfrom(A_np)

    W_packed_tvm = tvm.runtime.empty((N, K // 2), dtype="uint8", device=device)
    W_packed_tvm.copyfrom(W_packed)

    scales_tvm = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
    scales_tvm.copyfrom(scales)

    out_tvm = tvm.runtime.empty((1, N), dtype="float32", device=device)

    results = []

    # Test fast kernel
    print(f"\n--- Fast Kernel (K-tiling + parallel reduction) ---")
    try:
        kernel_func = create_w4a16_gemv_fast(N, K)
        mod = build_kernel(kernel_func)
        func = mod["w4a16_gemv_fast"]
        print("  Build successful!")

        # Warmup
        for _ in range(warmup):
            func(A_tvm, W_packed_tvm, scales_tvm, out_tvm)
        device.sync()

        # Benchmark
        device.sync()
        start = time.time()
        for _ in range(runs):
            func(A_tvm, W_packed_tvm, scales_tvm, out_tvm)
        device.sync()
        fast_ms = (time.time() - start) / runs * 1000

        # Verify
        out_np = out_tvm.numpy()
        cos_sim = np.dot(out_np.flatten(), out_ref.flatten()) / (
            np.linalg.norm(out_np) * np.linalg.norm(out_ref) + 1e-8)

        print(f"  Time:    {fast_ms:.4f} ms")
        print(f"  Cos sim: {cos_sim:.6f}")

        results.append(("Fast", fast_ms, cos_sim > 0.99))
    except Exception as e:
        print(f"  Failed: {e}")

    # Test simple kernel
    print(f"\n--- Simple Kernel (1 thread/output) ---")
    try:
        kernel_func = create_w4a16_gemv_simple(N, K)
        mod = build_kernel(kernel_func)
        func = mod["w4a16_gemv_simple"]
        print("  Build successful!")

        # Warmup
        for _ in range(warmup):
            func(A_tvm, W_packed_tvm, scales_tvm, out_tvm)
        device.sync()

        # Benchmark
        device.sync()
        start = time.time()
        for _ in range(runs):
            func(A_tvm, W_packed_tvm, scales_tvm, out_tvm)
        device.sync()
        simple_ms = (time.time() - start) / runs * 1000

        # Verify
        out_np = out_tvm.numpy()
        cos_sim = np.dot(out_np.flatten(), out_ref.flatten()) / (
            np.linalg.norm(out_np) * np.linalg.norm(out_ref) + 1e-8)

        print(f"  Time:    {simple_ms:.4f} ms")
        print(f"  Cos sim: {cos_sim:.6f}")

        results.append(("Simple", simple_ms, cos_sim > 0.99))
    except Exception as e:
        print(f"  Failed: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")

    TRT_FP8_MS = 0.53  # Baseline from measurements

    for name, time_ms, correct in results:
        status = "OK" if correct else "FAIL"
        speedup = TRT_FP8_MS / time_ms
        print(f"  {name:<10} {time_ms:.4f}ms  vs TRT FP8: {speedup:.2f}x  [{status}]")

    print(f"\nBaseline: TRT FP8 = {TRT_FP8_MS:.2f}ms")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=16384)
    parser.add_argument("--K", type=int, default=2048)
    args = parser.parse_args()

    # Run both dimensions
    benchmark_gemv(N=16384, K=2048)   # gate/up_proj
    benchmark_gemv(N=2048, K=16384)   # down_proj
