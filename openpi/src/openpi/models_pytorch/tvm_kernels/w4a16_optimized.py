#!/usr/bin/env python3
"""
W4A16 GEMM - Optimized with Shared Memory and Register-level Dequant

Strategy:
1. Tile computation with shared memory
2. Load A tiles to shared memory (cooperative loading)
3. Load W_packed tiles to shared memory
4. Dequant W in registers on-the-fly
5. Accumulate using FMA operations
6. Use proper memory coalescing

This is an intermediate step before full WMMA tensorization.

Author: Claude Code
Date: 2026-02-11
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import tir, runtime
from tvm.script import tir as T
import numpy as np
import time


# ==============================================================================
# Parameters
# ==============================================================================

QUANT_BLOCK = 32

# Tile sizes - optimized for Orin/Thor
BLOCK_M = 128  # Tile rows
BLOCK_N = 128  # Tile columns
BLOCK_K = 32   # Tile depth

# Thread block config
THREADS_PER_BLOCK_M = 16
THREADS_PER_BLOCK_N = 16
THREADS_PER_BLOCK = THREADS_PER_BLOCK_M * THREADS_PER_BLOCK_N

# Each thread computes BLOCK_M/THREADS_PER_BLOCK_M x BLOCK_N/THREADS_PER_BLOCK_N elements
THREAD_TILE_M = BLOCK_M // THREADS_PER_BLOCK_M  # 8
THREAD_TILE_N = BLOCK_N // THREADS_PER_BLOCK_N  # 8


def create_w4a16_optimized_kernel(M, N, K):
    """Create optimized W4A16 GEMM kernel with shared memory."""
    num_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    num_tiles_m = (M + BLOCK_M - 1) // BLOCK_M
    num_tiles_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles_k = (K + BLOCK_K - 1) // BLOCK_K

    @T.prim_func
    def w4a16_optimized(
        A: T.Buffer((M, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_blocks), "float16"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_optimized", "tir.noalias": True})

        # Allocate shared memory
        A_shared = T.alloc_buffer((BLOCK_M, BLOCK_K), "float16", scope="shared")
        W_shared_packed = T.alloc_buffer((BLOCK_N, BLOCK_K // 2), "uint8", scope="shared")
        scales_shared = T.alloc_buffer((BLOCK_N,), "float16", scope="shared")

        for tile_m in T.thread_binding(num_tiles_m, thread="blockIdx.y"):
            for tile_n in T.thread_binding(num_tiles_n, thread="blockIdx.x"):
                for thread_m in T.thread_binding(THREADS_PER_BLOCK_M, thread="threadIdx.y"):
                    for thread_n in T.thread_binding(THREADS_PER_BLOCK_N, thread="threadIdx.x"):
                        # Thread-local accumulator array
                        # Use direct output writes instead of local array

                        # Initialize output tile to zero
                        for i in range(THREAD_TILE_M):
                            for j in range(THREAD_TILE_N):
                                row = tile_m * BLOCK_M + thread_m * THREAD_TILE_M + i
                                col = tile_n * BLOCK_N + thread_n * THREAD_TILE_N + j
                                if row < M and col < N:
                                    C[row, col] = T.float32(0)

                        # Iterate over K tiles
                        for tile_k in range(num_tiles_k):
                            k_start = tile_k * BLOCK_K

                            # Load A tile to shared memory (cooperative)
                            # Each thread loads BLOCK_M * BLOCK_K / THREADS elements
                            for load_idx in range((BLOCK_M * BLOCK_K) // THREADS_PER_BLOCK):
                                flat_idx = (thread_m * THREADS_PER_BLOCK_N + thread_n) + load_idx * THREADS_PER_BLOCK
                                sm = flat_idx // BLOCK_K
                                sk = flat_idx % BLOCK_K
                                gm = tile_m * BLOCK_M + sm
                                gk = k_start + sk
                                if gm < M and gk < K:
                                    A_shared[sm, sk] = A[gm, gk]
                                else:
                                    A_shared[sm, sk] = T.float16(0)

                            # Load W_packed tile to shared memory
                            for load_idx in range((BLOCK_N * (BLOCK_K // 2)) // THREADS_PER_BLOCK):
                                flat_idx = (thread_m * THREADS_PER_BLOCK_N + thread_n) + load_idx * THREADS_PER_BLOCK
                                sn = flat_idx // (BLOCK_K // 2)
                                sk = flat_idx % (BLOCK_K // 2)
                                gn = tile_n * BLOCK_N + sn
                                gk = (k_start // 2) + sk
                                if sn < BLOCK_N and gn < N and gk < K_packed:
                                    W_shared_packed[sn, sk] = W_packed[gn, gk]
                                else:
                                    W_shared_packed[sn, sk] = T.uint8(0x88)  # zero in signed

                            # Load scales for this K block
                            block_idx = k_start // QUANT_BLOCK
                            for load_idx in range(BLOCK_N // THREADS_PER_BLOCK):
                                sn = (thread_m * THREADS_PER_BLOCK_N + thread_n) + load_idx * THREADS_PER_BLOCK
                                if sn < BLOCK_N:
                                    gn = tile_n * BLOCK_N + sn
                                    if gn < N and block_idx < num_blocks:
                                        scales_shared[sn] = scales[gn, block_idx]
                                    else:
                                        scales_shared[sn] = T.float16(1)

                            # Sync threads after loading shared memory
                            T.tvm_storage_sync("shared")

                            # Compute partial result for this K tile
                            for i in range(THREAD_TILE_M):
                                for j in range(THREAD_TILE_N):
                                    row = tile_m * BLOCK_M + thread_m * THREAD_TILE_M + i
                                    col = tile_n * BLOCK_N + thread_n * THREAD_TILE_N + j
                                    sm = thread_m * THREAD_TILE_M + i
                                    sn = thread_n * THREAD_TILE_N + j

                                    if row < M and col < N:
                                        for k in range(BLOCK_K):
                                            gk = k_start + k
                                            if gk < K:
                                                # Dequant on-the-fly in registers
                                                byte_idx = k // 2
                                                is_high = k % 2
                                                packed = W_shared_packed[sn, byte_idx]

                                                int4_val = T.if_then_else(
                                                    is_high == 0,
                                                    packed & T.uint8(0xF),
                                                    (packed >> 4) & T.uint8(0xF)
                                                )
                                                signed_val = T.Cast("float16", int4_val) - T.float16(8.0)
                                                w = signed_val * scales_shared[sn]

                                                # FMA
                                                a_val = A_shared[sm, k]
                                                C[row, col] = C[row, col] + T.Cast("float32", a_val * w)

                            # Sync before next tile load
                            T.tvm_storage_sync("shared")

    return w4a16_optimized


# ==============================================================================
# Quantization Utilities
# ==============================================================================

def quantize_int4(weight, block_size=QUANT_BLOCK):
    """Quantize to INT4."""
    N, K = weight.shape
    num_blocks = (K + block_size - 1) // block_size
    K_packed = K // 2
    W_packed = np.zeros((N, K_packed), dtype=np.uint8)
    scales = np.zeros((N, num_blocks), dtype=np.float16)

    for n in range(N):
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, K)
            block = weight[n, start:end]
            max_abs = np.max(np.abs(block))
            scale = max_abs / 7.0 if max_abs > 0 else 1.0
            scales[n, b] = scale
            for k in range(start, end):
                val = block[k - start] / scale if scale > 0 else 0
                quantized = int(np.clip(np.round(val + 8), 0, 15))
                byte_idx = k // 2
                if k % 2 == 0:
                    W_packed[n, byte_idx] = (W_packed[n, byte_idx] & 0xF0) | quantized
                else:
                    W_packed[n, byte_idx] = (W_packed[n, byte_idx] & 0x0F) | (quantized << 4)

    return W_packed, scales


def dequant_int4(W_packed, scales, K, block_size=QUANT_BLOCK):
    """Dequantize INT4."""
    N = W_packed.shape[0]
    W = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            byte_idx = k // 2
            packed = W_packed[n, byte_idx]
            int4_val = (packed & 0xF) if k % 2 == 0 else ((packed >> 4) & 0xF)
            block_idx = k // block_size
            W[n, k] = (int4_val - 8) * scales[n, block_idx]
    return W


# ==============================================================================
# Test
# ==============================================================================

def test_optimized_kernel(M=256, N=512, K=256):
    """Test optimized W4A16 kernel."""
    print(f"\n{'='*60}")
    print(f"W4A16 Optimized Kernel Test")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"Tiles: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    print(f"{'='*60}")

    # Data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Build
    print("Building kernel...")
    kernel = create_w4a16_optimized_kernel(M, N, K)
    mod = tvm.IRModule({"main": kernel})
    target = tvm.target.Target("cuda -arch=sm_110")

    try:
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(mod, target=target)
        print("Build successful!")
    except Exception as e:
        print(f"Build failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run
    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = runtime.empty((M, N), "float32", device)

    func = lib["w4a16_optimized"]
    func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    # Verify
    C_result = C_tvm.numpy()
    max_diff = np.abs(C_result - C_ref).max()
    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)

    print(f"\nResults:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Cos sim:  {cos_sim:.6f}")
    print(f"  Status: {'PASS' if cos_sim > 0.99 else 'FAIL'}")

    if cos_sim <= 0.99:
        print(f"\n  Ref sample:\n{C_ref[:2, :4]}")
        print(f"  TVM sample:\n{C_result[:2, :4]}")
        return

    # Benchmark
    warmup = 20
    runs = 100

    for _ in range(warmup):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12

    print(f"\n  Time:    {avg_ms:.4f} ms")
    print(f"  TFLOPS:  {tflops:.4f}")

    return avg_ms


if __name__ == "__main__":
    # Test with small size first
    test_optimized_kernel(256, 512, 256)
