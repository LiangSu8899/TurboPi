#!/usr/bin/env python3
"""
W4A16 GEMM - Shared Memory Tiled Version

Strategy:
1. Tile M, N, K dimensions
2. Load A tile to shared memory (cooperative)
3. Load W_packed tile to shared memory
4. Dequant W tile IN shared memory (per-tile, not global)
5. Compute tile GEMM from shared memory
6. Accumulate and write back

This avoids writing dequantized weights to GLOBAL memory.
The dequantized FP16 weights only exist in shared memory for each tile.

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


QUANT_BLOCK = 32

# Tile sizes
BLOCK_M = 64    # Tile rows
BLOCK_N = 64    # Tile cols
BLOCK_K = 32    # Tile depth (must be <= QUANT_BLOCK for single scale)

# Thread config: 16x16 = 256 threads
THREADS_M = 16
THREADS_N = 16


def create_w4a16_smem_kernel(M, N, K):
    """Create W4A16 kernel with shared memory tiling."""
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    num_tiles_m = (M + BLOCK_M - 1) // BLOCK_M
    num_tiles_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles_k = (K + BLOCK_K - 1) // BLOCK_K

    # Each thread computes this many elements
    THREAD_M = BLOCK_M // THREADS_M  # 4
    THREAD_N = BLOCK_N // THREADS_N  # 4

    @T.prim_func
    def w4a16_smem(
        A: T.Buffer((M, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_smem", "tir.noalias": True})

        # Shared memory
        A_smem = T.alloc_buffer((BLOCK_M, BLOCK_K), "float16", scope="shared")
        W_smem = T.alloc_buffer((BLOCK_N, BLOCK_K), "float16", scope="shared")

        # Block loops
        for bm in T.thread_binding(num_tiles_m, thread="blockIdx.y"):
            for bn in T.thread_binding(num_tiles_n, thread="blockIdx.x"):
                # Thread loops
                for ty in T.thread_binding(THREADS_M, thread="threadIdx.y"):
                    for tx in T.thread_binding(THREADS_N, thread="threadIdx.x"):
                        # Linear thread ID
                        tid = ty * THREADS_N + tx
                        NTHREADS = THREADS_M * THREADS_N

                        # Initialize accumulators for this thread's output elements
                        # Each thread computes THREAD_M x THREAD_N elements
                        for ii in range(THREAD_M):
                            for jj in range(THREAD_N):
                                row = bm * BLOCK_M + ty * THREAD_M + ii
                                col = bn * BLOCK_N + tx * THREAD_N + jj
                                if row < M and col < N:
                                    C[row, col] = T.float32(0)

                        # Loop over K tiles
                        for bk in range(num_tiles_k):
                            k_start = bk * BLOCK_K

                            # === Load A tile to shared memory ===
                            # Elements to load: BLOCK_M * BLOCK_K = 64 * 32 = 2048
                            # Threads: 256
                            # Each thread loads: 2048 / 256 = 8 elements
                            for load_iter in range(8):
                                flat_idx = tid + load_iter * NTHREADS
                                sm = flat_idx // BLOCK_K  # row in shared
                                sk = flat_idx % BLOCK_K   # col in shared
                                gm = bm * BLOCK_M + sm    # global row
                                gk = k_start + sk         # global k

                                if gm < M and gk < K and sm < BLOCK_M:
                                    A_smem[sm, sk] = A[gm, gk]
                                elif sm < BLOCK_M and sk < BLOCK_K:
                                    A_smem[sm, sk] = T.float16(0)

                            # === Load and dequant W tile to shared memory ===
                            # Elements to load (dequanted): BLOCK_N * BLOCK_K = 64 * 32 = 2048
                            # Each thread loads and dequants 8 elements
                            scale_block_idx = k_start // QUANT_BLOCK

                            for load_iter in range(8):
                                flat_idx = tid + load_iter * NTHREADS
                                sn = flat_idx // BLOCK_K  # row in shared (N dim)
                                sk = flat_idx % BLOCK_K   # col in shared (K dim)
                                gn = bn * BLOCK_N + sn    # global N
                                gk = k_start + sk         # global K

                                if gn < N and gk < K and sn < BLOCK_N and sk < BLOCK_K:
                                    # Dequant on-the-fly
                                    byte_idx = gk // 2
                                    is_high = gk % 2
                                    packed = W_packed[gn, byte_idx]

                                    int4_val = T.if_then_else(
                                        is_high == 0,
                                        packed & T.uint8(0xF),
                                        (packed >> 4) & T.uint8(0xF)
                                    )
                                    signed_val = T.Cast("float16", int4_val) - T.float16(8.0)
                                    scale = scales[gn, scale_block_idx] if scale_block_idx < num_scale_blocks else T.float16(1)
                                    W_smem[sn, sk] = signed_val * scale
                                elif sn < BLOCK_N and sk < BLOCK_K:
                                    W_smem[sn, sk] = T.float16(0)

                            # Sync before compute
                            T.tvm_storage_sync("shared")

                            # === Compute tile GEMM ===
                            for ii in range(THREAD_M):
                                for jj in range(THREAD_N):
                                    row = bm * BLOCK_M + ty * THREAD_M + ii
                                    col = bn * BLOCK_N + tx * THREAD_N + jj
                                    sm = ty * THREAD_M + ii  # shared A row
                                    sn = tx * THREAD_N + jj  # shared W row

                                    if row < M and col < N:
                                        # Reduction over K tile
                                        for sk in range(BLOCK_K):
                                            gk = k_start + sk
                                            if gk < K and sm < BLOCK_M and sn < BLOCK_N:
                                                a_val = A_smem[sm, sk]
                                                w_val = W_smem[sn, sk]
                                                C[row, col] = C[row, col] + T.Cast("float32", a_val * w_val)

                            # Sync before next K tile
                            T.tvm_storage_sync("shared")

    return w4a16_smem


def quantize_int4(weight, block_size=QUANT_BLOCK):
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


def test_smem_kernel(M=128, N=256, K=128):
    """Test shared memory tiled kernel."""
    print(f"\n{'='*60}")
    print(f"W4A16 Shared Memory Tiled Kernel")
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
    kernel = create_w4a16_smem_kernel(M, N, K)
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
        return None

    # Run
    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = runtime.empty((M, N), "float32", device)

    func = lib["w4a16_smem"]
    func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    # Verify
    C_result = C_tvm.numpy()
    max_diff = np.abs(C_result - C_ref).max()

    if np.isnan(C_result).any():
        print("WARNING: Output contains NaN!")
        cos_sim = 0
    else:
        cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
            np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)

    print(f"\nResults:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Cos sim:  {cos_sim:.6f}")
    print(f"  Status: {'PASS' if cos_sim > 0.99 else 'FAIL'}")

    if cos_sim <= 0.99:
        print(f"\n  Ref sample:\n{C_ref[:2, :4]}")
        print(f"  TVM sample:\n{C_result[:2, :4]}")
        return None

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


def test_full_size(M=712, N=16384, K=2048):
    """Test with full problem size."""
    print(f"\n{'='*60}")
    print(f"W4A16 SMEM Kernel - Full Size")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    # Data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    print("Quantizing...")
    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Build
    print("Building kernel...")
    kernel = create_w4a16_smem_kernel(M, N, K)
    mod = tvm.IRModule({"main": kernel})
    target = tvm.target.Target("cuda -arch=sm_110")

    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.build(mod, target=target)
    print("Build successful!")

    # Run
    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = runtime.empty((M, N), "float32", device)

    func = lib["w4a16_smem"]
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

    # Benchmark
    warmup = 10
    runs = 50

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

    BF16_MS = 0.42  # From benchmark
    print(f"\n  vs BF16 baseline ({BF16_MS}ms): {BF16_MS / avg_ms:.2f}x")

    return avg_ms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    test_smem_kernel()

    if args.full:
        test_full_size()
