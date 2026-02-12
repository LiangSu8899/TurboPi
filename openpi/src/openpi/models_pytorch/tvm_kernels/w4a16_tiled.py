#!/usr/bin/env python3
"""
W4A16 GEMM - Tiled version without shared memory (for correctness baseline)

This version tiles the K dimension but doesn't use shared memory.
It serves as a correctness baseline before adding shared memory optimizations.

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


def create_w4a16_tiled_kernel(M, N, K, TILE_M=32, TILE_N=32):
    """Create tiled W4A16 kernel."""
    num_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_tiles_m = (M + TILE_M - 1) // TILE_M
    num_tiles_n = (N + TILE_N - 1) // TILE_N

    @T.prim_func
    def w4a16_tiled(
        A: T.Buffer((M, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_blocks), "float16"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_tiled", "tir.noalias": True})

        for tile_m in T.thread_binding(num_tiles_m, thread="blockIdx.y"):
            for tile_n in T.thread_binding(num_tiles_n, thread="blockIdx.x"):
                for tm in T.thread_binding(TILE_M, thread="threadIdx.y"):
                    for tn in T.thread_binding(TILE_N, thread="threadIdx.x"):
                        m = tile_m * TILE_M + tm
                        n = tile_n * TILE_N + tn

                        if m < M and n < N:
                            # Initialize
                            C[m, n] = T.float32(0)

                            # Accumulate over K
                            for k in range(K):
                                # Dequant
                                byte_idx = k // 2
                                is_high = k % 2
                                packed = W_packed[n, byte_idx]

                                int4_val = T.if_then_else(
                                    is_high == 0,
                                    packed & T.uint8(0xF),
                                    (packed >> 4) & T.uint8(0xF)
                                )
                                signed_val = T.Cast("float16", int4_val) - T.float16(8.0)
                                block_idx = k // QUANT_BLOCK
                                scale = scales[n, block_idx]
                                w = signed_val * scale

                                # Accumulate
                                C[m, n] = C[m, n] + T.Cast("float32", A[m, k] * w)

    return w4a16_tiled


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


def test_tiled_kernel(M=128, N=256, K=128):
    """Test tiled W4A16 kernel."""
    print(f"\n{'='*60}")
    print(f"W4A16 Tiled Kernel Test")
    print(f"Shape: M={M}, N={N}, K={K}")
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
    kernel = create_w4a16_tiled_kernel(M, N, K)
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

    func = lib["w4a16_tiled"]
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
    print(f"W4A16 Tiled Kernel - Full Size")
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
    kernel = create_w4a16_tiled_kernel(M, N, K)
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

    func = lib["w4a16_tiled"]
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

    BF16_MS = 0.58
    print(f"\n  vs BF16 baseline ({BF16_MS}ms): {BF16_MS / avg_ms:.2f}x")

    return avg_ms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    test_tiled_kernel()

    if args.full:
        test_full_size()
