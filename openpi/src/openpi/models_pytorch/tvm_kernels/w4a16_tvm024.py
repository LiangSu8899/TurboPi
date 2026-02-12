#!/usr/bin/env python3
"""
W4A16 GEMM for TVM 0.24 (New API)

Uses TIR Script with dlight GPU scheduling.

Author: Claude Code
Date: 2026-02-11
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import tir
from tvm.script import tir as T
import numpy as np
import time


# ==============================================================================
# Parameters
# ==============================================================================

QUANT_BLOCK = 32


# ==============================================================================
# W4A16 Kernel
# ==============================================================================

def create_w4a16_kernel(M, N, K):
    """Create W4A16 GEMM kernel using TIR Script with explicit GPU binding."""
    num_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    # Tile sizes
    BLOCK_M = 32
    BLOCK_N = 32

    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N

    @T.prim_func
    def w4a16_gemm(
        A: T.Buffer((M, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_blocks), "float16"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_gemm", "tir.noalias": True})

        # Block and thread binding for CUDA
        for bm in T.thread_binding(num_blocks_m, thread="blockIdx.x"):
            for bn in T.thread_binding(num_blocks_n, thread="blockIdx.y"):
                for tm in T.thread_binding(BLOCK_M, thread="threadIdx.x"):
                    for tn in T.thread_binding(BLOCK_N, thread="threadIdx.y"):
                        m = bm * BLOCK_M + tm
                        n = bn * BLOCK_N + tn

                        if m < M and n < N:
                            # Initialize output
                            C[m, n] = T.float32(0)

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

                                # Accumulate directly to output buffer
                                C[m, n] = C[m, n] + T.Cast("float32", A[m, k] * w)

    return w4a16_gemm


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
# Build and Test
# ==============================================================================

def test_w4a16_small(M=64, N=256, K=128):
    """Test W4A16 with small size."""
    print(f"\n{'='*60}")
    print(f"W4A16 GEMM Test (TVM 0.24)")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    # Generate data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Create kernel
    print("Creating kernel...")
    kernel = create_w4a16_kernel(M, N, K)

    # Build
    print("Building...")
    try:
        from tvm import dlight as dl

        mod = tvm.IRModule({"main": kernel})
        target = tvm.target.Target("cuda -arch=sm_110")

        with tvm.transform.PassContext(opt_level=3):
            with target:
                mod = dl.ApplyDefaultSchedule(
                    dl.gpu.Fallback(),
                )(mod)

        lib = tvm.build(mod, target=target)
        print("dlight build successful!")

    except Exception as e:
        print(f"dlight failed: {e}, trying direct build...")
        target = tvm.target.Target("cuda -arch=sm_110")
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(kernel, target=target)
        print("Direct build successful!")

    # Run
    device = tvm.runtime.cuda(0)
    A_tvm = tvm.runtime.empty(A_np.shape, A_np.dtype.name, device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = tvm.runtime.empty(W_packed_np.shape, W_packed_np.dtype.name, device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = tvm.runtime.empty(scales_np.shape, scales_np.dtype.name, device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = tvm.runtime.empty((M, N), "float32", device)

    func = lib["w4a16_gemm"]
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

    avg_us = (time.time() - start) / runs * 1e6
    print(f"\n  Time: {avg_us:.2f} us")

    return avg_us


def test_w4a16_full(M=712, N=16384, K=2048):
    """Test W4A16 with full size."""
    print(f"\n{'='*60}")
    print(f"W4A16 GEMM Full Test (TVM 0.24)")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    # Generate data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    print("Quantizing weights...")
    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    print(f"  Original: {W_np.nbytes / 1e6:.2f} MB")
    print(f"  Packed:   {(W_packed_np.nbytes + scales_np.nbytes) / 1e6:.2f} MB")

    # Create kernel
    print("\nCreating kernel...")
    kernel = create_w4a16_kernel(M, N, K)

    # Build with dlight
    print("Building with dlight...")
    try:
        from tvm import dlight as dl

        mod = tvm.IRModule({"main": kernel})
        target = tvm.target.Target("cuda -arch=sm_110")

        with tvm.transform.PassContext(opt_level=3):
            with target:
                mod = dl.ApplyDefaultSchedule(
                    dl.gpu.Fallback(),
                )(mod)

        lib = tvm.build(mod, target=target)
        print("Build successful!")

    except Exception as e:
        print(f"Build failed: {e}")
        return None

    # Run
    device = tvm.runtime.cuda(0)
    A_tvm = tvm.runtime.empty(A_np.shape, A_np.dtype.name, device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = tvm.runtime.empty(W_packed_np.shape, W_packed_np.dtype.name, device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = tvm.runtime.empty(scales_np.shape, scales_np.dtype.name, device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = tvm.runtime.empty((M, N), "float32", device)

    func = lib["w4a16_gemm"]
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

    BF16_MS = 0.58
    print(f"\n  vs BF16 baseline ({BF16_MS}ms): {BF16_MS / avg_ms:.2f}x")

    return avg_ms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Run small test only")
    args = parser.parse_args()

    if args.small:
        test_w4a16_small()
    else:
        test_w4a16_small()
        print("\n" + "="*60)
        print("Running full size test...")
        test_w4a16_full()
