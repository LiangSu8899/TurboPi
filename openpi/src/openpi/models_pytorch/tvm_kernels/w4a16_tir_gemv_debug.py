#!/usr/bin/env python3
"""
W4A16 GEMV Debug - Check what's happening in the kernel.
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import tir, runtime
from tvm.script import tir as T
import numpy as np


QUANT_BLOCK = 32


def create_simple_gemv(N, K, THREADS=256):
    """Simplest possible W4A16 GEMV."""
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid
                if n < N:
                    # Initialize output to 0
                    C[0, n] = T.float32(0)

                    # Accumulate over K
                    for k in range(K):
                        byte_idx = k // 2
                        is_high = k % 2
                        packed = W_packed[n, byte_idx]

                        # Extract INT4
                        int4_val = T.if_then_else(
                            is_high == 0,
                            packed & T.uint8(0xF),
                            (packed >> 4) & T.uint8(0xF)
                        )

                        # Dequant
                        signed_val = T.Cast("float16", int4_val) - T.float16(8.0)
                        scale_idx = k // QUANT_BLOCK
                        scale = scales[n, scale_idx]
                        w = signed_val * scale

                        # Accumulate
                        a = A[0, k]
                        C[0, n] = C[0, n] + T.Cast("float32", a * w)

    return gemv


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


def test_small(N=64, K=64):
    """Test with small size for debugging."""
    print(f"\n{'='*60}")
    print(f"Debug W4A16 GEMV")
    print(f"N={N}, K={K}")
    print(f"{'='*60}")

    # Data
    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    print(f"\nA shape: {A_np.shape}, dtype: {A_np.dtype}")
    print(f"W_packed shape: {W_packed_np.shape}, dtype: {W_packed_np.dtype}")
    print(f"scales shape: {scales_np.shape}, dtype: {scales_np.dtype}")
    print(f"C_ref shape: {C_ref.shape}")

    print(f"\nA sample: {A_np[0, :4]}")
    print(f"W sample: {W_np[:2, :4]}")
    print(f"W_dequant sample: {W_dequant_np[:2, :4]}")
    print(f"C_ref sample: {C_ref[0, :4]}")

    # Build kernel
    print("\nBuilding kernel...")
    kernel = create_simple_gemv(N, K, THREADS=64)
    mod = tvm.IRModule({"main": kernel})
    target = tvm.target.Target("cuda -arch=sm_110")

    print("\nKernel IR:")
    print(mod.script()[:2000])

    try:
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(mod, target=target)
        print("\nBuild successful!")
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
    C_tvm = runtime.empty((1, N), "float32", device)

    # Initialize to -999 to check if kernel writes
    C_init = np.full((1, N), -999.0, dtype=np.float32)
    C_tvm.copyfrom(C_init)

    lib["gemv"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    C_result = C_tvm.numpy()

    print(f"\nC_result sample: {C_result[0, :4]}")
    print(f"C_ref sample: {C_ref[0, :4]}")

    # Check if kernel wrote anything
    if np.allclose(C_result, -999.0):
        print("\nERROR: Kernel did not write to output!")
    elif np.allclose(C_result, 0):
        print("\nERROR: Output is all zeros!")
    else:
        max_diff = np.abs(C_result - C_ref).max()
        cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
            np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)
        print(f"\nMax diff: {max_diff:.6f}")
        print(f"Cos sim: {cos_sim:.6f}")
        print(f"Status: {'PASS' if cos_sim > 0.99 else 'FAIL'}")


def test_full(N=16384, K=2048):
    """Test with full size."""
    import time

    print(f"\n{'='*60}")
    print(f"Full Size W4A16 GEMV")
    print(f"N={N}, K={K}")
    print(f"{'='*60}")

    # Data
    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    print("Quantizing...")
    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Build kernel
    print("Building kernel...")
    kernel = create_simple_gemv(N, K, THREADS=256)
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
    C_tvm = runtime.empty((1, N), "float32", device)

    lib["gemv"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    C_result = C_tvm.numpy()

    # Verify
    if np.isnan(C_result).any():
        print("ERROR: Output contains NaN!")
        return None

    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)
    print(f"\nCos sim: {cos_sim:.6f}")
    print(f"Status: {'PASS' if cos_sim > 0.99 else 'FAIL'}")

    if cos_sim <= 0.99:
        return None

    # Benchmark
    warmup = 50
    runs = 200

    for _ in range(warmup):
        lib["gemv"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        lib["gemv"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000
    flops = 2.0 * N * K
    gflops = flops / (avg_ms / 1000) / 1e9

    print(f"\nTime: {avg_ms:.4f} ms")
    print(f"GFLOPS: {gflops:.2f}")

    target_ms = 1.0
    if avg_ms < target_ms:
        print(f"ACHIEVED target of < {target_ms}ms!")
    else:
        print(f"Need {avg_ms/target_ms:.1f}x speedup to reach {target_ms}ms target")

    return avg_ms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Run small test")
    parser.add_argument("--full", action="store_true", help="Run full test")
    args = parser.parse_args()

    if args.small or not (args.small or args.full):
        test_small()

    if args.full:
        test_full()
