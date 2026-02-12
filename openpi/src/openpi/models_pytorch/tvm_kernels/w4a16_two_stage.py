#!/usr/bin/env python3
"""
W4A16 GEMM - Two-Stage Approach

Stage 1: Dequant W_packed -> W_fp16 (in global memory, one-time)
Stage 2: Standard FP16 GEMM with Tensor Cores

This is the "Dequant + cuBLAS" approach but using TVM's optimized GEMM.
While it uses extra memory, it should be much faster than fused approaches.

Author: Claude Code
Date: 2026-02-11
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import te, tir, runtime, relax
from tvm.script import tir as T
import numpy as np
import time


QUANT_BLOCK = 32


def create_dequant_kernel(N, K):
    """Create INT4 dequant kernel."""
    num_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    @T.prim_func
    def dequant_int4(
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_blocks), "float16"),
        W_fp16: T.Buffer((N, K), "float16"),
    ):
        T.func_attr({"global_symbol": "dequant_int4", "tir.noalias": True})

        # Simple parallelization: each thread handles one element
        TILE = 32
        num_tiles_n = (N + TILE - 1) // TILE
        num_tiles_k = (K + TILE - 1) // TILE

        for tn in T.thread_binding(num_tiles_n, thread="blockIdx.y"):
            for tk in T.thread_binding(num_tiles_k, thread="blockIdx.x"):
                for ty in T.thread_binding(TILE, thread="threadIdx.y"):
                    for tx in T.thread_binding(TILE, thread="threadIdx.x"):
                        n = tn * TILE + ty
                        k = tk * TILE + tx

                        if n < N and k < K:
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
                            W_fp16[n, k] = signed_val * scale

    return dequant_int4


def create_fp16_gemm_kernel(M, N, K):
    """Create FP16 GEMM kernel using TE."""
    A = te.placeholder((M, K), dtype="float16", name="A")
    W = te.placeholder((N, K), dtype="float16", name="W")

    k_axis = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(
            A[m, k_axis].astype("float32") * W[n, k_axis].astype("float32"),
            axis=k_axis
        ),
        name="C"
    )

    return A, W, C


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


def dequant_int4_np(W_packed, scales, K, block_size=QUANT_BLOCK):
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


def test_two_stage(M=256, N=512, K=256):
    """Test two-stage approach."""
    print(f"\n{'='*60}")
    print(f"W4A16 Two-Stage (Dequant + GEMM)")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    num_blocks = K // QUANT_BLOCK

    # Data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4_np(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Build dequant kernel
    print("Building dequant kernel...")
    dequant_kernel = create_dequant_kernel(N, K)
    dequant_mod = tvm.IRModule({"main": dequant_kernel})
    target = tvm.target.Target("cuda -arch=sm_110")

    with tvm.transform.PassContext(opt_level=3):
        dequant_lib = tvm.build(dequant_mod, target=target)
    print("Dequant kernel built!")

    # Build GEMM kernel with dlight
    print("Building GEMM kernel...")
    from tvm import dlight as dl

    A, W, C = create_fp16_gemm_kernel(M, N, K)
    gemm_func = te.create_prim_func([A, W, C])
    gemm_mod = tvm.IRModule({"main": gemm_func})

    with tvm.transform.PassContext(opt_level=3):
        with target:
            # Try Matmul schedule for Tensor Cores
            try:
                gemm_mod = dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                )(gemm_mod)
                print("Applied Matmul schedule")
            except:
                gemm_mod = dl.ApplyDefaultSchedule(
                    dl.gpu.Fallback(),
                )(gemm_mod)
                print("Applied Fallback schedule")

        gemm_lib = tvm.build(gemm_mod, target=target)
    print("GEMM kernel built!")

    # Create TVM arrays
    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    W_fp16_tvm = runtime.empty((N, K), "float16", device)
    C_tvm = runtime.empty((M, N), "float32", device)

    # Run dequant
    dequant_lib["dequant_int4"](W_packed_tvm, scales_tvm, W_fp16_tvm)
    device.sync()

    # Verify dequant
    W_fp16_result = W_fp16_tvm.numpy()
    dequant_diff = np.abs(W_fp16_result - W_dequant_np.astype(np.float16)).max()
    print(f"Dequant max diff: {dequant_diff:.6f}")

    # Run GEMM
    gemm_lib["main"](A_tvm, W_fp16_tvm, C_tvm)
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
        return None

    # Benchmark dequant
    warmup = 20
    runs = 100

    for _ in range(warmup):
        dequant_lib["dequant_int4"](W_packed_tvm, scales_tvm, W_fp16_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        dequant_lib["dequant_int4"](W_packed_tvm, scales_tvm, W_fp16_tvm)
    device.sync()
    dequant_ms = (time.time() - start) / runs * 1000
    print(f"\n  Dequant time: {dequant_ms:.4f} ms")

    # Benchmark GEMM
    for _ in range(warmup):
        gemm_lib["main"](A_tvm, W_fp16_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        gemm_lib["main"](A_tvm, W_fp16_tvm, C_tvm)
    device.sync()
    gemm_ms = (time.time() - start) / runs * 1000

    total_ms = dequant_ms + gemm_ms
    flops = 2.0 * M * N * K
    tflops = flops / (total_ms / 1000) / 1e12

    print(f"  GEMM time:    {gemm_ms:.4f} ms")
    print(f"  Total time:   {total_ms:.4f} ms")
    print(f"  TFLOPS:       {tflops:.4f}")

    return total_ms


def test_full_size(M=712, N=16384, K=2048):
    """Test with full problem size."""
    print(f"\n{'='*60}")
    print(f"W4A16 Two-Stage - Full Size")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    num_blocks = K // QUANT_BLOCK

    # Data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    print("Quantizing...")
    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4_np(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Build dequant kernel
    print("Building dequant kernel...")
    dequant_kernel = create_dequant_kernel(N, K)
    dequant_mod = tvm.IRModule({"main": dequant_kernel})
    target = tvm.target.Target("cuda -arch=sm_110")

    with tvm.transform.PassContext(opt_level=3):
        dequant_lib = tvm.build(dequant_mod, target=target)

    # Build GEMM kernel with dlight
    print("Building GEMM kernel...")
    from tvm import dlight as dl

    A, W, C = create_fp16_gemm_kernel(M, N, K)
    gemm_func = te.create_prim_func([A, W, C])
    gemm_mod = tvm.IRModule({"main": gemm_func})

    with tvm.transform.PassContext(opt_level=3):
        with target:
            gemm_mod = dl.ApplyDefaultSchedule(
                dl.gpu.Fallback(),
            )(gemm_mod)

        gemm_lib = tvm.build(gemm_mod, target=target)
    print("Kernels built!")

    # Create TVM arrays
    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    W_fp16_tvm = runtime.empty((N, K), "float16", device)
    C_tvm = runtime.empty((M, N), "float32", device)

    # Run
    dequant_lib["dequant_int4"](W_packed_tvm, scales_tvm, W_fp16_tvm)
    gemm_lib["main"](A_tvm, W_fp16_tvm, C_tvm)
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
        dequant_lib["dequant_int4"](W_packed_tvm, scales_tvm, W_fp16_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        dequant_lib["dequant_int4"](W_packed_tvm, scales_tvm, W_fp16_tvm)
    device.sync()
    dequant_ms = (time.time() - start) / runs * 1000

    for _ in range(warmup):
        gemm_lib["main"](A_tvm, W_fp16_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        gemm_lib["main"](A_tvm, W_fp16_tvm, C_tvm)
    device.sync()
    gemm_ms = (time.time() - start) / runs * 1000

    total_ms = dequant_ms + gemm_ms
    flops = 2.0 * M * N * K
    tflops = flops / (total_ms / 1000) / 1e12

    print(f"\n  Dequant time: {dequant_ms:.4f} ms")
    print(f"  GEMM time:    {gemm_ms:.4f} ms")
    print(f"  Total time:   {total_ms:.4f} ms")
    print(f"  TFLOPS:       {tflops:.4f}")

    BF16_MS = 0.42
    print(f"\n  vs BF16 baseline ({BF16_MS}ms): {BF16_MS / total_ms:.2f}x")

    return total_ms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    test_two_stage()

    if args.full:
        test_full_size()
