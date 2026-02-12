#!/usr/bin/env python3
"""
BF16/FP16 Baseline - Compare against W4A16.
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import te, tir, runtime
import numpy as np
import time


def create_fp16_gemm(M, N, K):
    """Create FP16 GEMM compute."""
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


def benchmark_fp16(M, N, K):
    """Benchmark FP16 GEMM with dlight."""
    from tvm import dlight as dl

    print(f"\n{'='*60}")
    print(f"FP16 Baseline: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    # Data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float16)
    C_ref = A_np.astype(np.float32) @ W_np.astype(np.float32).T

    # Build
    A, W, C = create_fp16_gemm(M, N, K)
    func = te.create_prim_func([A, W, C])
    mod = tvm.IRModule({"main": func})
    target = tvm.target.Target("cuda -arch=sm_110")

    with tvm.transform.PassContext(opt_level=3):
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
        lib = tvm.build(mod, target=target)

    # Run
    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_tvm = runtime.empty(W_np.shape, "float16", device)
    W_tvm.copyfrom(W_np)
    C_tvm = runtime.empty((M, N), "float32", device)

    lib["main"](A_tvm, W_tvm, C_tvm)
    device.sync()

    # Verify
    C_result = C_tvm.numpy()
    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)
    print(f"Cos sim: {cos_sim:.6f}")

    # Benchmark
    warmup, runs = 50, 200
    for _ in range(warmup):
        lib["main"](A_tvm, W_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        lib["main"](A_tvm, W_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12

    print(f"Time: {avg_ms:.4f} ms")
    print(f"TFLOPS: {tflops:.4f}")

    # Memory comparison
    fp16_mem = N * K * 2  # FP16 weights
    w4a16_mem = N * K // 2 + N * (K // 32) * 2  # INT4 packed + scales
    print(f"\nMemory: FP16={fp16_mem/1e6:.1f}MB, W4A16={w4a16_mem/1e6:.1f}MB, Ratio={fp16_mem/w4a16_mem:.1f}x")

    return avg_ms


def main():
    N, K = 16384, 2048

    print("\n" + "="*60)
    print("FP16 vs W4A16 Comparison")
    print("="*60)

    results = {}
    for M in [1, 4, 16, 64, 256, 712]:
        try:
            ms = benchmark_fp16(M, N, K)
            results[M] = ms
        except Exception as e:
            print(f"M={M} failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: FP16 Baseline")
    print("="*60)
    print(f"{'M':>6} | {'FP16 (ms)':>10} | {'TFLOPS':>8}")
    print("-"*40)
    for M, ms in results.items():
        flops = 2.0 * M * N * K
        tflops = flops / (ms / 1000) / 1e12
        print(f"{M:>6} | {ms:>10.4f} | {tflops:>8.4f}")


if __name__ == "__main__":
    main()
