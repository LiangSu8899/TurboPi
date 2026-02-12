#!/usr/bin/env python3
"""
Benchmark TVM Fused Kernels vs TRT FP8 Baseline.

Baseline: TRT FP8 Static Graph = 12.0 Hz (83.5 ms total, 47.4 ms KV Cache)
Goal: Determine if W4A4/W4A8/W4A16 can beat TRT FP8

Key insight from previous benchmarks:
- Separate dequant + GEMM: ~0.9-1.0 ms (TOO SLOW)
- TVM Fused kernel: dequant + GEMM in one pass (should be much faster)

Run with TVM environment:
    source /home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/venv/bin/activate
    python benchmark_tvm_vs_trt_fp8.py

Author: Claude Code
Date: 2026-02-10
"""

import sys
import os
import time

# Add TVM to path
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import te
from tvm.script import tir as T
import numpy as np

print(f"TVM Version: {tvm.__version__}")
print(f"CUDA Available: {tvm.cuda().exist}")

# Constants
BLOCK_SIZE = 32
M = 1  # Batch size (single token inference)
K = 3072  # Pi0 hidden dim
N = 3072  # Output features

WARMUP = 50
RUNS = 200


def create_w4a4_fused_kernel(M, N, K, block_size=32):
    """W4A4: Both weight and activation are nvFP4 with scales."""
    num_blocks_k = (K + block_size - 1) // block_size
    THREADS = 256
    total = M * N
    num_blocks = (total + THREADS - 1) // THREADS

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a4_fused", "tir.noalias": True})
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N
                    C[i, j] = T.float32(0)
                    for k in T.serial(K):
                        block_idx = k // block_size
                        a_val = A[i, k] * scale_A[i, block_idx]
                        w_val = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + a_val * w_val
    return kernel


def create_w4a8_fused_kernel(M, N, K, block_size=32):
    """W4A8: Weight is nvFP4, Activation is FP8 with scales."""
    num_blocks_k = (K + block_size - 1) // block_size
    THREADS = 256
    total = M * N
    num_blocks = (total + THREADS - 1) // THREADS

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),  # FP8 activation
        W: T.Buffer((N, K), "float32"),  # nvFP4 weight
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a8_fused", "tir.noalias": True})
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N
                    C[i, j] = T.float32(0)
                    for k in T.serial(K):
                        block_idx = k // block_size
                        a_val = A[i, k] * scale_A[i, block_idx]
                        w_val = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + a_val * w_val
    return kernel


def create_w4a16_fused_kernel(M, N, K, block_size=32):
    """W4A16: Weight is nvFP4 with scale, Activation is full precision."""
    num_blocks_k = (K + block_size - 1) // block_size
    THREADS = 256
    total = M * N
    num_blocks = (total + THREADS - 1) // THREADS

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),  # Full precision activation
        W: T.Buffer((N, K), "float32"),  # nvFP4 weight
        scale_W: T.Buffer((N, num_blocks_k), "float32"),  # Only weight scale
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_fused", "tir.noalias": True})
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N
                    C[i, j] = T.float32(0)
                    for k in T.serial(K):
                        block_idx = k // block_size
                        w_dequant = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + A[i, k] * w_dequant
    return kernel


def build_kernel(tir_func, target="cuda -arch=sm_110"):
    """Build TVM kernel."""
    target_obj = tvm.target.Target(target)
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(tir_func, target=target_obj)
    return mod


def benchmark_tvm_kernel(name, kernel_func, has_scale_a=True):
    """Benchmark a TVM fused kernel."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    num_blocks_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Build
    print("  Building kernel...")
    build_start = time.time()
    mod = build_kernel(kernel_func)
    build_time = (time.time() - build_start) * 1000
    print(f"  Build time: {build_time:.2f} ms")

    # Get function - use the global_symbol from kernel definition
    func_name = name.split()[0].lower().replace(" ", "_")
    if "w4a4" in name.lower():
        func = mod["w4a4_fused"]
    elif "w4a8" in name.lower():
        func = mod["w4a8_fused"]
    elif "w4a16" in name.lower():
        func = mod["w4a16_fused"]
    else:
        func = mod.get_function("default")

    # Prepare data using TVM runtime.empty + copyfrom
    device = tvm.runtime.cuda(0)

    A = tvm.runtime.empty((M, K), dtype="float32", device=device)
    A.copyfrom(np.random.randn(M, K).astype("float32"))

    W = tvm.runtime.empty((N, K), dtype="float32", device=device)
    W.copyfrom(np.random.randn(N, K).astype("float32"))

    C = tvm.runtime.empty((M, N), dtype="float32", device=device)
    C.copyfrom(np.zeros((M, N)).astype("float32"))

    if has_scale_a:
        scale_A = tvm.runtime.empty((M, num_blocks_k), dtype="float32", device=device)
        scale_A.copyfrom((np.random.rand(M, num_blocks_k) * 0.1 + 0.01).astype("float32"))

        scale_W = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
        scale_W.copyfrom((np.random.rand(N, num_blocks_k) * 0.1 + 0.01).astype("float32"))

        # Warmup
        for _ in range(WARMUP):
            func(A, W, scale_A, scale_W, C)
        tvm.runtime.cuda(0).sync()

        # Benchmark
        start = time.time()
        for _ in range(RUNS):
            func(A, W, scale_A, scale_W, C)
        tvm.runtime.cuda(0).sync()
    else:
        scale_W = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
        scale_W.copyfrom((np.random.rand(N, num_blocks_k) * 0.1 + 0.01).astype("float32"))

        # Warmup
        for _ in range(WARMUP):
            func(A, W, scale_W, C)
        tvm.runtime.cuda(0).sync()

        # Benchmark
        start = time.time()
        for _ in range(RUNS):
            func(A, W, scale_W, C)
        tvm.runtime.cuda(0).sync()

    elapsed = (time.time() - start) / RUNS * 1000
    print(f"  Avg time: {elapsed:.4f} ms")

    # Calculate TFLOPS
    flops = 2.0 * M * N * K
    tflops = flops / (elapsed / 1000) / 1e12
    print(f"  Throughput: {tflops:.2f} TFLOPS")

    return elapsed


def main():
    print("="*70)
    print("TVM Fused Kernels vs TRT FP8 Static Graph Benchmark")
    print("="*70)
    print(f"\nMatrix: M={M}, N={N}, K={K}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Warmup: {WARMUP}, Runs: {RUNS}")

    # Baseline from debug-08.md
    print("\n" + "="*70)
    print("BASELINE: TRT FP8 Static Graph (from debug-08.md)")
    print("="*70)
    print("  Full Pipeline: 83.5 ms = 12.0 Hz")
    print("  KV Cache only: 47.4 ms (56.8%)")
    print("  Vision TRT:    17.2 ms")
    print("  Denoise:       10.1 ms")
    print("  Other:          8.8 ms")

    # Per-GEMM estimate for KV Cache
    # KV Cache has: 18 layers × (QKV + O + MLP)
    # MLP: gate_proj + up_proj + down_proj = 3 GEMMs per layer
    # QKV + O: ~2 GEMMs per layer
    # Total: ~5 GEMMs × 18 layers = ~90 GEMMs
    # 47.4 ms / 90 = ~0.53 ms per GEMM average
    trt_fp8_per_gemm = 47.4 / 90  # ~0.53 ms

    print(f"\n  Estimated per-GEMM (KV Cache): {trt_fp8_per_gemm:.4f} ms")
    print("  (90 GEMMs in 18 layers × 5 GEMMs/layer)")

    results = {}

    # Benchmark TVM kernels
    print("\n" + "="*70)
    print("TVM FUSED KERNELS")
    print("="*70)

    # W4A4
    w4a4_kernel = create_w4a4_fused_kernel(M, N, K)
    w4a4_time = benchmark_tvm_kernel("W4A4 Fused (dequant+GEMM)", w4a4_kernel, has_scale_a=True)
    results["W4A4"] = w4a4_time

    # W4A8
    w4a8_kernel = create_w4a8_fused_kernel(M, N, K)
    w4a8_time = benchmark_tvm_kernel("W4A8 Fused (dequant+GEMM)", w4a8_kernel, has_scale_a=True)
    results["W4A8"] = w4a8_time

    # W4A16
    w4a16_kernel = create_w4a16_fused_kernel(M, N, K)
    w4a16_time = benchmark_tvm_kernel("W4A16 Fused (dequant+GEMM)", w4a16_kernel, has_scale_a=False)
    results["W4A16"] = w4a16_time

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: TVM Fused vs TRT FP8")
    print("="*70)
    print(f"""
{'Kernel':<25} {'Time (ms)':<12} {'vs TRT FP8':<15} {'Status'}
{'-'*65}
{'TRT FP8 (per GEMM)':<25} {trt_fp8_per_gemm:<12.4f} {'1.00x (baseline)':<15}
{'W4A4 Fused':<25} {results['W4A4']:<12.4f} {trt_fp8_per_gemm/results['W4A4']:.2f}x{' ':<11} {'✅ FASTER' if results['W4A4'] < trt_fp8_per_gemm else '❌ SLOWER'}
{'W4A8 Fused':<25} {results['W4A8']:<12.4f} {trt_fp8_per_gemm/results['W4A8']:.2f}x{' ':<11} {'✅ FASTER' if results['W4A8'] < trt_fp8_per_gemm else '❌ SLOWER'}
{'W4A16 Fused':<25} {results['W4A16']:<12.4f} {trt_fp8_per_gemm/results['W4A16']:.2f}x{' ':<11} {'✅ FASTER' if results['W4A16'] < trt_fp8_per_gemm else '❌ SLOWER'}
""")

    # Projected KV Cache time
    print("\n" + "="*70)
    print("PROJECTED KV CACHE LATENCY (90 GEMMs)")
    print("="*70)
    for name, t in results.items():
        kv_time = t * 90
        total_time = kv_time + 17.2 + 10.1 + 8.8  # Vision + Denoise + Other
        hz = 1000.0 / total_time
        print(f"  {name}: {kv_time:.1f} ms KV → {total_time:.1f} ms total → {hz:.1f} Hz")

    print(f"\n  TRT FP8 Baseline: 47.4 ms KV → 83.5 ms total → 12.0 Hz")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    best_kernel = min(results, key=results.get)
    best_time = results[best_kernel]

    if best_time < trt_fp8_per_gemm:
        speedup = trt_fp8_per_gemm / best_time
        print(f"  ✅ {best_kernel} is {speedup:.2f}x faster than TRT FP8 per-GEMM")
        print(f"     Proceed with TensorRT Plugin integration")
    else:
        slowdown = best_time / trt_fp8_per_gemm
        print(f"  ⚠️  Best TVM kernel ({best_kernel}) is {slowdown:.2f}x SLOWER than TRT FP8")
        print(f"     TVM naive kernel needs optimization:")
        print(f"     - Shared memory tiling")
        print(f"     - Vectorized memory access")
        print(f"     - TVM auto-scheduler tuning")
        print(f"     - Or use TRT FP8 and focus on other optimizations")

    print("="*70)


if __name__ == "__main__":
    main()
