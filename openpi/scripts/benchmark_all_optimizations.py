#!/usr/bin/env python3
"""
Benchmark All NVFP4 Optimizations.

Tests:
1. Multi-Layer Persistent MLP kernel
2. Optimized Triton kernels (vectorized + shared memory)
3. cuBLAS BF16 baseline
4. TRT FP8 estimates

Goal: Find the best approach to beat TRT FP8 (0.53ms per GEMM).

Author: Claude Code
Date: 2026-02-10
"""

import sys
import os
import time

# Add project paths
PROJECT_ROOT = "/home/heima-thor/suliang/Turbo-Pi"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "openpi/src"))

import torch
import torch.nn.functional as F
import numpy as np


def benchmark_bf16_cublas():
    """Benchmark cuBLAS BF16 as baseline."""
    print("\n" + "=" * 70)
    print("cuBLAS BF16 Baseline (Best Known)")
    print("=" * 70)

    device = torch.device('cuda')

    configs = [
        (2048, 16384, "MLP gate_proj: 2048 -> 16384"),
        (2048, 16384, "MLP up_proj: 2048 -> 16384"),
        (16384, 2048, "MLP down_proj: 16384 -> 2048"),
    ]

    results = {}
    warmup = 100
    runs = 500

    for K, N, desc in configs:
        x = torch.randn(1, K, device=device, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(warmup):
            _ = F.linear(x, w)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(runs):
            _ = F.linear(x, w)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / runs * 1000

        results[desc] = elapsed
        print(f"  {desc}: {elapsed:.4f} ms")

    # Full MLP
    mlp_time = sum(results.values())
    print(f"\n  Full MLP (3 GEMMs): {mlp_time:.4f} ms")

    return results


def benchmark_triton_optimized():
    """Benchmark optimized Triton kernels."""
    print("\n" + "=" * 70)
    print("Optimized Triton Kernels")
    print("=" * 70)

    try:
        from openpi.models_pytorch.nvfp4_triton_optimized import (
            quantize_weight_nvfp4,
            _nvfp4_gemv_shared_mem_kernel,
            _nvfp4_gemv_parallel_n_kernel,
        )
        import triton
    except ImportError as e:
        print(f"  Import error: {e}")
        return {}

    device = torch.device('cuda')

    configs = [
        (2048, 16384, "MLP gate/up: 2048 -> 16384"),
        (16384, 2048, "MLP down: 16384 -> 2048"),
    ]

    results = {}
    warmup = 100
    runs = 500

    for K, N, desc in configs:
        weight = torch.randn(N, K, device=device, dtype=torch.float32)
        w_packed, w_scale = quantize_weight_nvfp4(weight, block_size=32)
        x = torch.randn(K, device=device, dtype=torch.float32)
        out = torch.empty(N, device=device, dtype=torch.float32)
        num_blocks = K // 32

        # Test SharedMem kernel
        def run_kernel():
            grid = (triton.cdiv(N, 32),)
            _nvfp4_gemv_shared_mem_kernel[grid](
                x, w_packed, w_scale, x, out,
                N, K, num_blocks,
                HAS_BIAS=False,
                BLOCK_SIZE=32,
                BLOCK_N=32,
                NUM_WARPS=4,
            )

        # Warmup
        for _ in range(warmup):
            run_kernel()
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(runs):
            run_kernel()
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / runs * 1000

        results[desc] = elapsed
        print(f"  {desc}: {elapsed:.4f} ms")

    return results


def benchmark_persistent_mlp():
    """Benchmark persistent MLP kernel."""
    print("\n" + "=" * 70)
    print("Persistent MLP Kernel")
    print("=" * 70)

    try:
        from openpi.models_pytorch.nvfp4_persistent_mlp import (
            NVFP4PersistentMLP,
            benchmark_persistent_mlp as run_persistent_benchmark
        )
    except ImportError as e:
        print(f"  Import error: {e}")
        return {}

    device = torch.device('cuda')

    hidden_size = 2048
    mlp_dim = 16384

    # Create random weights
    gate_proj = torch.randn(mlp_dim, hidden_size, device=device, dtype=torch.float32)
    up_proj = torch.randn(mlp_dim, hidden_size, device=device, dtype=torch.float32)
    down_proj = torch.randn(hidden_size, mlp_dim, device=device, dtype=torch.float32)

    # Create module
    mlp = NVFP4PersistentMLP(hidden_size, mlp_dim, device=device)
    mlp.load_weights(gate_proj, up_proj, down_proj)

    # Input
    x = torch.randn(1, hidden_size, device=device, dtype=torch.float32)

    warmup = 50
    runs = 200

    # Warmup
    for _ in range(warmup):
        _ = mlp(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(runs):
        _ = mlp(x)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / runs * 1000

    print(f"  Full MLP (fused): {elapsed:.4f} ms")
    return {"Persistent MLP": elapsed}


def main():
    print("=" * 70)
    print("NVFP4 All Optimizations Benchmark")
    print("=" * 70)
    print(f"\nTarget: Beat TRT FP8 = 0.53 ms per GEMM")
    print(f"        Full MLP = 3 GEMMs × 0.53 = 1.59 ms")

    all_results = {}

    # 1. cuBLAS BF16
    try:
        bf16_results = benchmark_bf16_cublas()
        all_results["cuBLAS BF16"] = bf16_results
    except Exception as e:
        print(f"  cuBLAS BF16 benchmark failed: {e}")

    # 2. Optimized Triton
    try:
        triton_results = benchmark_triton_optimized()
        all_results["Triton Optimized"] = triton_results
    except Exception as e:
        print(f"  Triton benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # 3. Persistent MLP
    try:
        persistent_results = benchmark_persistent_mlp()
        all_results["Persistent MLP"] = persistent_results
    except Exception as e:
        print(f"  Persistent MLP benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    TRT_FP8_GEMM = 0.53
    TRT_FP8_MLP = TRT_FP8_GEMM * 3

    print(f"\n{'Approach':<30} {'Time (ms)':<12} {'vs TRT FP8':<12} {'Status'}")
    print("-" * 70)
    print(f"{'TRT FP8 (per GEMM)':<30} {TRT_FP8_GEMM:<12.4f} {'1.00x':<12} baseline")
    print(f"{'TRT FP8 (full MLP)':<30} {TRT_FP8_MLP:<12.4f} {'1.00x':<12} baseline")

    # cuBLAS BF16
    if "cuBLAS BF16" in all_results:
        total_bf16 = sum(all_results["cuBLAS BF16"].values())
        speedup = TRT_FP8_MLP / total_bf16
        status = "FASTER" if total_bf16 < TRT_FP8_MLP else "slower"
        print(f"{'cuBLAS BF16 (full MLP)':<30} {total_bf16:<12.4f} {speedup:.2f}x{' ':<8} {status}")

    # Triton Optimized
    if "Triton Optimized" in all_results and all_results["Triton Optimized"]:
        total_triton = sum(all_results["Triton Optimized"].values()) * 1.5  # Estimate 3 GEMMs
        speedup = TRT_FP8_MLP / total_triton
        status = "FASTER" if total_triton < TRT_FP8_MLP else "slower"
        print(f"{'Triton Optimized (full MLP)':<30} {total_triton:<12.4f} {speedup:.2f}x{' ':<8} {status}")

    # Persistent MLP
    if "Persistent MLP" in all_results and all_results["Persistent MLP"]:
        persistent_time = all_results["Persistent MLP"]["Persistent MLP"]
        speedup = TRT_FP8_MLP / persistent_time
        status = "FASTER" if persistent_time < TRT_FP8_MLP else "slower"
        print(f"{'Persistent MLP (full MLP)':<30} {persistent_time:<12.4f} {speedup:.2f}x{' ':<8} {status}")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find best approach
    best_time = float('inf')
    best_name = None

    if "cuBLAS BF16" in all_results:
        total = sum(all_results["cuBLAS BF16"].values())
        if total < best_time:
            best_time = total
            best_name = "cuBLAS BF16"

    if "Persistent MLP" in all_results and all_results["Persistent MLP"]:
        t = all_results["Persistent MLP"]["Persistent MLP"]
        if t < best_time:
            best_time = t
            best_name = "Persistent MLP"

    if best_time < TRT_FP8_MLP:
        print(f"\n  Best: {best_name} at {best_time:.4f} ms")
        print(f"  This is {TRT_FP8_MLP/best_time:.2f}x FASTER than TRT FP8!")
        print(f"\n  Next step: Integrate into TRT Plugin")
    else:
        print(f"\n  Current best: {best_name} at {best_time:.4f} ms")
        print(f"  Still {best_time/TRT_FP8_MLP:.2f}x slower than TRT FP8")
        print(f"\n  Further optimization needed:")
        print(f"  - Register-level accumulation")
        print(f"  - WMMA/Tensor Core utilization")
        print(f"  - Multi-stage pipelining")


if __name__ == "__main__":
    main()
