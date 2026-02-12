#!/usr/bin/env python3
"""
TVM TensorIR Kernels - Comprehensive Benchmark

Compares W4A4, W4A8, W4A16 against FP8 baseline (12 Hz)
Target: Break through FP8 12 Hz to reach 14-18 Hz

Run with MLC-LLM venv:
    source /home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/venv/bin/activate
    python benchmark_all.py

Author: Claude Code
Date: 2026-02-10
"""

import sys
import os

# Add TVM to path
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import time
import torch
import numpy as np

# Check TVM
try:
    import tvm
    print(f"TVM Version: {tvm.__version__}")
    print(f"CUDA Available: {tvm.cuda().exist}")
except ImportError:
    print("ERROR: TVM not found. Please activate MLC-LLM venv:")
    print("  source /home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/venv/bin/activate")
    sys.exit(1)


# Constants
BLOCK_SIZE = 32
M = 1  # Batch size (inference)
K = 3072  # Hidden dim (Pi0)
N_VALUES = [3072, 12288]  # gate_proj/up_proj, down_proj

WARMUP = 20
RUNS = 100


def benchmark_fp8_baseline(M, N, K, runs=RUNS):
    """Benchmark FP8 baseline (current best: 12 Hz)."""
    print(f"\n--- FP8 Baseline (cuBLAS) ---")

    # Simulate FP8 with BF16 (Thor uses TensorRT FP8)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(WARMUP):
        _ = torch.matmul(A, W.T)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(runs):
        C = torch.matmul(A, W.T)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / runs * 1000

    print(f"  Shape: [{M}, {K}] x [{K}, {N}]")
    print(f"  Time: {elapsed:.4f} ms")
    return elapsed


def benchmark_w4a4(M, N, K, runs=RUNS):
    """Benchmark W4A4 TVM kernel."""
    from nvfp4_quantize import build_nvfp4_quantize_kernel, BLOCK_SIZE as BS
    from nvfp4_gemm import build_nvfp4_gemm_kernel

    print(f"\n--- W4A4 (TVM TensorIR) ---")

    num_blocks = (K + BS - 1) // BS

    # Create nvFP4 values
    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6])
    A = fp4_values[torch.randint(0, len(fp4_values), (M, K))].float().cuda()
    W = fp4_values[torch.randint(0, len(fp4_values), (N, K))].float().cuda()
    scale_A = torch.rand(M, num_blocks, device="cuda") * 0.1 + 0.01
    scale_W = torch.rand(N, num_blocks, device="cuda") * 0.1 + 0.01

    # Build kernel
    print("  Building kernel...")
    build_start = time.time()
    try:
        gemm_mod = build_nvfp4_gemm_kernel(M, N, K)
        build_time = time.time() - build_start
        print(f"  Build time: {build_time*1000:.2f} ms")
    except Exception as e:
        print(f"  Build FAILED: {e}")
        return None

    # Prepare TVM arrays using DLPack (TVM 0.24 API)
    device = tvm.runtime.cuda(0)

    # Convert to float32 and use DLPack
    A_f32 = A.float().contiguous()
    W_f32 = W.float().contiguous()
    scale_A_f32 = scale_A.float().contiguous()
    scale_W_f32 = scale_W.float().contiguous()
    C_out = torch.empty(M, N, dtype=torch.float32, device="cuda")

    a_tvm = tvm.runtime.from_dlpack(A_f32)
    w_tvm = tvm.runtime.from_dlpack(W_f32)
    scale_a_tvm = tvm.runtime.from_dlpack(scale_A_f32)
    scale_w_tvm = tvm.runtime.from_dlpack(scale_W_f32)
    c_tvm = tvm.runtime.from_dlpack(C_out)

    func = gemm_mod["nvfp4_gemm"]

    # Warmup
    for _ in range(WARMUP):
        func(a_tvm, w_tvm, scale_a_tvm, scale_w_tvm, c_tvm)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(runs):
        func(a_tvm, w_tvm, scale_a_tvm, scale_w_tvm, c_tvm)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / runs * 1000

    print(f"  Shape: [{M}, {K}] x [{K}, {N}]")
    print(f"  Time: {elapsed:.4f} ms")
    return elapsed


def benchmark_w4a8(M, N, K, runs=RUNS):
    """Benchmark W4A8 TVM kernel (bypasses mxf8f6f4)."""
    from w4a8_gemm import build_w4a8_gemm_kernel, quantize_to_fp8_e4m3

    print(f"\n--- W4A8 (TVM TensorIR - mxf8f6f4 bypass) ---")

    num_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    # FP8 activation
    A = torch.randn(M, K, device="cuda")
    A_fp8, scale_A = quantize_to_fp8_e4m3(A)

    # nvFP4 weight
    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6])
    W = fp4_values[torch.randint(0, len(fp4_values), (N, K))].float().cuda()
    scale_W = torch.rand(N, num_blocks, device="cuda") * 0.1 + 0.01

    # Build kernel
    print("  Building kernel...")
    build_start = time.time()
    try:
        gemm_mod = build_w4a8_gemm_kernel(M, N, K)
        build_time = time.time() - build_start
        print(f"  Build time: {build_time*1000:.2f} ms")
    except Exception as e:
        print(f"  Build FAILED: {e}")
        return None

    # Prepare TVM arrays using DLPack (TVM 0.24 API)
    device = tvm.runtime.cuda(0)

    # Convert to float32 and use DLPack
    A_f32 = A_fp8.float().contiguous()
    W_f32 = W.float().contiguous()
    scale_A_f32 = scale_A.float().contiguous()
    scale_W_f32 = scale_W.float().contiguous()
    C_out = torch.empty(M, N, dtype=torch.float32, device="cuda")

    a_tvm = tvm.runtime.from_dlpack(A_f32)
    w_tvm = tvm.runtime.from_dlpack(W_f32)
    scale_a_tvm = tvm.runtime.from_dlpack(scale_A_f32)
    scale_w_tvm = tvm.runtime.from_dlpack(scale_W_f32)
    c_tvm = tvm.runtime.from_dlpack(C_out)

    func = gemm_mod["w4a8_gemm"]

    # Warmup
    for _ in range(WARMUP):
        func(a_tvm, w_tvm, scale_a_tvm, scale_w_tvm, c_tvm)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(runs):
        func(a_tvm, w_tvm, scale_a_tvm, scale_w_tvm, c_tvm)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / runs * 1000

    print(f"  Shape: [{M}, {K}] x [{K}, {N}]")
    print(f"  Time: {elapsed:.4f} ms")
    return elapsed


def benchmark_w4a16(M, N, K, runs=RUNS):
    """Benchmark W4A16 TVM kernel (dequant + cuBLAS)."""
    from w4a16_dequant import build_w4a16_dequant_kernel, build_w4a16_fused_kernel

    print(f"\n--- W4A16 (TVM dequant + cuBLAS) ---")

    num_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    # BF16 activation
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    # nvFP4 weight
    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6])
    W = fp4_values[torch.randint(0, len(fp4_values), (N, K))].float().cuda()
    scales = torch.rand(N, num_blocks, device="cuda") * 0.1 + 0.01

    # Build kernels
    print("  Building kernels...")
    build_start = time.time()
    try:
        dequant_mod = build_w4a16_dequant_kernel(N, K)
        fused_mod = build_w4a16_fused_kernel(M, N, K)
        build_time = time.time() - build_start
        print(f"  Build time: {build_time*1000:.2f} ms")
    except Exception as e:
        print(f"  Build FAILED: {e}")
        return None, None

    # Prepare TVM arrays using DLPack (TVM 0.24 API)
    device = tvm.runtime.cuda(0)

    # Convert to float32 and use DLPack
    W_f32 = W.float().contiguous()
    scales_f32 = scales.float().contiguous()
    W_dequant_out = torch.empty(N, K, dtype=torch.float32, device="cuda")

    w_tvm = tvm.runtime.from_dlpack(W_f32)
    scales_tvm = tvm.runtime.from_dlpack(scales_f32)
    w_dequant_tvm = tvm.runtime.from_dlpack(W_dequant_out)

    dequant_func = dequant_mod["dequant_nvfp4"]

    # Benchmark hybrid (dequant + cuBLAS)
    # Warmup
    for _ in range(WARMUP):
        dequant_func(w_tvm, scales_tvm, w_dequant_tvm)
        W_bf16 = W_dequant_out.to(torch.bfloat16)
        _ = torch.matmul(A, W_bf16.T)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        dequant_func(w_tvm, scales_tvm, w_dequant_tvm)
        W_bf16 = W_dequant_out.to(torch.bfloat16)
        C = torch.matmul(A, W_bf16.T)
    torch.cuda.synchronize()
    hybrid_time = (time.time() - start) / runs * 1000

    # Benchmark fused
    A_f32 = A.float().contiguous()
    C_fused_out = torch.empty(M, N, dtype=torch.float32, device="cuda")

    a_tvm = tvm.runtime.from_dlpack(A_f32)
    c_tvm = tvm.runtime.from_dlpack(C_fused_out)
    fused_func = fused_mod["w4a16_fused_gemm"]

    for _ in range(WARMUP):
        fused_func(a_tvm, w_tvm, scales_tvm, c_tvm)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        fused_func(a_tvm, w_tvm, scales_tvm, c_tvm)
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / runs * 1000

    print(f"  Shape: [{M}, {K}] x [{K}, {N}]")
    print(f"  Hybrid (dequant+cuBLAS): {hybrid_time:.4f} ms")
    print(f"  Fused (TVM):             {fused_time:.4f} ms")

    return hybrid_time, fused_time


def main():
    print("=" * 70)
    print("TVM TensorIR Kernels - Comprehensive Benchmark")
    print("Target: Break through FP8 12 Hz baseline")
    print("=" * 70)

    results = {}

    for N in N_VALUES:
        print(f"\n{'='*70}")
        print(f"Testing N={N} (K={K})")
        print("=" * 70)

        # FP8 baseline
        fp8_time = benchmark_fp8_baseline(M, N, K)
        results[f"FP8_{N}"] = fp8_time

        # W4A4
        w4a4_time = benchmark_w4a4(M, N, K)
        results[f"W4A4_{N}"] = w4a4_time

        # W4A8
        w4a8_time = benchmark_w4a8(M, N, K)
        results[f"W4A8_{N}"] = w4a8_time

        # W4A16
        w4a16_hybrid, w4a16_fused = benchmark_w4a16(M, N, K)
        results[f"W4A16_hybrid_{N}"] = w4a16_hybrid
        results[f"W4A16_fused_{N}"] = w4a16_fused

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Kernel':<25} {'N=3072 (ms)':<15} {'N=12288 (ms)':<15} {'vs FP8':<10}")
    print("-" * 70)

    for name, n_vals in [("FP8 Baseline", "FP8"), ("W4A4", "W4A4"),
                         ("W4A8", "W4A8"), ("W4A16 Hybrid", "W4A16_hybrid"),
                         ("W4A16 Fused", "W4A16_fused")]:
        t1 = results.get(f"{n_vals}_3072", None)
        t2 = results.get(f"{n_vals}_12288", None)

        t1_str = f"{t1:.4f}" if t1 else "FAIL"
        t2_str = f"{t2:.4f}" if t2 else "FAIL"

        # Calculate speedup vs FP8
        if t1 and results.get("FP8_3072"):
            speedup = results["FP8_3072"] / t1
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "-"

        print(f"{name:<25} {t1_str:<15} {t2_str:<15} {speedup_str:<10}")

    print("-" * 70)

    # Estimate inference Hz
    print("\n" + "=" * 70)
    print("INFERENCE RATE ESTIMATE (Pi0 MLP)")
    print("=" * 70)

    # Pi0 MLP has 4 matmuls per layer: gate_proj, up_proj, down_proj, gate*up
    # 24 layers total
    # gate_proj: [1, 3072] x [3072, 12288]
    # up_proj:   [1, 3072] x [3072, 12288]
    # down_proj: [1, 12288] x [12288, 3072]

    for name, prefix in [("FP8", "FP8"), ("W4A4", "W4A4"),
                         ("W4A8", "W4A8"), ("W4A16 Fused", "W4A16_fused")]:
        t_3072 = results.get(f"{prefix}_3072")
        t_12288 = results.get(f"{prefix}_12288")

        if t_3072 and t_12288:
            # Per layer: 2x [3072, 12288] + 1x [12288, 3072]
            # Approximation: 2 * t_3072 + t_12288
            layer_time = 2 * t_3072 + t_12288
            total_time = layer_time * 24  # 24 layers
            hz = 1000.0 / total_time

            print(f"{name}:")
            print(f"  Per layer:  {layer_time:.2f} ms")
            print(f"  24 layers:  {total_time:.2f} ms")
            print(f"  Est. Hz:    {hz:.1f} Hz")
            print()

    print("=" * 70)
    print("Note: FP8 baseline is ~12 Hz. Target is 14-18 Hz.")
    print("=" * 70)


if __name__ == "__main__":
    main()
