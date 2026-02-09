#!/usr/bin/env python3
"""
NVFP4 vs cuBLAS BF16 Benchmark
"""
import torch
import time
import subprocess
import os

def benchmark_cublas_bf16(M, N, K, iterations=100):
    """Benchmark cuBLAS BF16 GEMM"""
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = torch.matmul(x, w.T)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(x, w.T)
    torch.cuda.synchronize()

    ms = (time.perf_counter() - start) / iterations * 1000
    gflops = 2 * M * N * K / (ms / 1000) / 1e9
    return ms, gflops

def run_cutlass_nvfp4(M, N, K, iterations=100):
    """Run CUTLASS NVFP4 GEMM benchmark"""
    cmd = f"/workspace/external/cutlass_sm110_build/nvfp4_gemm_sm110a --m={M} --n={N} --k={K} --iterations={iterations}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if "Failed" in output:
        return None, None

    # Parse output
    ms = None
    gflops = None
    for line in output.split("\n"):
        if "Avg runtime" in line:
            ms = float(line.split()[-2])
        if "GFLOPS" in line:
            gflops = float(line.split()[-1])

    return ms, gflops

def main():
    print("=" * 70)
    print("NVFP4 vs cuBLAS BF16 Benchmark on Thor SM110")
    print("=" * 70)
    print()

    test_cases = [
        # Pi0.5 MLP sizes (approximate working sizes)
        (256, 16384, 2048, "Pi0.5 MLP gate/up (padded to 256)"),
        (256, 2048, 16384, "Pi0.5 MLP down (padded to 256)"),
        (512, 8192, 2048, "Pi0.5 MLP gate/up split"),
        (512, 2048, 8192, "Pi0.5 MLP down split"),
        # Larger sizes
        (1024, 4096, 2048, "Large batch"),
    ]

    header = f"{'Problem Size':<30} | {'BF16 (ms)':<12} | {'NVFP4 (ms)':<12} | {'Speedup':<10} | {'Notes'}"
    print(header)
    print("-" * 90)

    for M, N, K, desc in test_cases:
        bf16_ms, bf16_gflops = benchmark_cublas_bf16(M, N, K)
        nvfp4_ms, nvfp4_gflops = run_cutlass_nvfp4(M, N, K)

        size_str = f"{M}x{N}x{K}"

        if nvfp4_ms is None:
            print(f"{size_str:<30} | {bf16_ms:<12.3f} | {'FAILED':<12} | {'N/A':<10} | {desc}")
        else:
            speedup = bf16_ms / nvfp4_ms
            print(f"{size_str:<30} | {bf16_ms:<12.3f} | {nvfp4_ms:<12.3f} | {speedup:<10.2f}x | {desc}")

    print()
    print("=" * 70)
    print("Summary: NVFP4 GEMM on Thor SM110 with CUTLASS")
    print("=" * 70)

if __name__ == "__main__":
    main()
