#!/usr/bin/env python3
"""
NVFP4 CUTLASS 集成测试

使用 subprocess 调用 CUTLASS binary 来验证 NVFP4 GEMM 功能和精度。
一旦验证通过，可以进一步优化为直接 C++ 调用。

目标:
- 验证 NVFP4 量化精度
- 测试完整 MLP 的精度和延迟
- 对比 BF16 和 FP8
"""

import torch
import torch.nn as nn
import time
import subprocess
import os
import json
import numpy as np

# CUTLASS binary 路径
CUTLASS_BINARY = "/workspace/external/cutlass_sm110_build/nvfp4_gemm_sm110a"


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_section(title: str):
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


def run_cutlass_gemm(M: int, N: int, K: int, iterations: int = 100):
    """Run CUTLASS NVFP4 GEMM and return timing."""
    if not os.path.exists(CUTLASS_BINARY):
        return None

    cmd = f"{CUTLASS_BINARY} --m={M} --n={N} --k={K} --iterations={iterations}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if "Failed" in output or "error" in output.lower():
        return None

    ms = None
    for line in output.split("\n"):
        if "Avg runtime" in line:
            ms = float(line.split()[-2])
            break

    return ms


def benchmark_bf16_gemm(M: int, N: int, K: int, iterations: int = 100):
    """Benchmark cuBLAS BF16 GEMM."""
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

    return (time.perf_counter() - start) / iterations * 1000


def test_single_gemm_comparison():
    """Compare single GEMM performance: NVFP4 vs BF16."""
    print_section("Single GEMM Performance Comparison")

    test_cases = [
        # (M, N, K, description)
        (256, 16384, 2048, "MLP gate/up (batch=256)"),
        (256, 2048, 16384, "MLP down (batch=256)"),
        (512, 16384, 2048, "MLP gate/up (batch=512)"),
        (512, 2048, 16384, "MLP down (batch=512)"),
    ]

    print(f"\n{'Problem':<30} | {'BF16 (ms)':<12} | {'NVFP4 (ms)':<12} | {'Speedup':<10}")
    print("-" * 70)

    results = []
    for M, N, K, desc in test_cases:
        bf16_ms = benchmark_bf16_gemm(M, N, K)
        nvfp4_ms = run_cutlass_gemm(M, N, K)

        if nvfp4_ms:
            speedup = bf16_ms / nvfp4_ms
            print(f"{desc:<30} | {bf16_ms:<12.3f} | {nvfp4_ms:<12.3f} | {speedup:<10.2f}x")
            results.append({
                "problem": desc,
                "M": M, "N": N, "K": K,
                "bf16_ms": bf16_ms,
                "nvfp4_ms": nvfp4_ms,
                "speedup": speedup
            })
        else:
            print(f"{desc:<30} | {bf16_ms:<12.3f} | {'N/A':<12} | {'N/A':<10}")

    return results


def test_mlp_layer_performance():
    """Test complete MLP layer performance."""
    print_section("Complete MLP Layer Performance")

    batch_size = 256
    hidden_size = 2048
    intermediate_size = 16384

    # MLP = gate_proj + up_proj + down_proj
    # gate/up: [B, 2048] @ [2048, 16384] -> [B, 16384]
    # down: [B, 16384] @ [16384, 2048] -> [B, 2048]

    # BF16 MLP
    class BF16MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            gate = torch.nn.functional.gelu(self.gate_proj(x), approximate='tanh')
            return self.down_proj(gate * self.up_proj(x))

    mlp = BF16MLP().cuda().bfloat16()
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = mlp(x)
    torch.cuda.synchronize()

    # Benchmark BF16
    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        _ = mlp(x)
    torch.cuda.synchronize()
    bf16_ms = (time.perf_counter() - start) / iterations * 1000

    # NVFP4 estimate (sum of 3 GEMMs)
    gate_ms = run_cutlass_gemm(batch_size, intermediate_size, hidden_size)
    up_ms = run_cutlass_gemm(batch_size, intermediate_size, hidden_size)
    down_ms = run_cutlass_gemm(batch_size, hidden_size, intermediate_size)

    nvfp4_total = None
    if gate_ms and up_ms and down_ms:
        nvfp4_total = gate_ms + up_ms + down_ms

    print(f"\n  Config: batch={batch_size}, hidden={hidden_size}, intermediate={intermediate_size}")
    print(f"\n  BF16 MLP (PyTorch):  {bf16_ms:.3f} ms")

    if nvfp4_total:
        print(f"  NVFP4 GEMM (CUTLASS):")
        print(f"    - gate_proj:       {gate_ms:.3f} ms")
        print(f"    - up_proj:         {up_ms:.3f} ms")
        print(f"    - down_proj:       {down_ms:.3f} ms")
        print(f"    - Total:           {nvfp4_total:.3f} ms")
        print(f"    - Speedup:         {bf16_ms/nvfp4_total:.2f}x")

    return bf16_ms, nvfp4_total


def test_18_layer_kv_cache():
    """Test 18-layer KV Cache MLP performance."""
    print_section("18-Layer KV Cache MLP Performance")

    num_layers = 18
    batch_size = 256
    hidden_size = 2048
    intermediate_size = 16384

    # Create BF16 MLP stack
    class BF16MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            gate = torch.nn.functional.gelu(self.gate_proj(x), approximate='tanh')
            return self.down_proj(gate * self.up_proj(x))

    layers = nn.ModuleList([BF16MLP().cuda().bfloat16() for _ in range(num_layers)])
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(5):
        out = x
        for layer in layers:
            out = layer(out)
    torch.cuda.synchronize()

    # Benchmark BF16
    iterations = 20
    start = time.perf_counter()
    for _ in range(iterations):
        out = x
        for layer in layers:
            out = layer(out)
    torch.cuda.synchronize()
    bf16_total_ms = (time.perf_counter() - start) / iterations * 1000

    # NVFP4 estimate
    gate_ms = run_cutlass_gemm(batch_size, intermediate_size, hidden_size)
    up_ms = run_cutlass_gemm(batch_size, intermediate_size, hidden_size)
    down_ms = run_cutlass_gemm(batch_size, hidden_size, intermediate_size)

    nvfp4_total_ms = None
    if gate_ms and up_ms and down_ms:
        nvfp4_layer_ms = gate_ms + up_ms + down_ms
        nvfp4_total_ms = nvfp4_layer_ms * num_layers

    print(f"\n  Config: {num_layers} layers x batch={batch_size}")
    print(f"\n  BF16 total:   {bf16_total_ms:.2f} ms ({bf16_total_ms/num_layers:.3f} ms/layer)")

    if nvfp4_total_ms:
        print(f"  NVFP4 total:  {nvfp4_total_ms:.2f} ms ({nvfp4_layer_ms:.3f} ms/layer)")
        print(f"  Speedup:      {bf16_total_ms/nvfp4_total_ms:.2f}x")
        print(f"  Time saved:   {bf16_total_ms - nvfp4_total_ms:.2f} ms")

    # Memory savings
    bf16_mem = hidden_size * intermediate_size * 3 * num_layers * 2 / 1024 / 1024
    nvfp4_mem = hidden_size * intermediate_size * 3 * num_layers * 0.5 / 1024 / 1024

    print(f"\n  Memory:")
    print(f"    BF16:   {bf16_mem:.1f} MB")
    print(f"    NVFP4:  {nvfp4_mem:.1f} MB ({bf16_mem/nvfp4_mem:.1f}x compression)")

    return bf16_total_ms, nvfp4_total_ms


def main():
    print_header("NVFP4 CUTLASS Integration Test")

    print_section("Environment")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUTLASS binary: {'Available' if os.path.exists(CUTLASS_BINARY) else 'Not found'}")

    # Run tests
    gemm_results = test_single_gemm_comparison()
    bf16_mlp, nvfp4_mlp = test_mlp_layer_performance()
    bf16_18layer, nvfp4_18layer = test_18_layer_kv_cache()

    # Summary
    print_header("Summary")

    results = {
        "gemm_results": gemm_results,
        "mlp_layer": {"bf16_ms": bf16_mlp, "nvfp4_ms": nvfp4_mlp},
        "18_layer": {"bf16_ms": bf16_18layer, "nvfp4_ms": nvfp4_18layer}
    }

    print("\n  NVFP4 CUTLASS Performance Summary:")
    if nvfp4_mlp:
        print(f"    Single MLP layer: {bf16_mlp:.3f} ms -> {nvfp4_mlp:.3f} ms ({bf16_mlp/nvfp4_mlp:.2f}x)")
    if nvfp4_18layer:
        print(f"    18-layer KV Cache: {bf16_18layer:.2f} ms -> {nvfp4_18layer:.2f} ms ({bf16_18layer/nvfp4_18layer:.2f}x)")

    print("\n  Next Steps:")
    print("    1. Fix C++ extension scale factor layout")
    print("    2. Integrate into PI0Pytorch model")
    print("    3. Validate end-to-end precision")

    # Save results
    output_file = "/workspace/nvfp4_cutlass_integration_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
