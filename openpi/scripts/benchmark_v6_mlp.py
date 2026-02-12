#!/usr/bin/env python3
"""
Quick benchmark for V6 NVFP4 MLP kernel.

Tests the actual MLP layers from Pi0 model to measure:
1. V6 kernel performance on Pi0 MLP dimensions
2. Comparison with cuBLAS BF16 baseline
"""

import sys
import os
import time

# Setup paths
for path in [
    os.path.join(os.path.dirname(__file__), "..", "src"),
    os.path.join(os.path.dirname(__file__), "..", "nvfp4_packed_plugin", "python"),
    "/workspace/src",
    "/workspace/nvfp4_packed_plugin/python",
]:
    if os.path.exists(path):
        sys.path.insert(0, path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.backends.cudnn.enabled = False


def benchmark_layer(layer, x, warmup=50, runs=200, name=""):
    """Benchmark a layer."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = layer(x)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = layer(x)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    avg = np.mean(latencies)
    std = np.std(latencies)
    print(f"  {name}: {avg:.3f} ms (std: {std:.3f})")
    return avg


def main():
    print("=" * 60)
    print("V6 NVFP4 MLP Kernel Benchmark")
    print("=" * 60)

    device = torch.device('cuda')

    # Pi0 MLP dimensions (hidden_size=2048, intermediate_size=16384)
    # but for Gemma 2b variant, it's typically:
    # hidden_size=2048, intermediate_size=16384
    # Gate/Up proj: [2048] -> [16384]
    # Down proj: [16384] -> [2048]

    # Actually the model uses action expert which is smaller
    # Let's test both the PaLiGemma (larger) and action expert (smaller) sizes

    configs = [
        # (name, hidden_size, intermediate_size, batch_size)
        ("Action Expert MLP (bs=1)", 1024, 4096, 1),
        ("Action Expert MLP (bs=16)", 1024, 4096, 16),
        ("PaLiGemma MLP (bs=1)", 2048, 16384, 1),
    ]

    from nvfp4_packed import NVFP4PackedLinear

    results = {}

    for name, hidden_size, intermediate_size, batch_size in configs:
        print(f"\n--- {name} ---")
        print(f"hidden_size={hidden_size}, intermediate_size={intermediate_size}")

        # Create BF16 baseline MLP
        gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False).to(device, torch.bfloat16)
        up_proj = nn.Linear(hidden_size, intermediate_size, bias=False).to(device, torch.bfloat16)
        down_proj = nn.Linear(intermediate_size, hidden_size, bias=False).to(device, torch.bfloat16)

        # Create NVFP4 MLP with V6 kernel
        # Note: NVFP4PackedLinear.from_linear expects float32 weights on CPU
        gate_proj_f32 = nn.Linear(hidden_size, intermediate_size, bias=False)
        gate_proj_f32.weight.data = gate_proj.weight.data.float()
        up_proj_f32 = nn.Linear(hidden_size, intermediate_size, bias=False)
        up_proj_f32.weight.data = up_proj.weight.data.float()
        down_proj_f32 = nn.Linear(intermediate_size, hidden_size, bias=False)
        down_proj_f32.weight.data = down_proj.weight.data.float()

        gate_proj_nvfp4 = NVFP4PackedLinear.from_linear(gate_proj_f32.cuda(), activation='silu')
        up_proj_nvfp4 = NVFP4PackedLinear.from_linear(up_proj_f32.cuda(), activation='none')
        down_proj_nvfp4 = NVFP4PackedLinear.from_linear(down_proj_f32.cuda(), activation='none')

        # Move to CUDA
        gate_proj_nvfp4 = gate_proj_nvfp4.cuda()
        up_proj_nvfp4 = up_proj_nvfp4.cuda()
        down_proj_nvfp4 = down_proj_nvfp4.cuda()

        # Create input
        x_bf16 = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
        x_float = x_bf16.float()

        print("\nBF16 cuBLAS baseline:")
        t_gate_bf16 = benchmark_layer(gate_proj, x_bf16, name="gate_proj")
        t_up_bf16 = benchmark_layer(up_proj, x_bf16, name="up_proj")

        # For down_proj, input comes from intermediate layer
        intermediate_x_bf16 = torch.randn(batch_size, intermediate_size, device=device, dtype=torch.bfloat16)
        t_down_bf16 = benchmark_layer(down_proj, intermediate_x_bf16, name="down_proj")
        total_bf16 = t_gate_bf16 + t_up_bf16 + t_down_bf16
        print(f"  Total: {total_bf16:.3f} ms")

        print("\nNVFP4 V6 kernel:")
        t_gate_nvfp4 = benchmark_layer(gate_proj_nvfp4, x_float, name="gate_proj")
        t_up_nvfp4 = benchmark_layer(up_proj_nvfp4, x_float, name="up_proj")

        # For down_proj, input comes from intermediate layer
        intermediate_x_float = torch.randn(batch_size, intermediate_size, device=device, dtype=torch.float32)
        t_down_nvfp4 = benchmark_layer(down_proj_nvfp4, intermediate_x_float, name="down_proj")
        total_nvfp4 = t_gate_nvfp4 + t_up_nvfp4 + t_down_nvfp4
        print(f"  Total: {total_nvfp4:.3f} ms")

        speedup = total_bf16 / total_nvfp4
        print(f"\n=> Speedup: {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}")

        results[name] = {
            'bf16': total_bf16,
            'nvfp4': total_nvfp4,
            'speedup': speedup
        }

        # Clean up
        del gate_proj, up_proj, down_proj
        del gate_proj_nvfp4, up_proj_nvfp4, down_proj_nvfp4
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, data in results.items():
        print(f"{name}:")
        print(f"  BF16:  {data['bf16']:.3f} ms")
        print(f"  NVFP4: {data['nvfp4']:.3f} ms")
        print(f"  Speedup: {data['speedup']:.2f}x")

    # Estimate full model impact
    # Pi0 has 18 transformer layers each with MLP
    print("\n" + "=" * 60)
    print("Estimated Full Model MLP Contribution (18 layers)")
    print("=" * 60)

    # Use Action Expert results (bs=1) for estimation
    if "Action Expert MLP (bs=1)" in results:
        data = results["Action Expert MLP (bs=1)"]
        print(f"BF16 MLP total:  {data['bf16'] * 18:.2f} ms")
        print(f"NVFP4 MLP total: {data['nvfp4'] * 18:.2f} ms")
        print(f"MLP time saved:  {(data['bf16'] - data['nvfp4']) * 18:.2f} ms")


if __name__ == "__main__":
    main()
