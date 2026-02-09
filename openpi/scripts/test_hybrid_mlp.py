#!/usr/bin/env python3
"""
Test Hybrid Precision MLP (Gate/Up NVFP4, Down BF16)

This is Plan A: A stop-gap solution before Scale Layout is fixed.

Expected results:
- Precision: 0.97+ cosine similarity (vs 0.93 for full NVFP4)
- Speed: ~2x acceleration (vs 5.88x for full NVFP4)
- Bandwidth saving: 50% (2/3 weights use NVFP4, 1/3 stays BF16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    NVFP4MLP,
    HybridPrecisionMLP,
    BLOCK_SIZE,
)


def test_precision():
    """Test precision comparison between Full NVFP4 and Hybrid MLP."""
    print("=" * 70)
    print("Precision Test: Full NVFP4 vs Hybrid (Gate/Up NVFP4, Down BF16)")
    print("=" * 70)

    device = torch.device('cuda')
    batch_size = 256
    hidden_size = 2048
    intermediate_size = 16384

    # Create original BF16 MLP
    class BF16MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            gate = F.gelu(self.gate_proj(x), approximate='tanh')
            return self.down_proj(gate * self.up_proj(x))

    bf16_mlp = BF16MLP().to(device).bfloat16()

    # Create NVFP4 MLP (full quantization)
    nvfp4_mlp = NVFP4MLP.from_gemma_mlp(bf16_mlp, BLOCK_SIZE).to(device).bfloat16()

    # Create Hybrid MLP (Gate/Up NVFP4, Down BF16)
    hybrid_mlp = HybridPrecisionMLP.from_gemma_mlp(bf16_mlp, BLOCK_SIZE).to(device).bfloat16()

    # Test input
    torch.manual_seed(42)
    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    # Get outputs
    with torch.no_grad():
        bf16_out = bf16_mlp(x)
        nvfp4_out = nvfp4_mlp(x)
        hybrid_out = hybrid_mlp(x)

    # Calculate metrics
    def calc_metrics(pred, ref):
        cos_sim = F.cosine_similarity(
            pred.flatten().float().unsqueeze(0),
            ref.flatten().float().unsqueeze(0)
        ).item()

        mse = ((pred.float() - ref.float()) ** 2).mean().item()
        rmse = mse ** 0.5

        rel_error = ((pred.float() - ref.float()).abs() / (ref.float().abs() + 1e-8)).mean().item()

        return cos_sim, rmse, rel_error

    nvfp4_cos, nvfp4_rmse, nvfp4_rel = calc_metrics(nvfp4_out, bf16_out)
    hybrid_cos, hybrid_rmse, hybrid_rel = calc_metrics(hybrid_out, bf16_out)

    print(f"\nConfig: batch={batch_size}, hidden={hidden_size}, intermediate={intermediate_size}")
    print()
    print(f"{'Method':<25} {'Cosine Sim':>12} {'RMSE':>12} {'Rel Error':>12}")
    print("-" * 65)
    print(f"{'Full NVFP4':<25} {nvfp4_cos:>12.6f} {nvfp4_rmse:>12.6f} {nvfp4_rel*100:>11.2f}%")
    print(f"{'Hybrid (Gate/Up NVFP4)':<25} {hybrid_cos:>12.6f} {hybrid_rmse:>12.6f} {hybrid_rel*100:>11.2f}%")
    print()

    # Analysis
    if hybrid_cos > nvfp4_cos + 0.01:
        print(f"[IMPROVEMENT] Hybrid is better: {(hybrid_cos - nvfp4_cos)*100:.2f}% higher cosine sim")
    elif hybrid_cos > 0.97:
        print(f"[GOOD] Hybrid achieves 0.97+ cosine similarity!")
    else:
        print(f"[NOTE] Hybrid cosine: {hybrid_cos:.4f}")

    # Get stats
    stats = hybrid_mlp.get_stats()
    print(f"\nHybrid MLP Stats:")
    print(f"  Total params:     {stats['total_params']:,}")
    print(f"  NVFP4 ratio:      {stats['nvfp4_ratio']*100:.1f}% (Gate + Up)")
    print(f"  BF16 ratio:       {stats['bf16_ratio']*100:.1f}% (Down)")
    print(f"  Bandwidth saving: {stats['bandwidth_saving']*100:.1f}%")
    print(f"  Expected speedup: ~{stats['expected_speedup']:.1f}x")

    return hybrid_cos, nvfp4_cos


def test_latency():
    """Test latency comparison."""
    print("\n" + "=" * 70)
    print("Latency Test")
    print("=" * 70)

    device = torch.device('cuda')
    batch_size = 256
    hidden_size = 2048
    intermediate_size = 16384

    class BF16MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            gate = F.gelu(self.gate_proj(x), approximate='tanh')
            return self.down_proj(gate * self.up_proj(x))

    bf16_mlp = BF16MLP().to(device).bfloat16()
    nvfp4_mlp = NVFP4MLP.from_gemma_mlp(bf16_mlp, BLOCK_SIZE).to(device).bfloat16()
    hybrid_mlp = HybridPrecisionMLP.from_gemma_mlp(bf16_mlp, BLOCK_SIZE).to(device).bfloat16()

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = bf16_mlp(x)
        _ = nvfp4_mlp(x)
        _ = hybrid_mlp(x)
    torch.cuda.synchronize()

    iterations = 50

    # BF16
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = bf16_mlp(x)
    torch.cuda.synchronize()
    bf16_ms = (time.perf_counter() - start) / iterations * 1000

    # NVFP4 (simulation mode - actual CUTLASS would be faster)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = nvfp4_mlp(x)
    torch.cuda.synchronize()
    nvfp4_ms = (time.perf_counter() - start) / iterations * 1000

    # Hybrid (simulation mode)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = hybrid_mlp(x)
    torch.cuda.synchronize()
    hybrid_ms = (time.perf_counter() - start) / iterations * 1000

    print(f"\nLatency (simulation mode - NOT final performance):")
    print(f"  BF16 MLP:    {bf16_ms:.3f} ms")
    print(f"  NVFP4 MLP:   {nvfp4_ms:.3f} ms (sim, actual CUTLASS ~{bf16_ms/5.88:.3f} ms)")
    print(f"  Hybrid MLP:  {hybrid_ms:.3f} ms (sim, actual ~{bf16_ms/2:.3f} ms)")
    print()
    print("Note: Simulation is slower than actual CUTLASS NVFP4 GEMM.")
    print("      Expected speedups with CUTLASS:")
    print(f"      - Full NVFP4: ~5.88x ({bf16_ms:.3f} ms -> {bf16_ms/5.88:.3f} ms)")
    print(f"      - Hybrid:     ~2.0x  ({bf16_ms:.3f} ms -> {bf16_ms/2:.3f} ms)")


def test_layer_by_layer():
    """Test each layer's contribution to error."""
    print("\n" + "=" * 70)
    print("Layer-by-Layer Error Analysis")
    print("=" * 70)

    device = torch.device('cuda')
    batch_size = 256
    hidden_size = 2048
    intermediate_size = 16384

    class BF16MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    bf16_mlp = BF16MLP().to(device).bfloat16()
    nvfp4_mlp = NVFP4MLP.from_gemma_mlp(bf16_mlp, BLOCK_SIZE).to(device).bfloat16()

    torch.manual_seed(42)
    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        # BF16 reference
        gate_bf16 = F.gelu(bf16_mlp.gate_proj(x), approximate='tanh')
        up_bf16 = bf16_mlp.up_proj(x)
        hidden_bf16 = gate_bf16 * up_bf16
        out_bf16 = bf16_mlp.down_proj(hidden_bf16)

        # NVFP4
        gate_nvfp4 = F.gelu(nvfp4_mlp.gate_proj(x), approximate='tanh')
        up_nvfp4 = nvfp4_mlp.up_proj(x)
        hidden_nvfp4 = gate_nvfp4 * up_nvfp4
        out_nvfp4 = nvfp4_mlp.down_proj(hidden_nvfp4)

    # Calculate cosine similarity for each stage
    def cos_sim(a, b):
        return F.cosine_similarity(
            a.flatten().float().unsqueeze(0),
            b.flatten().float().unsqueeze(0)
        ).item()

    gate_cos = cos_sim(gate_nvfp4, gate_bf16)
    up_cos = cos_sim(up_nvfp4, up_bf16)
    hidden_cos = cos_sim(hidden_nvfp4, hidden_bf16)
    out_cos = cos_sim(out_nvfp4, out_bf16)

    print(f"\nCosine similarity at each stage (NVFP4 vs BF16):")
    print(f"  After gate_proj: {gate_cos:.6f}")
    print(f"  After up_proj:   {up_cos:.6f}")
    print(f"  After gate*up:   {hidden_cos:.6f}")
    print(f"  After down_proj: {out_cos:.6f}")
    print()

    # Identify the biggest drop
    drops = [
        ("gate_proj", 1.0 - gate_cos),
        ("up_proj", 1.0 - up_cos),
        ("gate*up", hidden_cos - min(gate_cos, up_cos)),  # Combined error
        ("down_proj", hidden_cos - out_cos),
    ]

    max_drop_layer, max_drop = max(drops, key=lambda x: abs(x[1]))
    print(f"Biggest precision drop: {max_drop_layer} (delta={max_drop:.6f})")

    if max_drop_layer == "down_proj":
        print("\n[CONFIRM] down_proj is the most sensitive layer!")
        print("         Hybrid MLP strategy (keeping down_proj in BF16) is correct.")


def main():
    print("Hybrid Precision MLP Test")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    # Run tests
    hybrid_cos, nvfp4_cos = test_precision()
    test_latency()
    test_layer_by_layer()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
Hybrid MLP Strategy (Gate/Up NVFP4, Down BF16):

Precision:
  - Full NVFP4:  {nvfp4_cos:.4f} cosine sim
  - Hybrid:      {hybrid_cos:.4f} cosine sim
  - Improvement: {(hybrid_cos - nvfp4_cos)*100:.2f}%

Performance:
  - Bandwidth saving: 50% (2/3 weights use NVFP4)
  - Expected speedup: ~2x (vs BF16)

Status: {"RECOMMENDED" if hybrid_cos > 0.97 else "NEEDS MORE WORK"}

Next Steps:
  1. If hybrid works (0.97+): Deploy and move to Plan B (Grid Search)
  2. Run Grid Search to find correct Scale Layout
  3. Once Layout is fixed, switch to full NVFP4 for 5.88x speedup
""")


if __name__ == "__main__":
    main()
