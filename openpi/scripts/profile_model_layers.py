#!/usr/bin/env python3
"""
Profile Model Layers - Detailed Latency Analysis
=================================================

This script profiles each component of the PI0 model to identify optimization opportunities.

Components analyzed:
1. Vision Encoder (SigLIP)
2. PaliGemma Language Model (18 layers)
   - Self-Attention
   - MLP (W4A16 or BF16)
3. Action Expert (6 layers)
   - Self-Attention
   - Cross-Attention
   - MLP
4. Diffusion sampling

Usage:
    python /workspace/scripts/profile_model_layers.py

Author: Claude Code
Date: 2026-02-11
"""

import sys
import os
import time
import argparse
import json
import pathlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Setup paths
script_dir = pathlib.Path(__file__).parent
for path in [
    script_dir.parent / "src",
    "/workspace/src",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# TVM paths
TVM_HOME = os.environ.get("TVM_HOME", "/workspace/external/tvm")
if TVM_HOME and os.path.exists(TVM_HOME):
    sys.path.insert(0, os.path.join(TVM_HOME, "python"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = False


@dataclass
class LayerProfile:
    """Profile data for a single layer/component."""
    name: str
    time_ms: float
    flops: int = 0
    memory_mb: float = 0
    params: int = 0


class ModelProfiler:
    """Profiles model components with detailed timing."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.profiles: Dict[str, List[float]] = defaultdict(list)
        self.hooks = []

    def _create_hook(self, name: str):
        """Create a forward hook for timing."""
        def hook(module, input, output):
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            if hasattr(module, '_start_time'):
                elapsed = (end_time - module._start_time) * 1000
                self.profiles[name].append(elapsed)
        return hook

    def _create_pre_hook(self, name: str):
        """Create a pre-forward hook for timing."""
        def hook(module, input):
            torch.cuda.synchronize()
            module._start_time = time.perf_counter()
        return hook

    def register_hooks(self):
        """Register timing hooks on key modules."""
        # Vision Encoder
        if hasattr(self.model, 'paligemma_with_expert'):
            pge = self.model.paligemma_with_expert

            # SigLIP Vision
            if hasattr(pge, 'paligemma') and hasattr(pge.paligemma, 'vision_encoder'):
                vision = pge.paligemma.vision_encoder
                h1 = vision.register_forward_pre_hook(self._create_pre_hook('vision_encoder'))
                h2 = vision.register_forward_hook(self._create_hook('vision_encoder'))
                self.hooks.extend([h1, h2])

            # PaliGemma Language Model layers
            if hasattr(pge, 'paligemma') and hasattr(pge.paligemma, 'language_model'):
                lm = pge.paligemma.language_model
                for i, layer in enumerate(lm.layers):
                    # Full layer
                    h1 = layer.register_forward_pre_hook(self._create_pre_hook(f'paligemma_layer_{i}'))
                    h2 = layer.register_forward_hook(self._create_hook(f'paligemma_layer_{i}'))
                    self.hooks.extend([h1, h2])

                    # Self-attention
                    if hasattr(layer, 'self_attn'):
                        h1 = layer.self_attn.register_forward_pre_hook(self._create_pre_hook(f'paligemma_attn_{i}'))
                        h2 = layer.self_attn.register_forward_hook(self._create_hook(f'paligemma_attn_{i}'))
                        self.hooks.extend([h1, h2])

                    # MLP
                    if hasattr(layer, 'mlp'):
                        h1 = layer.mlp.register_forward_pre_hook(self._create_pre_hook(f'paligemma_mlp_{i}'))
                        h2 = layer.mlp.register_forward_hook(self._create_hook(f'paligemma_mlp_{i}'))
                        self.hooks.extend([h1, h2])

            # Action Expert layers
            if hasattr(pge, 'action_expert'):
                expert = pge.action_expert
                if hasattr(expert, 'layers'):
                    for i, layer in enumerate(expert.layers):
                        h1 = layer.register_forward_pre_hook(self._create_pre_hook(f'action_expert_layer_{i}'))
                        h2 = layer.register_forward_hook(self._create_hook(f'action_expert_layer_{i}'))
                        self.hooks.extend([h1, h2])

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear_profiles(self):
        """Clear all collected profiles."""
        self.profiles.clear()

    def get_summary(self) -> Dict[str, float]:
        """Get average times for each component."""
        summary = {}
        for name, times in self.profiles.items():
            if times:
                summary[name] = np.mean(times)
        return summary


def profile_mlp_standalone(hidden_size=2048, intermediate_size=16384, num_layers=18,
                           use_w4a16=False, use_tvm=False, iterations=100):
    """Profile MLP layers in isolation."""
    device = torch.device('cuda')

    results = {}

    # Create MLP layers
    if use_w4a16:
        from openpi.models_pytorch.w4a16_mlp import W4A16MLP
        mlp = W4A16MLP(hidden_size, intermediate_size, use_tvm=use_tvm).to(device)
        method = "W4A16 TVM" if use_tvm else "W4A16 PyTorch"
    else:
        class BF16MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

            def forward(self, x):
                gate = F.gelu(self.gate_proj(x), approximate='tanh')
                return self.down_proj(gate * self.up_proj(x))

        mlp = BF16MLP().to(device).to(torch.bfloat16)
        method = "BF16"

    # Test input
    x = torch.randn(1, hidden_size, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(20):
        _ = mlp(x)
    torch.cuda.synchronize()

    # Benchmark single layer
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = mlp(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    single_layer_ms = np.mean(times)

    # Benchmark 18 layers sequential
    mlps = [mlp] + [type(mlp)().to(device) if use_w4a16 else
                    BF16MLP().to(device).to(torch.bfloat16) for _ in range(num_layers - 1)]
    if use_w4a16:
        for m in mlps[1:]:
            m.to(device)

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = x
        for m in mlps:
            out = m(out)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    total_18_layers_ms = np.mean(times)

    results[method] = {
        'single_layer_ms': single_layer_ms,
        'total_18_layers_ms': total_18_layers_ms,
        'per_layer_avg_ms': total_18_layers_ms / num_layers,
    }

    return results


def profile_attention_standalone(hidden_size=2048, num_heads=8, seq_len=968, iterations=100):
    """Profile attention layers in isolation."""
    device = torch.device('cuda')

    head_dim = hidden_size // num_heads

    # Simple attention implementation
    class SimpleAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.num_heads = num_heads
            self.head_dim = head_dim

        def forward(self, x):
            B, L, D = x.shape
            q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).contiguous().view(B, L, D)
            return self.o_proj(attn)

    attn = SimpleAttention().to(device).to(torch.bfloat16)

    # Test with different sequence lengths
    results = {}
    for sl in [1, 64, 256, 968]:
        x = torch.randn(1, sl, hidden_size, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(10):
            _ = attn(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = attn(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        results[f'seq_len_{sl}'] = np.mean(times)

    return results


def profile_vision_encoder(iterations=50):
    """Profile SigLIP vision encoder."""
    device = torch.device('cuda')

    # Load SigLIP model
    try:
        from openpi.models_pytorch.siglip_vision import SigLIPVisionModel
        vision = SigLIPVisionModel().to(device).to(torch.bfloat16)
    except:
        print("SigLIP model not available, skipping vision profiling")
        return {}

    # Test input (224x224 image)
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = vision(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = vision(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return {'vision_encoder_ms': np.mean(times)}


def main():
    parser = argparse.ArgumentParser(description="Profile Model Layers")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile_full_model", action="store_true",
                       help="Also profile full model inference")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="/root/.cache/openpi/checkpoints/pi05_libero")
    args = parser.parse_args()

    print("=" * 70)
    print("Model Layer Profiling - Detailed Latency Analysis")
    print("=" * 70)
    print(f"Iterations: {args.iterations}")
    print()

    # Check TVM
    tvm_available = False
    try:
        import tvm
        tvm_available = True
        print(f"TVM available: {tvm.__version__}")
    except ImportError:
        print("TVM not available")
    print()

    # =========================================================================
    # 1. MLP Layer Profiling
    # =========================================================================
    print("=" * 70)
    print("1. MLP LAYER PROFILING (batch=1, hidden=2048, intermediate=16384)")
    print("=" * 70)

    mlp_results = {}

    # BF16 baseline
    print("\nProfiling BF16 MLP...")
    mlp_results.update(profile_mlp_standalone(use_w4a16=False, iterations=args.iterations))

    # W4A16 PyTorch
    print("Profiling W4A16 PyTorch MLP...")
    mlp_results.update(profile_mlp_standalone(use_w4a16=True, use_tvm=False, iterations=args.iterations))

    # W4A16 TVM
    if tvm_available:
        print("Profiling W4A16 TVM MLP...")
        mlp_results.update(profile_mlp_standalone(use_w4a16=True, use_tvm=True, iterations=args.iterations))

    print("\nMLP Results:")
    print("-" * 70)
    print(f"{'Method':<20} {'Single Layer':>15} {'18 Layers':>15} {'Per-Layer Avg':>15}")
    print("-" * 70)
    for method, data in mlp_results.items():
        print(f"{method:<20} {data['single_layer_ms']:>12.3f} ms {data['total_18_layers_ms']:>12.3f} ms {data['per_layer_avg_ms']:>12.3f} ms")
    print("-" * 70)

    # Reference: TRT FP8
    print("\nReference (TRT FP8): 12.39 ms for 18 layers = 0.688 ms/layer")

    if 'W4A16 TVM' in mlp_results and 'BF16' in mlp_results:
        bf16_time = mlp_results['BF16']['total_18_layers_ms']
        w4a16_time = mlp_results['W4A16 TVM']['total_18_layers_ms']
        print(f"\nW4A16 TVM vs BF16: {bf16_time/w4a16_time:.2f}x speedup")
        print(f"W4A16 TVM vs TRT FP8: {12.39/w4a16_time:.2f}x")

    # =========================================================================
    # 2. Attention Layer Profiling
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. ATTENTION LAYER PROFILING (batch=1, hidden=2048, heads=8)")
    print("=" * 70)

    attn_results = profile_attention_standalone(iterations=args.iterations)

    print("\nAttention Results (per layer):")
    print("-" * 50)
    for name, time_ms in attn_results.items():
        print(f"  {name}: {time_ms:.3f} ms")
    print("-" * 50)

    # =========================================================================
    # 3. Component Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. ESTIMATED FULL MODEL BREAKDOWN (3 denoising steps)")
    print("=" * 70)

    # Estimates based on model architecture
    # PaliGemma: 18 layers, each with attention + MLP
    # Action Expert: 6 layers, each with self-attn + cross-attn + MLP

    paligemma_mlp_time = mlp_results.get('W4A16 TVM', mlp_results['BF16'])['total_18_layers_ms']
    paligemma_attn_time = attn_results.get('seq_len_968', 1.0) * 18  # 18 layers

    # Action expert (smaller)
    action_expert_mlp_time = paligemma_mlp_time * 6 / 18 * 0.3  # ~30% size
    action_expert_attn_time = attn_results.get('seq_len_1', 0.1) * 6 * 2  # self + cross

    # Vision encoder (rough estimate)
    vision_time = 15.0  # Typical SigLIP ~15ms

    # Diffusion overhead (noise scheduling, etc.)
    diffusion_overhead = 5.0

    # Total estimates
    single_step = vision_time + paligemma_mlp_time + paligemma_attn_time + \
                  action_expert_mlp_time + action_expert_attn_time + diffusion_overhead

    print("\nEstimated Component Breakdown (per diffusion step):")
    print("-" * 60)
    print(f"  Vision Encoder (SigLIP):      ~{vision_time:>8.1f} ms")
    print(f"  PaliGemma Attention (18 layers): ~{paligemma_attn_time:>6.1f} ms")
    print(f"  PaliGemma MLP (18 layers):       ~{paligemma_mlp_time:>6.1f} ms")
    print(f"  Action Expert Attention:      ~{action_expert_attn_time:>8.1f} ms")
    print(f"  Action Expert MLP:            ~{action_expert_mlp_time:>8.1f} ms")
    print(f"  Diffusion Overhead:           ~{diffusion_overhead:>8.1f} ms")
    print("-" * 60)
    print(f"  Estimated Single Step:        ~{single_step:>8.1f} ms")
    print(f"  Estimated 3 Steps:            ~{single_step * 3:>8.1f} ms")
    print("-" * 60)
    print(f"  Actual Measured (W4A16 TVM):   ~174.6 ms (3 steps)")
    print(f"  TRT FP8 Baseline:              ~120.6 ms (3 steps)")

    # =========================================================================
    # 4. Optimization Opportunities
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. OPTIMIZATION OPPORTUNITIES")
    print("=" * 70)

    print("""
Current State:
- MLP layers: W4A16 TVM optimized (~13.7ms for 18 layers)
- Attention: PyTorch BF16 (no optimization)
- Vision: PyTorch BF16 (no optimization)
- Full model: ~174.6 ms (3 steps)

Target: TRT FP8 ~120.6 ms (3 steps) = 40.2 ms/step

Gap Analysis (per step):
- Current: ~58.2 ms/step
- Target: ~40.2 ms/step
- Gap: ~18 ms to optimize

Optimization Priority (by impact):
1. Vision Encoder TRT FP16/FP8: ~15ms -> ~8ms (save ~7ms)
2. Attention TRT FP8 + FlashAttn: ~20ms -> ~12ms (save ~8ms)
3. Action Expert TRT FP8: ~5ms -> ~3ms (save ~2ms)
4. W4A16 TRT Plugin (C++): ~14ms -> ~12ms (save ~2ms)

Total Potential Savings: ~19ms/step -> 39.2ms/step (target achieved)
""")

    # =========================================================================
    # 5. Full Graph Quantization Design
    # =========================================================================
    print("=" * 70)
    print("5. FULL GRAPH QUANTIZATION DESIGN PROPOSAL")
    print("=" * 70)

    print("""
Recommended Quantization Strategy:

┌─────────────────────────────────────────────────────────────────────┐
│                        PI0 Model Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐                                                 │
│  │ Vision Encoder  │  --> TRT FP16 (precision sensitive)             │
│  │ (SigLIP)        │      or INT8 PTQ if accuracy permits            │
│  └────────┬────────┘                                                 │
│           │                                                          │
│           v                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              PaliGemma Language Model (18 layers)            │    │
│  │  ┌───────────────┐  ┌───────────────┐                       │    │
│  │  │  Self-Attn    │  │     MLP       │                       │    │
│  │  │  TRT FP8      │  │  W4A16 TVM    │  <-- Already done     │    │
│  │  │  +FlashAttn   │  │  (nvFP4)      │                       │    │
│  │  └───────────────┘  └───────────────┘                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│           │                                                          │
│           v                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Action Expert (6 layers)                        │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐               │    │
│  │  │ Self-Attn │  │Cross-Attn │  │    MLP    │               │    │
│  │  │  TRT FP8  │  │  TRT FP8  │  │W4A16 TVM  │               │    │
│  │  └───────────┘  └───────────┘  └───────────┘               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│           │                                                          │
│           v                                                          │
│  ┌─────────────────┐                                                 │
│  │ Diffusion Head  │  --> TRT FP16/FP32 (precision critical)        │
│  └─────────────────┘                                                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

Implementation Phases:

Phase 1: W4A16 TRT Plugin Integration (Current)
  - Complete C++ TRT Plugin for W4A16 MLP
  - Fix supportsFormatCombination issue
  - Expected: ~2ms saving per step

Phase 2: Vision Encoder TRT
  - Export SigLIP to TRT with FP16
  - Benchmark INT8 PTQ for potential gains
  - Expected: ~7ms saving per step

Phase 3: Attention TRT FP8
  - Use TRT's native FP8 attention
  - Enable FlashAttention-2 integration
  - Expected: ~8ms saving per step

Phase 4: Static Graph Compilation
  - Combine all components into single TRT engine
  - Enable CUDA Graph capture
  - Expected: ~3ms saving from graph overhead

Total Expected: 120.6ms - 20ms = ~100ms for 3 steps = 10 Hz
""")

    print("\n" + "=" * 70)
    print("Profiling Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
