#!/usr/bin/env python3
"""
Benchmark VLM CUDA Graph vs Eager Execution.

This script validates that CUDA Graph capture eliminates the 171ms
Python dispatch overhead observed in eager mode.

Expected results:
    - Eager W4A16: ~226ms (3 denoising steps)
    - Graph W4A16: <50ms (target)
    - TRT FP8:     ~120ms (baseline)

Author: Claude Code
Date: 2026-02-11
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import numpy as np
import json


def benchmark_paligemma_graph():
    """Benchmark CUDA Graph on real PaliGemma model with W4A16."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from openpi.utils.model_patcher import patch_paligemma_decode_path
    from safetensors.torch import load_file

    device = 'cuda'
    checkpoint_path = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()

    # Load config
    with open(checkpoint_path / "config.json") as f:
        model_config = json.load(f)

    max_token_len = model_config.get("tokenizer_max_length", 200)

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=model_config.get("action_dim", 32),
        action_horizon=model_config.get("action_horizon", 50),
        max_token_len=max_token_len,
        pi05=True,
        dtype="bfloat16",
    )

    print("=" * 70)
    print("VLM CUDA Graph Benchmark")
    print("=" * 70)

    # Load model
    print("\n1. Loading model...")
    model = PI0Pytorch(pi0_config)
    state_dict = load_file(checkpoint_path / "model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Apply W4A16 quantization
    print("\n2. Applying W4A16 INT4 TVM quantization...")
    stats = patch_paligemma_decode_path(model, verbose=False)
    print(f"   Replaced {stats['replaced']} MLP layers")

    # Get PaliGemma language model
    paligemma_lm = model.paligemma_with_expert.paligemma.language_model
    config = paligemma_lm.config

    print(f"\n3. Model config:")
    print(f"   num_layers: {config.num_hidden_layers}")
    print(f"   hidden_size: {config.hidden_size}")
    print(f"   num_kv_heads: {config.num_key_value_heads}")
    print(f"   head_dim: {config.head_dim}")

    # =========================================================================
    # Benchmark: Eager mode (per-layer timing)
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. Benchmarking Eager Mode (18 layers)")
    print("=" * 70)

    # Create test input
    seq_len = 1
    hidden = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16, device=device)
    position_ids = torch.tensor([[0]], dtype=torch.long, device=device)

    # Create position embeddings (rotary)
    dummy_for_rope = torch.zeros(
        1, seq_len, config.head_dim,
        device=device, dtype=torch.bfloat16
    )
    cos, sin = paligemma_lm.rotary_emb(dummy_for_rope, position_ids)
    position_embeddings = (cos, sin)

    # Create attention mask
    attention_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bfloat16, device=device)

    # Warmup
    print("   Warming up...")
    for _ in range(5):
        with torch.no_grad():
            h = hidden.clone()
            for layer in paligemma_lm.layers:
                h = layer(
                    h,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
    torch.cuda.synchronize()

    # Benchmark single forward pass (18 layers)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(20):
        start.record()
        with torch.no_grad():
            h = hidden.clone()
            for layer in paligemma_lm.layers:
                h = layer(
                    h,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    eager_18_layers = np.mean(times)
    print(f"   18 layers (eager): {eager_18_layers:.2f} ms")
    print(f"   Per layer: {eager_18_layers/18:.3f} ms")

    # =========================================================================
    # Benchmark: CUDA Graph mode
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. Benchmarking CUDA Graph Mode")
    print("=" * 70)

    # Create graphed callable for the layer loop
    def forward_all_layers(hidden_states):
        h = hidden_states
        for layer in paligemma_lm.layers:
            h = layer(
                h,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
        return h

    # Warmup
    print("   Warming up for graph capture...")
    for _ in range(3):
        with torch.no_grad():
            _ = forward_all_layers(hidden)
    torch.cuda.synchronize()

    # Capture CUDA Graph
    print("   Capturing CUDA Graph...")

    # Static buffers
    static_hidden = hidden.clone()
    static_output = torch.zeros_like(hidden)

    # Capture
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            output = forward_all_layers(static_hidden)
            static_output.copy_(output)

    torch.cuda.synchronize()
    print("   Graph captured!")

    # Benchmark graph replay
    times = []
    for _ in range(100):
        static_hidden.copy_(hidden)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    graph_18_layers = np.mean(times)
    print(f"   18 layers (graph): {graph_18_layers:.2f} ms")
    print(f"   Per layer: {graph_18_layers/18:.3f} ms")

    # =========================================================================
    # Speedup analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. RESULTS")
    print("=" * 70)

    speedup = eager_18_layers / graph_18_layers

    print(f"""
   18 PaliGemma layers (decode, seq_len=1):
     Eager:  {eager_18_layers:.2f} ms
     Graph:  {graph_18_layers:.2f} ms
     Speedup: {speedup:.2f}x

   Estimated full model (54 MLP layers, 3 denoising steps):
     Vision:  ~11 ms
     Eager:   ~{eager_18_layers * 3:.0f} ms (×3 for 54 layers)
     Graph:   ~{graph_18_layers * 3:.0f} ms (×3 for 54 layers)

   Projected total (3-step denoising):
     Current Eager: ~226 ms
     With Graph:    ~{11 + graph_18_layers * 9:.0f} ms (vision + 3×{graph_18_layers*3:.0f}ms)
     TRT FP8:       ~120 ms (baseline)
""")

    if graph_18_layers * 9 + 11 < 120:
        print("   ✅ CUDA Graph W4A16 can beat TRT FP8!")
    else:
        print("   ⚠️ Need more optimization to beat TRT FP8")

    return {
        'eager_18_layers_ms': eager_18_layers,
        'graph_18_layers_ms': graph_18_layers,
        'speedup': speedup,
    }


def benchmark_simple_graph():
    """Quick test with simple model to verify Graph capture works."""
    print("=" * 70)
    print("Simple CUDA Graph Test")
    print("=" * 70)

    device = 'cuda'
    dtype = torch.bfloat16
    hidden_size = 2048
    intermediate_size = 16384
    num_layers = 18

    # Simple MLP stack (simulating Transformer without attention)
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.gelu(self.gate_proj(x)) * self.up_proj(x))

    mlps = nn.ModuleList([SimpleMLP() for _ in range(num_layers)]).to(device).to(dtype)

    # Test input
    x = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            h = x.clone()
            for mlp in mlps:
                h = mlp(h)
    torch.cuda.synchronize()

    # Benchmark Eager
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(50):
        start.record()
        with torch.no_grad():
            h = x.clone()
            for mlp in mlps:
                h = mlp(h)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    eager_time = np.mean(times)
    print(f"\n  Eager (18 MLP): {eager_time:.2f} ms")

    # Capture Graph
    def forward_all_mlps(hidden):
        h = hidden
        for mlp in mlps:
            h = mlp(h)
        return h

    static_x = x.clone()
    static_output = torch.zeros_like(x)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward_all_mlps(static_x)
        static_output.copy_(output)
    torch.cuda.synchronize()

    # Benchmark Graph
    times = []
    for _ in range(100):
        static_x.copy_(x)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    graph_time = np.mean(times)
    print(f"  Graph (18 MLP): {graph_time:.2f} ms")
    print(f"  Speedup: {eager_time/graph_time:.2f}x")


if __name__ == "__main__":
    # Quick sanity check
    benchmark_simple_graph()

    print("\n")

    # Real PaliGemma benchmark
    try:
        benchmark_paligemma_graph()
    except Exception as e:
        print(f"PaliGemma benchmark failed: {e}")
        import traceback
        traceback.print_exc()
