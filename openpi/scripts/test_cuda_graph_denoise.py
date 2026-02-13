#!/usr/bin/env python3
"""
Test CUDA Graph capture for denoise_step_with_cache.

This script tests:
1. Whether denoise_step_with_cache is CUDA Graph compatible
2. The overhead elimination from graph capture
3. Different graph capture strategies

Usage:
    docker exec turbo_pi_eval python /workspace/scripts/test_cuda_graph_denoise.py

Author: Turbo-Pi Team
Date: 2026-02-13
"""

import sys
import os
import time

# Skip TVM to avoid libfpA_intB_gemm.so error
os.environ["OPENPI_SKIP_TVM"] = "1"

sys.path.insert(0, "/workspace/src")
os.chdir("/workspace")

import torch
import torch.nn.functional as F
import numpy as np


def test_basic_graph_capture():
    """Test basic CUDA graph capture capability."""
    print("=" * 60)
    print("Test 1: Basic CUDA Graph Capture")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Simple computation to capture
    batch_size = 1
    hidden_size = 1024
    seq_len = 83

    # Static buffers
    static_input = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    static_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)
    static_output = torch.zeros_like(static_input)

    def simple_forward():
        x = static_input @ static_weight
        x = F.gelu(x)
        return x @ static_weight.T

    # Warmup
    for _ in range(3):
        _ = simple_forward()
    torch.cuda.synchronize()

    # Capture graph
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = simple_forward()
            static_output.copy_(out)

        # Test replay
        graph.replay()
        torch.cuda.synchronize()

        print("  ✓ Basic CUDA Graph capture succeeded!")

        # Benchmark
        warmup, runs = 20, 100

        # Non-graph
        for _ in range(warmup):
            _ = simple_forward()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            _ = simple_forward()
        torch.cuda.synchronize()
        no_graph_ms = (time.perf_counter() - start) / runs * 1000

        # Graph
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            graph.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - start) / runs * 1000

        print(f"\n  Performance comparison:")
        print(f"    No graph: {no_graph_ms:.4f} ms")
        print(f"    Graph:    {graph_ms:.4f} ms")
        print(f"    Speedup:  {no_graph_ms / graph_ms:.2f}x")

        return True

    except Exception as e:
        print(f"  ✗ CUDA Graph capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sdpa_graph_capture():
    """Test SDPA in CUDA graph."""
    print("\n" + "=" * 60)
    print("Test 2: SDPA in CUDA Graph")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch_size = 1
    num_heads = 8
    head_dim = 256
    prefix_len = 1050
    suffix_len = 83
    kv_len = prefix_len + suffix_len

    # Static buffers
    static_q = torch.randn(batch_size, num_heads, suffix_len, head_dim, device=device, dtype=dtype)
    static_k = torch.randn(batch_size, num_heads, kv_len, head_dim, device=device, dtype=dtype)
    static_v = torch.randn(batch_size, num_heads, kv_len, head_dim, device=device, dtype=dtype)
    static_out = torch.zeros(batch_size, num_heads, suffix_len, head_dim, device=device, dtype=dtype)

    def sdpa_forward():
        return F.scaled_dot_product_attention(static_q, static_k, static_v)

    # Warmup
    for _ in range(3):
        _ = sdpa_forward()
    torch.cuda.synchronize()

    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = sdpa_forward()
            static_out.copy_(out)

        graph.replay()
        torch.cuda.synchronize()

        print("  ✓ SDPA CUDA Graph capture succeeded!")

        # Benchmark for 18 layers
        warmup, runs = 20, 100
        num_layers = 18

        # No graph
        for _ in range(warmup):
            for _ in range(num_layers):
                _ = sdpa_forward()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            for _ in range(num_layers):
                _ = sdpa_forward()
        torch.cuda.synchronize()
        no_graph_ms = (time.perf_counter() - start) / runs * 1000

        # Graph
        for _ in range(warmup):
            for _ in range(num_layers):
                graph.replay()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            for _ in range(num_layers):
                graph.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - start) / runs * 1000

        print(f"\n  Performance (18 layers):")
        print(f"    No graph: {no_graph_ms:.4f} ms")
        print(f"    Graph:    {graph_ms:.4f} ms")
        print(f"    Savings:  {no_graph_ms - graph_ms:.4f} ms")

        return True

    except Exception as e:
        print(f"  ✗ SDPA Graph capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_layer_graph():
    """Test capturing a full transformer layer."""
    print("\n" + "=" * 60)
    print("Test 3: Full Transformer Layer Graph")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Simulate Gemma 300M dimensions
    batch_size = 1
    seq_len = 83
    hidden_size = 1024
    num_heads = 8
    head_dim = hidden_size // num_heads  # 128
    mlp_dim = 4096
    prefix_len = 1050
    kv_len = prefix_len + seq_len

    # Static buffers for one layer
    static_hidden = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    static_kv_key = torch.randn(batch_size, num_heads, prefix_len, head_dim, device=device, dtype=dtype)
    static_kv_val = torch.randn(batch_size, num_heads, prefix_len, head_dim, device=device, dtype=dtype)
    static_adarms_cond = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

    # Simulated layer weights
    ln_weight = torch.randn(hidden_size, device=device, dtype=dtype)
    q_proj = torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)
    k_proj = torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)
    v_proj = torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)
    o_proj = torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)
    up_proj = torch.randn(hidden_size, mlp_dim, device=device, dtype=dtype)
    down_proj = torch.randn(mlp_dim, hidden_size, device=device, dtype=dtype)
    gate_proj = torch.randn(hidden_size, mlp_dim, device=device, dtype=dtype)

    static_output = torch.zeros_like(static_hidden)

    def layer_forward():
        # Input norm (simplified RMSNorm)
        variance = static_hidden.pow(2).mean(-1, keepdim=True)
        normed = static_hidden * torch.rsqrt(variance + 1e-6) * ln_weight

        # QKV projections
        q = (normed @ q_proj).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = (normed @ k_proj).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = (normed @ v_proj).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Concatenate with KV cache
        full_k = torch.cat([static_kv_key, k], dim=2)
        full_v = torch.cat([static_kv_val, v], dim=2)

        # SDPA attention
        attn_out = F.scaled_dot_product_attention(q, full_k, full_v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)

        # O projection + residual
        out = attn_out @ o_proj
        out = static_hidden + out

        # Post-attn norm
        variance = out.pow(2).mean(-1, keepdim=True)
        normed = out * torch.rsqrt(variance + 1e-6) * ln_weight

        # MLP (SwiGLU)
        gate = F.silu(normed @ gate_proj)
        up = normed @ up_proj
        mlp_out = (gate * up) @ down_proj

        # Final residual
        return out + mlp_out

    # Warmup
    for _ in range(3):
        _ = layer_forward()
    torch.cuda.synchronize()

    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = layer_forward()
            static_output.copy_(out)

        graph.replay()
        torch.cuda.synchronize()

        print("  ✓ Full layer CUDA Graph capture succeeded!")

        # Benchmark 18 layers
        warmup, runs = 20, 100
        num_layers = 18

        # No graph
        for _ in range(warmup):
            for _ in range(num_layers):
                _ = layer_forward()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            for _ in range(num_layers):
                _ = layer_forward()
        torch.cuda.synchronize()
        no_graph_ms = (time.perf_counter() - start) / runs * 1000

        # Graph (same graph for all layers - simplified)
        for _ in range(warmup):
            for _ in range(num_layers):
                graph.replay()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            for _ in range(num_layers):
                graph.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - start) / runs * 1000

        print(f"\n  Performance (18 layers):")
        print(f"    No graph: {no_graph_ms:.4f} ms")
        print(f"    Graph:    {graph_ms:.4f} ms")
        print(f"    Savings:  {no_graph_ms - graph_ms:.4f} ms per step")
        print(f"    10 steps: {(no_graph_ms - graph_ms) * 10:.4f} ms total")

        return True

    except Exception as e:
        print(f"  ✗ Full layer Graph capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chained_graphs():
    """Test chained CUDA graphs for 10 denoise steps."""
    print("\n" + "=" * 60)
    print("Test 4: Chained CUDA Graphs (10 steps)")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch_size = 1
    action_horizon = 50
    action_dim = 32
    hidden_size = 1024

    # Static buffers - shared across steps
    static_x_t = torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=dtype)
    static_v_t = torch.zeros_like(static_x_t)
    static_state = torch.randn(batch_size, 8, device=device, dtype=dtype)

    # Simulated denoise step (simplified)
    proj_in = torch.randn(action_dim, hidden_size, device=device, dtype=dtype)
    proj_out = torch.randn(hidden_size, action_dim, device=device, dtype=dtype)

    # Pre-computed timestep embeddings for 10 steps
    dt = -0.1
    timesteps = [1.0 + i * dt for i in range(10)]

    def simple_denoise_step():
        """Simplified denoise step for testing."""
        h = static_x_t @ proj_in
        h = F.gelu(h)
        v_t = h @ proj_out
        return v_t

    # Capture 10 graphs (one per step)
    graphs = []

    # First step graph
    for _ in range(3):
        _ = simple_denoise_step()
    torch.cuda.synchronize()

    try:
        for step_idx in range(10):
            graph = torch.cuda.CUDAGraph()

            # Warmup for this step
            for _ in range(2):
                v = simple_denoise_step()
                static_x_t.add_(v, alpha=dt)
            torch.cuda.synchronize()

            # Reset for capture
            static_x_t.normal_()

            with torch.cuda.graph(graph):
                v_t = simple_denoise_step()
                static_v_t.copy_(v_t)
                static_x_t.add_(static_v_t, alpha=dt)

            graphs.append(graph)

        torch.cuda.synchronize()
        print(f"  ✓ Captured {len(graphs)} chained graphs!")

        # Benchmark
        warmup, runs = 20, 100

        # No graph - Python loop
        for _ in range(warmup):
            static_x_t.normal_()
            for _ in range(10):
                v = simple_denoise_step()
                static_x_t.add_(v, alpha=dt)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            static_x_t.normal_()
            for _ in range(10):
                v = simple_denoise_step()
                static_x_t.add_(v, alpha=dt)
        torch.cuda.synchronize()
        no_graph_ms = (time.perf_counter() - start) / runs * 1000

        # Graph chain
        for _ in range(warmup):
            static_x_t.normal_()
            for g in graphs:
                g.replay()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            static_x_t.normal_()
            for g in graphs:
                g.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - start) / runs * 1000

        print(f"\n  Performance (10 step chain):")
        print(f"    Python loop: {no_graph_ms:.4f} ms")
        print(f"    Graph chain: {graph_ms:.4f} ms")
        print(f"    Savings:     {no_graph_ms - graph_ms:.4f} ms")

        return True

    except Exception as e:
        print(f"  ✗ Chained graphs failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_denoise_step_compatibility():
    """Test actual denoise_step_with_cache CUDA Graph compatibility."""
    print("\n" + "=" * 60)
    print("Test 5: denoise_step_with_cache Compatibility")
    print("=" * 60)

    try:
        from openpi.inference.unified_policy import UnifiedPolicy

        # Try to load model via UnifiedPolicy
        print("  Loading model via UnifiedPolicy...")

        device = torch.device("cuda")

        policy = UnifiedPolicy(
            checkpoint_dir="/root/.cache/openpi/pytorch_checkpoints/pi05_libero",
            backend="pytorch",
            num_denoising_steps=10,
            device="cuda",
        )
        policy.warmup(num_iterations=2)

        # Get the model from the backend
        model = policy._backend.model
        batch_size = 1

        # Create dummy observation for prefix cache
        test_obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "test cuda graph",
        }

        # Process observation through policy backend
        obs = policy._backend._preprocess(test_obs)

        print(f"  Model loaded!")

        # Compute prefix embeddings and KV cache
        print("  Computing prefix KV cache...")
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(obs, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_kv_cache = model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)

        print(f"  Prefix length: {prefix_pad_masks.shape[1]}")
        print(f"  KV cache: {len(prefix_kv_cache)} layers")

        # Create static buffers for denoise step
        action_horizon = model.config.action_horizon
        action_dim = model.config.action_dim

        static_state = state.clone()
        static_x_t = torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=torch.bfloat16)
        static_timestep = torch.tensor([1.0], device=device, dtype=torch.float32)
        static_v_t = torch.zeros_like(static_x_t)

        def denoise_forward():
            return model.denoise_step_with_cache(
                static_state,
                prefix_kv_cache,
                prefix_pad_masks,
                static_x_t,
                static_timestep,
            )

        # Warmup
        print("  Warming up...")
        for _ in range(3):
            with torch.no_grad():
                _ = denoise_forward()
        torch.cuda.synchronize()

        # Try to capture
        print("  Attempting CUDA Graph capture...")

        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph):
            with torch.no_grad():
                v_t = denoise_forward()
                static_v_t.copy_(v_t)

        torch.cuda.synchronize()

        # Test replay
        graph.replay()
        torch.cuda.synchronize()

        print("  ✓ denoise_step_with_cache is CUDA Graph compatible!")

        # Benchmark
        warmup, runs = 10, 50

        # No graph
        for _ in range(warmup):
            with torch.no_grad():
                _ = denoise_forward()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                _ = denoise_forward()
        torch.cuda.synchronize()
        no_graph_ms = (time.perf_counter() - start) / runs * 1000

        # Graph
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            graph.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - start) / runs * 1000

        print(f"\n  Performance (single denoise step):")
        print(f"    No graph: {no_graph_ms:.2f} ms")
        print(f"    Graph:    {graph_ms:.2f} ms")
        print(f"    Savings:  {no_graph_ms - graph_ms:.2f} ms per step")
        print(f"    10 steps: {(no_graph_ms - graph_ms) * 10:.2f} ms total")

        return True, no_graph_ms, graph_ms

    except Exception as e:
        print(f"  ✗ denoise_step_with_cache Graph capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0


def main():
    print("=" * 60)
    print("CUDA Graph Denoise Test")
    print("=" * 60)

    props = torch.cuda.get_device_properties(0)
    print(f"\nDevice: {props.name}")
    print(f"Compute: {props.major}.{props.minor}")
    print(f"PyTorch: {torch.__version__}")

    # Run tests
    results = []

    results.append(("Basic Graph", test_basic_graph_capture()))
    results.append(("SDPA Graph", test_sdpa_graph_capture()))
    results.append(("Full Layer Graph", test_full_layer_graph()))
    results.append(("Chained Graphs", test_chained_graphs()))

    # Test actual denoise step
    denoise_result = test_denoise_step_compatibility()
    results.append(("Denoise Step", denoise_result[0]))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    if all(r[1] for r in results):
        print("""
  All tests passed! CUDA Graph optimization is feasible.

  Expected optimization potential:
  - Per-step savings: 3-5 ms (Python/kernel launch overhead)
  - 10 steps total:   30-50 ms savings
  - Current denoise:  ~95 ms
  - Target denoise:   ~45-65 ms
""")
    else:
        print("""
  Some tests failed. CUDA Graph may need workarounds.
  Check the specific failures above.
""")


if __name__ == "__main__":
    main()
