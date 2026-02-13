#!/usr/bin/env python3
"""
Benchmark: eager vs SDPA attention implementation.

This script measures the performance difference between eager and SDPA
attention in the denoise step.

Usage:
    docker exec turbo_pi_eval python /workspace/scripts/benchmark_attn_implementation.py

Author: Turbo-Pi Team
Date: 2026-02-13
"""

import sys
import time
import logging

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def benchmark_attention_kernel(
    batch_size: int = 1,
    seq_len: int = 712,
    num_heads: int = 8,
    head_dim: int = 256,
    warmup: int = 20,
    runs: int = 100,
):
    """Benchmark attention kernel performance."""

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create test tensors
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Create causal mask
    attn_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype) * float("-inf"),
        diagonal=1
    )

    print(f"\nBenchmark: B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}")
    print("=" * 60)

    results = {}

    # 1. Eager Attention (manual)
    def eager_attention(q, k, v, mask):
        scale = head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    # Warmup
    for _ in range(warmup):
        _ = eager_attention(query, key, value, attn_mask)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        _ = eager_attention(query, key, value, attn_mask)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - start) / runs * 1000
    results["eager"] = eager_ms
    print(f"  Eager attention:  {eager_ms:.4f} ms")

    # 2. SDPA (PyTorch native)
    # Warmup
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        _ = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    torch.cuda.synchronize()
    sdpa_ms = (time.perf_counter() - start) / runs * 1000
    results["sdpa"] = sdpa_ms
    print(f"  SDPA attention:   {sdpa_ms:.4f} ms")

    # 3. SDPA with is_causal=True (no explicit mask)
    # Warmup
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        _ = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.cuda.synchronize()
    sdpa_causal_ms = (time.perf_counter() - start) / runs * 1000
    results["sdpa_causal"] = sdpa_causal_ms
    print(f"  SDPA (is_causal): {sdpa_causal_ms:.4f} ms")

    # Speedup
    print(f"\n  Speedup:")
    print(f"    SDPA vs Eager:         {eager_ms / sdpa_ms:.2f}x")
    print(f"    SDPA_causal vs Eager:  {eager_ms / sdpa_causal_ms:.2f}x")

    return results


def estimate_denoise_impact():
    """Estimate impact on full denoise loop."""

    print("\n" + "=" * 60)
    print("Estimating Impact on Denoise Loop")
    print("=" * 60)

    # Run benchmark with denoise-like dimensions
    # Pi0.5: 8 heads, 256 head_dim, sequence ~712
    results = benchmark_attention_kernel(
        batch_size=1,
        seq_len=712,
        num_heads=8,
        head_dim=256,
    )

    # DiT has 18 layers, 10 steps
    num_layers = 18
    num_steps = 10

    eager_total = results["eager"] * num_layers * num_steps
    sdpa_total = results["sdpa"] * num_layers * num_steps
    sdpa_causal_total = results["sdpa_causal"] * num_layers * num_steps

    print(f"\n  Estimated total attention time (18 layers × 10 steps):")
    print(f"    Eager:       {eager_total:.2f} ms")
    print(f"    SDPA:        {sdpa_total:.2f} ms")
    print(f"    SDPA_causal: {sdpa_causal_total:.2f} ms")

    print(f"\n  Potential savings:")
    print(f"    SDPA:        {eager_total - sdpa_total:.2f} ms")
    print(f"    SDPA_causal: {eager_total - sdpa_causal_total:.2f} ms")

    current_denoise = 94.9  # ms
    print(f"\n  Impact on current denoise ({current_denoise:.1f} ms):")
    print(f"    With SDPA:        {current_denoise - (eager_total - sdpa_total):.1f} ms")
    print(f"    With SDPA_causal: {current_denoise - (eager_total - sdpa_causal_total):.1f} ms")


def test_cuda_graph_compatibility():
    """Test if SDPA works with CUDA Graph capture."""

    print("\n" + "=" * 60)
    print("Testing CUDA Graph Compatibility")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch_size, seq_len, num_heads, head_dim = 1, 712, 8, 256

    # Static buffers
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    output = torch.empty(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        result = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.cuda.synchronize()

    # Try CUDA Graph capture
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

        # Test replay
        graph.replay()
        torch.cuda.synchronize()

        print("  ✓ SDPA works with CUDA Graph capture!")

        # Benchmark graph replay
        warmup, runs = 20, 100
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            graph.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - start) / runs * 1000

        print(f"  Graph replay time: {graph_ms:.4f} ms")

        return True

    except Exception as e:
        print(f"  ✗ CUDA Graph capture failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Attention Implementation Benchmark")
    print("=" * 60)

    # Device info
    props = torch.cuda.get_device_properties(0)
    print(f"\nDevice: {props.name}")
    print(f"Compute: {props.major}.{props.minor}")

    # Test various sequence lengths
    for seq_len in [256, 512, 712, 1024]:
        benchmark_attention_kernel(seq_len=seq_len)

    # Estimate impact
    estimate_denoise_impact()

    # Test CUDA Graph
    test_cuda_graph_compatibility()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
  If SDPA shows significant speedup over Eager:

  1. Edit pi0_pytorch.py line 889:
     - Change: _attn_implementation = "eager"
     + To:     _attn_implementation = "sdpa"

  2. Verify CUDA Graph still works

  3. Run full benchmark to confirm improvement
""")


if __name__ == "__main__":
    main()
