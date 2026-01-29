#!/usr/bin/env python3
"""Benchmark KV cache optimization for Pi0.5 inference."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


def benchmark():
    print("=" * 60)
    print("Pi0.5 KV Cache Optimization Benchmark")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation

    config = Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        max_state_dim=32,
        pi05=True,
        dtype="bfloat16",
    )

    print("Loading model...")
    model = PI0Pytorch(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    batch_size = 1
    observation = Observation(
        images={
            "base_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype),
            "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype),
            "right_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224, device=device, dtype=dtype),
        },
        image_masks={
            "base_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(batch_size, device=device, dtype=torch.bool),
        },
        state=torch.randn(batch_size, 32, device=device, dtype=dtype),
        tokenized_prompt=torch.zeros(batch_size, 200, device=device, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(batch_size, 200, device=device, dtype=torch.bool),
    )

    # Warmup
    print("\nWarmup...")
    with torch.no_grad():
        for _ in range(3):
            _ = model.sample_actions(device, observation, num_steps=1, use_kv_cache=True)
            torch.cuda.synchronize()

    print("\n" + "-" * 60)
    print("Benchmark Results")
    print("-" * 60)

    # Test configurations
    configs = [
        ("KV Cache (10 steps)", True, 10),
        ("No Cache (10 steps)", False, 10),
        ("KV Cache (5 steps)", True, 5),
        ("KV Cache (3 steps)", True, 3),
        ("KV Cache (1 step)", True, 1),
    ]

    for name, use_kv_cache, num_steps in configs:
        # Use CUDA events for accurate timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Run multiple iterations for averaging
        num_iters = 5
        latencies = []

        for _ in range(num_iters):
            with torch.no_grad():
                start_event.record()
                _ = model.sample_actions(device, observation, num_steps=num_steps, use_kv_cache=use_kv_cache)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))

        avg_latency = sum(latencies) / len(latencies)
        throughput = 1000.0 / avg_latency

        print(f"  {name}: {avg_latency:.1f} ms ({throughput:.2f} Hz)")

    # Compare memory usage
    print("\n" + "-" * 60)
    print("Memory Usage")
    print("-" * 60)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model.sample_actions(device, observation, num_steps=10, use_kv_cache=True)
        torch.cuda.synchronize()
    kv_cache_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"  With KV Cache: {kv_cache_memory:.2f} GB")

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model.sample_actions(device, observation, num_steps=10, use_kv_cache=False)
        torch.cuda.synchronize()
    no_cache_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Without KV Cache: {no_cache_memory:.2f} GB")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Expected speedup from KV caching: ~7-8x per denoising step")
    print("(Only 50 suffix tokens processed vs ~1000 total tokens)")


if __name__ == "__main__":
    benchmark()
