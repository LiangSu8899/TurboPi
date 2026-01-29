#!/usr/bin/env python3
"""
Baseline benchmark with torch.compile optimization.
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_benchmark():
    print("=" * 60)
    print("Pi0.5 VLA Compiled Benchmark (torch.compile)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Precision: {dtype}")
    print(f"  torch.compile: enabled")

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

    print("\nInitializing model...")
    model = PI0Pytorch(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Apply torch.compile
    print("Compiling model with torch.compile...")
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("  torch.compile applied successfully")
    except Exception as e:
        print(f"  torch.compile failed: {e}")
        print("  Continuing without compilation")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e9:.2f}B")

    # Create observation
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

    # Extended warmup for compilation
    num_warmup = 5
    num_denoising_steps = 10
    print(f"\nWarming up and compiling ({num_warmup} iterations)...")
    with torch.no_grad():
        for i in range(num_warmup):
            start = time.perf_counter()
            _ = model.sample_actions(device, observation, num_steps=num_denoising_steps)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(f"  Warmup {i+1}: {elapsed*1000:.1f} ms")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    num_runs = 20
    print(f"\nRunning {num_runs} benchmark iterations...")
    times = []

    with torch.no_grad():
        for i in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            actions = model.sample_actions(device, observation, num_steps=num_denoising_steps)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            elapsed = end - start
            times.append(elapsed)

            if (i + 1) % 5 == 0:
                hz = 1.0 / elapsed
                print(f"  Iteration {i + 1}/{num_runs}: {hz:.2f} Hz ({elapsed * 1000:.1f} ms)")

    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = 1.0 / avg_time

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nThroughput: {throughput:.3f} Hz")
    print(f"Avg Latency: {avg_time * 1000:.1f} ± {std_time * 1000:.1f} ms")
    print(f"Per-step Latency: {avg_time * 1000 / num_denoising_steps:.1f} ms")

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak Memory: {peak_memory:.2f} GB")

    print(f"\n{'='*60}")
    if throughput >= 12.0:
        print("[PASS] Phase 2 target (12-15 Hz) achieved!")
    elif throughput >= 3.0:
        print("[PASS] Phase 1 target (3-4 Hz) achieved")
    else:
        print(f"[INFO] Current: {throughput:.2f} Hz, Target: 3-4 Hz (Phase 1)")

    return throughput


if __name__ == "__main__":
    run_benchmark()
