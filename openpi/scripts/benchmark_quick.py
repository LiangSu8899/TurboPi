#!/usr/bin/env python3
"""
Quick benchmark with various configurations to diagnose performance issues.
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_test(dtype, num_steps):
    device = torch.device("cuda")

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation

    dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float32"

    config = Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        max_state_dim=32,
        pi05=True,
        dtype=dtype_str,
    )

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
    with torch.no_grad():
        for _ in range(2):
            _ = model.sample_actions(device, observation, num_steps=num_steps)
            torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model.sample_actions(device, observation, num_steps=num_steps)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    avg = np.mean(times) * 1000
    hz = 1.0 / (np.mean(times))

    # Cleanup
    del model, observation
    torch.cuda.empty_cache()

    return avg, hz


def main():
    print("=" * 60)
    print("Pi0.5 Performance Diagnostics")
    print("=" * 60)

    print("\nGPU Info:")
    print(f"  {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  cuDNN: {torch.backends.cudnn.version()}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n" + "-" * 60)
    print("Configuration Tests:")
    print("-" * 60)
    print(f"{'Dtype':<12} {'Steps':<8} {'Latency (ms)':<15} {'Throughput (Hz)':<15}")
    print("-" * 60)

    for dtype, dtype_name in [(torch.bfloat16, "bfloat16"), (torch.float32, "float32")]:
        for steps in [10, 5, 3, 1]:
            try:
                latency, hz = run_test(dtype, steps)
                print(f"{dtype_name:<12} {steps:<8} {latency:<15.1f} {hz:<15.2f}")
            except Exception as e:
                print(f"{dtype_name:<12} {steps:<8} ERROR: {e}")

    print("-" * 60)
    print("\nExpected from Phase 1 Report:")
    print("  bfloat16, 10 steps: 280.9 ms, 3.56 Hz")
    print("  bfloat16, 5 steps:  205.3 ms, 4.87 Hz")
    print("  bfloat16, 3 steps:  175.0 ms, 5.72 Hz")


if __name__ == "__main__":
    main()
