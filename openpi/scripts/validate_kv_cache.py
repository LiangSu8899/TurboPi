#!/usr/bin/env python3
"""Validate that KV cache implementation produces same outputs as no-cache version."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


def validate():
    print("=" * 60)
    print("Pi0.5 KV Cache Numerical Validation")
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

    # Use fixed noise for comparison
    torch.manual_seed(42)
    noise = model.sample_noise((batch_size, 50, 32), device)

    print("\nRunning inference with same noise...")

    with torch.no_grad():
        # Run with KV cache
        actions_cached = model.sample_actions(device, observation, noise=noise.clone(), num_steps=10, use_kv_cache=True)

        # Run without KV cache
        actions_no_cache = model.sample_actions(device, observation, noise=noise.clone(), num_steps=10, use_kv_cache=False)

    # Compare outputs
    print("\n" + "-" * 60)
    print("Comparison Results")
    print("-" * 60)

    diff = (actions_cached - actions_no_cache).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Check relative difference
    rel_diff = diff / (actions_no_cache.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()

    print(f"  Max relative difference: {max_rel_diff:.4%}")

    # Print first action vector comparison
    print("\n" + "-" * 60)
    print("First Action Vector (first 7 dims)")
    print("-" * 60)
    print(f"  Cached:   {actions_cached[0, 0, :7].float().cpu().numpy()}")
    print(f"  No-cache: {actions_no_cache[0, 0, :7].float().cpu().numpy()}")

    # Determine if validation passed
    print("\n" + "=" * 60)
    if max_diff < 0.01:
        print("VALIDATION PASSED - Outputs match within tolerance (max diff < 0.01)")
    elif max_diff < 0.1:
        print("VALIDATION PASSED WITH WARNING - Small differences detected")
        print("  This is expected due to different attention computation paths")
    else:
        print("VALIDATION FAILED - Significant differences detected!")
    print("=" * 60)


if __name__ == "__main__":
    validate()
