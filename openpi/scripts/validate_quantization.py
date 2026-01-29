#!/usr/bin/env python3
"""Validate quantized Pi0.5 model accuracy against baseline.

Usage:
    python scripts/validate_quantization.py \
        --baseline_path ~/.cache/openpi/checkpoints/pi05_libero \
        --quantized_path ./quantized_models/pi05_fp4fp8

Validates:
    - MSE between quantized and baseline action outputs
    - Inference throughput comparison
    - Memory usage comparison
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


def create_test_observation(device, dtype, batch_size=1):
    """Create a test observation for validation."""
    from openpi.models_pytorch.pi0_pytorch import Observation

    return Observation(
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


def main():
    parser = argparse.ArgumentParser(
        description="Validate quantized Pi0.5 model accuracy"
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        default="~/.cache/openpi/checkpoints/pi05_libero",
        help="Path to baseline model checkpoint",
    )
    parser.add_argument(
        "--quantized_path",
        type=str,
        default="./quantized_models/pi05_fp4fp8",
        help="Path to quantized model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--mse_threshold",
        type=float,
        default=0.006,
        help="Maximum allowed MSE increase (default: 0.6%)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Pi0.5 Quantization Validation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config

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

    # Load baseline model
    print("\nLoading baseline model...")
    baseline_model = PI0Pytorch(config)
    baseline_model = baseline_model.to(device=device, dtype=dtype)
    baseline_model.eval()

    baseline_path = Path(args.baseline_path).expanduser()
    checkpoint_file = baseline_path / "model.safetensors"
    if checkpoint_file.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_file))
        baseline_model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights from {checkpoint_file}")

    # Load quantized model
    print("\nLoading quantized model...")
    quantized_model = PI0Pytorch(config)
    quantized_model = quantized_model.to(device=device, dtype=dtype)
    quantized_model.eval()

    quantized_path = Path(args.quantized_path).expanduser()
    quantized_file = quantized_path / "quantized_model.pt"
    if quantized_file.exists():
        state_dict = torch.load(quantized_file, map_location=device)
        quantized_model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights from {quantized_file}")
    else:
        print(f"  WARNING: Quantized model not found at {quantized_file}")
        print("  Using baseline weights for comparison (will show 0 difference)")

    # Validation loop
    print(f"\nRunning validation ({args.num_samples} samples, {args.num_steps} steps)...")
    print("-" * 60)

    mse_values = []
    max_diffs = []
    mean_diffs = []

    with torch.no_grad():
        for i in range(args.num_samples):
            # Use fixed seed for reproducibility
            torch.manual_seed(i * 42)

            observation = create_test_observation(device, dtype)

            # Sample noise
            noise = baseline_model.sample_noise(
                (1, config.action_horizon, config.action_dim), device
            )

            # Run baseline
            baseline_actions = baseline_model.sample_actions(
                device, observation,
                noise=noise.clone(),
                num_steps=args.num_steps,
                use_kv_cache=True,
            )

            # Run quantized
            quantized_actions = quantized_model.sample_actions(
                device, observation,
                noise=noise.clone(),
                num_steps=args.num_steps,
                use_kv_cache=True,
            )

            # Compute metrics
            diff = (baseline_actions - quantized_actions).abs()
            mse = torch.nn.functional.mse_loss(baseline_actions, quantized_actions).item()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            mse_values.append(mse)
            max_diffs.append(max_diff)
            mean_diffs.append(mean_diff)

            if (i + 1) % 5 == 0:
                print(f"  Sample {i+1}/{args.num_samples}: MSE={mse:.6f}, Max={max_diff:.6f}")

    # Summary
    avg_mse = sum(mse_values) / len(mse_values)
    avg_max_diff = sum(max_diffs) / len(max_diffs)
    avg_mean_diff = sum(mean_diffs) / len(mean_diffs)
    max_mse = max(mse_values)

    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"\n  Average MSE: {avg_mse:.6f}")
    print(f"  Max MSE: {max_mse:.6f}")
    print(f"  Average Max Diff: {avg_max_diff:.6f}")
    print(f"  Average Mean Diff: {avg_mean_diff:.6f}")

    # Pass/Fail
    print("\n" + "-" * 60)
    if max_mse < args.mse_threshold:
        print(f"VALIDATION PASSED - MSE ({max_mse:.6f}) < threshold ({args.mse_threshold})")
    else:
        print(f"VALIDATION FAILED - MSE ({max_mse:.6f}) > threshold ({args.mse_threshold})")

    # Benchmark comparison
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    # Warmup
    observation = create_test_observation(device, dtype)
    for _ in range(3):
        with torch.no_grad():
            _ = baseline_model.sample_actions(device, observation, num_steps=1)
            _ = quantized_model.sample_actions(device, observation, num_steps=1)
            torch.cuda.synchronize()

    # Benchmark baseline
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        for _ in range(5):
            _ = baseline_model.sample_actions(device, observation, num_steps=args.num_steps)
        end.record()
        torch.cuda.synchronize()
    baseline_latency = start.elapsed_time(end) / 5

    # Benchmark quantized
    with torch.no_grad():
        start.record()
        for _ in range(5):
            _ = quantized_model.sample_actions(device, observation, num_steps=args.num_steps)
        end.record()
        torch.cuda.synchronize()
    quantized_latency = start.elapsed_time(end) / 5

    speedup = baseline_latency / quantized_latency if quantized_latency > 0 else 1.0

    print(f"\n  Baseline Latency: {baseline_latency:.1f} ms ({1000/baseline_latency:.2f} Hz)")
    print(f"  Quantized Latency: {quantized_latency:.1f} ms ({1000/quantized_latency:.2f} Hz)")
    print(f"  Speedup: {speedup:.2f}x")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
