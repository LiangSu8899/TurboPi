#!/usr/bin/env python3
"""Benchmark quantized Pi0.5 model directly after quantization.

This script performs quantization and benchmarking in the same session
to properly evaluate FP4/FP8 performance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


def main():
    print("=" * 60)
    print("Pi0.5 Quantized Model Benchmark")
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

    print("\nLoading baseline model...")
    model = PI0Pytorch(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Load weights
    model_path = Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()
    checkpoint_file = model_path / "model.safetensors"
    if checkpoint_file.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_file))
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights from {checkpoint_file}")

    # Create test observation
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
            _ = model.sample_actions(device, observation, num_steps=1)
            torch.cuda.synchronize()

    # Benchmark baseline
    print("\nBenchmarking baseline (bfloat16)...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    num_iters = 5
    with torch.no_grad():
        start.record()
        for _ in range(num_iters):
            _ = model.sample_actions(device, observation, num_steps=10)
        end.record()
        torch.cuda.synchronize()
    baseline_latency = start.elapsed_time(end) / num_iters
    baseline_hz = 1000 / baseline_latency

    print(f"  Latency: {baseline_latency:.1f} ms ({baseline_hz:.2f} Hz)")

    # Get baseline action for comparison
    torch.manual_seed(42)
    noise = model.sample_noise((1, config.action_horizon, config.action_dim), device)
    with torch.no_grad():
        baseline_actions = model.sample_actions(device, observation, noise=noise.clone(), num_steps=10)

    # Apply quantization
    print("\n" + "=" * 60)
    print("Applying FP4/FP8 Quantization...")
    print("=" * 60)

    from openpi.quantization import (
        PrecisionConfig,
        SyntheticCalibrationDataset,
        calibration_forward_loop,
    )

    try:
        import modelopt.torch.quantization as mtq
    except ImportError:
        print("ModelOpt not available!")
        return

    precision_config = PrecisionConfig()

    # Build quantization config
    quant_config = {
        "quant_cfg": {
            "*": {"enable": False},
        },
        "algorithm": "max",
    }

    # Add FP4 patterns
    for pattern in precision_config.fp4_patterns:
        glob_pattern = pattern.replace(r".*\.", "*").replace(r"$", "")
        quant_config["quant_cfg"][glob_pattern] = {
            "num_bits": 4,
            "axis": None,
            "enable": True,
        }

    # Add FP8 patterns
    for pattern in precision_config.fp8_patterns:
        glob_pattern = pattern.replace(r".*\.", "*").replace(r"$", "")
        quant_config["quant_cfg"][glob_pattern] = {
            "num_bits": 8,
            "axis": None,
            "enable": True,
        }

    # Create calibration data
    print("\nRunning calibration (32 samples)...")
    dataset = SyntheticCalibrationDataset(num_samples=32, device="cuda")

    def calibration_fn():
        calibration_forward_loop(model, dataset, 32)

    # Apply quantization
    quantized_model = mtq.quantize(model, quant_config, forward_loop=calibration_fn)
    print("Quantization applied!")

    # Warmup quantized model
    print("\nWarmup quantized model...")
    with torch.no_grad():
        for _ in range(3):
            _ = quantized_model.sample_actions(device, observation, num_steps=1)
            torch.cuda.synchronize()

    # Benchmark quantized model
    print("\nBenchmarking quantized model...")
    with torch.no_grad():
        start.record()
        for _ in range(num_iters):
            _ = quantized_model.sample_actions(device, observation, num_steps=10)
        end.record()
        torch.cuda.synchronize()
    quantized_latency = start.elapsed_time(end) / num_iters
    quantized_hz = 1000 / quantized_latency

    print(f"  Latency: {quantized_latency:.1f} ms ({quantized_hz:.2f} Hz)")

    # Get quantized action for comparison
    with torch.no_grad():
        quantized_actions = quantized_model.sample_actions(device, observation, noise=noise.clone(), num_steps=10)

    # Compare outputs
    diff = (baseline_actions - quantized_actions).abs()
    mse = torch.nn.functional.mse_loss(baseline_actions, quantized_actions).item()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\nBaseline (bfloat16):")
    print(f"  Latency: {baseline_latency:.1f} ms ({baseline_hz:.2f} Hz)")
    print(f"\nQuantized (FP4/FP8):")
    print(f"  Latency: {quantized_latency:.1f} ms ({quantized_hz:.2f} Hz)")
    print(f"\nSpeedup: {baseline_latency/quantized_latency:.2f}x")
    print(f"\nNumerical Accuracy:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max Diff: {max_diff:.6f}")
    print(f"  Mean Diff: {mean_diff:.6f}")

    # Memory usage
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = quantized_model.sample_actions(device, observation, num_steps=10)
        torch.cuda.synchronize()
    memory_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nMemory Usage: {memory_gb:.2f} GB")

    print("=" * 60)


if __name__ == "__main__":
    main()
