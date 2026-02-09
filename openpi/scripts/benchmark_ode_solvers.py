#!/usr/bin/env python3
"""Benchmark different ODE solvers for Pi0.5 denoising.

Compares Euler (baseline) vs higher-order methods:
- midpoint (2nd order)
- heun (2nd order)
- dpm_solver_2 (2nd order, optimized for flow)
- rk4 (4th order)

Usage:
    python scripts/benchmark_ode_solvers.py
"""

import sys
import os
import time
import json
import pathlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np


def benchmark_solver(model, observation, solver_type, num_warmup=5, num_trials=20):
    """Benchmark a specific ODE solver."""
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.sample_actions(device, observation, num_steps=10, solver_type=solver_type)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_trials):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.sample_actions(device, observation, num_steps=10, solver_type=solver_type)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "output": output,
    }


def compare_outputs(baseline_output, test_output):
    """Compare two outputs and compute similarity metrics."""
    baseline_np = baseline_output.float().cpu().numpy()
    test_np = test_output.float().cpu().numpy()

    max_diff = np.abs(baseline_np - test_np).max()
    mean_diff = np.abs(baseline_np - test_np).mean()
    cosine_sim = np.dot(baseline_np.flatten(), test_np.flatten()) / (
        np.linalg.norm(baseline_np.flatten()) * np.linalg.norm(test_np.flatten())
    )

    return {
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "cosine_sim": float(cosine_sim),
    }


def main():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
    from safetensors.torch import load_file

    print("=" * 70)
    print("ODE SOLVER BENCHMARK FOR PI0.5 DENOISING")
    print("=" * 70)

    device = "cuda"
    checkpoint_dir = pathlib.Path.home() / ".cache/openpi/checkpoints/pi05_libero"

    # Load model
    print("\nLoading model...")
    config_path = checkpoint_dir / "config.json"
    with open(config_path) as f:
        model_config = json.load(f)

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=model_config.get("action_dim", 32),
        action_horizon=model_config.get("action_horizon", 50),
        max_token_len=model_config.get("tokenizer_max_length", 200),
        pi05=True,
        dtype="bfloat16",
    )

    model = PI0Pytorch(pi0_config)
    weights_path = checkpoint_dir / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device)
    model.eval()
    print("Model loaded.")

    # Create dummy observation with fixed seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    dummy_obs = Observation(
        images={
            "base_0_rgb": torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16),
            "left_wrist_0_rgb": torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16),
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, device=device, dtype=torch.bfloat16) - 1,
        },
        image_masks={
            "base_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(1, device=device, dtype=torch.bool),
        },
        state=torch.randn(1, 32, device=device, dtype=torch.bfloat16),
        tokenized_prompt=torch.ones(1, 200, device=device, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(1, 200, device=device, dtype=torch.bool),
    )

    # Solvers to test
    solvers = ["euler", "midpoint", "heun", "dpm_solver_2", "rk4"]

    print("\n" + "=" * 70)
    print("BENCHMARKING SOLVERS (10 denoising steps)")
    print("=" * 70)

    results = {}

    # Baseline: Euler
    print("\n[1/5] Testing euler (baseline)...")
    euler_result = benchmark_solver(model, dummy_obs, "euler")
    results["euler"] = euler_result
    baseline_output = euler_result["output"]
    print(f"  Latency: {euler_result['mean_ms']:.2f} ± {euler_result['std_ms']:.2f} ms")

    # Test other solvers
    for i, solver in enumerate(solvers[1:], start=2):
        print(f"\n[{i}/5] Testing {solver}...")
        result = benchmark_solver(model, dummy_obs, solver)

        # Compare with baseline
        comparison = compare_outputs(baseline_output, result["output"])
        result["comparison"] = comparison
        results[solver] = result

        speedup = euler_result["mean_ms"] / result["mean_ms"]
        print(f"  Latency: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Cosine sim vs Euler: {comparison['cosine_sim']:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Solver':<15} | {'Latency (ms)':<15} | {'Speedup':<10} | {'NN Evals':<10} | {'Cosine':<10}")
    print("-" * 70)

    nn_evals = {"euler": 1, "midpoint": 2, "heun": 2, "dpm_solver_2": "1-2", "rk4": 4}

    print(f"{'euler':<15} | {euler_result['mean_ms']:.2f} ms        | 1.00x      | 1/step     | -")

    for solver in solvers[1:]:
        r = results[solver]
        speedup = euler_result["mean_ms"] / r["mean_ms"]
        cosine = r["comparison"]["cosine_sim"]
        print(f"{solver:<15} | {r['mean_ms']:.2f} ms        | {speedup:.2f}x      | {nn_evals[solver]}/step    | {cosine:.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nKey Observations:")
    print("1. Higher-order solvers use more NN evaluations per step")
    print("2. midpoint/heun use 2x evals, rk4 uses 4x evals")
    print("3. dpm_solver_2 uses 2 evals for first step, 1 for subsequent steps")
    print("\nFor SAME accuracy, higher-order solvers can use FEWER steps.")
    print("For SAME steps, higher-order solvers give BETTER quality (smoother actions).")

    # Save results
    output_file = "ode_solver_benchmark.json"
    save_results = {
        solver: {
            "mean_ms": r["mean_ms"],
            "std_ms": r["std_ms"],
            "comparison": r.get("comparison", {}),
        }
        for solver, r in results.items()
    }
    with open(output_file, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
