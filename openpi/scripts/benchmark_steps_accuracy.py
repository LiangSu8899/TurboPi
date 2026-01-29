#!/usr/bin/env python3
"""
Benchmark denoising steps vs accuracy tradeoff for Pi0.5 VLA.

This script evaluates the impact of reducing denoising steps on action quality
to find the optimal steps-performance balance.

Usage:
    python scripts/benchmark_steps_accuracy.py --steps 10 5 3 2 1
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class StepsBenchmarkResult:
    """Results for a single step configuration."""
    num_steps: int
    latency_ms: float
    throughput_hz: float
    mse_vs_10steps: float
    max_diff_vs_10steps: float
    action_mean: float
    action_std: float


def load_model_and_observation(
    model_path: str,
    device: str = "cuda",
) -> Tuple[torch.nn.Module, "Observation"]:
    """Load model and create test observation."""
    from openpi.models_pytorch.transformers_replace import ensure_patched
    ensure_patched()

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
    from safetensors.torch import load_file

    model_path = Path(model_path).expanduser()
    dtype = torch.bfloat16

    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        model_config = json.load(f)

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=32,
        action_horizon=50,
        max_token_len=model_config.get("tokenizer_max_length", 200),
        max_state_dim=model_config.get("max_state_dim", 32),
        pi05=True,
        dtype="bfloat16",
    )

    logger.info("Loading model...")
    model = PI0Pytorch(pi0_config)
    model = model.to(device=device, dtype=dtype)

    weights_path = model_path / "model.safetensors"
    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

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

    return model, observation


def benchmark_steps(
    model: torch.nn.Module,
    observation: "Observation",
    steps_to_test: List[int],
    num_warmup: int = 5,
    num_runs: int = 20,
) -> List[StepsBenchmarkResult]:
    """Benchmark different denoising step configurations."""
    device = torch.device("cuda")
    results = []

    # First, get 10-step reference
    logger.info("\nGenerating 10-step reference...")
    torch.manual_seed(42)
    with torch.no_grad():
        ref_actions = model.sample_actions(device, observation, num_steps=10, use_kv_cache=True)
    ref_actions = ref_actions.float()

    for num_steps in steps_to_test:
        logger.info(f"\nBenchmarking {num_steps} steps...")

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model.sample_actions(device, observation, num_steps=num_steps, use_kv_cache=True)
                torch.cuda.synchronize()

        # Benchmark latency
        latencies = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(num_runs):
                start_event.record()
                actions = model.sample_actions(device, observation, num_steps=num_steps, use_kv_cache=True)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))

        avg_latency = np.mean(latencies)
        throughput = 1000.0 / avg_latency

        # Get actions for accuracy comparison (fixed seed)
        torch.manual_seed(42)
        with torch.no_grad():
            test_actions = model.sample_actions(device, observation, num_steps=num_steps, use_kv_cache=True)
        test_actions = test_actions.float()

        # Calculate metrics
        diff = test_actions - ref_actions
        mse = (diff ** 2).mean().item()
        max_diff = diff.abs().max().item()

        result = StepsBenchmarkResult(
            num_steps=num_steps,
            latency_ms=avg_latency,
            throughput_hz=throughput,
            mse_vs_10steps=mse,
            max_diff_vs_10steps=max_diff,
            action_mean=test_actions.mean().item(),
            action_std=test_actions.std().item(),
        )
        results.append(result)

        logger.info(f"  Latency: {avg_latency:.1f} ms ({throughput:.2f} Hz)")
        logger.info(f"  MSE vs 10 steps: {mse:.6f}")
        logger.info(f"  Max diff vs 10 steps: {max_diff:.4f}")

    return results


def print_summary(results: List[StepsBenchmarkResult]):
    """Print benchmark summary table."""
    print("\n" + "=" * 100)
    print("DENOISING STEPS vs PERFORMANCE BENCHMARK")
    print("=" * 100)
    print(f"{'Steps':<8} {'Latency (ms)':<15} {'Throughput (Hz)':<18} {'MSE vs 10':<15} {'Max Diff':<12} {'Recommendation':<20}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x.num_steps, reverse=True):
        rec = ""
        if r.throughput_hz >= 20 and r.mse_vs_10steps < 0.01:
            rec = "OPTIMAL"
        elif r.throughput_hz >= 20:
            rec = "HIGH PERF"
        elif r.mse_vs_10steps < 0.001:
            rec = "HIGH PRECISION"

        print(f"{r.num_steps:<8} {r.latency_ms:<15.1f} {r.throughput_hz:<18.2f} {r.mse_vs_10steps:<15.6f} {r.max_diff_vs_10steps:<12.4f} {rec:<20}")

    print("=" * 100)

    # Find optimal configuration
    optimal = None
    for r in results:
        if r.throughput_hz >= 20:
            if optimal is None or r.mse_vs_10steps < optimal.mse_vs_10steps:
                optimal = r

    if optimal:
        print(f"\nOptimal configuration for >20 Hz: {optimal.num_steps} steps")
        print(f"  Throughput: {optimal.throughput_hz:.1f} Hz")
        print(f"  Quality loss (MSE): {optimal.mse_vs_10steps:.6f}")
    else:
        print("\nNo configuration achieved >20 Hz")
        fastest = max(results, key=lambda x: x.throughput_hz)
        print(f"Fastest available: {fastest.num_steps} steps @ {fastest.throughput_hz:.1f} Hz")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark denoising steps vs accuracy"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="~/.cache/openpi/checkpoints/pi05_libero",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[10, 5, 3, 2, 1],
        help="Denoising step counts to test",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=20,
        help="Number of benchmark runs per configuration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./steps_benchmark_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Load model
    model, observation = load_model_and_observation(args.model_path)

    # Benchmark
    results = benchmark_steps(
        model,
        observation,
        steps_to_test=args.steps,
        num_runs=args.num_runs,
    )

    # Print summary
    print_summary(results)

    # Save results
    results_dict = {
        "model_path": args.model_path,
        "steps_tested": args.steps,
        "results": [
            {
                "num_steps": r.num_steps,
                "latency_ms": r.latency_ms,
                "throughput_hz": r.throughput_hz,
                "mse_vs_10steps": r.mse_vs_10steps,
                "max_diff_vs_10steps": r.max_diff_vs_10steps,
                "action_mean": r.action_mean,
                "action_std": r.action_std,
            }
            for r in results
        ],
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
