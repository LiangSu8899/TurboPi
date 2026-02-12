#!/usr/bin/env python3
"""
Simple Denoise Profiling with NVTX Markers.

Uses the existing FullOptimizedPolicy but adds NVTX markers
to profile kernel gaps and identify bottlenecks.

Usage:
    nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas \
        --output=denoise_profile \
        python scripts/profile_denoise_simple.py --steps 10

Author: Turbo-Pi Team
Date: 2026-02-12
"""

import sys
import os
import time
import pathlib
import logging
import argparse

import numpy as np
import torch

# Setup paths
script_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

# Disable cuDNN for Jetson
torch.backends.cudnn.enabled = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def nvtx_range(name: str):
    """NVTX range context manager."""
    class NVTXRange:
        def __enter__(self):
            torch.cuda.nvtx.range_push(name)
            return self
        def __exit__(self, *args):
            torch.cuda.nvtx.range_pop()
    return NVTXRange()


def run_profiling(checkpoint_dir: str, num_steps: int, warmup: int, iterations: int):
    """Run profiling using FullOptimizedPolicy."""
    from libero_eval_full_optimized import FullOptimizedPolicy

    device = "cuda"

    logger.info("=" * 70)
    logger.info(f"Denoise Profiling: {num_steps} steps, {iterations} iterations")
    logger.info("=" * 70)

    # Create policy
    logger.info("Loading FullOptimizedPolicy...")
    policy = FullOptimizedPolicy(
        checkpoint_dir=checkpoint_dir,
        device=device,
        num_denoising_steps=num_steps,
    )

    # Create dummy observation
    obs_dict = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.randn(8).astype(np.float32),
        "prompt": "pick up the black bowl",
    }

    # Warmup
    logger.info(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = policy.infer(obs_dict)
    torch.cuda.synchronize()

    # Profile iterations with NVTX
    logger.info(f"Profiling ({iterations} iterations)...")
    latencies = []

    for i in range(iterations):
        torch.cuda.nvtx.mark(f"Iteration_{i}_Start")

        torch.cuda.synchronize()
        start = time.perf_counter()

        with nvtx_range(f"Iteration_{i}"):
            with nvtx_range("E2E_Inference"):
                result = policy.infer(obs_dict)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        torch.cuda.nvtx.mark(f"Iteration_{i}_End")

    # Report
    latencies = np.array(latencies)
    logger.info("=" * 70)
    logger.info("Profiling Results")
    logger.info("=" * 70)
    logger.info(f"  Mean: {latencies.mean():.2f} ms")
    logger.info(f"  Std:  {latencies.std():.2f} ms")
    logger.info(f"  Min:  {latencies.min():.2f} ms")
    logger.info(f"  Max:  {latencies.max():.2f} ms")
    logger.info(f"  Per-step: {latencies.mean() / num_steps:.2f} ms/step (denoise only estimate)")
    logger.info("=" * 70)

    # Get component breakdown
    stats = policy.get_latency_stats()
    logger.info("\nComponent Breakdown:")
    for name in ['vision', 'kv_cache', 'denoise']:
        if name in stats.get('components', {}):
            comp = stats['components'][name]
            logger.info(f"  {name}: {comp.get('mean_ms', 0):.2f} ms")

    return latencies


def main():
    parser = argparse.ArgumentParser(description="Simple Denoise Profiling")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero"),
        help="Model checkpoint directory"
    )
    parser.add_argument("--steps", type=int, default=10, help="Denoising steps")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, help="Profile iterations")
    args = parser.parse_args()

    run_profiling(
        args.checkpoint,
        num_steps=args.steps,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
