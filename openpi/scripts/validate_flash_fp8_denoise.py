#!/usr/bin/env python3
"""
Validate FlashAttention + FP8 MLP Denoising Precision.

This script compares the output of the optimized FlashFP8DenoiseEngine
against the baseline PyTorch implementation to ensure precision is maintained.

Validation criteria:
1. Cosine similarity > 0.999
2. Max absolute difference < 0.05
3. Mean absolute difference < 0.01

Usage:
    python validate_flash_fp8_denoise.py --checkpoint /path/to/checkpoint
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_single_step_precision(model, engine, prefix_kv_cache, prefix_pad_masks, device="cuda"):
    """Validate precision of a single denoise step."""
    logger.info("=" * 60)
    logger.info("Validating Single Denoise Step Precision")
    logger.info("=" * 60)

    action_horizon = model.config.action_horizon
    action_dim = model.config.action_dim

    # Use fixed seed for reproducibility
    torch.manual_seed(42)

    # Create test inputs
    state = torch.randn(1, 32, device=device, dtype=torch.bfloat16)
    x_t = torch.randn(1, action_horizon, action_dim, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor(0.5, device=device, dtype=torch.float32)

    # Baseline: Original PyTorch implementation
    with torch.no_grad():
        baseline_output = model.denoise_step_with_cache(
            state, prefix_kv_cache, prefix_pad_masks, x_t, timestep
        )

    # Optimized: FlashFP8 implementation
    with torch.no_grad():
        optimized_output = engine.denoise_step_with_cache(
            state, prefix_kv_cache, prefix_pad_masks, x_t, timestep
        )

    # Compute precision metrics
    baseline_flat = baseline_output.float().flatten()
    optimized_flat = optimized_output.float().flatten()

    cos_sim = F.cosine_similarity(baseline_flat, optimized_flat, dim=0).item()
    max_diff = (baseline_output - optimized_output).abs().max().item()
    mean_diff = (baseline_output - optimized_output).abs().mean().item()

    logger.info(f"Precision Metrics (Single Step):")
    logger.info(f"  Cosine Similarity: {cos_sim:.6f}")
    logger.info(f"  Max Absolute Diff: {max_diff:.6e}")
    logger.info(f"  Mean Absolute Diff: {mean_diff:.6e}")

    # Validation criteria
    passed = cos_sim > 0.999 and max_diff < 0.05 and mean_diff < 0.01
    status = "PASSED" if passed else "FAILED"
    logger.info(f"  Status: {status}")

    return {
        "cosine_similarity": cos_sim,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "passed": passed,
    }


def validate_full_denoise_loop_precision(model, engine, prefix_kv_cache, prefix_pad_masks, device="cuda", num_steps=10):
    """Validate precision of full denoising loop."""
    logger.info("=" * 60)
    logger.info(f"Validating Full Denoise Loop Precision ({num_steps} steps)")
    logger.info("=" * 60)

    action_horizon = model.config.action_horizon
    action_dim = model.config.action_dim

    # Use fixed seed for reproducibility
    torch.manual_seed(42)

    # Create initial noise (same for both)
    noise = torch.randn(1, action_horizon, action_dim, device=device, dtype=torch.bfloat16)
    state = torch.randn(1, 32, device=device, dtype=torch.bfloat16)

    # Baseline: Original PyTorch implementation
    dt = torch.tensor(-1.0 / num_steps, device=device)
    x_t_baseline = noise.clone()
    time_val = torch.tensor(1.0, device=device)

    with torch.no_grad():
        for _ in range(num_steps):
            v_t = model.denoise_step_with_cache(
                state, prefix_kv_cache, prefix_pad_masks, x_t_baseline, time_val
            )
            x_t_baseline = x_t_baseline + dt * v_t
            time_val = time_val + dt

    # Optimized: FlashFP8 implementation
    x_t_optimized = noise.clone()
    time_val = torch.tensor(1.0, device=device)

    with torch.no_grad():
        for _ in range(num_steps):
            v_t = engine.denoise_step_with_cache(
                state, prefix_kv_cache, prefix_pad_masks, x_t_optimized, time_val
            )
            x_t_optimized = x_t_optimized + dt * v_t
            time_val = time_val + dt

    # Compute precision metrics
    baseline_flat = x_t_baseline.float().flatten()
    optimized_flat = x_t_optimized.float().flatten()

    cos_sim = F.cosine_similarity(baseline_flat, optimized_flat, dim=0).item()
    max_diff = (x_t_baseline - x_t_optimized).abs().max().item()
    mean_diff = (x_t_baseline - x_t_optimized).abs().mean().item()

    logger.info(f"Precision Metrics (Full Loop):")
    logger.info(f"  Cosine Similarity: {cos_sim:.6f}")
    logger.info(f"  Max Absolute Diff: {max_diff:.6e}")
    logger.info(f"  Mean Absolute Diff: {mean_diff:.6e}")

    # Validation criteria (relaxed for accumulated error over steps)
    passed = cos_sim > 0.995 and max_diff < 0.1 and mean_diff < 0.02
    status = "PASSED" if passed else "FAILED"
    logger.info(f"  Status: {status}")

    return {
        "cosine_similarity": cos_sim,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "passed": passed,
        "num_steps": num_steps,
    }


def benchmark_latency(model, engine, prefix_kv_cache, prefix_pad_masks, device="cuda", num_iterations=50, num_steps=10):
    """Benchmark latency comparison."""
    logger.info("=" * 60)
    logger.info(f"Benchmarking Latency ({num_iterations} iterations, {num_steps} steps)")
    logger.info("=" * 60)

    action_horizon = model.config.action_horizon
    action_dim = model.config.action_dim

    state = torch.randn(1, 32, device=device, dtype=torch.bfloat16)

    # Warmup
    logger.info("Warming up...")
    for _ in range(5):
        x_t = torch.randn(1, action_horizon, action_dim, device=device, dtype=torch.bfloat16)
        dt = torch.tensor(-1.0 / num_steps, device=device)
        time_val = torch.tensor(1.0, device=device)
        for _ in range(num_steps):
            v_t = model.denoise_step_with_cache(state, prefix_kv_cache, prefix_pad_masks, x_t, time_val)
            x_t = x_t + dt * v_t
            time_val = time_val + dt
        v_t = engine.denoise_step_with_cache(state, prefix_kv_cache, prefix_pad_masks, x_t, time_val)
    torch.cuda.synchronize()

    # Benchmark baseline
    logger.info("Benchmarking baseline (PyTorch)...")
    baseline_times = []
    for _ in range(num_iterations):
        x_t = torch.randn(1, action_horizon, action_dim, device=device, dtype=torch.bfloat16)
        dt = torch.tensor(-1.0 / num_steps, device=device)
        time_val = torch.tensor(1.0, device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_steps):
            v_t = model.denoise_step_with_cache(state, prefix_kv_cache, prefix_pad_masks, x_t, time_val)
            x_t = x_t + dt * v_t
            time_val = time_val + dt
        torch.cuda.synchronize()
        baseline_times.append((time.perf_counter() - start) * 1000)

    # Benchmark optimized
    logger.info("Benchmarking optimized (FlashFP8)...")
    optimized_times = []
    for _ in range(num_iterations):
        x_t = torch.randn(1, action_horizon, action_dim, device=device, dtype=torch.bfloat16)
        dt = torch.tensor(-1.0 / num_steps, device=device)
        time_val = torch.tensor(1.0, device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_steps):
            v_t = engine.denoise_step_with_cache(state, prefix_kv_cache, prefix_pad_masks, x_t, time_val)
            x_t = x_t + dt * v_t
            time_val = time_val + dt
        torch.cuda.synchronize()
        optimized_times.append((time.perf_counter() - start) * 1000)

    baseline_mean = np.mean(baseline_times)
    baseline_std = np.std(baseline_times)
    optimized_mean = np.mean(optimized_times)
    optimized_std = np.std(optimized_times)
    speedup = baseline_mean / optimized_mean

    logger.info(f"\nLatency Results:")
    logger.info(f"  Baseline (PyTorch):  {baseline_mean:.2f} ± {baseline_std:.2f} ms")
    logger.info(f"  Optimized (FlashFP8): {optimized_mean:.2f} ± {optimized_std:.2f} ms")
    logger.info(f"  Speedup: {speedup:.2f}x")
    logger.info(f"  Per-step baseline: {baseline_mean/num_steps:.2f} ms")
    logger.info(f"  Per-step optimized: {optimized_mean/num_steps:.2f} ms")

    return {
        "baseline_mean_ms": baseline_mean,
        "baseline_std_ms": baseline_std,
        "optimized_mean_ms": optimized_mean,
        "optimized_std_ms": optimized_std,
        "speedup": speedup,
        "per_step_baseline_ms": baseline_mean / num_steps,
        "per_step_optimized_ms": optimized_mean / num_steps,
    }


def main(checkpoint_dir: str, compile_trt: bool = True):
    """Main validation function."""
    import json

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from openpi.inference.flash_fp8_denoise import FlashFP8DenoiseEngine, EXPERT_NUM_LAYERS, EXPERT_NUM_KV_HEADS, EXPERT_HEAD_DIM
    from safetensors.torch import load_file

    device = "cuda"

    logger.info("=" * 60)
    logger.info("FlashFP8 Denoise Precision Validation")
    logger.info("=" * 60)

    # Load config
    config_path = Path(checkpoint_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
        config = Pi0Config(
            action_dim=model_config.get("action_dim", 32),
            action_horizon=model_config.get("action_horizon", 50),
            max_token_len=model_config.get("max_token_len", 200),
            max_state_dim=model_config.get("max_state_dim", 32),
        )
    else:
        config = Pi0Config(action_dim=32, action_horizon=50, max_token_len=200, max_state_dim=32)

    # Load model
    logger.info(f"Loading model from {checkpoint_dir}...")
    model = PI0Pytorch(config)
    weights_path = Path(checkpoint_dir) / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    logger.info("Model loaded successfully")

    # Create FlashFP8 engine
    logger.info("Creating FlashFP8DenoiseEngine...")
    engine = FlashFP8DenoiseEngine(model, device=device, compile_trt=compile_trt)
    logger.info(f"Engine stats: {engine.get_stats()}")

    # Create dummy prefix KV cache (simulating output from Vision+Language processing)
    logger.info("Creating dummy prefix KV cache...")
    batch_size = 1
    prefix_len = 968

    prefix_kv_cache = []
    for _ in range(EXPERT_NUM_LAYERS):
        k = torch.randn(batch_size, EXPERT_NUM_KV_HEADS, prefix_len, EXPERT_HEAD_DIM,
                       device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, EXPERT_NUM_KV_HEADS, prefix_len, EXPERT_HEAD_DIM,
                       device=device, dtype=torch.bfloat16)
        prefix_kv_cache.append((k, v))

    prefix_pad_masks = torch.ones(batch_size, prefix_len, device=device, dtype=torch.bool)

    # Run validations
    results = {}

    # 1. Single step precision
    results["single_step"] = validate_single_step_precision(
        model, engine, prefix_kv_cache, prefix_pad_masks, device
    )

    # 2. Full loop precision (3 steps)
    results["full_loop_3_steps"] = validate_full_denoise_loop_precision(
        model, engine, prefix_kv_cache, prefix_pad_masks, device, num_steps=3
    )

    # 3. Full loop precision (10 steps)
    results["full_loop_10_steps"] = validate_full_denoise_loop_precision(
        model, engine, prefix_kv_cache, prefix_pad_masks, device, num_steps=10
    )

    # 4. Latency benchmark
    results["latency"] = benchmark_latency(
        model, engine, prefix_kv_cache, prefix_pad_masks, device, num_iterations=50, num_steps=10
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    all_passed = all([
        results["single_step"]["passed"],
        results["full_loop_3_steps"]["passed"],
        results["full_loop_10_steps"]["passed"],
    ])

    logger.info(f"Single Step Precision: {'PASSED' if results['single_step']['passed'] else 'FAILED'}")
    logger.info(f"  Cosine Similarity: {results['single_step']['cosine_similarity']:.6f}")

    logger.info(f"Full Loop (3 steps) Precision: {'PASSED' if results['full_loop_3_steps']['passed'] else 'FAILED'}")
    logger.info(f"  Cosine Similarity: {results['full_loop_3_steps']['cosine_similarity']:.6f}")

    logger.info(f"Full Loop (10 steps) Precision: {'PASSED' if results['full_loop_10_steps']['passed'] else 'FAILED'}")
    logger.info(f"  Cosine Similarity: {results['full_loop_10_steps']['cosine_similarity']:.6f}")

    logger.info(f"\nLatency Performance:")
    logger.info(f"  Baseline: {results['latency']['baseline_mean_ms']:.2f} ms")
    logger.info(f"  Optimized: {results['latency']['optimized_mean_ms']:.2f} ms")
    logger.info(f"  Speedup: {results['latency']['speedup']:.2f}x")

    overall_status = "ALL VALIDATIONS PASSED" if all_passed else "SOME VALIDATIONS FAILED"
    logger.info(f"\nOverall Status: {overall_status}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/root/.cache/openpi/checkpoints/pi05_libero")
    parser.add_argument("--no-trt", action="store_true", help="Disable TRT compilation (for precision testing only)")
    args = parser.parse_args()

    results = main(args.checkpoint, compile_trt=not args.no_trt)
    sys.exit(0 if all([
        results["single_step"]["passed"],
        results["full_loop_3_steps"]["passed"],
        results["full_loop_10_steps"]["passed"],
    ]) else 1)
