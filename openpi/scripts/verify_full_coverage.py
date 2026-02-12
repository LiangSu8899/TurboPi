#!/usr/bin/env python3
"""Verify Full W4A16 Coverage: MLP + Attention Quantization.

This script benchmarks the performance improvement from quantizing
both MLP and Attention projection layers (q, k, v, o).

Expected improvements over MLP-only:
- Additional ~40-60ms savings from attention quantization
- ~3x total speedup vs baseline

Author: Claude Code
Date: 2026-02-11
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BenchmarkResult:
    """Results from benchmark."""
    mode: str
    total_ms: float
    hz: float
    mlp_layers: int
    attention_layers: int
    memory_saved_mb: float


def load_model_base():
    """Load base model without quantization."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from safetensors.torch import load_file

    device = 'cuda'
    checkpoint_path = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()

    # Load config
    with open(checkpoint_path / "config.json") as f:
        model_config = json.load(f)

    max_token_len = model_config.get("tokenizer_max_length", 200)

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=model_config.get("action_dim", 32),
        action_horizon=model_config.get("action_horizon", 50),
        max_token_len=max_token_len,
        pi05=True,
        dtype="bfloat16",
    )

    print("Loading model...")
    model = PI0Pytorch(pi0_config)
    state_dict = load_file(checkpoint_path / "model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, pi0_config, device, max_token_len


def create_test_observation(device, max_token_len):
    """Create a test observation."""
    from openpi.models_pytorch.pi0_pytorch import Observation

    return Observation(
        images={
            "base_0_rgb": torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device=device),
            "left_wrist_0_rgb": torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device=device),
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, dtype=torch.bfloat16, device=device),
        },
        image_masks={
            "base_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
        },
        state=torch.randn(1, 32, dtype=torch.bfloat16, device=device),
        tokenized_prompt=torch.randint(0, 1000, (1, max_token_len), dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.ones(1, max_token_len, dtype=torch.bool, device=device),
    )


def benchmark_inference(model, device, observation, num_steps=3, num_iterations=20, warmup=5):
    """Benchmark inference."""
    # Warmup
    print("  Warming up...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=num_steps, use_kv_cache=True)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(num_iterations):
        start_event.record()
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=num_steps, use_kv_cache=True)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    hz = 1000 / mean_ms

    return mean_ms, std_ms, hz


def run_baseline(model, device, observation, num_steps=3) -> BenchmarkResult:
    """Run baseline (BF16, no quantization)."""
    print("\n" + "=" * 70)
    print("BASELINE: BF16 (No Quantization)")
    print("=" * 70)

    mean_ms, std_ms, hz = benchmark_inference(model, device, observation, num_steps)

    print(f"\n  Results:")
    print(f"    Mean:     {mean_ms:.2f} ms")
    print(f"    Std:      {std_ms:.2f} ms")
    print(f"    Hz:       {hz:.1f}")

    return BenchmarkResult(
        mode="BF16 Baseline",
        total_ms=mean_ms,
        hz=hz,
        mlp_layers=0,
        attention_layers=0,
        memory_saved_mb=0,
    )


def run_mlp_only(model, device, observation, num_steps=3) -> BenchmarkResult:
    """Run MLP-only W4A16 quantization."""
    from openpi.utils.model_patcher import patch_paligemma_decode_path, patch_expert_decode_path

    print("\n" + "=" * 70)
    print("W4A16 MLP-ONLY: Quantizing gate/up/down projections")
    print("=" * 70)

    # Patch PaliGemma
    pg_stats = patch_paligemma_decode_path(model, quantize_attention=False, verbose=True)

    # Patch Expert
    ex_stats = patch_expert_decode_path(model, quantize_attention=False, verbose=True)

    total_mlp = pg_stats['replaced'] + ex_stats['replaced']
    total_memory = pg_stats['memory_saved_mb'] + ex_stats['memory_saved_mb']

    print(f"\n  Total MLP layers replaced: {total_mlp}")
    print(f"  Total memory saved: {total_memory:.1f} MB")

    # Benchmark
    mean_ms, std_ms, hz = benchmark_inference(model, device, observation, num_steps)

    print(f"\n  Results:")
    print(f"    Mean:     {mean_ms:.2f} ms")
    print(f"    Std:      {std_ms:.2f} ms")
    print(f"    Hz:       {hz:.1f}")

    return BenchmarkResult(
        mode="W4A16 MLP-only",
        total_ms=mean_ms,
        hz=hz,
        mlp_layers=total_mlp,
        attention_layers=0,
        memory_saved_mb=total_memory,
    )


def run_full_coverage() -> BenchmarkResult:
    """Run full W4A16 coverage (MLP + Attention)."""
    from openpi.utils.model_patcher import patch_full_coverage

    print("\n" + "=" * 70)
    print("W4A16 FULL COVERAGE: MLP + Attention (Q, K, V, O)")
    print("=" * 70)

    # Need to reload model fresh since we can't unpatch
    model, config, device, max_token_len = load_model_base()
    observation = create_test_observation(device, max_token_len)

    # Apply full coverage
    stats = patch_full_coverage(model, verbose=True)

    print(f"\n  Total layers replaced: {stats['total_replaced']}")
    print(f"    - MLP: {stats['total_mlp_layers']}")
    print(f"    - Attention: {stats['total_attention_layers']}")
    print(f"  Total memory saved: {stats['total_memory_saved_mb']:.1f} MB")

    # Benchmark
    mean_ms, std_ms, hz = benchmark_inference(model, device, observation, num_steps=3)

    print(f"\n  Results:")
    print(f"    Mean:     {mean_ms:.2f} ms")
    print(f"    Std:      {std_ms:.2f} ms")
    print(f"    Hz:       {hz:.1f}")

    return BenchmarkResult(
        mode="W4A16 Full Coverage",
        total_ms=mean_ms,
        hz=hz,
        mlp_layers=stats['total_mlp_layers'],
        attention_layers=stats['total_attention_layers'],
        memory_saved_mb=stats['total_memory_saved_mb'],
    )


def verify_accuracy(model, device, observation, reference_actions=None):
    """Verify action output accuracy after quantization."""
    print("\n" + "=" * 70)
    print("ACCURACY VERIFICATION")
    print("=" * 70)

    with torch.no_grad():
        actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)

    print(f"  Action shape: {actions.shape}")
    print(f"  Action dtype: {actions.dtype}")
    print(f"  Action range: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
    print(f"  Action mean:  {actions.mean().item():.4f}")
    print(f"  Action std:   {actions.std().item():.4f}")

    if reference_actions is not None:
        l2_dist = torch.sqrt(torch.mean((actions - reference_actions) ** 2)).item()
        cos_sim = F.cosine_similarity(
            actions.flatten().unsqueeze(0),
            reference_actions.flatten().unsqueeze(0)
        ).item()
        print(f"\n  vs Reference:")
        print(f"    L2 Distance:      {l2_dist:.6f}")
        print(f"    Cosine Similarity: {cos_sim:.6f}")

    return actions


def run_verification():
    """Run complete verification."""
    print("=" * 70)
    print("W4A16 FULL COVERAGE VERIFICATION")
    print("MLP + Attention Quantization Performance Test")
    print("=" * 70)
    print()

    # Test 1: Baseline
    model, config, device, max_token_len = load_model_base()
    observation = create_test_observation(device, max_token_len)

    # Get baseline actions for accuracy check
    print("Getting reference actions from BF16 baseline...")
    with torch.no_grad():
        reference_actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)

    baseline_result = run_baseline(model, device, observation)

    # Test 2: MLP-only
    mlp_result = run_mlp_only(model, device, observation)
    mlp_actions = verify_accuracy(model, device, observation, reference_actions)

    # Test 3: Full coverage (need fresh model)
    full_result = run_full_coverage()

    # Re-load for accuracy check
    model2, _, device2, max_token_len2 = load_model_base()
    observation2 = create_test_observation(device2, max_token_len2)
    from openpi.utils.model_patcher import patch_full_coverage
    patch_full_coverage(model2, verbose=False)
    full_actions = verify_accuracy(model2, device2, observation2, reference_actions)

    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    # TRT FP8 baseline for comparison
    trt_fp8_ms = 120.0
    trt_fp8_hz = 8.3

    print(f"""
   +----------------------------------+----------+--------+----------+
   | Mode                             | Time(ms) | Hz     | Speedup  |
   +----------------------------------+----------+--------+----------+
   | TRT FP8 (reference)              | {trt_fp8_ms:7.1f}  | {trt_fp8_hz:5.1f}  |   1.0x   |
   | BF16 Baseline                    | {baseline_result.total_ms:7.1f}  | {baseline_result.hz:5.1f}  | {trt_fp8_ms/baseline_result.total_ms:5.1f}x   |
   | W4A16 MLP-only ({mlp_result.mlp_layers} layers)       | {mlp_result.total_ms:7.1f}  | {mlp_result.hz:5.1f}  | {trt_fp8_ms/mlp_result.total_ms:5.1f}x   |
   | W4A16 Full Coverage              | {full_result.total_ms:7.1f}  | {full_result.hz:5.1f}  | {trt_fp8_ms/full_result.total_ms:5.1f}x   |
   |   ({full_result.mlp_layers} MLP + {full_result.attention_layers} ATT layers)   |          |        |          |
   +----------------------------------+----------+--------+----------+

   Performance Gains from Full Coverage:
   - vs MLP-only: {mlp_result.total_ms - full_result.total_ms:.1f} ms faster ({(mlp_result.total_ms/full_result.total_ms - 1)*100:.1f}% improvement)
   - vs Baseline: {baseline_result.total_ms - full_result.total_ms:.1f} ms faster ({(baseline_result.total_ms/full_result.total_ms - 1)*100:.1f}% improvement)

   Memory Savings:
   - MLP-only:      {mlp_result.memory_saved_mb:.1f} MB
   - Full Coverage: {full_result.memory_saved_mb:.1f} MB

   Accuracy Check:
   - MLP-only vs BF16:      L2={torch.sqrt(torch.mean((mlp_actions - reference_actions) ** 2)).item():.6f}
   - Full Coverage vs BF16: L2={torch.sqrt(torch.mean((full_actions - reference_actions) ** 2)).item():.6f}
""")

    # Verdict
    if full_result.hz > mlp_result.hz * 1.1:
        print("   VERDICT: Full Coverage provides significant speedup!")
    elif full_result.hz > mlp_result.hz:
        print("   VERDICT: Full Coverage provides modest speedup.")
    else:
        print("   VERDICT: Full Coverage may not provide additional benefit on this hardware.")

    return {
        'baseline': baseline_result,
        'mlp_only': mlp_result,
        'full_coverage': full_result,
    }


if __name__ == "__main__":
    results = run_verification()
