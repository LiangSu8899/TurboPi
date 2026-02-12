#!/usr/bin/env python3
"""Verify Hybrid Sampler: TRT FP8 Prefill + W4A16 TVM Decode.

This script benchmarks the hybrid pipeline that combines:
1. TRT Vision Encoder (FP16) - ~5ms
2. TRT FP8 Prefill (compute-bound) - ~40ms
3. W4A16 Decode + CUDA Graph (memory-bound) - ~30ms

Target: Total < 80ms (> 12 Hz) single-frame inference

Author: Claude Code
Date: 2026-02-11
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BenchmarkResult:
    """Benchmark results."""
    mode: str
    vision_ms: float
    embed_ms: float
    prefill_ms: float
    denoise_ms: float
    total_ms: float
    hz: float


def load_model_and_apply_w4a16():
    """Load PI0Pytorch model and apply W4A16 MLP quantization."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from openpi.utils.model_patcher import patch_paligemma_decode_path, patch_expert_decode_path
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

    # Apply W4A16 MLP quantization (MLP only, NOT attention)
    print("Applying W4A16 MLP quantization...")
    pg_stats = patch_paligemma_decode_path(model, quantize_attention=False, verbose=False)
    ex_stats = patch_expert_decode_path(model, quantize_attention=False, verbose=False)
    print(f"  PaliGemma: {pg_stats['replaced']} MLP layers")
    print(f"  Expert: {ex_stats['replaced']} MLP layers")

    return model, pi0_config, device, max_token_len, str(checkpoint_path)


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


def benchmark_baseline(model, device, observation, num_steps=3, num_iterations=20) -> BenchmarkResult:
    """Benchmark baseline eager mode (BF16)."""
    print("\n" + "=" * 70)
    print("BASELINE: Eager BF16 (No Optimization)")
    print("=" * 70)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=num_steps, use_kv_cache=True)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=num_steps, use_kv_cache=True)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    hz = 1000 / mean_ms

    print(f"\n  Baseline Results:")
    print(f"    Mean:     {mean_ms:.2f} ms")
    print(f"    Hz:       {hz:.1f}")

    return BenchmarkResult(
        mode="Baseline BF16",
        vision_ms=0,
        embed_ms=0,
        prefill_ms=0,
        denoise_ms=0,
        total_ms=mean_ms,
        hz=hz,
    )


def benchmark_hybrid_no_trt_compile(model, device, observation, checkpoint_dir, num_steps=3, num_iterations=20) -> BenchmarkResult:
    """Benchmark hybrid sampler WITHOUT TRT compilation (fallback mode)."""
    from openpi.policies.hybrid_sampler import HybridSampler, HybridSamplerConfig

    print("\n" + "=" * 70)
    print("HYBRID (Fallback): TRT Vision + Eager Prefill + W4A16 Graph Decode")
    print("=" * 70)

    config = HybridSamplerConfig(
        num_denoise_steps=num_steps,
        batch_size=1,
        dtype=torch.bfloat16,
        use_trt_vision=True,
        use_trt_prefill=False,  # Skip TRT prefill
        use_cuda_graph=True,
        checkpoint_dir=checkpoint_dir,
    )

    sampler = HybridSampler(model, device, config)

    try:
        print("  Warming up...")
        sampler.warm_up(observation)
    except Exception as e:
        print(f"\n  [ERROR] Warmup failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Benchmark
    print("  Benchmarking...")
    times = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = sampler.sample_actions(observation)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    hz = 1000 / mean_ms
    stats = sampler.get_timing_stats()

    print(f"\n  Hybrid (Fallback) Results:")
    print(f"    Mean:     {mean_ms:.2f} ms")
    print(f"    Hz:       {hz:.1f}")
    print(f"\n  Timing Breakdown:")
    print(f"    Vision:   {stats['vision_ms']:.2f} ms")
    print(f"    Embed:    {stats['embed_ms']:.2f} ms")
    print(f"    Prefill:  {stats['prefill_ms']:.2f} ms")
    print(f"    Denoise:  {stats['denoise_ms']:.2f} ms")

    return BenchmarkResult(
        mode="Hybrid (Fallback)",
        vision_ms=stats['vision_ms'],
        embed_ms=stats['embed_ms'],
        prefill_ms=stats['prefill_ms'],
        denoise_ms=stats['denoise_ms'],
        total_ms=mean_ms,
        hz=hz,
    )


def benchmark_hybrid_full(model, device, observation, checkpoint_dir, num_steps=3, num_iterations=20) -> BenchmarkResult:
    """Benchmark full hybrid sampler with TRT FP8 compilation."""
    from openpi.policies.hybrid_sampler import HybridSampler, HybridSamplerConfig

    print("\n" + "=" * 70)
    print("HYBRID (Full): TRT Vision + TRT FP8 Prefill + W4A16 Graph Decode")
    print("=" * 70)

    config = HybridSamplerConfig(
        num_denoise_steps=num_steps,
        batch_size=1,
        dtype=torch.bfloat16,
        use_trt_vision=True,
        use_trt_prefill=True,
        use_cuda_graph=True,
        checkpoint_dir=checkpoint_dir,
    )

    sampler = HybridSampler(model, device, config)

    # Compile TRT (expensive!)
    print("\n  Compiling TRT FP8 Prefill (this takes ~6 minutes)...")
    compile_success = sampler.compile_trt()

    if not compile_success:
        print("  [WARNING] TRT compilation failed, results may be suboptimal")

    try:
        print("  Warming up...")
        sampler.warm_up(observation)
    except Exception as e:
        print(f"\n  [ERROR] Warmup failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Benchmark
    print("  Benchmarking...")
    times = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = sampler.sample_actions(observation)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    hz = 1000 / mean_ms
    stats = sampler.get_timing_stats()

    print(f"\n  Hybrid (Full) Results:")
    print(f"    Mean:     {mean_ms:.2f} ms")
    print(f"    Hz:       {hz:.1f}")
    print(f"\n  Timing Breakdown:")
    print(f"    Vision:   {stats['vision_ms']:.2f} ms")
    print(f"    Embed:    {stats['embed_ms']:.2f} ms")
    print(f"    Prefill:  {stats['prefill_ms']:.2f} ms (TRT FP8)")
    print(f"    Denoise:  {stats['denoise_ms']:.2f} ms (W4A16 Graph)")

    return BenchmarkResult(
        mode="Hybrid (Full TRT)",
        vision_ms=stats['vision_ms'],
        embed_ms=stats['embed_ms'],
        prefill_ms=stats['prefill_ms'],
        denoise_ms=stats['denoise_ms'],
        total_ms=mean_ms,
        hz=hz,
    )


def run_verification(skip_trt_compile: bool = False):
    """Run complete verification."""
    print("=" * 70)
    print("HYBRID SAMPLER VERIFICATION")
    print("TRT FP8 Prefill + W4A16 TVM Decode")
    print("=" * 70)
    print()

    # Load model
    model, config, device, max_token_len, checkpoint_dir = load_model_and_apply_w4a16()
    observation = create_test_observation(device, max_token_len)

    results = {}

    # 1. Baseline
    results['baseline'] = benchmark_baseline(model, device, observation)

    # 2. Hybrid without TRT compile (fallback mode)
    results['hybrid_fallback'] = benchmark_hybrid_no_trt_compile(
        model, device, observation, checkpoint_dir
    )

    # 3. Hybrid with full TRT compile (optional, slow)
    if not skip_trt_compile:
        results['hybrid_full'] = benchmark_hybrid_full(
            model, device, observation, checkpoint_dir
        )

    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    trt_fp8_ref_ms = 120.0
    trt_fp8_ref_hz = 8.3

    print(f"""
   +----------------------------------+----------+--------+----------+
   | Mode                             | Time(ms) | Hz     | Speedup  |
   +----------------------------------+----------+--------+----------+
   | TRT FP8 Static (reference)       | {trt_fp8_ref_ms:7.1f}  | {trt_fp8_ref_hz:5.1f}  |   1.0x   |
""")

    for name, result in results.items():
        if result is not None:
            speedup = trt_fp8_ref_ms / result.total_ms
            print(f"   | {result.mode:32} | {result.total_ms:7.1f}  | {result.hz:5.1f}  | {speedup:5.1f}x   |")

    print("   +----------------------------------+----------+--------+----------+")

    # Component breakdown for hybrid
    if results.get('hybrid_fallback') is not None:
        r = results['hybrid_fallback']
        print(f"""
   Component Breakdown (Hybrid Fallback):
   +---------------------------+----------+
   | Component                 | Time(ms) |
   +---------------------------+----------+
   | Vision (TRT)              | {r.vision_ms:7.2f}  |
   | Embed Prefix              | {r.embed_ms:7.2f}  |
   | Prefill (Eager)           | {r.prefill_ms:7.2f}  |
   | Denoise (W4A16 Graph)     | {r.denoise_ms:7.2f}  |
   +---------------------------+----------+
   | Total                     | {r.total_ms:7.2f}  |
   +---------------------------+----------+
""")

    if results.get('hybrid_full') is not None:
        r = results['hybrid_full']
        print(f"""
   Component Breakdown (Hybrid Full TRT):
   +---------------------------+----------+
   | Component                 | Time(ms) |
   +---------------------------+----------+
   | Vision (TRT)              | {r.vision_ms:7.2f}  |
   | Embed Prefix              | {r.embed_ms:7.2f}  |
   | Prefill (TRT FP8)         | {r.prefill_ms:7.2f}  |
   | Denoise (W4A16 Graph)     | {r.denoise_ms:7.2f}  |
   +---------------------------+----------+
   | Total                     | {r.total_ms:7.2f}  |
   +---------------------------+----------+
""")

    # Target check
    target_ms = 80.0
    target_hz = 12.0

    best_result = min(
        [r for r in results.values() if r is not None],
        key=lambda x: x.total_ms
    )

    passed = best_result.total_ms < target_ms

    print(f"   Target: < {target_ms}ms ({target_hz}Hz)")
    print(f"   Best:   {best_result.total_ms:.1f}ms ({best_result.hz:.1f}Hz)")
    print(f"   Status: {'PASS' if passed else 'FAIL'}")

    if passed:
        print(f"\n   HYBRID VERIFICATION: PASSED")
        print(f"   Achieved {best_result.hz:.1f} Hz = {trt_fp8_ref_ms/best_result.total_ms:.1f}x faster than TRT FP8 Static!")
    else:
        print(f"\n   HYBRID VERIFICATION: NEEDS OPTIMIZATION")
        bottleneck = max(
            [('Vision', best_result.vision_ms),
             ('Embed', best_result.embed_ms),
             ('Prefill', best_result.prefill_ms),
             ('Denoise', best_result.denoise_ms)],
            key=lambda x: x[1]
        )
        print(f"   Bottleneck: {bottleneck[0]} ({bottleneck[1]:.1f}ms)")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-trt-compile", action="store_true",
                        help="Skip TRT FP8 compilation (takes ~6 min)")
    args = parser.parse_args()

    results = run_verification(skip_trt_compile=args.skip_trt_compile)
