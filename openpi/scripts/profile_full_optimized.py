#!/usr/bin/env python3
"""
Performance Profiler for Full Optimized Pipeline.

Uses the same FullOptimizedPolicy from libero_eval_full_optimized.py:
- Vision TRT (FP16) - torch_tensorrt.compile
- KV Cache TRT FP8 MLP - ModelOpt + torch_tensorrt
- Denoising CUDA Graph - Pre-captured graph replay

Expected performance (1 denoising step):
- Vision TRT: ~17ms (2 images)
- KV Cache TRT FP8: ~40ms
- Denoise CUDA Graph: ~10ms/step
- Total: ~80ms (12.5 Hz)

Usage:
    # Profile with default 3 steps
    python scripts/profile_full_optimized.py

    # Profile with 1, 3, 10 steps
    python scripts/profile_full_optimized.py --steps 1 3 10 --save-as-baseline

Author: Based on libero_eval_full_optimized.py
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
import torch

# Setup paths
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

# Disable cuDNN for Jetson compatibility
torch.backends.cudnn.enabled = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_dummy_observation(device, tokenizer, max_token_len=200, max_state_dim=32):
    """Create dummy observation for profiling."""
    from openpi.models_pytorch.pi0_pytorch import Observation

    # Dummy images (224x224 RGB)
    img = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    wrist_img = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)

    images = {
        "base_0_rgb": img,
        "left_wrist_0_rgb": wrist_img,
        "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, device=device, dtype=torch.bfloat16) - 1.0,
    }
    image_masks = {
        "base_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
        "right_wrist_0_rgb": torch.zeros(1, device=device, dtype=torch.bool),
    }

    # Dummy state
    state = torch.randn(1, max_state_dim, device=device, dtype=torch.bfloat16)

    # Tokenize prompt
    token_ids = tokenizer.Encode("pick up the black bowl", add_bos=True)
    padding_len = max_token_len - len(token_ids)
    attention_mask = [1] * len(token_ids) + [0] * padding_len
    token_ids = token_ids + [0] * padding_len

    tokenized_prompt = torch.tensor([token_ids], dtype=torch.long, device=device)
    tokenized_prompt_mask = torch.tensor([attention_mask], dtype=torch.bool, device=device)

    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )


def profile_full_optimized(checkpoint_dir: str, num_steps_list: list, iterations: int = 30):
    """Profile the full optimized pipeline for multiple denoising step counts."""
    from libero_eval_full_optimized import FullOptimizedPolicy

    all_results = {}
    device = "cuda"

    for num_steps in num_steps_list:
        logger.info(f"\n{'='*70}")
        logger.info(f"Profiling Full Optimized Pipeline with {num_steps} denoising steps")
        logger.info(f"{'='*70}")

        # Create policy (this will compile TRT and capture CUDA Graph)
        logger.info("Creating FullOptimizedPolicy...")
        policy = FullOptimizedPolicy(
            checkpoint_dir=checkpoint_dir,
            device=device,
            num_denoising_steps=num_steps,
        )

        # Create dummy observation using policy's tokenizer
        obs_dict = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.random.randn(8).astype(np.float32),
            "prompt": "pick up the black bowl",
        }

        # Warmup
        logger.info(f"Warming up ({5} iterations)...")
        for _ in range(5):
            _ = policy.infer(obs_dict)
        torch.cuda.synchronize()

        # Reset latency records
        policy.latency_records = []
        policy.component_latencies = {
            'vision': [],
            'kv_cache': [],
            'denoise': [],
            'total': [],
        }

        # Profile
        logger.info(f"Profiling ({iterations} iterations)...")
        for i in range(iterations):
            _ = policy.infer(obs_dict)
        torch.cuda.synchronize()

        # Get stats
        stats = policy.get_latency_stats()

        # Store results in standard format
        step_key = f"{num_steps}_steps"
        all_results[step_key] = {
            'e2e_total': {
                'mean': stats['mean_ms'],
                'std': stats['std_ms'],
                'min': stats['min_ms'],
                'max': stats['max_ms'],
            },
            'vision': stats['components'].get('vision', {}),
            'kv_cache': stats['components'].get('kv_cache', {}),
            'denoise': stats['components'].get('denoise', {}),
            'breakdown': stats.get('breakdown', {}),
            'hz': stats['hz'],
        }

        # Print summary
        print(f"\n{'='*60}")
        print(f" {num_steps} Denoising Steps - Performance Summary (FULL OPTIMIZED)")
        print(f"{'='*60}")

        total = stats['mean_ms']
        print(f"\n{'Component':<20} {'Mean (ms)':>12} {'Std':>10} {'%':>8}")
        print("-" * 55)

        breakdown = stats.get('breakdown', {})
        for name, display in [('vision', 'Vision TRT FP16'), ('kv_cache', 'KV Cache TRT FP8'), ('denoise', 'Denoise CUDA Graph')]:
            if name in breakdown:
                b = breakdown[name]
                comp = stats['components'].get(name, {})
                print(f"{display:<20} {b['ms']:>12.2f} {comp.get('std_ms', 0):>10.2f} {b['pct']:>7.1f}%")

        print("-" * 55)
        print(f"{'E2E Total':<20} {total:>12.2f} ms")
        print(f"{'Frequency':<20} {stats['hz']:>12.1f} Hz")
        print(f"{'='*60}")

        # Cleanup
        del policy
        torch.cuda.empty_cache()

    return all_results


def generate_markdown(results: dict, output_path: str, tag: str = ""):
    """Generate markdown report."""
    lines = []
    lines.append("# Pi0.5 Full Optimized Pipeline - Performance Breakdown")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Tag**: {tag}")
    lines.append(f"**Platform**: NVIDIA Jetson Thor (SM110)")
    lines.append("")

    lines.append("## Backend Configuration")
    lines.append("")
    lines.append("| Component | Backend | Status |")
    lines.append("|-----------|---------|--------|")
    lines.append("| Vision Encoder | TRT FP16 (torch_tensorrt.compile) | ✅ Optimized |")
    lines.append("| KV Cache Prefill | TRT FP8 MLP (ModelOpt + torch_tensorrt) | ✅ Optimized |")
    lines.append("| Denoise Loop | CUDA Graph (torch.cuda.CUDAGraph) | ✅ Optimized |")
    lines.append("")

    # Summary table
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| Denoising Steps | Total Latency | Frequency | Vision | KV Cache | Denoise |")
    lines.append("|-----------------|---------------|-----------|--------|----------|---------|")

    for key in sorted([k for k in results.keys() if k != '_metadata']):
        data = results[key]
        num_steps = key.split('_')[0]
        total = data['e2e_total']['mean']
        hz = data['hz']

        breakdown = data.get('breakdown', {})
        vision = breakdown.get('vision', {}).get('ms', 0)
        kv = breakdown.get('kv_cache', {}).get('ms', 0)
        denoise = breakdown.get('denoise', {}).get('ms', 0)

        lines.append(f"| **{num_steps}** | **{total:.1f} ms** | **{hz:.1f} Hz** | {vision:.1f} ms | {kv:.1f} ms | {denoise:.1f} ms |")

    lines.append("")

    # Detailed breakdown
    for key in sorted([k for k in results.keys() if k != '_metadata']):
        data = results[key]
        num_steps = key.split('_')[0]

        lines.append(f"## {num_steps} Denoising Steps - Detailed Breakdown")
        lines.append("")
        lines.append("| Component | Time (ms) | Std (ms) | % of Total |")
        lines.append("|-----------|-----------|----------|------------|")

        total = data['e2e_total']['mean']
        breakdown = data.get('breakdown', {})

        for name, display in [('vision', 'Vision TRT FP16'), ('kv_cache', 'KV Cache TRT FP8'), ('denoise', f'Denoise CUDA Graph ({num_steps}x)')]:
            if name in breakdown:
                b = breakdown[name]
                comp = data.get(name, {})
                std = comp.get('std_ms', 0)
                lines.append(f"| {display} | {b['ms']:.2f} | {std:.2f} | {b['pct']:.1f}% |")

        lines.append(f"| **E2E Total** | **{total:.2f}** | {data['e2e_total'].get('std', 0):.2f} | 100% |")
        lines.append("")

        # Denoise per-step
        denoise_total = breakdown.get('denoise', {}).get('ms', 0)
        per_step = denoise_total / int(num_steps) if int(num_steps) > 0 else 0
        lines.append(f"### Denoise Analysis ({num_steps} steps)")
        lines.append("")
        lines.append(f"- Total denoise time: {denoise_total:.2f} ms")
        lines.append(f"- Per-step average: {per_step:.2f} ms")
        lines.append(f"- CUDA Graph overhead: minimal (graph replay)")
        lines.append("")

    # Comparison with PyTorch baseline
    lines.append("## Optimization Speedup")
    lines.append("")
    lines.append("Expected speedup vs PyTorch BF16 baseline:")
    lines.append("")
    lines.append("| Component | PyTorch BF16 | TRT Optimized | Speedup |")
    lines.append("|-----------|--------------|---------------|---------|")
    lines.append("| Vision | ~23 ms | ~17 ms | 1.35x |")
    lines.append("| KV Cache | ~88 ms | ~40 ms | 2.2x |")
    lines.append("| Denoise (1 step) | ~18 ms | ~10 ms | 1.8x |")
    lines.append("| **E2E (1 step)** | **~142 ms** | **~80 ms** | **1.78x** |")
    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Markdown report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile Full Optimized Pipeline")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero"),
                        help="Model checkpoint directory")
    parser.add_argument("--steps", type=int, nargs='+', default=[1, 3, 10],
                        help="Denoising steps to profile")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Number of iterations for profiling")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--save-as-baseline", action="store_true",
                        help="Save results as the new baseline")
    parser.add_argument("--tag", type=str, default="full_optimized_v1.2.0",
                        help="Tag for this profiling run")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(script_dir.parent / "docs")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Denoising steps: {args.steps}")
    logger.info(f"Iterations: {args.iterations}")

    # Run profiling
    results = profile_full_optimized(args.checkpoint, args.steps, args.iterations)

    # Add metadata
    results['_metadata'] = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tag': args.tag,
        'iterations': args.iterations,
        'steps': args.steps,
        'checkpoint': args.checkpoint,
        'backend': 'full_optimized',
        'optimizations': {
            'vision': 'TRT_FP16',
            'kv_cache': 'TRT_FP8_MLP',
            'denoise': 'CUDA_Graph',
        }
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""

    json_path = os.path.join(args.output_dir, f"full_optimized_profile_{timestamp}{tag_suffix}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"JSON results saved to {json_path}")

    if args.save_as_baseline:
        baseline_path = os.path.join(args.output_dir, "baseline_profile.json")
        with open(baseline_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved as baseline: {baseline_path}")

    # Generate markdown
    md_path = os.path.join(args.output_dir, "trt-fp8-detailed-performance-breakdown.md")
    generate_markdown(results, md_path, args.tag)

    print(f"\n{'='*60}")
    print("Profiling Complete!")
    print(f"{'='*60}")
    print(f"JSON results: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
