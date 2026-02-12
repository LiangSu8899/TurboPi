#!/usr/bin/env python3
"""
Simple Baseline Performance Profiler for Pi0.5.

This script measures the end-to-end performance of the current TRT FP8 mixed
quantization scheme using the UnifiedPolicy interface.

Usage:
    python scripts/profile_simple_baseline.py --steps 1 3 10

Author: Claude Code
Date: 2026-02-12
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Setup paths
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def profile_with_hooks(model, obs, num_steps, iterations=30):
    """Profile model components using forward hooks."""
    device = torch.device("cuda")
    timings = {}

    # Warmup
    logger.info("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model.sample_actions(device, obs, num_steps=num_steps, use_kv_cache=True)
    torch.cuda.synchronize()

    # Profile full E2E
    logger.info(f"Profiling E2E ({num_steps} steps)...")
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.sample_actions(device, obs, num_steps=num_steps, use_kv_cache=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    timings['e2e_total'] = {'mean': np.mean(times), 'std': np.std(times), 'min': np.min(times), 'max': np.max(times)}

    # Profile individual stages
    images = list(obs.images.values())
    img_masks = list(obs.image_masks.values())

    # Vision
    logger.info("Profiling Vision...")
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.paligemma_with_expert.embed_image(images[0])
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    timings['vision_base'] = {'mean': np.mean(times), 'std': np.std(times), 'min': np.min(times), 'max': np.max(times)}

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.paligemma_with_expert.embed_image(images[1])
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    timings['vision_wrist'] = {'mean': np.mean(times), 'std': np.std(times), 'min': np.min(times), 'max': np.max(times)}

    # Embed Prefix
    logger.info("Profiling Embed Prefix...")
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, obs.tokenized_prompt, obs.tokenized_prompt_mask
            )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    timings['embed_prefix'] = {'mean': np.mean(times), 'std': np.std(times), 'min': np.min(times), 'max': np.max(times)}

    # KV Cache Prefill
    logger.info("Profiling KV Cache Prefill...")
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, obs.tokenized_prompt, obs.tokenized_prompt_mask
        )

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            prefix_kv_cache = model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    timings['kv_cache_prefill'] = {'mean': np.mean(times), 'std': np.std(times), 'min': np.min(times), 'max': np.max(times)}

    # Denoise Loop
    logger.info(f"Profiling Denoise Loop ({num_steps} steps)...")
    with torch.no_grad():
        prefix_kv_cache = model.compute_prefix_kv_cache(
            prefix_embs, prefix_pad_masks, prefix_att_masks
        )

    times = []
    for _ in range(iterations):
        x_t = torch.randn(1, 50, 32, device=device, dtype=torch.bfloat16)
        dt = -1.0 / num_steps
        t = 1.0

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for step in range(num_steps):
                timestep = torch.tensor([t], device=device, dtype=torch.float32)
                v_t = model.denoise_step_with_cache(
                    obs.state, prefix_kv_cache, prefix_pad_masks, x_t, timestep
                )
                x_t = x_t + dt * v_t
                t += dt
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    timings['denoise_loop'] = {'mean': np.mean(times), 'std': np.std(times), 'min': np.min(times), 'max': np.max(times)}

    # Single denoise step
    logger.info("Profiling Single Denoise Step...")
    times = []
    x_t = torch.randn(1, 50, 32, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], device=device, dtype=torch.float32)

    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.denoise_step_with_cache(
                obs.state, prefix_kv_cache, prefix_pad_masks, x_t, timestep
            )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    timings['denoise_single_step'] = {'mean': np.mean(times), 'std': np.std(times), 'min': np.min(times), 'max': np.max(times)}

    return timings


def main():
    parser = argparse.ArgumentParser(description="Simple Baseline Performance Profiler")
    parser.add_argument("--checkpoint", type=str,
                        default="/root/.cache/openpi/checkpoints/pi05_libero",
                        help="Model checkpoint directory")
    parser.add_argument("--steps", type=int, nargs='+', default=[1, 3, 10],
                        help="Denoising steps to profile")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Number of iterations for profiling")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--save-as-baseline", action="store_true",
                        help="Save results as the new baseline")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for this profiling run")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(script_dir.parent / "docs")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
    from safetensors.torch import load_file

    config = Pi0Config(
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        max_state_dim=32,
        pi05=True,
    )

    model = PI0Pytorch(config)
    weights_path = Path(args.checkpoint) / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    logger.info("Model loaded successfully")

    # Create dummy observation
    img = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    wrist_img = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)

    # Tokenize
    import sentencepiece as spm
    tokenizer_path = Path(args.checkpoint) / "tokenizer.model"
    if not tokenizer_path.exists():
        tokenizer_path = Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"
    if not tokenizer_path.exists():
        tokenizer_path = Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(tokenizer_path))

    token_ids = tokenizer.Encode("pick up the black bowl", add_bos=True)
    padding_len = 200 - len(token_ids)
    token_ids = token_ids + [0] * padding_len
    token_masks = [1] * (200 - padding_len) + [0] * padding_len

    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    token_masks_t = torch.tensor([token_masks], dtype=torch.bool, device=device)
    state = torch.randn(1, 32, device=device, dtype=torch.bfloat16)

    obs = Observation(
        images={"base_0_rgb": img, "left_wrist_0_rgb": wrist_img,
                "right_wrist_0_rgb": torch.zeros_like(img)},
        image_masks={"base_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
                     "left_wrist_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
                     "right_wrist_0_rgb": torch.zeros(1, device=device, dtype=torch.bool)},
        state=state,
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_masks_t,
    )

    # Profile for each step count
    all_results = {}

    for num_steps in args.steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Profiling with {num_steps} denoising steps")
        logger.info(f"{'='*60}")

        timings = profile_with_hooks(model, obs, num_steps, args.iterations)
        all_results[f"{num_steps}_steps"] = timings

        # Print summary
        print(f"\n{'='*60}")
        print(f" {num_steps} Denoising Steps - Performance Summary")
        print(f"{'='*60}")

        total = timings['e2e_total']['mean']
        print(f"\n{'Component':<25} {'Mean (ms)':>10} {'Std':>8} {'%':>6}")
        print("-" * 55)

        components = [
            ('Vision (base)', 'vision_base'),
            ('Vision (wrist)', 'vision_wrist'),
            ('Embed Prefix', 'embed_prefix'),
            ('KV Cache Prefill', 'kv_cache_prefill'),
            (f'Denoise Loop ({num_steps}x)', 'denoise_loop'),
        ]

        for name, key in components:
            t = timings[key]
            pct = t['mean'] / total * 100
            print(f"{name:<25} {t['mean']:>10.2f} {t['std']:>8.2f} {pct:>5.1f}%")

        print("-" * 55)
        print(f"{'E2E Total':<25} {total:>10.2f} ms")
        print(f"{'Frequency':<25} {1000/total:>10.1f} Hz")
        print(f"{'='*60}")

        # Additional info
        single_step = timings['denoise_single_step']['mean']
        print(f"\nSingle denoise step: {single_step:.2f} ms")
        print(f"Denoise loop overhead: {timings['denoise_loop']['mean'] - num_steps * single_step:.2f} ms")

    # Add metadata
    all_results['_metadata'] = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tag': args.tag,
        'iterations': args.iterations,
        'steps': args.steps,
        'checkpoint': args.checkpoint,
    }

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""

    json_path = os.path.join(args.output_dir, f"baseline_profile_{timestamp}{tag_suffix}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    if args.save_as_baseline:
        baseline_path = os.path.join(args.output_dir, "baseline_profile.json")
        with open(baseline_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved as baseline: {baseline_path}")

    # Generate markdown
    md_path = os.path.join(args.output_dir, "trt-fp8-detailed-performance-breakdown.md")
    generate_markdown(all_results, md_path)
    logger.info(f"Markdown report saved to {md_path}")

    print(f"\n{'='*60}")
    print("Profiling Complete!")
    print(f"{'='*60}")


def generate_markdown(results: dict, output_path: str):
    """Generate markdown report."""
    lines = []
    lines.append("# Pi0.5 TRT FP8 Mixed Quantization - Detailed Performance Breakdown")
    lines.append("")

    metadata = results.get('_metadata', {})
    lines.append(f"**Generated**: {metadata.get('timestamp', 'N/A')}")
    lines.append(f"**Tag**: {metadata.get('tag', 'N/A')}")
    lines.append(f"**Platform**: NVIDIA Jetson Thor (SM110)")
    lines.append(f"**Backend**: Torch-TRT FP8 Static Graph Mixed Quantization (v1.2.0)")
    lines.append("")

    # Summary table
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| Denoising Steps | Total Latency | Frequency | Vision | Embed | KV Cache | Denoise |")
    lines.append("|-----------------|---------------|-----------|--------|-------|----------|---------|")

    for key in sorted([k for k in results.keys() if k != '_metadata']):
        data = results[key]
        num_steps = key.split('_')[0]
        total = data['e2e_total']['mean']
        hz = 1000 / total
        vision = data['vision_base']['mean'] + data['vision_wrist']['mean']
        embed = data['embed_prefix']['mean']
        kv = data['kv_cache_prefill']['mean']
        denoise = data['denoise_loop']['mean']

        lines.append(f"| {num_steps} | {total:.1f} ms | {hz:.1f} Hz | {vision:.1f} ms | {embed:.1f} ms | {kv:.1f} ms | {denoise:.1f} ms |")

    lines.append("")

    # Detailed breakdown for each step count
    for key in sorted([k for k in results.keys() if k != '_metadata']):
        data = results[key]
        num_steps = key.split('_')[0]

        lines.append(f"## {num_steps} Denoising Steps - Detailed Breakdown")
        lines.append("")
        lines.append("| Component | Time (ms) | % of Total | Notes |")
        lines.append("|-----------|-----------|------------|-------|")

        total = data['e2e_total']['mean']

        components = [
            ('Vision (base)', 'vision_base', 'SigLIP encoder'),
            ('Vision (wrist)', 'vision_wrist', 'SigLIP encoder'),
            ('Embed Prefix', 'embed_prefix', 'Image + Language fusion'),
            ('KV Cache Prefill', 'kv_cache_prefill', 'PaliGemma 18 layers'),
            (f'Denoise Loop ({num_steps}x)', 'denoise_loop', 'Action Expert'),
        ]

        for name, key_name, notes in components:
            t = data[key_name]['mean']
            pct = t / total * 100
            lines.append(f"| {name} | {t:.2f} | {pct:.1f}% | {notes} |")

        lines.append(f"| **Total** | **{total:.2f}** | 100% | |")
        lines.append("")

        # Denoise details
        single = data['denoise_single_step']['mean']
        loop = data['denoise_loop']['mean']
        overhead = loop - int(num_steps) * single

        lines.append(f"### Denoise Analysis ({num_steps} steps)")
        lines.append("")
        lines.append(f"- Single step: {single:.2f} ms")
        lines.append(f"- {num_steps} steps total: {loop:.2f} ms")
        lines.append(f"- Overhead: {overhead:.2f} ms ({overhead/loop*100:.1f}%)")
        lines.append("")

    # Optimization opportunities
    lines.append("## Optimization Opportunities")
    lines.append("")
    lines.append("Based on the profiling results:")
    lines.append("")
    lines.append("1. **KV Cache MLP** - Primary target for FP8 quantization")
    lines.append("2. **Vision Encoder** - Can use TRT FP16/INT8")
    lines.append("3. **Embed Prefix** - Potential for kernel fusion")
    lines.append("4. **Denoise Loop** - CUDA Graph optimization")
    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    main()
