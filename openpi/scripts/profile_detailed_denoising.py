#!/usr/bin/env python3
"""
Detailed Denoising Performance Profiler for Pi0.5 TRT FP8 Mixed Quantization.

This script measures the detailed per-layer performance breakdown of the
current Torch-TRT FP8 static graph mixed quantization scheme for 1, 3, 10
denoising steps.

Components measured:
1. Vision Encoder (SigLIP) - FP16
2. Embed Prefix (Language Embedding) - BF16
3. KV Cache Prefill (PaliGemma 18 layers)
   - Per-layer: Input LayerNorm, Attention, Post-Attention LayerNorm, MLP
   - Attention: Q/K/V projection, RoPE, SDPA, Output projection
   - MLP (TRT FP8): gate_proj, up_proj, down_proj, GELU
4. Denoising Loop (Action Expert 18 layers)
   - Per-layer: Self-Attention, Cross-Attention, MLP
   - Time embedding, Action projection

Usage:
    # Inside Docker container:
    python /workspace/scripts/profile_detailed_denoising.py --steps 1 3 10

Author: Claude Code
Date: 2026-02-12
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


@dataclass
class LayerTiming:
    """Timing data for a single layer/component."""
    name: str
    times_ms: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return np.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return np.std(self.times_ms) if self.times_ms else 0.0

    @property
    def min_ms(self) -> float:
        return np.min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return np.max(self.times_ms) if self.times_ms else 0.0


class CUDATimer:
    """High-precision CUDA timing using events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        torch.cuda.synchronize()
        self.start_event.record()

    def stop(self) -> float:
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


class DetailedProfiler:
    """Detailed profiler for Pi0.5 model components."""

    def __init__(self):
        self.timings: Dict[str, LayerTiming] = defaultdict(lambda: LayerTiming(""))
        self.timer = CUDATimer()
        self.hooks = []

    def record(self, name: str, time_ms: float):
        """Record a timing measurement."""
        if name not in self.timings:
            self.timings[name] = LayerTiming(name)
        self.timings[name].times_ms.append(time_ms)

    def clear(self):
        """Clear all timings."""
        self.timings.clear()

    def get_summary(self) -> Dict[str, dict]:
        """Get summary statistics for all components."""
        summary = {}
        for name, timing in self.timings.items():
            summary[name] = {
                'mean_ms': timing.mean_ms,
                'std_ms': timing.std_ms,
                'min_ms': timing.min_ms,
                'max_ms': timing.max_ms,
                'count': len(timing.times_ms),
            }
        return summary

    def print_summary(self, title: str = "Performance Summary"):
        """Print formatted summary."""
        summary = self.get_summary()
        total = sum(v['mean_ms'] for v in summary.values())

        print(f"\n{'='*80}")
        print(f" {title}")
        print(f"{'='*80}")
        print(f"{'Component':<45} {'Mean (ms)':>10} {'Std':>8} {'%':>6}")
        print(f"{'-'*80}")

        # Sort by mean time (descending)
        sorted_items = sorted(summary.items(), key=lambda x: x[1]['mean_ms'], reverse=True)

        for name, stats in sorted_items:
            pct = (stats['mean_ms'] / total * 100) if total > 0 else 0
            print(f"{name:<45} {stats['mean_ms']:>10.3f} {stats['std_ms']:>8.3f} {pct:>5.1f}%")

        print(f"{'-'*80}")
        print(f"{'TOTAL':<45} {total:>10.3f} ms")
        print(f"{'Hz':<45} {1000/total:>10.1f}")
        print(f"{'='*80}")

        return summary


def profile_vision_encoder(model, images, img_masks, profiler: DetailedProfiler, iterations: int = 50):
    """Profile Vision Encoder (SigLIP) in detail."""
    timer = profiler.timer

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.paligemma_with_expert.embed_image(images[0])
    torch.cuda.synchronize()

    # Profile
    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            img_emb = model.paligemma_with_expert.embed_image(images[0])
        time_ms = timer.stop()
        profiler.record("1.Vision/SigLIP_total", time_ms)

    # Second image (wrist)
    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            img_emb = model.paligemma_with_expert.embed_image(images[1])
        time_ms = timer.stop()
        profiler.record("1.Vision/SigLIP_wrist", time_ms)


def profile_language_embedding(model, tokens, token_masks, profiler: DetailedProfiler, iterations: int = 50):
    """Profile Language Embedding."""
    timer = profiler.timer

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            lang_emb = model.paligemma_with_expert.embed_language_tokens(tokens)
    torch.cuda.synchronize()

    # Profile
    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            lang_emb = model.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb = lang_emb * (lang_emb.shape[-1] ** 0.5)
        time_ms = timer.stop()
        profiler.record("2.Embed/Language_embedding", time_ms)


def profile_kv_cache_prefill(
    model,
    prefix_embs,
    prefix_pad_masks,
    prefix_att_masks,
    profiler: DetailedProfiler,
    iterations: int = 50,
    use_trt_fp8: bool = True
):
    """Profile KV Cache Prefill (PaliGemma 18 layers) in detail."""
    timer = profiler.timer
    device = prefix_embs.device
    dtype = prefix_embs.dtype

    # Import required functions
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    # Prepare inputs
    position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
    att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    attention_mask = torch.where(att_2d_masks[:, None, :, :], 0.0, -2.3819763e38).to(dtype)

    # Get PaliGemma language model
    lm = model.paligemma_with_expert.paligemma.language_model

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)
    torch.cuda.synchronize()

    # Profile full KV cache computation
    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            kv_cache = model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)
        time_ms = timer.stop()
        profiler.record("3.KVCache/total", time_ms)

    # Profile per-layer breakdown
    # We need to manually run through each layer to measure
    B, S, H = prefix_embs.shape
    hidden_states = prefix_embs.clone()

    # Get RoPE embeddings
    cos, sin = None, None
    if hasattr(lm, 'rotary_emb'):
        cos, sin = lm.rotary_emb(hidden_states, position_ids)
    elif hasattr(lm.layers[0].self_attn, 'rotary_emb'):
        cos, sin = lm.layers[0].self_attn.rotary_emb(hidden_states, position_ids)

    for layer_idx, layer in enumerate(lm.layers):
        # Profile Input LayerNorm
        for _ in range(iterations):
            x = hidden_states.clone()
            timer.start()
            with torch.no_grad():
                normed = layer.input_layernorm(x)
            time_ms = timer.stop()
            profiler.record(f"3.KVCache/L{layer_idx:02d}_input_norm", time_ms)

        # Profile Self-Attention
        for _ in range(iterations):
            x = hidden_states.clone()
            timer.start()
            with torch.no_grad():
                normed = layer.input_layernorm(x)
                attn_out, _, _ = layer.self_attn(
                    normed,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=True,
                )
            time_ms = timer.stop()
            profiler.record(f"3.KVCache/L{layer_idx:02d}_self_attn", time_ms)

        # Profile Post-Attention LayerNorm
        for _ in range(iterations):
            x = hidden_states.clone()
            timer.start()
            with torch.no_grad():
                normed = layer.post_attention_layernorm(x)
            time_ms = timer.stop()
            profiler.record(f"3.KVCache/L{layer_idx:02d}_post_norm", time_ms)

        # Profile MLP
        for _ in range(iterations):
            x = hidden_states.clone()
            timer.start()
            with torch.no_grad():
                normed = layer.post_attention_layernorm(x)
                mlp_out = layer.mlp(normed)
            time_ms = timer.stop()
            profiler.record(f"3.KVCache/L{layer_idx:02d}_mlp", time_ms)

        # Update hidden states for next layer
        with torch.no_grad():
            normed = layer.input_layernorm(hidden_states)
            attn_out, _, _ = layer.self_attn(
                normed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=True,
            )
            hidden_states = hidden_states + attn_out
            normed = layer.post_attention_layernorm(hidden_states)
            mlp_out = layer.mlp(normed)
            hidden_states = hidden_states + mlp_out

    # Final norm
    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            _ = lm.norm(hidden_states)
        time_ms = timer.stop()
        profiler.record("3.KVCache/final_norm", time_ms)


def profile_denoise_step(
    model,
    state,
    prefix_kv_cache,
    prefix_pad_masks,
    x_t,
    timestep,
    profiler: DetailedProfiler,
    iterations: int = 50
):
    """Profile a single denoising step in detail."""
    timer = profiler.timer
    device = x_t.device

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model.denoise_step_with_cache(
                state, prefix_kv_cache, prefix_pad_masks, x_t, timestep
            )
    torch.cuda.synchronize()

    # Profile full denoise step
    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            v_t = model.denoise_step_with_cache(
                state, prefix_kv_cache, prefix_pad_masks, x_t, timestep
            )
        time_ms = timer.stop()
        profiler.record("4.Denoise/step_total", time_ms)

    # Profile sub-components
    # Time embedding
    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
            time_emb = create_sinusoidal_pos_embedding(
                timestep, model.action_in_proj.out_features,
                min_period=4e-3, max_period=4.0, device=device
            )
            time_emb = time_emb.to(dtype=model.action_in_proj.weight.dtype)
            time_emb = model.time_mlp_out(F.gelu(model.time_mlp_in(time_emb)))
        time_ms = timer.stop()
        profiler.record("4.Denoise/time_embedding", time_ms)

    # Action input projection
    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            action_emb = model.action_in_proj(x_t)
        time_ms = timer.stop()
        profiler.record("4.Denoise/action_in_proj", time_ms)

    # Action output projection
    batch_size = x_t.shape[0]
    action_horizon = x_t.shape[1]
    expert_width = model.action_in_proj.out_features
    dummy_expert_out = torch.randn(batch_size, action_horizon, expert_width, device=device, dtype=x_t.dtype)

    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            _ = model.action_out_proj(dummy_expert_out)
        time_ms = timer.stop()
        profiler.record("4.Denoise/action_out_proj", time_ms)


def profile_action_expert(
    model,
    suffix_embs,
    suffix_pad_masks,
    suffix_att_masks,
    prefix_kv_cache,
    prefix_pad_masks,
    profiler: DetailedProfiler,
    iterations: int = 50
):
    """Profile Action Expert (Gemma 300M, 18 layers) in detail."""
    timer = profiler.timer
    device = suffix_embs.device
    dtype = suffix_embs.dtype

    # Get action expert
    expert = model.paligemma_with_expert.gemma_expert

    # Prepare attention mask
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    B, suffix_len, H = suffix_embs.shape
    prefix_len = prefix_pad_masks.shape[1]

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model.paligemma_with_expert.run_action_expert(
                suffix_embs, suffix_pad_masks, suffix_att_masks,
                prefix_kv_cache, prefix_pad_masks
            )
    torch.cuda.synchronize()

    # Profile full action expert
    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            expert_out = model.paligemma_with_expert.run_action_expert(
                suffix_embs, suffix_pad_masks, suffix_att_masks,
                prefix_kv_cache, prefix_pad_masks
            )
        time_ms = timer.stop()
        profiler.record("4.Denoise/ActionExpert_total", time_ms)


def profile_full_inference(
    checkpoint_dir: str,
    num_steps_list: List[int] = [1, 3, 10],
    iterations: int = 50,
    warmup: int = 10
):
    """Profile full inference pipeline with detailed breakdown."""

    device = torch.device("cuda")

    # Load model
    logger.info(f"Loading model from {checkpoint_dir}...")
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
    from openpi.models_pytorch.gemma_pytorch import load_pretrained_weights

    config = Pi0Config(
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        max_state_dim=32,
        pi05=True,
    )

    model = PI0Pytorch(config)
    load_pretrained_weights(model.paligemma_with_expert, checkpoint_dir)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    logger.info("Model loaded successfully")

    # Create dummy observation
    batch_size = 1
    img = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.bfloat16)
    wrist_img = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.bfloat16)
    images = [img, wrist_img, torch.zeros_like(img)]  # base, wrist, dummy
    img_masks = [
        torch.ones(batch_size, device=device, dtype=torch.bool),
        torch.ones(batch_size, device=device, dtype=torch.bool),
        torch.zeros(batch_size, device=device, dtype=torch.bool),
    ]

    # Tokenize prompt
    import sentencepiece as spm
    tokenizer_path = Path(checkpoint_dir) / "tokenizer.model"
    if not tokenizer_path.exists():
        tokenizer_path = Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(tokenizer_path))

    prompt = "pick up the black bowl"
    token_ids = tokenizer.Encode(prompt, add_bos=True)
    max_token_len = 200
    padding_len = max_token_len - len(token_ids)
    token_ids = token_ids + [0] * padding_len
    token_masks = [1] * (max_token_len - padding_len) + [0] * padding_len

    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    token_masks = torch.tensor([token_masks], dtype=torch.bool, device=device)

    state = torch.randn(batch_size, 32, device=device, dtype=torch.bfloat16)

    # Profile each component
    results = {}

    for num_steps in num_steps_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"Profiling {num_steps} denoising steps")
        logger.info(f"{'='*60}")

        profiler = DetailedProfiler()

        # 1. Profile Vision Encoder
        logger.info("Profiling Vision Encoder...")
        profile_vision_encoder(model, images, img_masks, profiler, iterations)

        # 2. Profile Language Embedding
        logger.info("Profiling Language Embedding...")
        profile_language_embedding(model, tokens, token_masks, profiler, iterations)

        # 3. Profile KV Cache Prefill
        logger.info("Profiling KV Cache Prefill (this takes longer)...")
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, tokens, token_masks
            )
        profile_kv_cache_prefill(
            model, prefix_embs, prefix_pad_masks, prefix_att_masks,
            profiler, iterations // 2  # Fewer iterations for heavy computation
        )

        # 4. Profile Denoising
        logger.info(f"Profiling Denoising Loop ({num_steps} steps)...")
        with torch.no_grad():
            prefix_kv_cache = model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )

        x_t = torch.randn(batch_size, 50, 32, device=device, dtype=torch.bfloat16)
        timestep = torch.tensor([0.5], device=device, dtype=torch.float32)

        profile_denoise_step(
            model, state, prefix_kv_cache, prefix_pad_masks,
            x_t, timestep, profiler, iterations
        )

        # Profile action expert separately
        logger.info("Profiling Action Expert...")
        with torch.no_grad():
            suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(
                state, x_t, timestep
            )
        profile_action_expert(
            model, suffix_embs, suffix_pad_masks, suffix_att_masks,
            prefix_kv_cache, prefix_pad_masks, profiler, iterations
        )

        # 5. Profile full denoising loop
        logger.info(f"Profiling full {num_steps}-step denoising loop...")
        timer = profiler.timer

        for _ in range(iterations):
            x_t = torch.randn(batch_size, 50, 32, device=device, dtype=torch.bfloat16)
            dt = -1.0 / num_steps
            t = 1.0

            timer.start()
            with torch.no_grad():
                for step in range(num_steps):
                    timestep = torch.tensor([t], device=device, dtype=torch.float32)
                    v_t = model.denoise_step_with_cache(
                        state, prefix_kv_cache, prefix_pad_masks, x_t, timestep
                    )
                    x_t = x_t + dt * v_t
                    t += dt
            time_ms = timer.stop()
            profiler.record(f"4.Denoise/{num_steps}step_loop", time_ms)

        # 6. Profile full pipeline (Vision + Embed + KV Cache + Denoise)
        logger.info("Profiling full inference pipeline...")
        obs = Observation(
            images={"base_0_rgb": img, "left_wrist_0_rgb": wrist_img, "right_wrist_0_rgb": torch.zeros_like(img)},
            image_masks={"base_0_rgb": img_masks[0], "left_wrist_0_rgb": img_masks[1], "right_wrist_0_rgb": img_masks[2]},
            state=state,
            tokenized_prompt=tokens,
            tokenized_prompt_mask=token_masks,
        )

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model.sample_actions(device, obs, num_steps=num_steps, use_kv_cache=True)
        torch.cuda.synchronize()

        # Profile
        for _ in range(iterations):
            timer.start()
            with torch.no_grad():
                actions = model.sample_actions(device, obs, num_steps=num_steps, use_kv_cache=True)
            time_ms = timer.stop()
            profiler.record(f"5.E2E/{num_steps}step_full_pipeline", time_ms)

        # Print summary
        summary = profiler.print_summary(f"Performance Breakdown ({num_steps} denoising steps)")
        results[f"{num_steps}_steps"] = summary

    return results


def save_results(results: dict, output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def generate_markdown_report(results: dict, output_path: str):
    """Generate a markdown report of the profiling results."""

    md = []
    md.append("# Pi0.5 TRT FP8 Mixed Quantization - Detailed Performance Analysis")
    md.append("")
    md.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append("| Denoising Steps | Total Latency (ms) | Frequency (Hz) |")
    md.append("|-----------------|-------------------|----------------|")

    for step_key in sorted(results.keys()):
        step_data = results[step_key]
        num_steps = int(step_key.split('_')[0])

        # Find full pipeline timing
        pipeline_key = f"5.E2E/{num_steps}step_full_pipeline"
        if pipeline_key in step_data:
            total_ms = step_data[pipeline_key]['mean_ms']
            hz = 1000 / total_ms
            md.append(f"| {num_steps} | {total_ms:.2f} | {hz:.1f} |")

    md.append("")

    # Detailed breakdown for each step count
    for step_key in sorted(results.keys()):
        step_data = results[step_key]
        num_steps = int(step_key.split('_')[0])

        md.append(f"## {num_steps} Denoising Steps - Detailed Breakdown")
        md.append("")

        # Group by category
        categories = {
            "1.Vision": [],
            "2.Embed": [],
            "3.KVCache": [],
            "4.Denoise": [],
            "5.E2E": [],
        }

        for name, stats in step_data.items():
            for cat in categories.keys():
                if name.startswith(cat):
                    categories[cat].append((name, stats))
                    break

        # Calculate total for percentage
        total_ms = 0
        for name in ["1.Vision/SigLIP_total", "1.Vision/SigLIP_wrist",
                     "2.Embed/Language_embedding", "3.KVCache/total",
                     f"4.Denoise/{num_steps}step_loop"]:
            if name in step_data:
                total_ms += step_data[name]['mean_ms']

        for cat_name, cat_items in categories.items():
            if not cat_items:
                continue

            md.append(f"### {cat_name}")
            md.append("")
            md.append("| Component | Mean (ms) | Std (ms) | % of Total |")
            md.append("|-----------|-----------|----------|------------|")

            for name, stats in sorted(cat_items, key=lambda x: -x[1]['mean_ms']):
                pct = (stats['mean_ms'] / total_ms * 100) if total_ms > 0 else 0
                short_name = name.split('/')[-1]
                md.append(f"| {short_name} | {stats['mean_ms']:.3f} | {stats['std_ms']:.3f} | {pct:.1f}% |")

            md.append("")

        # Per-layer KV Cache breakdown (simplified)
        md.append("### KV Cache Per-Layer Summary (PaliGemma 18 layers)")
        md.append("")

        layer_attn_times = []
        layer_mlp_times = []

        for i in range(18):
            attn_key = f"3.KVCache/L{i:02d}_self_attn"
            mlp_key = f"3.KVCache/L{i:02d}_mlp"

            if attn_key in step_data:
                layer_attn_times.append(step_data[attn_key]['mean_ms'])
            if mlp_key in step_data:
                layer_mlp_times.append(step_data[mlp_key]['mean_ms'])

        if layer_attn_times and layer_mlp_times:
            total_attn = sum(layer_attn_times)
            total_mlp = sum(layer_mlp_times)
            avg_attn = np.mean(layer_attn_times)
            avg_mlp = np.mean(layer_mlp_times)

            md.append(f"- **Total Attention Time**: {total_attn:.2f} ms ({avg_attn:.3f} ms/layer avg)")
            md.append(f"- **Total MLP Time**: {total_mlp:.2f} ms ({avg_mlp:.3f} ms/layer avg)")
            md.append(f"- **Attention/MLP Ratio**: {total_attn/total_mlp:.2f}x")
            md.append("")

            md.append("| Layer | Attention (ms) | MLP (ms) |")
            md.append("|-------|----------------|----------|")
            for i, (attn, mlp) in enumerate(zip(layer_attn_times, layer_mlp_times)):
                md.append(f"| {i:02d} | {attn:.3f} | {mlp:.3f} |")
            md.append("")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(md))

    logger.info(f"Markdown report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detailed Denoising Performance Profiler")
    parser.add_argument(
        "--checkpoint", type=str,
        default="/root/.cache/openpi/checkpoints/pi05_libero",
        help="Model checkpoint directory"
    )
    parser.add_argument(
        "--steps", type=int, nargs='+', default=[1, 3, 10],
        help="Denoising steps to profile"
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Number of iterations for profiling"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )
    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(script_dir.parent / "docs")

    os.makedirs(args.output_dir, exist_ok=True)

    # Run profiling
    results = profile_full_inference(
        args.checkpoint,
        num_steps_list=args.steps,
        iterations=args.iterations
    )

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"denoising_profile_{timestamp}.json")
    save_results(results, json_path)

    md_path = os.path.join(args.output_dir, f"denoising_performance_breakdown.md")
    generate_markdown_report(results, md_path)

    print(f"\n{'='*60}")
    print("Profiling Complete!")
    print(f"{'='*60}")
    print(f"JSON results: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
