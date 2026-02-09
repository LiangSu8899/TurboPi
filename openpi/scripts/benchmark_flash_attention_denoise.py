#!/usr/bin/env python3
"""
Benchmark FlashAttention vs Eager/SDPA for Denoising.

This script benchmarks the denoising component with different attention implementations.
Does not require LIBERO - uses synthetic data for pure latency benchmarking.

Reports:
- Denoise latency breakdown
- FlashAttention speedup
- Full pipeline latency estimation

Usage:
    docker exec turbo_pi_eval python3 /workspace/scripts/benchmark_flash_attention_denoise.py
"""

import os
import sys
import json
import math
import time
import logging
import argparse
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), "src")
sys.path.insert(0, src_dir)

# Check FlashAttention availability
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    logger.info("FlashAttention 2 available")
except ImportError:
    HAS_FLASH_ATTN = False
    logger.warning("FlashAttention not available")

# Constants
SEQ_LEN = 968
NUM_LAYERS = 18


class FlashDenoiseStepWrapper(nn.Module):
    """FlashAttention-optimized denoising step wrapper."""

    def __init__(self, pi0_model, prefix_len: int = SEQ_LEN, use_flash_attn: bool = True):
        super().__init__()
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN

        self.gemma_expert = pi0_model.paligemma_with_expert.gemma_expert.model
        self.paligemma_lm = pi0_model.paligemma_with_expert.paligemma.language_model

        self.action_in_proj = pi0_model.action_in_proj
        self.action_out_proj = pi0_model.action_out_proj
        self.time_mlp_in = pi0_model.time_mlp_in
        self.time_mlp_out = pi0_model.time_mlp_out

        self.action_horizon = pi0_model.config.action_horizon
        self.action_dim = pi0_model.config.action_dim
        self.num_layers = self.gemma_expert.config.num_hidden_layers
        self.hidden_size = self.gemma_expert.config.hidden_size
        self.head_dim = self.gemma_expert.layers[0].self_attn.head_dim
        self.num_heads = self.gemma_expert.layers[0].self_attn.config.num_attention_heads
        self.num_kv_heads = self.gemma_expert.layers[0].self_attn.config.num_key_value_heads
        self.prefix_len = prefix_len

        self.register_buffer('_suffix_pad_masks', torch.ones(1, self.action_horizon, dtype=torch.bool))
        self._attn_mask_cache = None

    def _flash_attention(self, query, key, value, softmax_scale):
        """
        FlashAttention forward with native GQA support.

        When seqlen_q < seqlen_k with causal=True, FlashAttention aligns the causal
        mask to the end, giving us exactly the pattern we need:
        Q[i] can attend to K[0 : prefix_len + i + 1]
        """
        # FlashAttention expects (B, seq, heads, dim) layout
        q = query.transpose(1, 2).contiguous()  # (B, suffix_len, H, D)
        k = key.transpose(1, 2).contiguous()    # (B, total_len, KV_H, D)
        v = value.transpose(1, 2).contiguous()  # (B, total_len, KV_H, D)

        # causal=True with different lengths gives correct pattern:
        # Q[i] sees K[0 : seqlen_k - seqlen_q + i + 1] = K[0 : prefix_len + i + 1]
        out = flash_attn_func(q, k, v, causal=True, softmax_scale=softmax_scale)

        return out.transpose(1, 2).contiguous()

    def _sdpa_attention(self, query, key, value, softmax_scale, batch_size):
        """SDPA fallback with proper masking."""
        num_kv_groups = self.num_heads // self.num_kv_heads
        total_len = key.shape[2]

        key_expanded = key[:, :, None, :, :].expand(
            batch_size, self.num_kv_heads, num_kv_groups, total_len, self.head_dim
        ).reshape(batch_size, self.num_heads, total_len, self.head_dim)
        value_expanded = value[:, :, None, :, :].expand(
            batch_size, self.num_kv_heads, num_kv_groups, total_len, self.head_dim
        ).reshape(batch_size, self.num_heads, total_len, self.head_dim)

        if self._attn_mask_cache is None or self._attn_mask_cache.shape[-1] != total_len:
            suffix_len = self.action_horizon
            attn_mask = torch.zeros(suffix_len, total_len, device=query.device, dtype=query.dtype)
            attn_mask[:, :self.prefix_len] = 0
            suffix_mask = torch.triu(torch.ones(suffix_len, suffix_len, device=query.device), diagonal=1) * -1e9
            attn_mask[:, self.prefix_len:] = suffix_mask
            self._attn_mask_cache = attn_mask[None, None, :, :]

        return F.scaled_dot_product_attention(
            query, key_expanded, value_expanded,
            attn_mask=self._attn_mask_cache,
            scale=softmax_scale,
        )

    def _eager_attention(self, query, key, value, softmax_scale, batch_size):
        """Eager attention (baseline)."""
        from transformers.models.gemma import modeling_gemma

        num_kv_groups = self.num_heads // self.num_kv_heads
        total_len = key.shape[2]

        key_expanded = key[:, :, None, :, :].expand(
            batch_size, self.num_kv_heads, num_kv_groups, total_len, self.head_dim
        ).reshape(batch_size, self.num_heads, total_len, self.head_dim)
        value_expanded = value[:, :, None, :, :].expand(
            batch_size, self.num_kv_heads, num_kv_groups, total_len, self.head_dim
        ).reshape(batch_size, self.num_heads, total_len, self.head_dim)

        # Create mask
        suffix_len = self.action_horizon
        if self._attn_mask_cache is None or self._attn_mask_cache.shape[-1] != total_len:
            attn_mask = torch.zeros(suffix_len, total_len, device=query.device, dtype=query.dtype)
            attn_mask[:, :self.prefix_len] = 0
            suffix_mask = torch.triu(torch.ones(suffix_len, suffix_len, device=query.device), diagonal=1) * -1e9
            attn_mask[:, self.prefix_len:] = suffix_mask
            self._attn_mask_cache = attn_mask[None, None, :, :]

        # Eager attention
        attn_weights = torch.matmul(query, key_expanded.transpose(2, 3)) * softmax_scale
        attn_weights = attn_weights + self._attn_mask_cache
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn_weights, value_expanded)

    def forward(
        self,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        attention_mode: str = "flash",  # "flash", "sdpa", or "eager"
    ) -> torch.Tensor:
        from transformers.models.gemma import modeling_gemma

        batch_size = x_t.shape[0]
        device = x_t.device

        adarms_cond = self._compute_time_embedding(timestep, batch_size, device)

        action_embs = self.action_in_proj(x_t.to(self.action_in_proj.weight.dtype))
        suffix_embs = action_embs.to(torch.bfloat16)

        prefix_offsets = torch.sum(prefix_pad_masks.long(), dim=-1, keepdim=True)
        suffix_position_ids = prefix_offsets + torch.arange(self.action_horizon, device=device)

        hidden_states = suffix_embs

        for layer_idx in range(self.num_layers):
            layer = self.gemma_expert.layers[layer_idx]
            cached_key = prefix_keys[:, layer_idx]
            cached_value = prefix_values[:, layer_idx]

            normed_hidden, gate = layer.input_layernorm(hidden_states, cond=adarms_cond)

            input_shape = normed_hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            query_states = layer.self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)

            dummy = torch.zeros(
                query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                device=device, dtype=query_states.dtype
            )
            cos, sin = self.paligemma_lm.rotary_emb(dummy, suffix_position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            full_key = torch.cat([cached_key, key_states], dim=2)
            full_value = torch.cat([cached_value, value_states], dim=2)

            scaling = layer.self_attn.scaling

            if attention_mode == "flash" and self.use_flash_attn:
                att_output = self._flash_attention(query_states, full_key, full_value, scaling)
            elif attention_mode == "sdpa":
                att_output = self._sdpa_attention(query_states, full_key, full_value, scaling, batch_size)
            else:  # eager
                att_output = self._eager_attention(query_states, full_key, full_value, scaling, batch_size)

            att_output = att_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)

            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output)

            out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gate)
            after_first_residual = out_emb.clone()

            out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond)
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb)

            hidden_states = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)

        hidden_states, _ = self.gemma_expert.norm(hidden_states, cond=adarms_cond)
        suffix_out = hidden_states.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    def _compute_time_embedding(self, timestep, batch_size, device):
        half_dim = self.action_in_proj.out_features // 2
        fraction = torch.linspace(0.0, 1.0, half_dim, dtype=torch.float64, device=device)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * timestep[:, None].to(torch.float64)
        time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).float()
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        return F.silu(time_emb)


def benchmark_denoise_attention(checkpoint_dir: str, num_steps: int = 10, num_iterations: int = 50):
    """Benchmark denoising with different attention implementations."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from safetensors.torch import load_file

    logger.info("=" * 60)
    logger.info("Benchmarking Denoising Attention Implementations")
    logger.info(f"Steps: {num_steps}, Iterations: {num_iterations}")
    logger.info("=" * 60)

    device = "cuda"
    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()

    # Load model
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
        pi0_config = Pi0Config(
            action_dim=model_config.get("action_dim", 32),
            action_horizon=model_config.get("action_horizon", 50),
            max_token_len=model_config.get("max_token_len", 200),
            max_state_dim=model_config.get("max_state_dim", 32),
        )
    else:
        pi0_config = Pi0Config(action_dim=32, action_horizon=50, max_state_dim=32)

    model = PI0Pytorch(pi0_config)
    weights_path = checkpoint_path / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    logger.info(f"Loaded model from {checkpoint_path}")

    # Create wrapper
    wrapper = FlashDenoiseStepWrapper(model, prefix_len=SEQ_LEN, use_flash_attn=True)
    wrapper = wrapper.to(device).eval()

    # Create dummy KV cache
    num_kv_heads = 1
    head_dim = wrapper.head_dim
    prefix_keys = torch.randn(1, NUM_LAYERS, num_kv_heads, SEQ_LEN, head_dim,
                             device=device, dtype=torch.bfloat16)
    prefix_values = torch.randn(1, NUM_LAYERS, num_kv_heads, SEQ_LEN, head_dim,
                               device=device, dtype=torch.bfloat16)
    prefix_pad_masks = torch.ones(1, SEQ_LEN, device=device, dtype=torch.bool)

    def run_denoise_loop(attention_mode: str, x_t_init: torch.Tensor):
        """Run denoising loop with specified attention mode."""
        x_t = x_t_init.clone()
        dt = torch.tensor(-1.0 / num_steps, device=device, dtype=torch.float32)
        time_val = 1.0

        for step in range(num_steps):
            timestep = torch.tensor([time_val], device=device, dtype=torch.float32)
            v_t = wrapper(prefix_keys, prefix_values, prefix_pad_masks, x_t, timestep,
                         attention_mode=attention_mode)
            x_t = x_t + dt * v_t
            time_val += dt.item()

        return x_t

    # Warmup
    logger.info("Warming up...")
    action_horizon = wrapper.action_horizon
    action_dim = wrapper.action_dim
    x_t_init = torch.randn(1, action_horizon, action_dim, device=device, dtype=torch.bfloat16)
    for mode in ["eager", "sdpa", "flash"]:
        for _ in range(3):
            _ = run_denoise_loop(mode, x_t_init)
    torch.cuda.synchronize()

    results = {}

    # Benchmark each mode
    for mode in ["eager", "sdpa", "flash"]:
        if mode == "flash" and not HAS_FLASH_ATTN:
            logger.info(f"Skipping {mode} (not available)")
            continue

        logger.info(f"Benchmarking {mode}...")
        times = []

        for _ in range(num_iterations):
            x_t_init = torch.randn(1, action_horizon, action_dim, device=device, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = run_denoise_loop(mode, x_t_init)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        results[mode] = {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "per_step_ms": np.mean(times) / num_steps,
        }

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    print(f"\n{'Mode':<10} {'Total (ms)':<15} {'Per Step (ms)':<15} {'Speedup':<10}")
    print("-" * 55)

    baseline = results.get("eager", {}).get("mean_ms", 1)
    for mode, data in results.items():
        speedup = baseline / data["mean_ms"]
        print(f"{mode:<10} {data['mean_ms']:.2f} ± {data['std_ms']:.2f}{'':<5} "
              f"{data['per_step_ms']:.2f}{'':<10} {speedup:.2f}x")

    # Precision check
    logger.info("\n" + "=" * 60)
    logger.info("PRECISION CHECK")
    logger.info("=" * 60)

    torch.manual_seed(42)
    x_t_fixed = torch.randn(1, action_horizon, action_dim, device=device, dtype=torch.bfloat16)

    outputs = {}
    for mode in results.keys():
        outputs[mode] = run_denoise_loop(mode, x_t_fixed.clone())

    if "eager" in outputs:
        baseline_out = outputs["eager"].float()
        for mode, out in outputs.items():
            if mode == "eager":
                continue
            out_float = out.float()
            max_diff = (baseline_out - out_float).abs().max().item()
            cos_sim = F.cosine_similarity(baseline_out.flatten(), out_float.flatten(), dim=0).item()
            status = "✅" if max_diff < 0.1 and cos_sim > 0.99 else "⚠️"
            logger.info(f"{mode} vs eager: max_diff={max_diff:.4e}, cos_sim={cos_sim:.6f} {status}")

    # Save results
    output_path = "/workspace/flash_attention_benchmark.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default=os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    benchmark_denoise_attention(args.checkpoint, args.steps, args.iterations)


if __name__ == "__main__":
    main()
