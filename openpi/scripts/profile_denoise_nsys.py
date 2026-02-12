#!/usr/bin/env python3
"""
Denoise Module Deep Profiling with NVTX Markers.

Purpose: 精密诊断 Denoise 模块的 100ms 延迟来源
- Kernel Launch Overhead (CPU-GPU Gap)
- Memory Bandwidth Bottleneck (HBM Traffic)
- Stream Synchronization Issues

Usage:
    # Step 1: 运行带 NVTX 标记的脚本
    nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas \
        --gpu-metrics-device=all \
        --output=denoise_profile \
        python scripts/profile_denoise_nsys.py

    # Step 2: 导出为 sqlite 并分析
    nsys export --type=sqlite denoise_profile.nsys-rep
    python scripts/analyze_nsys_gaps.py denoise_profile.sqlite

Author: Turbo-Pi Team
Date: 2026-02-12
"""

import sys
import os
import time
import math
import pathlib
import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup paths
script_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

# Disable cuDNN for Jetson compatibility
torch.backends.cudnn.enabled = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# NVTX Utilities
# ============================================================================

def nvtx_range(name: str):
    """Context manager for NVTX range."""
    class NVTXRange:
        def __enter__(self):
            torch.cuda.nvtx.range_push(name)
            return self
        def __exit__(self, *args):
            torch.cuda.nvtx.range_pop()
    return NVTXRange()


def nvtx_mark(name: str):
    """NVTX instant marker."""
    torch.cuda.nvtx.mark(name)


# ============================================================================
# NVTX-Instrumented Denoise Step Wrapper
# ============================================================================

class NVTXDenoiseStepWrapper(nn.Module):
    """
    DenoiseStepWrapper with NVTX instrumentation for fine-grained profiling.

    NVTX Markers:
    - "Step_N": Each denoising step
    - "Time_Embed": Time embedding computation
    - "Action_Proj_In": Action input projection
    - "Layer_N": Each transformer layer
    - "Layer_N/Attn": Attention within layer
    - "Layer_N/MLP": MLP within layer
    - "Action_Proj_Out": Final output projection
    """

    def __init__(self, pi0_model, prefix_len: int = 1037):
        super().__init__()
        self.pi0_model = pi0_model
        self.prefix_len = prefix_len

        # Extract components
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

        self.register_buffer('_suffix_pad_masks', torch.ones(1, self.action_horizon, dtype=torch.bool))

    def forward(
        self,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        step_idx: int = 0,  # For NVTX naming
    ) -> torch.Tensor:
        from transformers.models.gemma import modeling_gemma

        batch_size = x_t.shape[0]
        device = x_t.device

        # ================================================================
        # Time Embedding with NVTX
        # ================================================================
        with nvtx_range(f"Step_{step_idx}/Time_Embed"):
            adarms_cond = self._compute_time_embedding(timestep, batch_size, device)

        # ================================================================
        # Action Input Projection with NVTX
        # ================================================================
        with nvtx_range(f"Step_{step_idx}/Action_Proj_In"):
            action_embs = self.action_in_proj(x_t.to(self.action_in_proj.weight.dtype))
            suffix_embs = action_embs.to(torch.bfloat16)

        # ================================================================
        # Attention Mask Preparation
        # ================================================================
        with nvtx_range(f"Step_{step_idx}/Mask_Prep"):
            suffix_pad_masks = self._suffix_pad_masks.expand(batch_size, -1)
            suffix_att_masks = torch.zeros(batch_size, self.action_horizon, device=device, dtype=torch.bfloat16)
            suffix_att_masks[:, 0] = 1.0

            suffix_att_2d = self._make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            suffix_to_prefix = prefix_pad_masks[:, None, :].expand(batch_size, self.action_horizon, -1)
            full_att_masks = torch.cat([suffix_to_prefix, suffix_att_2d], dim=2)
            full_att_masks_4d = self._prepare_4d_mask(full_att_masks)

            prefix_offsets = torch.sum(prefix_pad_masks.long(), dim=-1, keepdim=True)
            suffix_position_ids = prefix_offsets + torch.arange(self.action_horizon, device=device)

        hidden_states = suffix_embs

        # ================================================================
        # Transformer Layers Loop with Fine-Grained NVTX
        # ================================================================
        for layer_idx in range(self.num_layers):
            with nvtx_range(f"Step_{step_idx}/Layer_{layer_idx}"):
                layer = self.gemma_expert.layers[layer_idx]
                cached_key = prefix_keys[:, layer_idx]
                cached_value = prefix_values[:, layer_idx]

                # LayerNorm + AdaRMS
                with nvtx_range(f"Step_{step_idx}/Layer_{layer_idx}/LN"):
                    normed_hidden, gate = layer.input_layernorm(hidden_states, cond=adarms_cond)

                # Q, K, V Projections
                with nvtx_range(f"Step_{step_idx}/Layer_{layer_idx}/QKV_Proj"):
                    input_shape = normed_hidden.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                    query_states = layer.self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
                    key_states = layer.self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
                    value_states = layer.self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)

                # RoPE
                with nvtx_range(f"Step_{step_idx}/Layer_{layer_idx}/RoPE"):
                    dummy = torch.zeros(
                        query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                        device=device, dtype=query_states.dtype
                    )
                    cos, sin = self.paligemma_lm.rotary_emb(dummy, suffix_position_ids)
                    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                        query_states, key_states, cos, sin, unsqueeze_dim=1
                    )

                # KV Cache Concat
                with nvtx_range(f"Step_{step_idx}/Layer_{layer_idx}/KV_Concat"):
                    full_key = torch.cat([cached_key, key_states], dim=2)
                    full_value = torch.cat([cached_value, value_states], dim=2)

                # Attention Compute
                with nvtx_range(f"Step_{step_idx}/Layer_{layer_idx}/Attn"):
                    scaling = layer.self_attn.scaling
                    att_output, _ = modeling_gemma.eager_attention_forward(
                        layer.self_attn, query_states, full_key, full_value,
                        full_att_masks_4d, scaling
                    )

                # Output Projection
                with nvtx_range(f"Step_{step_idx}/Layer_{layer_idx}/O_Proj"):
                    num_heads = layer.self_attn.config.num_attention_heads
                    att_output = att_output.reshape(batch_size, -1, num_heads * layer.self_attn.head_dim)

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output)

                # First Residual
                with nvtx_range(f"Step_{step_idx}/Layer_{layer_idx}/Res1"):
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gate)
                    after_first_residual = out_emb.clone()

                # Post-Attention LayerNorm
                with nvtx_range(f"Step_{step_idx}/Layer_{layer_idx}/PostLN"):
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond)

                # MLP (This is the BIG one - likely memory bound)
                with nvtx_range(f"Step_{step_idx}/Layer_{layer_idx}/MLP"):
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)
                    out_emb = layer.mlp(out_emb)

                # Second Residual
                with nvtx_range(f"Step_{step_idx}/Layer_{layer_idx}/Res2"):
                    hidden_states = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)

        # ================================================================
        # Final Output with NVTX
        # ================================================================
        with nvtx_range(f"Step_{step_idx}/Final_Norm"):
            hidden_states, _ = self.gemma_expert.norm(hidden_states, cond=adarms_cond)

        with nvtx_range(f"Step_{step_idx}/Action_Proj_Out"):
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

    def _make_att_2d_masks(self, pad_masks, att_masks):
        cumsum = torch.cumsum(att_masks.float(), dim=1)
        att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
        return att_2d & pad_2d

    def _prepare_4d_mask(self, mask):
        mask_4d = mask[:, None, :, :]
        return torch.where(mask_4d, 0.0, -2.3819763e38).to(torch.bfloat16)


# ============================================================================
# NVTX-Instrumented Denoise Loop
# ============================================================================

class NVTXDenoiseLoop:
    """
    Denoising loop with NVTX instrumentation for profiling.

    Two modes:
    1. Python Loop Mode: Standard for loop (for gap analysis)
    2. CUDA Graph Mode: Captured graph (for production comparison)
    """

    def __init__(self, wrapper: NVTXDenoiseStepWrapper, num_steps: int = 10):
        self.wrapper = wrapper
        self.num_steps = num_steps
        self.dt = -1.0 / num_steps

    def run_python_loop(
        self,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run denoising with Python for loop (NOT CUDA Graph).

        Purpose: Diagnose kernel launch gaps and CPU overhead.
        """
        device = x_t.device
        time_val = 1.0
        dt_tensor = torch.tensor(self.dt, device=device, dtype=torch.float32)

        # Mark overall denoise start
        nvtx_mark("Denoise_Loop_Start")

        with nvtx_range("Denoise_Full_Loop"):
            for step in range(self.num_steps):
                with nvtx_range(f"Denoise_Step_{step}"):
                    # Mark step boundary for gap analysis
                    nvtx_mark(f"Step_{step}_Start")

                    timestep = torch.tensor([time_val], device=device, dtype=torch.float32)

                    # DiT Forward with fine-grained NVTX
                    v_t = self.wrapper(
                        prefix_keys,
                        prefix_values,
                        prefix_pad_masks,
                        x_t,
                        timestep,
                        step_idx=step,
                    )

                    # Step update
                    with nvtx_range(f"Step_{step}/Update"):
                        x_t = x_t + dt_tensor * v_t

                    time_val += self.dt

                    nvtx_mark(f"Step_{step}_End")

        nvtx_mark("Denoise_Loop_End")
        return x_t


# ============================================================================
# Main Profiling Script
# ============================================================================

def load_model_and_create_inputs(checkpoint_dir: str, device: str = "cuda"):
    """Load model and create dummy inputs for profiling."""
    import json
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from safetensors.torch import load_file

    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()
    logger.info(f"Loading model from {checkpoint_path}...")

    # Load config
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
    else:
        model_config = {}

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=model_config.get("action_dim", 32),
        action_horizon=model_config.get("action_horizon", 50),
        max_token_len=model_config.get("tokenizer_max_length", 200),
        pi05=True,
        dtype="bfloat16",
    )

    # Create model
    model = PI0Pytorch(pi0_config)

    # Load weights
    weights_path = checkpoint_path / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device).eval()

    # Create NVTX wrapper
    wrapper = NVTXDenoiseStepWrapper(model, prefix_len=1037)
    wrapper = wrapper.to(device).eval()

    # Create dummy inputs
    batch_size = 1
    prefix_len = 1037
    num_layers = 18
    num_kv_heads = 8
    head_dim = 256

    prefix_keys = torch.randn(
        batch_size, num_layers, num_kv_heads, prefix_len, head_dim,
        device=device, dtype=torch.bfloat16
    )
    prefix_values = torch.randn(
        batch_size, num_layers, num_kv_heads, prefix_len, head_dim,
        device=device, dtype=torch.bfloat16
    )
    prefix_pad_masks = torch.ones(batch_size, prefix_len, device=device, dtype=torch.bool)
    x_t = torch.randn(
        batch_size, model.config.action_horizon, model.config.action_dim,
        device=device, dtype=torch.bfloat16
    )

    return wrapper, prefix_keys, prefix_values, prefix_pad_masks, x_t


def run_profiling(checkpoint_dir: str, num_steps: int = 10, warmup: int = 5, iterations: int = 10):
    """Run profiling with NVTX markers."""
    device = "cuda"

    logger.info("=" * 70)
    logger.info(f"Denoise Profiling: {num_steps} steps, {iterations} iterations")
    logger.info("=" * 70)

    # Load model
    wrapper, prefix_keys, prefix_values, prefix_pad_masks, x_t = \
        load_model_and_create_inputs(checkpoint_dir, device)

    # Create denoise loop
    denoise_loop = NVTXDenoiseLoop(wrapper, num_steps=num_steps)

    # Warmup (without NVTX to avoid polluting the trace)
    logger.info(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        x_t_init = torch.randn_like(x_t)
        _ = denoise_loop.run_python_loop(
            prefix_keys, prefix_values, prefix_pad_masks, x_t_init
        )
    torch.cuda.synchronize()

    # Profile iterations
    logger.info(f"Profiling ({iterations} iterations)...")
    latencies = []

    for i in range(iterations):
        nvtx_mark(f"Iteration_{i}_Start")

        torch.cuda.synchronize()
        start = time.perf_counter()

        x_t_init = torch.randn_like(x_t)

        with nvtx_range(f"Iteration_{i}"):
            result = denoise_loop.run_python_loop(
                prefix_keys, prefix_values, prefix_pad_masks, x_t_init
            )

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        nvtx_mark(f"Iteration_{i}_End")

    # Report
    latencies = np.array(latencies)
    logger.info("=" * 70)
    logger.info("Profiling Results")
    logger.info("=" * 70)
    logger.info(f"  Mean: {latencies.mean():.2f} ms")
    logger.info(f"  Std:  {latencies.std():.2f} ms")
    logger.info(f"  Min:  {latencies.min():.2f} ms")
    logger.info(f"  Max:  {latencies.max():.2f} ms")
    logger.info(f"  Per-step: {latencies.mean() / num_steps:.2f} ms/step")
    logger.info("=" * 70)

    return latencies


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Denoise Module NVTX Profiling")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero"),
        help="Model checkpoint directory"
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of denoising steps")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
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
