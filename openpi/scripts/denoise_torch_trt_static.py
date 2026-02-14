#!/usr/bin/env python3
"""Static Denoise TensorRT via torch_tensorrt (FP8).

This script implements FP8 TensorRT compilation following the SAME approach
as torch_trt_fp8_kv_cache.py, which successfully achieved 2.94x speedup.

Key differences from denoise_torch_trt.py:
1. Uses export_torch_mode() context (CRITICAL for FP8 to work!)
2. Creates simplified static modules (like SimpleMLP in VLM)
3. All shapes are STATIC (no dynamic reshape)
4. 10-step denoise loop unrolled into single graph

Expected performance:
- CUDA Graph BF16: 109 ms
- TRT FP8: ~40 ms (2.7x speedup)

Usage:
    docker exec -it turbo_pi_eval python /workspace/scripts/denoise_torch_trt_static.py \
        --checkpoint_dir /root/.cache/openpi/pytorch_checkpoints/pi05_libero \
        --output_path /workspace/denoise_trt \
        --precision fp8

Author: Turbo-Pi Team
Date: 2026-02-13
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch_tensorrt for direct compilation
try:
    import torch_tensorrt
    HAS_TORCH_TRT = True
except ImportError:
    HAS_TORCH_TRT = False
    print("Warning: torch_tensorrt not available.")

# nvidia-modelopt for FP8 quantization
try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode
    HAS_MODELOPT = True
except ImportError:
    HAS_MODELOPT = False
    print("Warning: nvidia-modelopt not available. FP8 quantization disabled.")


# ============================================================================
# Model Constants (pi0.5 / Gemma 300M Expert)
# ============================================================================
NUM_LAYERS = 18
NUM_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
HIDDEN_SIZE = 1024  # Gemma Expert hidden size
MLP_DIM = 4096      # Gemma Expert MLP dim (intermediate_size)
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10000.0

# Default shapes (STATIC - no dynamic shapes!)
BATCH_SIZE = 1
ACTION_HORIZON = 50
ACTION_DIM = 32
PREFIX_LEN = 968
NUM_STEPS = 10


# ============================================================================
# Static Module Definitions (following VLM's SimpleMLP pattern)
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm with FP32 variance computation (precision protected)."""
    def __init__(self, hidden_size: int, eps: float = RMS_NORM_EPS, cond_dim: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.eps = eps

        # Weight parameter (for non-adaptive or as placeholder)
        self.weight = nn.Parameter(torch.zeros(hidden_size))

        # Dense layer for adaptive normalization
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, hidden_size * 3, bias=True)
        else:
            self.dense = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_dtype = x.dtype
        # Variance in FP32 (CRITICAL for precision!)
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normed = x.float() * torch.rsqrt(variance + self.eps)

        if cond is None or self.dense is None:
            # Regular RMSNorm
            normed = normed * (1.0 + self.weight.float())
            return normed.to(input_dtype), None

        # Adaptive RMSNorm with conditioning
        modulation = self.dense(cond)
        if len(x.shape) == 3:  # [batch, seq, features]
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()

        return normed.to(input_dtype), gate.to(input_dtype)


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm with conditioning (gating).

    Simplified static version without dynamic dispatch.
    """
    def __init__(self, hidden_size: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        # Gate MLP: cond_dim -> 2 * hidden_size (gate + scale)
        self.cond_proj = nn.Linear(HIDDEN_SIZE, 2 * hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_dtype = x.dtype
        # Variance in FP32 (precision protected)
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_norm = x.float() * torch.rsqrt(variance + self.eps)
        x_norm = (x_norm * self.weight.float()).to(input_dtype)

        # Conditioning
        cond_out = self.cond_proj(cond)
        # Static split (no dynamic shape)
        gate = cond_out[..., :x.shape[-1]]
        scale = cond_out[..., x.shape[-1]:]

        # Apply adaptive scaling
        x_out = x_norm * (1.0 + scale.unsqueeze(1))

        return x_out, gate.unsqueeze(1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with BF16-compatible precision.

    IMPORTANT: We must match the original model's inv_freq precision (BF16)
    to get identical RoPE embeddings. The original model computes inv_freq
    in BF16 which has less precision than FP32.
    """
    def __init__(self, head_dim: int, base: float = ROPE_THETA):
        super().__init__()
        self.head_dim = head_dim
        # Compute inv_freq in BF16 to match original model's precision
        # Original: inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
        # Then stored in BF16 buffer
        inv_freq_fp32 = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        inv_freq_bf16 = inv_freq_fp32.to(torch.bfloat16)
        # Store as FP32 for computation but with BF16-quantized values
        self.register_buffer("inv_freq", inv_freq_bf16.float(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # All computation in FP32 (precision protected)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # CRITICAL: Disable autocast for RoPE computation
        with torch.autocast(device_type="cuda", enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims."""
    # Static slicing (no dynamic shape)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K."""
    cos = cos.unsqueeze(1)  # (B, 1, S, head_dim)
    sin = sin.unsqueeze(1)  # (B, 1, S, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SimpleMLP(nn.Module):
    """Simple MLP for TRT compilation (following VLM pattern)."""
    def __init__(self, hidden_size: int = HIDDEN_SIZE, mlp_dim: int = MLP_DIM):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, mlp_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x), approximate='tanh')
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class SimpleAttention(nn.Module):
    """Simple attention for TRT compilation (static shapes)."""
    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_heads: int = NUM_HEADS,
                 num_kv_heads: int = NUM_KV_HEADS, head_dim: int = HEAD_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cached_key: torch.Tensor,
        cached_value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with static shapes.

        IMPORTANT: No attention mask is used here to match the original model!
        The original model's denoise_step_with_cache uses SDPA without mask:
            att_output = F.scaled_dot_product_attention(query_states, full_key_states, full_value_states)
        The comment explains: "Since suffix attention mask is ALL TRUE (bidirectional),
        we can skip the mask entirely."

        Args:
            hidden_states: (B, action_horizon, hidden_size)
            cos, sin: (B, action_horizon, head_dim)
            cached_key: (B, num_kv_heads, prefix_len, head_dim)
            cached_value: (B, num_kv_heads, prefix_len, head_dim)
        """
        B, S, _ = hidden_states.shape

        # Q, K, V projections with STATIC reshape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Static reshape (no -1 dynamic)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Concatenate with cached KV
        k = torch.cat([cached_key, k], dim=2)
        v = torch.cat([cached_value, v], dim=2)

        # Expand KV for GQA (static repeat)
        # Note: SDPA can handle GQA via broadcasting, but we expand explicitly
        # for TRT compatibility
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Use SDPA without mask (matching original model behavior!)
        # The original uses: F.scaled_dot_product_attention(query_states, full_key_states, full_value_states)
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Static reshape
        attn_output = attn_output.view(B, S, self.num_heads * self.head_dim)

        return self.o_proj(attn_output)


class SimpleDenoiseLayer(nn.Module):
    """Single denoise transformer layer with adaptive RMSNorm."""
    def __init__(self):
        super().__init__()
        # Adaptive RMSNorm with conditioning from adarms_cond
        self.input_layernorm = RMSNorm(HIDDEN_SIZE, cond_dim=HIDDEN_SIZE)
        self.self_attn = SimpleAttention()
        self.post_attention_layernorm = RMSNorm(HIDDEN_SIZE, cond_dim=HIDDEN_SIZE)
        self.mlp = SimpleMLP()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cached_key: torch.Tensor,
        cached_value: torch.Tensor,
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-attention norm with adaptive conditioning
        normed, gate = self.input_layernorm(hidden_states, cond=adarms_cond)

        # Attention (no mask - matching original model)
        attn_output = self.self_attn(normed, cos, sin, cached_key, cached_value)

        # Gated residual connection
        if gate is not None:
            hidden_states = hidden_states + attn_output * gate
        else:
            hidden_states = hidden_states + attn_output

        # Post-attention norm with adaptive conditioning
        normed, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)

        # MLP
        mlp_output = self.mlp(normed)

        # Gated residual connection
        if gate is not None:
            hidden_states = hidden_states + mlp_output * gate
        else:
            hidden_states = hidden_states + mlp_output

        return hidden_states


class StaticDenoiseStep(nn.Module):
    """Single denoise step with FULLY STATIC shapes.

    All shapes are predetermined at compile time. No dynamic reshape.
    """
    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        action_horizon: int = ACTION_HORIZON,
        action_dim: int = ACTION_DIM,
        prefix_len: int = PREFIX_LEN,
        num_layers: int = NUM_LAYERS,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.prefix_len = prefix_len
        self.num_layers = num_layers

        # Action projection (IMPORTANT: must have bias=True to match checkpoint!)
        self.action_in_proj = nn.Linear(action_dim, HIDDEN_SIZE, bias=True)
        self.action_out_proj = nn.Linear(HIDDEN_SIZE, action_dim, bias=True)

        # Transformer layers
        self.layers = nn.ModuleList([SimpleDenoiseLayer() for _ in range(num_layers)])

        # Final norm (also adaptive)
        self.final_norm = RMSNorm(HIDDEN_SIZE, cond_dim=HIDDEN_SIZE)

        # RoPE
        self.rotary_emb = RotaryEmbedding(HEAD_DIM)

    def forward(
        self,
        x_t: torch.Tensor,              # (B, action_horizon, action_dim)
        suffix_position_ids: torch.Tensor,  # (B, action_horizon)
        adarms_cond: torch.Tensor,      # (B, hidden_size)
        cached_keys: torch.Tensor,      # (num_layers, B, num_kv_heads, prefix_len, head_dim)
        cached_values: torch.Tensor,    # (num_layers, B, num_kv_heads, prefix_len, head_dim)
    ) -> torch.Tensor:
        """Single denoise step forward.

        Note: No attention mask is used to match the original model behavior.
        The original model's denoise_step_with_cache uses SDPA without mask.
        """
        # Embed actions
        hidden_states = self.action_in_proj(x_t)

        # Compute RoPE for suffix positions
        cos, sin = self.rotary_emb(hidden_states, suffix_position_ids)

        # Process through all layers (no mask)
        for i in range(self.num_layers):
            hidden_states = self.layers[i](
                hidden_states,
                cos, sin,
                cached_keys[i], cached_values[i],
                adarms_cond,
            )

        # Final norm (with adaptive conditioning)
        hidden_states, _ = self.final_norm(hidden_states, cond=adarms_cond)

        # Project to action
        return self.action_out_proj(hidden_states)


class StaticDenoiseLoop(nn.Module):
    """10-step denoise loop unrolled into single graph (STATIC).

    This module contains the full 10-step denoising as a single TRT graph,
    eliminating Python loop overhead.
    """
    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        action_horizon: int = ACTION_HORIZON,
        action_dim: int = ACTION_DIM,
        prefix_len: int = PREFIX_LEN,
        num_layers: int = NUM_LAYERS,
        num_steps: int = NUM_STEPS,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.prefix_len = prefix_len
        self.num_layers = num_layers
        self.num_steps = num_steps

        # Single denoise step (shared across all steps)
        self.denoise_step = StaticDenoiseStep(
            batch_size, action_horizon, action_dim, prefix_len, num_layers
        )

        # Flow matching coefficients (pre-computed)
        dt = -1.0 / num_steps
        self.register_buffer("dt", torch.tensor(dt))

    def forward(
        self,
        noise: torch.Tensor,                # (B, action_horizon, action_dim)
        suffix_position_ids: torch.Tensor,  # (B, action_horizon)
        adarms_conds: torch.Tensor,         # (num_steps, B, hidden_size)
        cached_keys: torch.Tensor,          # (num_layers, B, num_kv_heads, prefix_len, head_dim)
        cached_values: torch.Tensor,        # (num_layers, B, num_kv_heads, prefix_len, head_dim)
    ) -> torch.Tensor:
        """Full denoise loop (unrolled, static).

        Note: No attention mask is used to match the original model behavior.
        """
        x_t = noise
        dt = self.dt

        # Unrolled 10-step loop
        for step in range(self.num_steps):
            # Get pre-computed adarms condition for this step
            adarms_cond = adarms_conds[step]

            # Single denoise step (no mask)
            v_t = self.denoise_step(
                x_t, suffix_position_ids, adarms_cond,
                cached_keys, cached_values,
            )

            # Flow matching update: x_{t-1} = x_t + v_t * dt
            x_t = x_t + v_t * dt

        return x_t


def load_weights_from_checkpoint(model: nn.Module, checkpoint_dir: str, device: str = "cuda"):
    """Load weights from pi0.5 checkpoint into static model.

    Note: Use /root/.cache/openpi/checkpoints/pi05_libero (PyTorch format),
    NOT /root/.cache/openpi/pytorch_checkpoints/pi05_libero (JAX format).
    """
    from safetensors import safe_open

    checkpoint_path = Path(checkpoint_dir)
    weights_path = checkpoint_path / "model.safetensors"

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    print(f"Loading weights from: {weights_path}")
    state_dict = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    dtype = torch.float16  # TRT uses FP16/FP8

    # Get denoise_step reference (handles both StaticDenoiseStep and StaticDenoiseLoop)
    if hasattr(model, 'denoise_step'):
        denoise_step = model.denoise_step
    else:
        denoise_step = model

    # Load action projections (weight and bias)
    denoise_step.action_in_proj.weight.data = state_dict["action_in_proj.weight"].to(dtype).to(device)
    denoise_step.action_in_proj.bias.data = state_dict["action_in_proj.bias"].to(dtype).to(device)
    denoise_step.action_out_proj.weight.data = state_dict["action_out_proj.weight"].to(dtype).to(device)
    denoise_step.action_out_proj.bias.data = state_dict["action_out_proj.bias"].to(dtype).to(device)

    # Load transformer layers from gemma_expert
    # Key format: paligemma_with_expert.gemma_expert.model.layers.{i}.{component}
    for i, layer in enumerate(denoise_step.layers):
        prefix = f"paligemma_with_expert.gemma_expert.model.layers.{i}"

        # Attention
        layer.self_attn.q_proj.weight.data = state_dict[f"{prefix}.self_attn.q_proj.weight"].to(dtype).to(device)
        layer.self_attn.k_proj.weight.data = state_dict[f"{prefix}.self_attn.k_proj.weight"].to(dtype).to(device)
        layer.self_attn.v_proj.weight.data = state_dict[f"{prefix}.self_attn.v_proj.weight"].to(dtype).to(device)
        layer.self_attn.o_proj.weight.data = state_dict[f"{prefix}.self_attn.o_proj.weight"].to(dtype).to(device)

        # MLP
        layer.mlp.gate_proj.weight.data = state_dict[f"{prefix}.mlp.gate_proj.weight"].to(dtype).to(device)
        layer.mlp.up_proj.weight.data = state_dict[f"{prefix}.mlp.up_proj.weight"].to(dtype).to(device)
        layer.mlp.down_proj.weight.data = state_dict[f"{prefix}.mlp.down_proj.weight"].to(dtype).to(device)

        # Adaptive RMSNorm dense layers (scale, shift, gate projection)
        layer.input_layernorm.dense.weight.data = state_dict[f"{prefix}.input_layernorm.dense.weight"].to(dtype).to(device)
        layer.input_layernorm.dense.bias.data = state_dict[f"{prefix}.input_layernorm.dense.bias"].to(dtype).to(device)

        layer.post_attention_layernorm.dense.weight.data = state_dict[f"{prefix}.post_attention_layernorm.dense.weight"].to(dtype).to(device)
        layer.post_attention_layernorm.dense.bias.data = state_dict[f"{prefix}.post_attention_layernorm.dense.bias"].to(dtype).to(device)

    # Final norm (also adaptive)
    denoise_step.final_norm.dense.weight.data = state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"].to(dtype).to(device)
    denoise_step.final_norm.dense.bias.data = state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"].to(dtype).to(device)

    print(f"Weights loaded successfully ({len(denoise_step.layers)} layers)")


def compile_trt_fp8(
    module: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    device: str = "cuda",
) -> nn.Module:
    """Compile module with TRT FP8 following VLM's successful approach.

    CRITICAL: Uses export_torch_mode() context which is the key to success!
    """
    if not HAS_TORCH_TRT:
        raise RuntimeError("torch_tensorrt not available")
    if not HAS_MODELOPT:
        raise RuntimeError("modelopt not available for FP8 quantization")

    print("="*60)
    print("Compiling with Torch-TRT FP8")
    print("="*60)

    # Step 1: Calibration function
    def calibrate(m):
        m.eval()
        with torch.no_grad():
            for _ in range(10):
                m(*example_inputs)

    # Step 2: FP8 quantization with modelopt
    print("\n[1/3] Applying FP8 quantization...")
    quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
    module_fp8 = mtq.quantize(module, quant_cfg, forward_loop=calibrate)

    print("\nQuantization summary:")
    mtq.print_quant_summary(module_fp8)

    # Step 3: Compile with TRT using export_torch_mode
    # THIS IS THE KEY DIFFERENCE FROM THE FAILED APPROACH!
    print("\n[2/3] Compiling with torch_tensorrt...")
    print("  Using export_torch_mode() context (critical for FP8!)")

    with export_torch_mode():
        trt_module = torch_tensorrt.compile(
            module_fp8,
            inputs=example_inputs,
            enabled_precisions={torch.float16, torch.float8_e4m3fn},
            workspace_size=8 << 30,  # 8GB workspace
        )

    print("\n[3/3] Compilation complete!")
    return trt_module


def compile_trt_fp16(
    module: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    device: str = "cuda",
) -> nn.Module:
    """Compile module with TRT FP16 (fallback if FP8 fails)."""
    if not HAS_TORCH_TRT:
        raise RuntimeError("torch_tensorrt not available")

    print("="*60)
    print("Compiling with Torch-TRT FP16")
    print("="*60)

    # Build input specs
    input_specs = []
    for inp in example_inputs:
        spec = torch_tensorrt.Input(
            min_shape=list(inp.shape),
            opt_shape=list(inp.shape),
            max_shape=list(inp.shape),
            dtype=inp.dtype,
        )
        input_specs.append(spec)

    trt_module = torch_tensorrt.compile(
        module,
        inputs=input_specs,
        enabled_precisions={torch.float16, torch.float32},
        truncate_long_and_double=True,
        workspace_size=8 << 30,
    )

    print("Compilation complete!")
    return trt_module


def benchmark_module(module: nn.Module, inputs: Tuple, warmup: int = 20, iterations: int = 100):
    """Benchmark module inference time."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = module(*inputs)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            _ = module(*inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000  # ms


def main():
    parser = argparse.ArgumentParser(description="Export Static Denoise to TRT FP8")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to PyTorch checkpoint directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for compiled model")
    parser.add_argument("--precision", type=str, default="fp8",
                        choices=["fp16", "fp8"],
                        help="Quantization precision")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prefix_len", type=int, default=968)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--action_horizon", type=int, default=50)
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark after compilation")
    parser.add_argument("--compile_loop", action="store_true",
                        help="Compile full 10-step loop (default: single step)")

    args = parser.parse_args()

    device = "cuda"

    print("="*60)
    print("Static Denoise TRT Compilation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Precision: {args.precision}")
    print(f"Batch size: {args.batch_size}")
    print(f"Prefix len: {args.prefix_len}")
    print(f"Action horizon: {args.action_horizon}")
    print(f"Num steps: {args.num_steps}")
    print(f"Compile loop: {args.compile_loop}")
    print("="*60)

    # Create model
    if args.compile_loop:
        print("\nCreating StaticDenoiseLoop (10-step unrolled)...")
        model = StaticDenoiseLoop(
            batch_size=args.batch_size,
            action_horizon=args.action_horizon,
            action_dim=ACTION_DIM,
            prefix_len=args.prefix_len,
            num_layers=NUM_LAYERS,
            num_steps=args.num_steps,
        )
    else:
        print("\nCreating StaticDenoiseStep (single step)...")
        model = StaticDenoiseStep(
            batch_size=args.batch_size,
            action_horizon=args.action_horizon,
            action_dim=ACTION_DIM,
            prefix_len=args.prefix_len,
            num_layers=NUM_LAYERS,
        )

    model = model.to(device).half()
    model.eval()

    # Load weights if checkpoint exists
    checkpoint_path = Path(args.checkpoint_dir)
    if (checkpoint_path / "model.safetensors").exists():
        try:
            load_weights_from_checkpoint(model, args.checkpoint_dir, device)
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
            print("Using random weights for compilation test")
    else:
        print("No checkpoint found, using random weights for compilation test")

    # Create example inputs (STATIC shapes, FP16)
    dtype = torch.float16

    # Create full attention mask (zeros = all valid for compilation test)
    # Shape: (B, 1, 1, prefix_len + action_horizon) - full size for attention
    # First prefix_len positions can be padding (-inf), suffix positions are always 0
    full_seq_len = args.prefix_len + args.action_horizon
    attn_mask = torch.zeros(args.batch_size, 1, 1, full_seq_len, dtype=dtype, device=device)

    if args.compile_loop:
        # Full loop inputs (6 arguments including mask)
        example_inputs = (
            torch.randn(args.batch_size, args.action_horizon, ACTION_DIM, dtype=dtype, device=device),
            torch.arange(args.prefix_len, args.prefix_len + args.action_horizon, dtype=torch.long, device=device).unsqueeze(0).expand(args.batch_size, -1),
            torch.randn(args.num_steps, args.batch_size, HIDDEN_SIZE, dtype=dtype, device=device),
            torch.randn(NUM_LAYERS, args.batch_size, NUM_KV_HEADS, args.prefix_len, HEAD_DIM, dtype=dtype, device=device),
            torch.randn(NUM_LAYERS, args.batch_size, NUM_KV_HEADS, args.prefix_len, HEAD_DIM, dtype=dtype, device=device),
            attn_mask,
        )
    else:
        # Single step inputs (6 arguments including mask)
        example_inputs = (
            torch.randn(args.batch_size, args.action_horizon, ACTION_DIM, dtype=dtype, device=device),
            torch.arange(args.prefix_len, args.prefix_len + args.action_horizon, dtype=torch.long, device=device).unsqueeze(0).expand(args.batch_size, -1),
            torch.randn(args.batch_size, HIDDEN_SIZE, dtype=dtype, device=device),
            torch.randn(NUM_LAYERS, args.batch_size, NUM_KV_HEADS, args.prefix_len, HEAD_DIM, dtype=dtype, device=device),
            torch.randn(NUM_LAYERS, args.batch_size, NUM_KV_HEADS, args.prefix_len, HEAD_DIM, dtype=dtype, device=device),
            attn_mask,
        )

    # Verify original module works and SAVE OUTPUT before compilation
    # (FP8 quantization modifies the model in-place!)
    print("\nTesting original module...")
    with torch.no_grad():
        orig_output = model(*example_inputs).clone()  # Save BEFORE compilation
    print(f"  Output shape: {orig_output.shape}")
    print(f"  Output mean: {orig_output.mean():.4f}, nan: {torch.isnan(orig_output).any()}")

    if args.benchmark:
        orig_time = benchmark_module(model, example_inputs)
        print(f"  Original time: {orig_time:.2f} ms")

    # Compile with TRT
    try:
        if args.precision == "fp8":
            compiled = compile_trt_fp8(model, example_inputs, device)
        else:
            compiled = compile_trt_fp16(model, example_inputs, device)

        # Verify compiled module
        print("\nTesting compiled module...")
        with torch.no_grad():
            compiled_output = compiled(*example_inputs)
        print(f"  Output shape: {compiled_output.shape}")
        print(f"  Output mean: {compiled_output.mean():.4f}, nan: {torch.isnan(compiled_output).any()}")

        # Check accuracy against saved original output
        diff = torch.abs(compiled_output.float() - orig_output.float()).max().item()
        cos_sim = F.cosine_similarity(
            compiled_output.float().flatten().unsqueeze(0),
            orig_output.float().flatten().unsqueeze(0)
        ).item()
        print(f"  Max diff: {diff:.6f}")
        print(f"  Cosine similarity: {cos_sim:.6f}")

        if args.benchmark:
            compiled_time = benchmark_module(compiled, example_inputs)
            print(f"  Compiled time: {compiled_time:.2f} ms")
            speedup = orig_time / compiled_time
            print(f"  Speedup: {speedup:.2f}x")

            # Project full pipeline improvement
            if args.compile_loop:
                print(f"\n  10-step projection:")
                print(f"    Original: {orig_time:.2f} ms")
                print(f"    TRT: {compiled_time:.2f} ms")
            else:
                print(f"\n  10-step projection:")
                print(f"    Original: {orig_time * 10:.2f} ms")
                print(f"    TRT: {compiled_time * 10:.2f} ms")

        # Save compiled model
        output_dir = Path(args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = "loop" if args.compile_loop else "step"
        save_path = output_dir / f"denoise_{suffix}_{args.precision}.pt"

        try:
            torch.save(compiled, str(save_path))
            print(f"\nCompiled model saved to: {save_path}")
        except Exception as e:
            print(f"\nWarning: Could not save model: {e}")

        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)

    except Exception as e:
        print(f"\nCompilation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
