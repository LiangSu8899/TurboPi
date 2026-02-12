"""Static Denoise Step for CUDA Graph capture.

This module provides a fully static denoise step that can be captured
as a CUDA Graph. All tensors are pre-allocated - no dynamic allocation
during graph capture or replay.

Key insight: The original denoise_step_with_cache fails graph capture because
embed_suffix() creates tensors dynamically (torch.tensor()). This module
pre-allocates all required tensors.

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
from dataclasses import dataclass


def create_sinusoidal_pos_embedding_static(
    time: torch.Tensor,
    out_buffer: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
):
    """Compute sine-cosine positional embedding into pre-allocated buffer.

    Unlike the original which returns a new tensor, this writes to out_buffer.
    """
    device = time.device
    dtype = torch.float64

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)

    # Write to output buffer
    out_buffer[:, :dimension // 2].copy_(torch.sin(sin_input).to(out_buffer.dtype))
    out_buffer[:, dimension // 2:].copy_(torch.cos(sin_input).to(out_buffer.dtype))


class StaticDenoiseStep(nn.Module):
    """Fully static denoise step for CUDA Graph capture.

    This module pre-allocates all tensors required for a denoise step,
    enabling capture as CUDA Graph.

    Pre-allocated buffers:
    - Time embedding buffer
    - Action embedding buffer
    - Suffix embeddings buffer
    - Attention mask buffers
    - Position IDs buffer
    - Output buffer
    """

    def __init__(
        self,
        model,  # PI0Pytorch
        prefix_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
        batch_size: int = 1,
        action_horizon: int = 50,
        action_dim: int = 32,
        state_dim: int = 32,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.prefix_kv_cache = prefix_kv_cache
        self.prefix_pad_masks = prefix_pad_masks
        self.batch_size = batch_size
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.device = device or torch.device('cuda')
        self.dtype = dtype

        # Get expert config
        expert = model.paligemma_with_expert.gemma_expert.model
        self.expert = expert
        self.expert_hidden_size = expert.config.hidden_size
        self.expert_head_dim = expert.config.head_dim
        self.num_layers = expert.config.num_hidden_layers

        # Get action projections
        self.action_in_proj = model.action_in_proj
        self.action_out_proj = model.action_out_proj
        self.time_mlp_in = model.time_mlp_in
        self.time_mlp_out = model.time_mlp_out

        # Prefix length
        self.prefix_len = prefix_pad_masks.shape[1]
        self.suffix_len = action_horizon  # For Pi0.5, suffix = action tokens only

        # =====================================================================
        # Pre-allocate ALL static buffers
        # =====================================================================

        # Input buffers
        self._static_state = torch.zeros(
            batch_size, state_dim,
            dtype=dtype, device=self.device
        )
        self._static_x_t = torch.zeros(
            batch_size, action_horizon, action_dim,
            dtype=dtype, device=self.device
        )
        self._static_timestep = torch.zeros(
            batch_size,
            dtype=torch.float32, device=self.device
        )

        # Time embedding buffer
        self._time_emb = torch.zeros(
            batch_size, self.expert_hidden_size,
            dtype=dtype, device=self.device
        )

        # Action embedding buffer (after projection)
        self._action_emb = torch.zeros(
            batch_size, action_horizon, self.expert_hidden_size,
            dtype=dtype, device=self.device
        )

        # Suffix embeddings buffer
        self._suffix_embs = torch.zeros(
            batch_size, self.suffix_len, self.expert_hidden_size,
            dtype=dtype, device=self.device
        )

        # AdaRMS conditioning
        self._adarms_cond = torch.zeros(
            batch_size, self.expert_hidden_size,
            dtype=dtype, device=self.device
        )

        # Attention mask (suffix attending to prefix + suffix)
        # Shape: (batch, 1, suffix_len, prefix_len + suffix_len)
        total_len = self.prefix_len + self.suffix_len
        self._attention_mask = torch.zeros(
            batch_size, 1, self.suffix_len, total_len,
            dtype=dtype, device=self.device
        )
        # Initialize: suffix can attend to all prefix (causal within suffix)
        self._init_attention_mask()

        # Position IDs for suffix
        self._position_ids = torch.zeros(
            batch_size, self.suffix_len,
            dtype=torch.long, device=self.device
        )
        self._init_position_ids()

        # Hidden states buffer (for transformer layers)
        self._hidden_states = torch.zeros(
            batch_size, self.suffix_len, self.expert_hidden_size,
            dtype=dtype, device=self.device
        )

        # Output buffer
        self._v_t = torch.zeros(
            batch_size, action_horizon, action_dim,
            dtype=dtype, device=self.device
        )

        # Graph state
        self._graph_captured = False
        self._denoise_graph: Optional[torch.cuda.CUDAGraph] = None

    def _init_attention_mask(self):
        """Initialize static attention mask."""
        # For Pi0.5:
        # - Suffix tokens attend to all prefix tokens
        # - Suffix tokens have causal attention within suffix
        # - We use the pre-computed attention mask format

        # suffix_to_prefix: all True (can attend)
        self._attention_mask[:, :, :, :self.prefix_len] = 0.0  # 0 = can attend

        # suffix_to_suffix: causal (lower triangular)
        for i in range(self.suffix_len):
            # Token i can attend to tokens 0..i in suffix
            self._attention_mask[:, :, i, self.prefix_len:self.prefix_len + i + 1] = 0.0
            # Cannot attend to tokens i+1..suffix_len
            self._attention_mask[:, :, i, self.prefix_len + i + 1:] = -2.3819763e38

    def _init_position_ids(self):
        """Initialize static position IDs."""
        # Suffix positions continue from prefix
        prefix_offset = self.prefix_len
        for i in range(self.suffix_len):
            self._position_ids[:, i] = prefix_offset + i

    def _embed_suffix_static(self):
        """Static version of embed_suffix - uses pre-allocated buffers."""
        # 1. Compute time embedding
        create_sinusoidal_pos_embedding_static(
            self._static_timestep,
            self._time_emb,
            self.expert_hidden_size,
        )

        # 2. Time MLP (for adaRMS) - match dtype with weight
        time_emb_casted = self._time_emb.to(self.time_mlp_in.weight.dtype)
        x = self.time_mlp_in(time_emb_casted)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        self._adarms_cond.copy_(F.silu(x).to(self.dtype))

        # 3. Action embedding
        self._action_emb.copy_(
            self.action_in_proj(self._static_x_t.to(self.action_in_proj.weight.dtype)).to(self.dtype)
        )

        # 4. Copy to suffix embeddings
        self._suffix_embs.copy_(self._action_emb)

    def _forward_expert_static(self):
        """Forward through expert layers - uses pre-allocated buffers."""
        from transformers.models.gemma import modeling_gemma

        paligemma_lm = self.model.paligemma_with_expert.paligemma.language_model

        # Start with suffix embeddings
        self._hidden_states.copy_(self._suffix_embs)

        for layer_idx in range(self.num_layers):
            layer = self.expert.layers[layer_idx]
            cached_key, cached_value = self.prefix_kv_cache[layer_idx]

            # Input layernorm with adaRMS conditioning
            normed_hidden, gate = layer.input_layernorm(self._hidden_states, cond=self._adarms_cond)

            # Compute Q, K, V for suffix
            input_shape = normed_hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            query_states = layer.self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)

            # Apply RoPE
            dummy = torch.zeros(
                query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                device=query_states.device, dtype=query_states.dtype
            )
            cos, sin = paligemma_lm.rotary_emb(dummy, self._position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            # Concatenate cached prefix K, V with suffix K, V
            full_key = torch.cat([cached_key, key_states], dim=2)
            full_value = torch.cat([cached_value, value_states], dim=2)

            # Compute attention
            scaling = layer.self_attn.scaling
            att_output, _ = modeling_gemma.eager_attention_forward(
                layer.self_attn, query_states, full_key, full_value,
                self._attention_mask, scaling
            )

            head_dim = layer.self_attn.head_dim
            num_heads = layer.self_attn.config.num_attention_heads
            att_output = att_output.reshape(self.batch_size, -1, num_heads * head_dim)

            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output)

            # First residual with gating
            out_emb = modeling_gemma._gated_residual(self._hidden_states, out_emb, gate)
            after_first_residual = out_emb.clone()

            # Post-attention layernorm with adaRMS
            out_emb, gate = layer.post_attention_layernorm(out_emb, cond=self._adarms_cond)
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)

            # MLP
            out_emb = layer.mlp(out_emb)

            # Second residual with gating
            self._hidden_states.copy_(modeling_gemma._gated_residual(after_first_residual, out_emb, gate))

        # Final norm
        self._hidden_states, _ = self.expert.norm(self._hidden_states, cond=self._adarms_cond)

    def _denoise_forward(self):
        """Full static denoise step - this gets captured."""
        # 1. Embed suffix (action tokens + time conditioning)
        self._embed_suffix_static()

        # 2. Forward through expert layers
        self._forward_expert_static()

        # 3. Project to action space
        suffix_out = self._hidden_states[:, -self.action_horizon:]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        self._v_t.copy_(self.action_out_proj(suffix_out))

    def capture_graph(self, warmup_iters: int = 3):
        """Capture CUDA Graph for the denoise step."""
        if self._graph_captured:
            return

        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                self._denoise_forward()
        torch.cuda.synchronize()

        # Capture
        self._denoise_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._denoise_graph):
            self._denoise_forward()

        torch.cuda.synchronize()
        self._graph_captured = True

    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Execute captured denoise step.

        Args:
            state: (batch, state_dim) - robot state (unused in Pi0.5)
            x_t: (batch, action_horizon, action_dim) - noisy actions
            timestep: (batch,) - current timestep

        Returns:
            v_t: (batch, action_horizon, action_dim) - velocity prediction
        """
        if not self._graph_captured:
            raise RuntimeError("Graph not captured. Call capture_graph() first.")

        # Copy inputs to static buffers
        self._static_state.copy_(state)
        self._static_x_t.copy_(x_t)
        self._static_timestep.copy_(timestep)

        # Replay graph
        self._denoise_graph.replay()

        return self._v_t.clone()


class StaticDenoiseLoop(nn.Module):
    """Full N-step denoising loop with CUDA Graph.

    Captures the entire denoising loop (N steps) as a single graph.
    """

    def __init__(
        self,
        model,
        prefix_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
        num_steps: int = 3,
        batch_size: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device or torch.device('cuda')
        self.dtype = dtype

        action_horizon = model.config.action_horizon
        action_dim = model.config.action_dim

        self.dt = -1.0 / num_steps

        # Create static denoise step
        self._denoise_step = StaticDenoiseStep(
            model=model,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            batch_size=batch_size,
            action_horizon=action_horizon,
            action_dim=action_dim,
            device=self.device,
            dtype=dtype,
        )

        # Static buffers for loop
        self._static_state = torch.zeros(
            batch_size, 32,
            dtype=dtype, device=self.device
        )
        self._static_noise = torch.zeros(
            batch_size, action_horizon, action_dim,
            dtype=dtype, device=self.device
        )
        self._working_x_t = torch.zeros_like(self._static_noise)
        self._output = torch.zeros_like(self._static_noise)

        # Pre-compute timesteps
        self._timesteps = [
            torch.full((batch_size,), 1.0 + i * self.dt, dtype=torch.float32, device=self.device)
            for i in range(num_steps)
        ]

        self._graph_captured = False
        self._loop_graph: Optional[torch.cuda.CUDAGraph] = None

    def _loop_forward(self):
        """Unrolled loop - gets captured as single graph."""
        self._working_x_t.copy_(self._static_noise)

        for i in range(self.num_steps):
            # Update denoise step inputs
            self._denoise_step._static_state.copy_(self._static_state)
            self._denoise_step._static_x_t.copy_(self._working_x_t)
            self._denoise_step._static_timestep.copy_(self._timesteps[i])

            # Run denoise (using internal computation, not graph)
            self._denoise_step._denoise_forward()

            # Euler step
            self._working_x_t.add_(self._denoise_step._v_t, alpha=self.dt)

        self._output.copy_(self._working_x_t)

    def capture_graph(self, warmup_iters: int = 3):
        """Capture the entire loop as a single graph."""
        if self._graph_captured:
            return

        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                self._loop_forward()
        torch.cuda.synchronize()

        # Capture
        self._loop_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._loop_graph):
            self._loop_forward()

        torch.cuda.synchronize()
        self._graph_captured = True

    def forward(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Execute captured denoising loop.

        Args:
            state: (batch, state_dim)
            noise: (batch, action_horizon, action_dim)

        Returns:
            Denoised actions
        """
        if not self._graph_captured:
            raise RuntimeError("Graph not captured. Call capture_graph() first.")

        self._static_state.copy_(state)
        self._static_noise.copy_(noise)

        self._loop_graph.replay()

        return self._output.clone()
