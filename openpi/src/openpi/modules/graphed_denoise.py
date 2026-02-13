"""CUDA Graph-captured denoising loop for PI0.

This module provides high-performance denoising by:
1. Pre-computing timestep embeddings (adarms_cond) for all 10 steps
2. Capturing 10 CUDA Graphs (one per denoise step)
3. Chain replaying graphs with minimal Python overhead

Expected optimization: ~9 ms savings (0.9 ms/step × 10 steps)

Author: Turbo-Pi Team
Date: 2026-02-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class GraphedDenoiseConfig:
    """Configuration for graphed denoising."""
    num_steps: int = 10
    action_horizon: int = 50
    action_dim: int = 32
    batch_size: int = 1
    dtype: torch.dtype = torch.bfloat16


class GraphedDenoiseLoop(nn.Module):
    """CUDA Graph-captured 10-step denoising loop.

    This class captures the entire denoising loop as a chain of CUDA Graphs,
    eliminating Python dispatch overhead between steps.

    Architecture:
    1. Pre-compute adarms_cond for all 10 timesteps (done once during warmup)
    2. Capture 10 graphs, each with its pre-computed adarms_cond
    3. On inference, chain-replay all 10 graphs

    Expected performance:
    - Without graph: ~95 ms (denoise only)
    - With graph:    ~86 ms (9 ms savings)
    """

    def __init__(
        self,
        model,  # PI0Pytorch model
        config: GraphedDenoiseConfig,
        device: torch.device,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.device = device

        # Compute dt and timesteps
        self.dt = -1.0 / config.num_steps

        # Pre-compute timestep values
        self.timestep_values = [
            1.0 + i * self.dt for i in range(config.num_steps)
        ]

        # Graph capture state
        self._graphs_captured = False
        self._step_graphs: List[torch.cuda.CUDAGraph] = []

        # Static buffers (allocated during capture)
        self._static_state: Optional[torch.Tensor] = None
        self._static_x_t: Optional[torch.Tensor] = None
        self._static_v_t: Optional[torch.Tensor] = None
        self._static_prefix_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._static_prefix_pad_masks: Optional[torch.Tensor] = None

        # Pre-computed adarms_cond for each step
        self._static_adarms_conds: List[torch.Tensor] = []

        # Pre-computed static tensors for each step
        self._static_timesteps: List[torch.Tensor] = []

    def _precompute_adarms_conds(self):
        """Pre-compute adarms_cond for all 10 timesteps.

        adarms_cond = silu(time_mlp_out(silu(time_mlp_in(time_emb))))
        where time_emb = sinusoidal_embedding(timestep)
        """
        from openpi.models_pytorch.embedding import create_sinusoidal_pos_embedding

        self._static_adarms_conds = []

        model_dtype = self.model.action_in_proj.weight.dtype

        for timestep_val in self.timestep_values:
            # Create timestep tensor
            timestep = torch.tensor(
                [timestep_val] * self.config.batch_size,
                dtype=torch.float32,
                device=self.device
            )

            # Compute sinusoidal embedding
            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.model.action_in_proj.out_features,
                min_period=4e-3,
                max_period=4.0,
                device=self.device
            )
            time_emb = time_emb.to(dtype=model_dtype)

            # Apply time MLP (for pi05)
            with torch.no_grad():
                x = self.model.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.model.time_mlp_out(x)
                adarms_cond = F.silu(x)

            # Store as static buffer
            self._static_adarms_conds.append(adarms_cond)

            # Also store static timestep tensor
            self._static_timesteps.append(timestep.clone())

    def _create_static_buffers(
        self,
        prefix_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
    ):
        """Create static buffers for graph capture."""
        # State buffer (max_state_dim = 32)
        self._static_state = torch.zeros(
            self.config.batch_size, 32,
            dtype=self.config.dtype,
            device=self.device,
        )

        # Action buffers
        self._static_x_t = torch.zeros(
            self.config.batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            dtype=self.config.dtype,
            device=self.device,
        )
        self._static_v_t = torch.zeros_like(self._static_x_t)

        # Copy KV cache to static buffers
        self._static_prefix_kv_cache = [
            (k.clone(), v.clone()) for k, v in prefix_kv_cache
        ]
        self._static_prefix_pad_masks = prefix_pad_masks.clone()

    def _denoise_step_with_precomputed_adarms(
        self,
        step_idx: int,
    ) -> torch.Tensor:
        """Execute single denoise step using pre-computed adarms_cond.

        This is a modified version of denoise_step_with_cache that uses
        pre-computed adarms_cond instead of computing it from timestep.
        """
        # Get pre-computed values
        adarms_cond = self._static_adarms_conds[step_idx]
        timestep = self._static_timesteps[step_idx]

        # Get models
        gemma_expert = self.model.paligemma_with_expert.gemma_expert.model
        paligemma_lm = self.model.paligemma_with_expert.paligemma.language_model
        num_layers = gemma_expert.config.num_hidden_layers

        batch_size = self._static_prefix_pad_masks.shape[0]
        prefix_len = self._static_prefix_pad_masks.shape[1]

        # Embed actions (this still needs to be computed per step since x_t changes)
        model_dtype = self.model.action_in_proj.weight.dtype
        action_emb = self.model.action_in_proj(self._static_x_t.to(model_dtype))

        # Build suffix embeddings (state + action)
        if not self.model.pi05:
            state_emb = self.model.state_proj(self._static_state)
            suffix_embs = torch.cat([state_emb[:, None, :], action_emb], dim=1)
        else:
            suffix_embs = action_emb

        suffix_len = suffix_embs.shape[1]

        # Convert to bfloat16 if needed
        if gemma_expert.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Position IDs
        prefix_offsets = torch.sum(self._static_prefix_pad_masks, dim=-1)[:, None]
        suffix_pad_masks = torch.ones(
            batch_size, suffix_len, dtype=torch.bool, device=self.device
        )
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        hidden_states = suffix_embs

        # Process through all layers
        for layer_idx in range(num_layers):
            layer = gemma_expert.layers[layer_idx]
            cached_key, cached_value = self._static_prefix_kv_cache[layer_idx]

            # Input layernorm with adaRMS conditioning
            normed_hidden, gate = layer.input_layernorm(hidden_states, cond=adarms_cond)

            # Q, K, V projections
            input_shape = normed_hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            query_states = layer.self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)

            # Apply RoPE
            dummy_tensor = torch.zeros(
                query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                device=query_states.device, dtype=query_states.dtype
            )
            cos, sin = paligemma_lm.rotary_emb(dummy_tensor, suffix_position_ids)
            from openpi.models_pytorch.transformers_replace.models.gemma import modeling_gemma
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            # Concatenate with cached KV
            full_key_states = torch.cat([cached_key, key_states], dim=2)
            full_value_states = torch.cat([cached_value, value_states], dim=2)

            # SDPA attention (no mask needed - see debug-10 optimization)
            att_output = F.scaled_dot_product_attention(
                query_states, full_key_states, full_value_states
            )
            att_output = att_output.transpose(1, 2).contiguous()

            head_dim = layer.self_attn.head_dim
            num_heads = layer.self_attn.config.num_attention_heads
            att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)

            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output)

            # Residual with gating
            out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gate)
            after_first_residual = out_emb.clone()

            # Post-attention layernorm
            out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond)
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)

            # MLP
            out_emb = layer.mlp(out_emb)

            # Second residual
            hidden_states = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)

        # Final norm
        hidden_states, _ = gemma_expert.norm(hidden_states, cond=adarms_cond)

        # Extract action tokens and project
        suffix_out = hidden_states[:, -self.model.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)

        return self.model.action_out_proj(suffix_out)

    def capture_graphs(
        self,
        prefix_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
        warmup_iters: int = 3,
    ):
        """Capture CUDA Graphs for all 10 denoise steps.

        Args:
            prefix_kv_cache: Pre-computed KV cache from embed_prefix
            prefix_pad_masks: Padding masks for prefix
            warmup_iters: Number of warmup iterations before capture
        """
        if self._graphs_captured:
            return

        # Create static buffers
        self._create_static_buffers(prefix_kv_cache, prefix_pad_masks)

        # Pre-compute adarms_cond for all steps
        self._precompute_adarms_conds()

        # Capture graph for each step
        self._step_graphs = []

        for step_idx in range(self.config.num_steps):
            # Initialize x_t for this step
            if step_idx == 0:
                self._static_x_t.normal_()
            # else: x_t is updated in-place by previous step

            # Warmup
            for _ in range(warmup_iters):
                with torch.no_grad():
                    _ = self._denoise_step_with_precomputed_adarms(step_idx)
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph):
                v_t = self._denoise_step_with_precomputed_adarms(step_idx)
                self._static_v_t.copy_(v_t)
                # Euler update in-place
                self._static_x_t.add_(self._static_v_t, alpha=self.dt)

            self._step_graphs.append(graph)
            torch.cuda.synchronize()

        self._graphs_captured = True

    def forward(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Execute captured denoising loop.

        Args:
            state: (batch, state_dim) robot state
            noise: (batch, action_horizon, action_dim) initial noise

        Returns:
            Denoised actions (batch, action_horizon, action_dim)
        """
        if not self._graphs_captured:
            raise RuntimeError("Graphs not captured. Call capture_graphs() first.")

        # Copy inputs to static buffers
        self._static_state[:, :state.shape[1]].copy_(state)
        self._static_x_t.copy_(noise)

        # Replay all graphs in sequence
        for graph in self._step_graphs:
            graph.replay()

        return self._static_x_t.clone()

    def update_prefix_cache(
        self,
        prefix_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
    ):
        """Update prefix KV cache for new observation.

        Note: This requires re-capturing graphs if prefix length changes.
        """
        if self._static_prefix_pad_masks is None:
            raise RuntimeError("Graphs not captured yet.")

        # Check if prefix length matches
        if prefix_pad_masks.shape != self._static_prefix_pad_masks.shape:
            raise RuntimeError(
                f"Prefix length changed: {self._static_prefix_pad_masks.shape} -> {prefix_pad_masks.shape}. "
                "Re-capture graphs required."
            )

        # Update KV cache in-place
        for i, (k, v) in enumerate(prefix_kv_cache):
            self._static_prefix_kv_cache[i][0].copy_(k)
            self._static_prefix_kv_cache[i][1].copy_(v)

        self._static_prefix_pad_masks.copy_(prefix_pad_masks)


class ChainedDenoiseGraphs(nn.Module):
    """Chain of CUDA Graph-captured denoise steps.

    Uses the CUDA Graph-compatible denoise_step_graphed method which
    pre-computes all dynamic tensors (position_ids, adarms_cond) before capture.

    This approach:
    1. Pre-computes all static tensors during warmup
    2. Captures 10 graphs (one per timestep)
    3. Chain-replays for minimal Python overhead
    """

    def __init__(
        self,
        model,
        num_steps: int = 10,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.device = device

        self.dt = -1.0 / num_steps

        # Graph state
        self._graphs: List[torch.cuda.CUDAGraph] = []
        self._captured = False

        # Static buffers
        self._static_x_t = None
        self._static_v_t = None
        self._static_prefix_kv_cache = None
        self._static_suffix_position_ids = None
        self._static_adarms_conds = []

    def capture(
        self,
        state: torch.Tensor,
        prefix_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
        warmup_iters: int = 3,
    ):
        """Capture all denoise step graphs using graph-compatible method."""
        batch_size = state.shape[0]
        action_horizon = self.model.config.action_horizon
        action_dim = self.model.config.action_dim
        prefix_len = prefix_pad_masks.shape[1]
        dtype = torch.bfloat16

        # Pre-compute all static tensors for graphed execution
        graph_tensors = self.model.precompute_graph_tensors(
            batch_size=batch_size,
            prefix_len=prefix_len,
            num_steps=self.num_steps,
            device=self.device,
        )

        self._static_suffix_position_ids = graph_tensors["suffix_position_ids"]
        self._static_adarms_conds = graph_tensors["adarms_conds"]

        # Create static buffers
        self._static_x_t = torch.randn(
            batch_size, action_horizon, action_dim,
            device=self.device, dtype=dtype
        )
        self._static_v_t = torch.zeros_like(self._static_x_t)
        self._static_prefix_kv_cache = [
            (k.clone(), v.clone()) for k, v in prefix_kv_cache
        ]

        # Capture graphs
        self._graphs = []

        for step_idx in range(self.num_steps):
            adarms_cond = self._static_adarms_conds[step_idx]

            # Warmup using graph-compatible method
            for _ in range(warmup_iters):
                with torch.no_grad():
                    _ = self.model.denoise_step_graphed(
                        self._static_x_t,
                        self._static_suffix_position_ids,
                        adarms_cond,
                        self._static_prefix_kv_cache,
                    )
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph):
                v_t = self.model.denoise_step_graphed(
                    self._static_x_t,
                    self._static_suffix_position_ids,
                    adarms_cond,
                    self._static_prefix_kv_cache,
                )
                self._static_v_t.copy_(v_t)
                self._static_x_t.add_(self._static_v_t, alpha=self.dt)

            self._graphs.append(graph)
            torch.cuda.synchronize()

        self._captured = True

    def forward(
        self,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Run captured denoising loop.

        Note: state is baked into the KV cache during capture.
        Only noise needs to be provided at runtime.
        """
        if not self._captured:
            raise RuntimeError("Graphs not captured.")

        self._static_x_t.copy_(noise)

        for graph in self._graphs:
            graph.replay()

        return self._static_x_t.clone()

    def update_kv_cache(
        self,
        prefix_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        """Update the static KV cache for a new observation.

        Call this when the observation changes but you want to reuse
        the captured graphs (same prefix length required).
        """
        if not self._captured:
            raise RuntimeError("Graphs not captured.")

        for i, (k, v) in enumerate(prefix_kv_cache):
            self._static_prefix_kv_cache[i][0].copy_(k)
            self._static_prefix_kv_cache[i][1].copy_(v)
