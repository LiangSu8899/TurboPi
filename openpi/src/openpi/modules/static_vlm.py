"""Static VLM wrapper for CUDA Graph capture.

This module provides static memory allocation for KV cache and CUDA Graph-captured
decode steps, eliminating Python dispatch overhead.

Key components:
- StaticKVCache: Pre-allocated KV cache with fixed max_seq_len
- StaticVLMDecode: CUDA Graph-captured decode step
- GraphedPaliGemma: Full VLM with captured graph

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class StaticKVCacheConfig:
    """Configuration for static KV cache."""
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int = 1024
    batch_size: int = 1
    dtype: torch.dtype = torch.bfloat16


class StaticKVCache:
    """Pre-allocated static KV cache for CUDA Graph compatibility.

    Unlike dynamic KV cache that grows with sequence length, this cache
    pre-allocates memory for max_seq_len tokens. This enables CUDA Graph
    capture since memory layout is fixed.

    Memory layout: [num_layers, 2, batch_size, num_kv_heads, max_seq_len, head_dim]
    where dim 1 is [key, value]
    """

    def __init__(
        self,
        config: StaticKVCacheConfig,
        device: torch.device = torch.device('cuda'),
    ):
        self.config = config
        self.device = device

        # Pre-allocate full cache buffer
        # Shape: [num_layers, 2, batch_size, num_kv_heads, max_seq_len, head_dim]
        self.cache = torch.zeros(
            config.num_layers,
            2,  # key, value
            config.batch_size,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
            dtype=config.dtype,
            device=device,
        )

        # Current position in cache (how many tokens have been written)
        self.position = 0

    def reset(self):
        """Reset cache position (zero-copy, just reset pointer)."""
        self.position = 0

    def get_kv_for_layer(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get K, V tensors for a specific layer up to current position.

        Returns:
            (key, value) each with shape (batch, num_kv_heads, seq_len, head_dim)
        """
        key = self.cache[layer_idx, 0, :, :, :self.position, :]
        value = self.cache[layer_idx, 1, :, :, :self.position, :]
        return key, value

    def update_layer(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """Update cache for a layer with new K, V values.

        Args:
            layer_idx: Which layer to update
            key: (batch, num_kv_heads, new_seq_len, head_dim)
            value: (batch, num_kv_heads, new_seq_len, head_dim)
        """
        new_seq_len = key.shape[2]
        end_pos = self.position + new_seq_len

        self.cache[layer_idx, 0, :, :, self.position:end_pos, :].copy_(key)
        self.cache[layer_idx, 1, :, :, self.position:end_pos, :].copy_(value)

    def advance_position(self, num_tokens: int):
        """Advance cache position after writing tokens."""
        self.position += num_tokens

    def write_prefix_cache(
        self,
        kv_list: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        """Write prefix KV cache from list of (K, V) tuples.

        Args:
            kv_list: List of (K, V) for each layer, where K/V have shape
                     (batch, num_kv_heads, seq_len, head_dim)
        """
        if len(kv_list) != self.config.num_layers:
            raise ValueError(f"Expected {self.config.num_layers} layers, got {len(kv_list)}")

        seq_len = kv_list[0][0].shape[2]

        for layer_idx, (key, value) in enumerate(kv_list):
            self.cache[layer_idx, 0, :, :, :seq_len, :].copy_(key)
            self.cache[layer_idx, 1, :, :, :seq_len, :].copy_(value)

        self.position = seq_len


class StaticVLMDecode(nn.Module):
    """CUDA Graph-captured VLM decode step.

    This wraps the VLM decode step (single token generation) in a way that's
    compatible with CUDA Graph capture:
    - Static input buffers
    - Static output buffers
    - No Python control flow during forward
    """

    def __init__(
        self,
        paligemma_lm: nn.Module,
        expert: nn.Module,
        static_kv_cache: StaticKVCache,
        max_action_horizon: int = 50,
        action_dim: int = 32,
    ):
        super().__init__()
        self.paligemma_lm = paligemma_lm
        self.expert = expert
        self.kv_cache = static_kv_cache
        self.max_action_horizon = max_action_horizon
        self.action_dim = action_dim

        config = paligemma_lm.config
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        # Static buffers for graph capture (will be allocated on first use)
        self._graph_captured = False
        self._decode_graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_hidden: Optional[torch.Tensor] = None
        self._static_position_ids: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None

    def _create_static_buffers(self, device: torch.device, dtype: torch.dtype, batch_size: int):
        """Create static buffers for CUDA Graph capture."""
        # Hidden state for decode step (seq_len=1 for autoregressive decode)
        self._static_hidden = torch.zeros(
            batch_size, 1, self.hidden_size,
            dtype=dtype, device=device
        )

        # Position IDs
        self._static_position_ids = torch.zeros(
            batch_size, 1,
            dtype=torch.long, device=device
        )

        # Output buffer
        self._static_output = torch.zeros(
            batch_size, 1, self.hidden_size,
            dtype=dtype, device=device
        )

    def _forward_one_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through one transformer layer with KV cache.

        This is the inner loop that will be captured in the CUDA Graph.
        """
        from transformers.models.gemma import modeling_gemma

        layer = self.paligemma_lm.layers[layer_idx]

        # Input layernorm
        normed_hidden, _ = layer.input_layernorm(hidden_states, cond=None)

        # Q, K, V projections
        input_shape = normed_hidden.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query = layer.self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
        key = layer.self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
        value = layer.self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        dummy = torch.zeros(
            query.shape[0], query.shape[2], query.shape[-1],
            device=query.device, dtype=query.dtype
        )
        cos, sin = self.paligemma_lm.rotary_emb(dummy, position_ids)
        query, key = modeling_gemma.apply_rotary_pos_emb(
            query, key, cos, sin, unsqueeze_dim=1
        )

        # Get cached K, V and concatenate
        cached_key, cached_value = self.kv_cache.get_kv_for_layer(layer_idx)
        full_key = torch.cat([cached_key, key], dim=2)
        full_value = torch.cat([cached_value, value], dim=2)

        # Update cache
        self.kv_cache.update_layer(layer_idx, key, value)

        # Compute attention with SDPA
        batch_size = query.shape[0]
        num_kv_groups = layer.self_attn.num_key_value_groups

        # Expand K, V for GQA
        key_expanded = full_key[:, :, None, :, :].expand(
            batch_size, full_key.shape[1], num_kv_groups, full_key.shape[2], full_key.shape[3]
        ).reshape(batch_size, -1, full_key.shape[2], full_key.shape[3])
        value_expanded = full_value[:, :, None, :, :].expand(
            batch_size, full_value.shape[1], num_kv_groups, full_value.shape[2], full_value.shape[3]
        ).reshape(batch_size, -1, full_value.shape[2], full_value.shape[3])

        att_output = torch.nn.functional.scaled_dot_product_attention(
            query, key_expanded, value_expanded,
            attn_mask=attention_mask.to(query.dtype),
            dropout_p=0.0,
            is_causal=False,
            scale=layer.self_attn.scaling,
        )

        # Reshape and project
        att_output = att_output.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_dim
        )

        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output)

        # First residual
        out_emb = hidden_states + out_emb
        after_first_residual = out_emb.clone()

        # Post-attention layernorm
        out_emb, _ = layer.post_attention_layernorm(out_emb, cond=None)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)

        # MLP
        out_emb = layer.mlp(out_emb)

        # Second residual
        hidden_states = after_first_residual + out_emb

        return hidden_states

    def forward_all_layers(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through all layers - this is what gets captured in CUDA Graph."""
        for layer_idx in range(self.num_layers):
            hidden_states = self._forward_one_layer(
                layer_idx, hidden_states, position_ids, attention_mask
            )
        return hidden_states

    def capture_graph(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 1,
        warmup_iters: int = 3,
    ):
        """Capture CUDA Graph for the decode step.

        This should be called once after model is loaded and warmed up.
        """
        if self._graph_captured:
            return

        self._create_static_buffers(device, dtype, batch_size)

        # Create attention mask for decode (can attend to all previous tokens)
        cache_len = self.kv_cache.position
        total_len = cache_len + 1  # +1 for current token

        # Attention mask: current token can attend to all previous
        static_attention_mask = torch.zeros(
            batch_size, 1, 1, total_len,
            dtype=dtype, device=device
        )

        # Warmup runs
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = self.forward_all_layers(
                    self._static_hidden,
                    self._static_position_ids,
                    static_attention_mask,
                )
        torch.cuda.synchronize()

        # Capture graph
        self._decode_graph = torch.cuda.CUDAGraph()
        self._static_attention_mask = static_attention_mask

        with torch.cuda.graph(self._decode_graph):
            output = self.forward_all_layers(
                self._static_hidden,
                self._static_position_ids,
                self._static_attention_mask,
            )
            self._static_output.copy_(output)

        torch.cuda.synchronize()
        self._graph_captured = True

    def decode_step(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Execute captured decode step.

        Args:
            hidden_states: (batch, 1, hidden_size) - current token embedding
            position_ids: (batch, 1) - position in sequence

        Returns:
            (batch, 1, hidden_size) - output hidden states
        """
        if not self._graph_captured:
            raise RuntimeError("Graph not captured. Call capture_graph() first.")

        # Copy inputs to static buffers
        self._static_hidden.copy_(hidden_states)
        self._static_position_ids.copy_(position_ids)

        # Replay graph
        self._decode_graph.replay()

        return self._static_output.clone()


class GraphedDenoiseStep(nn.Module):
    """CUDA Graph-captured full denoise step.

    This captures the entire denoise step (embed_suffix + expert forward + action projection)
    as a single CUDA Graph, eliminating all Python dispatch overhead.
    """

    def __init__(
        self,
        model,  # PI0Pytorch model
        static_kv_cache: StaticKVCache,
        action_horizon: int = 50,
        action_dim: int = 32,
    ):
        super().__init__()
        self.model = model
        self.kv_cache = static_kv_cache
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        expert = model.paligemma_with_expert.gemma_expert.model
        self.num_layers = expert.config.num_hidden_layers
        self.hidden_size = expert.config.hidden_size

        # Graph state
        self._graph_captured = False
        self._denoise_graph: Optional[torch.cuda.CUDAGraph] = None

        # Static buffers
        self._static_state: Optional[torch.Tensor] = None
        self._static_x_t: Optional[torch.Tensor] = None
        self._static_timestep: Optional[torch.Tensor] = None
        self._static_v_t: Optional[torch.Tensor] = None

    def _create_static_buffers(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ):
        """Create static buffers for graph capture."""
        self._static_state = torch.zeros(
            batch_size, 32,  # max_state_dim
            dtype=dtype, device=device
        )
        self._static_x_t = torch.zeros(
            batch_size, self.action_horizon, self.action_dim,
            dtype=dtype, device=device
        )
        self._static_timestep = torch.zeros(
            batch_size,
            dtype=torch.float32, device=device
        )
        self._static_v_t = torch.zeros(
            batch_size, self.action_horizon, self.action_dim,
            dtype=dtype, device=device
        )

    def _denoise_forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
    ) -> torch.Tensor:
        """The actual denoise step that gets captured.

        This is the full denoise_step_with_cache but using static KV cache.
        """
        # Convert static KV cache to list format
        kv_list = []
        for layer_idx in range(self.num_layers):
            k, v = self.kv_cache.get_kv_for_layer(layer_idx)
            kv_list.append((k, v))

        # Use model's denoise_step_with_cache
        return self.model.denoise_step_with_cache(
            state,
            kv_list,
            prefix_pad_masks,
            x_t,
            timestep,
        )

    def capture_graph(
        self,
        device: torch.device,
        prefix_pad_masks: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 1,
        warmup_iters: int = 3,
    ):
        """Capture CUDA Graph for the denoise step."""
        if self._graph_captured:
            return

        self._create_static_buffers(device, dtype, batch_size)

        # Store static prefix_pad_masks
        self._static_prefix_pad_masks = prefix_pad_masks.clone()

        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = self._denoise_forward(
                    self._static_state,
                    self._static_x_t,
                    self._static_timestep,
                    self._static_prefix_pad_masks,
                )
        torch.cuda.synchronize()

        # Capture
        self._denoise_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._denoise_graph):
            v_t = self._denoise_forward(
                self._static_state,
                self._static_x_t,
                self._static_timestep,
                self._static_prefix_pad_masks,
            )
            self._static_v_t.copy_(v_t)

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
            state: (batch, state_dim)
            x_t: (batch, action_horizon, action_dim) - noisy actions
            timestep: (batch,) - current timestep

        Returns:
            v_t: (batch, action_horizon, action_dim) - velocity prediction
        """
        if not self._graph_captured:
            raise RuntimeError("Graph not captured. Call capture_graph() first.")

        # Copy inputs
        self._static_state.copy_(state)
        self._static_x_t.copy_(x_t)
        self._static_timestep.copy_(timestep)

        # Replay
        self._denoise_graph.replay()

        return self._static_v_t.clone()


class GraphedPI0(nn.Module):
    """Full PI0 model with CUDA Graph-captured inference.

    This wraps PI0Pytorch with:
    1. Static KV cache for prefix
    2. CUDA Graph-captured denoise step
    3. Unrolled denoising loop

    Usage:
        graphed_model = GraphedPI0(model, device)
        graphed_model.warm_up(sample_observation)  # One-time warmup
        actions = graphed_model.sample_actions(observation)  # Fast inference
    """

    def __init__(
        self,
        model,  # PI0Pytorch
        device: torch.device,
        max_seq_len: int = 1024,
        num_denoise_steps: int = 3,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.max_seq_len = max_seq_len
        self.num_denoise_steps = num_denoise_steps

        # Get config from model
        paligemma_lm = model.paligemma_with_expert.paligemma.language_model
        config = paligemma_lm.config

        # Create static KV cache
        cache_config = StaticKVCacheConfig(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seq_len=max_seq_len,
            batch_size=1,
            dtype=torch.bfloat16,
        )
        self.static_kv_cache = StaticKVCache(cache_config, device)

        # Graphed denoise step (created during warm_up)
        self.graphed_denoise: Optional[GraphedDenoiseStep] = None

        # Warm-up state
        self._warmed_up = False

    def warm_up(self, observation, num_steps: int = 3):
        """Warm up and capture CUDA Graphs.

        This should be called once with a sample observation before inference.

        Args:
            observation: Sample Observation for shape inference
            num_steps: Number of denoise steps to use
        """
        self.num_denoise_steps = num_steps

        with torch.no_grad():
            # 1. Embed prefix (vision + language)
            images, img_masks, lang_tokens, lang_masks, state = \
                self.model._preprocess_observation(observation, train=False)

            prefix_embs, prefix_pad_masks, prefix_att_masks = \
                self.model.embed_prefix(images, img_masks, lang_tokens, lang_masks)

            # 2. Compute prefix KV cache and write to static cache
            prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )
            self.static_kv_cache.write_prefix_cache(prefix_kv_cache)

            # 3. Create and capture graphed denoise step
            self.graphed_denoise = GraphedDenoiseStep(
                self.model,
                self.static_kv_cache,
                action_horizon=self.model.config.action_horizon,
                action_dim=self.model.config.action_dim,
            )

            self.graphed_denoise.capture_graph(
                device=self.device,
                prefix_pad_masks=prefix_pad_masks,
                dtype=torch.bfloat16,
                batch_size=1,
                warmup_iters=3,
            )

            # Store prefix_pad_masks for inference
            self._prefix_pad_masks = prefix_pad_masks

        self._warmed_up = True

    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fast inference using captured CUDA Graphs.

        Args:
            observation: Input observation
            noise: Optional initial noise

        Returns:
            Denoised actions (batch, action_horizon, action_dim)
        """
        if not self._warmed_up:
            raise RuntimeError("Model not warmed up. Call warm_up() first.")

        batch_size = observation.state.shape[0]

        # 1. Embed prefix and update KV cache
        images, img_masks, lang_tokens, lang_masks, state = \
            self.model._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = \
            self.model.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        prefix_kv_cache = self.model.compute_prefix_kv_cache(
            prefix_embs, prefix_pad_masks, prefix_att_masks
        )
        self.static_kv_cache.write_prefix_cache(prefix_kv_cache)

        # 2. Initialize noise
        if noise is None:
            actions_shape = (batch_size, self.model.config.action_horizon, self.model.config.action_dim)
            noise = self.model.sample_noise(actions_shape, self.device)

        # Get model dtype
        model_dtype = next(self.model.parameters()).dtype
        x_t = noise.to(model_dtype)

        # 3. Denoising loop using captured graph
        dt = -1.0 / self.num_denoise_steps
        time = 1.0

        for step in range(self.num_denoise_steps):
            timestep = torch.tensor([time] * batch_size, dtype=torch.float32, device=self.device)

            # Use graphed denoise step
            v_t = self.graphed_denoise(state, x_t, timestep)

            # Euler step
            x_t = x_t + dt * v_t
            time += dt

        return x_t
