"""Static Prefill for CUDA Graph capture.

This module provides a fully static version of compute_prefix_kv_cache
that can be captured as a CUDA Graph.

The original compute_prefix_kv_cache has Python dispatch overhead (136ms).
With CUDA Graph, we expect to reduce this to ~15ms.

Architecture:
1. StaticPrefill: Computes KV cache from prefix embeddings (Graph captured)
2. StaticEmbedPrefix: Embeds images + language to prefix embeddings (eager)

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


class StaticEmbedPrefix(nn.Module):
    """Static prefix embedding for fixed-size inputs.

    This prepares embeddings for StaticPrefill by:
    1. Processing pre-embedded images (from TRT or eager)
    2. Embedding language tokens
    3. Concatenating to fixed-size prefix
    """

    def __init__(
        self,
        model,  # PI0Pytorch
        max_num_images: int = 3,
        num_img_tokens: int = 256,
        max_lang_tokens: int = 200,
        batch_size: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.max_num_images = max_num_images
        self.num_img_tokens = num_img_tokens
        self.max_lang_tokens = max_lang_tokens
        self.batch_size = batch_size
        self.device = device or torch.device('cuda')
        self.dtype = dtype

        # Get hidden size
        paligemma_lm = model.paligemma_with_expert.paligemma.language_model
        self.hidden_size = paligemma_lm.config.hidden_size

        # Total prefix length
        self.max_prefix_len = max_num_images * num_img_tokens + max_lang_tokens

        # Pre-allocate output buffer
        self._prefix_embs = torch.zeros(
            batch_size, self.max_prefix_len, self.hidden_size,
            dtype=dtype, device=self.device
        )
        self._prefix_pad_masks = torch.zeros(
            batch_size, self.max_prefix_len,
            dtype=torch.bool, device=self.device
        )
        self._prefix_att_masks = torch.zeros(
            batch_size, self.max_prefix_len,
            dtype=torch.bool, device=self.device
        )

    def forward(
        self,
        image_embeddings: List[torch.Tensor],  # From TRT or eager vision encoder
        image_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Embed prefix (images + language).

        Args:
            image_embeddings: List of (batch, num_patches, hidden) from vision encoder
            image_masks: List of (batch,) bool masks
            lang_tokens: (batch, max_lang_len) token IDs
            lang_masks: (batch, max_lang_len) bool masks

        Returns:
            prefix_embs: (batch, prefix_len, hidden)
            prefix_pad_masks: (batch, prefix_len)
            prefix_att_masks: (batch, prefix_len)
            actual_len: Actual prefix length
        """
        # Clear buffers
        self._prefix_embs.zero_()
        self._prefix_pad_masks.zero_()
        self._prefix_att_masks.zero_()

        offset = 0

        # Copy image embeddings
        for img_emb, img_mask in zip(image_embeddings, image_masks):
            num_patches = img_emb.shape[1]
            self._prefix_embs[:, offset:offset + num_patches].copy_(img_emb.to(self.dtype))
            self._prefix_pad_masks[:, offset:offset + num_patches] = img_mask[:, None].expand(-1, num_patches)
            # Image tokens: full attention (att_mask = 0)
            offset += num_patches

        # Embed and copy language tokens
        lang_emb = self.model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        lang_len = lang_emb.shape[1]

        self._prefix_embs[:, offset:offset + lang_len].copy_(lang_emb.to(self.dtype))
        self._prefix_pad_masks[:, offset:offset + lang_len] = lang_masks
        # Language tokens: full attention (att_mask = 0)
        offset += lang_len

        actual_len = offset

        return (
            self._prefix_embs[:, :actual_len],
            self._prefix_pad_masks[:, :actual_len],
            self._prefix_att_masks[:, :actual_len],
            actual_len
        )


class StaticPrefill(nn.Module):
    """Static prefill computation for CUDA Graph capture.

    Pre-allocates all buffers needed for computing prefix KV cache.
    """

    def __init__(
        self,
        model,  # PI0Pytorch
        max_prefix_len: int = 512,
        batch_size: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.max_prefix_len = max_prefix_len
        self.batch_size = batch_size
        self.device = device or torch.device('cuda')
        self.dtype = dtype

        # Get model configs
        paligemma_lm = model.paligemma_with_expert.paligemma.language_model
        self.paligemma_lm = paligemma_lm
        config = paligemma_lm.config

        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        # Pre-allocate input buffer
        self._static_prefix_embs = torch.zeros(
            batch_size, max_prefix_len, self.hidden_size,
            dtype=dtype, device=self.device
        )

        # Pre-allocate attention mask
        self._static_attention_mask = torch.zeros(
            batch_size, 1, max_prefix_len, max_prefix_len,
            dtype=dtype, device=self.device
        )

        # Pre-allocate position IDs
        self._static_position_ids = torch.zeros(
            batch_size, max_prefix_len,
            dtype=torch.long, device=self.device
        )

        # Pre-allocate KV cache output buffers
        self._kv_cache = []
        for _ in range(self.num_layers):
            key = torch.zeros(
                batch_size, self.num_kv_heads, max_prefix_len, self.head_dim,
                dtype=dtype, device=self.device
            )
            value = torch.zeros(
                batch_size, self.num_kv_heads, max_prefix_len, self.head_dim,
                dtype=dtype, device=self.device
            )
            self._kv_cache.append((key, value))

        # Hidden states buffer
        self._hidden_states = torch.zeros(
            batch_size, max_prefix_len, self.hidden_size,
            dtype=dtype, device=self.device
        )

        # Current sequence length (set before each inference)
        self._current_seq_len = max_prefix_len

        # Graph state
        self._graph_captured = False
        self._prefill_graph: Optional[torch.cuda.CUDAGraph] = None

    def _prefill_forward(self):
        """Forward pass for prefill - this gets captured."""
        from transformers.models.gemma import modeling_gemma

        seq_len = self._current_seq_len

        # Start with input embeddings
        self._hidden_states[:, :seq_len].copy_(self._static_prefix_embs[:, :seq_len])

        for layer_idx in range(self.num_layers):
            layer = self.paligemma_lm.layers[layer_idx]
            hidden = self._hidden_states[:, :seq_len]

            # Input layernorm
            normed_hidden, _ = layer.input_layernorm(hidden, cond=None)

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
            cos, sin = self.paligemma_lm.rotary_emb(dummy, self._static_position_ids[:, :seq_len])
            query, key = modeling_gemma.apply_rotary_pos_emb(
                query, key, cos, sin, unsqueeze_dim=1
            )

            # Store KV for this layer
            self._kv_cache[layer_idx][0][:, :, :seq_len].copy_(key)
            self._kv_cache[layer_idx][1][:, :, :seq_len].copy_(value)

            # Attention with SDPA
            num_kv_groups = layer.self_attn.num_key_value_groups

            # Expand K, V for GQA
            key_expanded = key[:, :, None, :, :].expand(
                self.batch_size, key.shape[1], num_kv_groups, seq_len, key.shape[-1]
            ).reshape(self.batch_size, -1, seq_len, key.shape[-1])
            value_expanded = value[:, :, None, :, :].expand(
                self.batch_size, value.shape[1], num_kv_groups, seq_len, value.shape[-1]
            ).reshape(self.batch_size, -1, seq_len, value.shape[-1])

            att_output = F.scaled_dot_product_attention(
                query, key_expanded, value_expanded,
                attn_mask=self._static_attention_mask[:, :, :seq_len, :seq_len].to(query.dtype),
                dropout_p=0.0,
                is_causal=False,
                scale=layer.self_attn.scaling,
            )

            # Reshape and project
            att_output = att_output.transpose(1, 2).reshape(
                self.batch_size, seq_len, self.num_heads * self.head_dim
            )

            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output)

            # First residual
            out_emb = hidden + out_emb
            after_first_residual = out_emb.clone()

            # Post-attention layernorm
            out_emb, _ = layer.post_attention_layernorm(out_emb, cond=None)
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)

            # MLP
            out_emb = layer.mlp(out_emb)

            # Second residual
            self._hidden_states[:, :seq_len].copy_(after_first_residual + out_emb)

    def set_sequence_length(self, seq_len: int):
        """Set current sequence length for prefill."""
        if seq_len > self.max_prefix_len:
            raise ValueError(f"seq_len {seq_len} > max_prefix_len {self.max_prefix_len}")
        self._current_seq_len = seq_len

        # Initialize position IDs
        for i in range(seq_len):
            self._static_position_ids[:, i] = i

    def capture_graph(self, seq_len: int, warmup_iters: int = 3):
        """Capture CUDA Graph for prefill.

        Note: Graph is captured for a fixed sequence length.
        """
        if self._graph_captured:
            return

        self.set_sequence_length(seq_len)

        # Initialize attention mask (prefix can attend to all prefix)
        self._static_attention_mask[:, :, :seq_len, :seq_len] = 0.0

        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                self._prefill_forward()
        torch.cuda.synchronize()

        # Capture
        self._prefill_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._prefill_graph):
            self._prefill_forward()

        torch.cuda.synchronize()
        self._graph_captured = True

    def forward(
        self,
        prefix_embs: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Execute captured prefill.

        Args:
            prefix_embs: (batch, seq_len, hidden_size)

        Returns:
            List of (K, V) for each layer
        """
        if not self._graph_captured:
            raise RuntimeError("Graph not captured. Call capture_graph() first.")

        seq_len = prefix_embs.shape[1]
        if seq_len != self._current_seq_len:
            raise RuntimeError(f"seq_len {seq_len} != captured seq_len {self._current_seq_len}")

        # Copy input
        self._static_prefix_embs[:, :seq_len].copy_(prefix_embs)

        # Replay
        self._prefill_graph.replay()

        # Return KV cache (cloned to avoid issues with buffer reuse)
        return [
            (k[:, :, :seq_len].clone(), v[:, :, :seq_len].clone())
            for k, v in self._kv_cache
        ]
