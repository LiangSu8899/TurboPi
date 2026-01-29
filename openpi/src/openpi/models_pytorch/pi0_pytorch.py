import logging
import math
from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

# Use pure Python config to avoid JAX/Flax dependencies
import openpi.models_pytorch.gemma_config as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


@dataclass
class Pi0Config:
    """Configuration for Pi0/Pi0.5 VLA model."""
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    max_state_dim: int = 32
    pi05: bool = True
    dtype: Union[str, torch.dtype] = "bfloat16"


@dataclass
class Observation:
    """Observation data structure for Pi0 model inference."""
    images: dict  # dict of image_name -> tensor [B, C, H, W]
    image_masks: dict  # dict of image_name -> tensor [B]
    state: Tensor  # [B, state_dim]
    tokenized_prompt: Tensor  # [B, max_token_len]
    tokenized_prompt_mask: Tensor  # [B, max_token_len]
    token_ar_mask: Tensor = None  # optional [B, max_token_len]
    token_loss_mask: Tensor = None  # optional [B, max_token_len]


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # Temporarily disabled for baseline testing on Jetson Thor
        # self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        # Cast to model dtype (not timestep dtype)
        model_dtype = self.action_in_proj.weight.dtype
        time_emb = time_emb.to(dtype=model_dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions.to(model_dtype))

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def compute_prefix_kv_cache(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute and cache K, V tensors for prefix tokens through PaliGemma layers.

        This computes K, V projections for the prefix (image + language) tokens using
        PaliGemma's projection weights. The cached K, V can be reused across all
        denoising steps since prefix content doesn't change.

        Args:
            prefix_embs: (B, prefix_len, hidden_dim) - prefix embeddings
            prefix_pad_masks: (B, prefix_len) - padding mask
            prefix_att_masks: (B, prefix_len) - attention pattern mask

        Returns:
            List of (K, V) tuples for each layer, where K and V have shape
            (B, num_kv_heads, prefix_len, head_dim)
        """
        from transformers.models.gemma import modeling_gemma

        # Get models and configs
        paligemma_lm = self.paligemma_with_expert.paligemma.language_model
        num_layers = paligemma_lm.config.num_hidden_layers

        # Convert to bfloat16 if needed
        if paligemma_lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Compute prefix-only attention mask
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Position IDs for prefix
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Store K, V for each layer
        kv_cache = []
        hidden_states = prefix_embs

        for layer_idx in range(num_layers):
            layer = paligemma_lm.layers[layer_idx]

            # Input layernorm
            normed_hidden, _ = layer.input_layernorm(hidden_states, cond=None)

            # Compute Q, K, V projections
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
            cos, sin = paligemma_lm.rotary_emb(dummy_tensor, position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            # Store K, V for this layer (after RoPE)
            kv_cache.append((key_states.clone(), value_states.clone()))

            # Complete the layer forward pass to get hidden_states for next layer
            scaling = layer.self_attn.scaling
            att_output, _ = modeling_gemma.eager_attention_forward(
                layer.self_attn, query_states, key_states, value_states,
                prefix_att_2d_masks_4d, scaling
            )

            head_dim = layer.self_attn.head_dim
            num_heads = layer.self_attn.config.num_attention_heads
            batch_size = query_states.shape[0]
            att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)

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

        return kv_cache

    def denoise_step_with_cache(
        self,
        state: torch.Tensor,
        prefix_kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one denoising step using cached prefix K, V.

        This only processes suffix (action) tokens through Gemma Expert, using the
        cached K, V from prefix for cross-attention. This is 7-8x faster than
        processing all tokens every step.

        Args:
            state: (B, state_dim) - robot state
            prefix_kv_cache: List of (K, V) tuples from compute_prefix_kv_cache
            prefix_pad_masks: (B, prefix_len) - padding mask for prefix
            x_t: (B, action_horizon, action_dim) - noisy actions
            timestep: (B,) - current timestep

        Returns:
            (B, action_horizon, action_dim) - velocity prediction
        """
        from transformers.models.gemma import modeling_gemma

        # Embed suffix (action tokens + timestep conditioning)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # Get models
        gemma_expert = self.paligemma_with_expert.gemma_expert.model
        paligemma_lm = self.paligemma_with_expert.paligemma.language_model
        num_layers = gemma_expert.config.num_hidden_layers

        batch_size = suffix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]

        # Convert to bfloat16 if needed
        if gemma_expert.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Suffix can attend to all prefix + causal within suffix
        # Compute suffix attention mask (causal within suffix)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # Suffix can attend to all valid prefix tokens
        suffix_to_prefix_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # Full attention mask: (B, suffix_len, prefix_len + suffix_len)
        full_att_masks = torch.cat([suffix_to_prefix_masks, suffix_att_2d_masks], dim=2)
        full_att_masks_4d = self._prepare_attention_masks_4d(full_att_masks)

        # Position IDs for suffix (continuing from prefix)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        hidden_states = suffix_embs

        for layer_idx in range(num_layers):
            layer = gemma_expert.layers[layer_idx]
            cached_key, cached_value = prefix_kv_cache[layer_idx]

            # Input layernorm with adaRMS conditioning
            normed_hidden, gate = layer.input_layernorm(hidden_states, cond=adarms_cond)

            # Compute Q, K, V for suffix only
            input_shape = normed_hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            query_states = layer.self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)

            # Apply RoPE to suffix Q, K
            dummy_tensor = torch.zeros(
                query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                device=query_states.device, dtype=query_states.dtype
            )
            cos, sin = paligemma_lm.rotary_emb(dummy_tensor, suffix_position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            # Concatenate cached prefix K, V with suffix K, V
            full_key_states = torch.cat([cached_key, key_states], dim=2)
            full_value_states = torch.cat([cached_value, value_states], dim=2)

            # Compute attention (suffix Q attending to prefix+suffix K, V)
            scaling = layer.self_attn.scaling
            att_output, _ = modeling_gemma.eager_attention_forward(
                layer.self_attn, query_states, full_key_states, full_value_states,
                full_att_masks_4d, scaling
            )

            head_dim = layer.self_attn.head_dim
            num_heads = layer.self_attn.config.num_attention_heads
            att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)

            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output)

            # First residual with gating
            out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gate)
            after_first_residual = out_emb.clone()

            # Post-attention layernorm with adaRMS
            out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond)
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)

            # MLP
            out_emb = layer.mlp(out_emb)

            # Second residual with gating
            hidden_states = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)

        # Final norm
        hidden_states, _ = gemma_expert.norm(hidden_states, cond=adarms_cond)

        # Extract action tokens and project
        suffix_out = hidden_states[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, use_kv_cache=True) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)

        Args:
            device: Device to run inference on
            observation: Input observation
            noise: Optional initial noise
            num_steps: Number of denoising steps
            use_kv_cache: If True, use optimized KV cache (7-8x faster). If False, use
                         no-cache implementation for debugging/validation.
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        # Get the model's dtype from the first parameter
        model_dtype = next(self.parameters()).dtype

        # Convert noise to model dtype if needed
        x_t = noise.to(model_dtype)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        if use_kv_cache:
            # Optimized path: compute prefix KV cache once, reuse for all denoising steps
            prefix_kv_cache = self.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)

            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = self.denoise_step_with_cache(
                    state,
                    prefix_kv_cache,
                    prefix_pad_masks,
                    x_t,
                    expanded_time,
                )
                x_t = x_t + dt * v_t
                time += dt
        else:
            # Fallback path: process all tokens every step (for debugging/validation)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = self.denoise_step_no_cache(
                    state,
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                    x_t,
                    expanded_time,
                )
                x_t = x_t + dt * v_t
                time += dt

        return x_t

    def denoise_step_no_cache(
        self,
        state,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        x_t,
        timestep,
    ):
        """Apply one denoising step WITHOUT KV cache - processes prefix + suffix together.

        This is the correct implementation that matches JAX behavior. The prefix and suffix
        are processed through shared attention layers, where each expert (PaliGemma for prefix,
        Gemma Expert for suffix) uses its own weights but attention is computed jointly.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]

        # Convert embeddings to bfloat16 if model weights are bfloat16
        if (
            self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Build combined attention mask for prefix + suffix
        # prefix_att_2d_masks: (B, prefix_len, prefix_len) - how prefix attends to prefix
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

        # suffix can attend to all prefix tokens
        suffix_to_prefix_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # suffix_att_2d_masks: (B, suffix_len, suffix_len) - how suffix attends to suffix
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # prefix cannot attend to suffix (causal)
        prefix_to_suffix_masks = torch.zeros(
            batch_size, prefix_len, suffix_len, dtype=torch.bool, device=prefix_pad_masks.device
        )

        # Combine into full attention mask: (B, prefix_len + suffix_len, prefix_len + suffix_len)
        # Row order: [prefix positions, suffix positions]
        # Col order: [prefix positions, suffix positions]
        top_row = torch.cat([prefix_att_2d_masks, prefix_to_suffix_masks], dim=2)  # (B, prefix_len, total_len)
        bottom_row = torch.cat([suffix_to_prefix_masks, suffix_att_2d_masks], dim=2)  # (B, suffix_len, total_len)
        full_att_2d_masks = torch.cat([top_row, bottom_row], dim=1)  # (B, total_len, total_len)

        # Position IDs: prefix positions, then suffix positions continuing from prefix
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=1)

        # Prepare attention masks for transformers format
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # Use SDPA for faster attention (was previously forced to "eager" which caused 4x slowdown)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "sdpa"  # noqa: SLF001
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "sdpa"  # noqa: SLF001

        # Forward with BOTH prefix and suffix - this uses the "else" branch in gemma_pytorch.py
        # which correctly processes both through shared attention layers
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        # Convert to action_out_proj weight dtype (may be bfloat16 or float32)
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep.

        WARNING: This method uses KV cache which has a known bug - the KV cache from
        PaliGemma is used with Gemma Expert, but these models have different K,V
        projection weights. Use denoise_step_no_cache() instead for correct behavior.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # Convert suffix_embs to bfloat16 if model weights are bfloat16
        if (
            self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)
