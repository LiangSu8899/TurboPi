"""
Patch transformers library to use custom Gemma modules with adaptive RMSNorm support.

This module should be imported BEFORE any transformers models are created.
It patches the transformers.models.gemma module with our custom implementation
that supports adaptive RMSNorm (adaRMS) for the Pi0.5 action expert.

Usage:
    # At the top of your script, before importing any models:
    from openpi.models_pytorch import transformers_replace
    transformers_replace.patch_transformers()

    # Then import and use transformers models normally
    from transformers import GemmaForCausalLM
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def patch_transformers():
    """
    Patch transformers library to use custom Gemma modules.

    This function replaces the transformers.models.gemma module with our custom
    implementation that includes:
    - GemmaRMSNorm with adaptive normalization support (cond_dim)
    - _gated_residual function for conditional residual connections
    - adarms_cond parameter support throughout the model
    - ONNX-compatible weight attribute (always defined)

    Must be called before creating any Gemma models.
    """
    import importlib

    # Get the path to our custom modules
    custom_path = Path(__file__).parent

    # First, we need to ensure transformers is imported
    import transformers

    # Patch the GemmaConfig to support our custom attributes
    _patch_gemma_config()

    # Patch the GemmaRMSNorm class
    _patch_gemma_rmsnorm()

    # Patch the _gated_residual function
    _patch_gated_residual()

    # Patch the GemmaDecoderLayer to support adarms_cond
    _patch_gemma_decoder_layer()

    # Patch the GemmaModel to support adarms_cond
    _patch_gemma_model()

    logger.info("Transformers library patched with custom Gemma modules")


def _patch_gemma_config():
    """Patch GemmaConfig to support use_adarms and adarms_cond_dim."""
    from transformers.models.gemma import configuration_gemma

    original_init = configuration_gemma.GemmaConfig.__init__

    def patched_init(self, *args, use_adarms=False, adarms_cond_dim=None, **kwargs):
        original_init(self, *args, **kwargs)
        self.use_adarms = use_adarms
        self.adarms_cond_dim = adarms_cond_dim
        if self.use_adarms and self.adarms_cond_dim is None:
            self.adarms_cond_dim = self.hidden_size

    configuration_gemma.GemmaConfig.__init__ = patched_init
    logger.debug("Patched GemmaConfig.__init__")


def _patch_gemma_rmsnorm():
    """Patch GemmaRMSNorm to support adaptive normalization and ONNX export."""
    import torch
    from torch import nn
    from transformers.models.gemma import modeling_gemma
    from typing import Optional

    class GemmaRMSNorm(nn.Module):
        """Custom GemmaRMSNorm with adaptive normalization support."""

        def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
            super().__init__()
            self.eps = eps
            self.dim = dim
            self.cond_dim = cond_dim
            self._use_adarms = cond_dim is not None

            # Dense layer for adaptive normalization (if cond_dim is provided)
            if cond_dim is not None:
                self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
                nn.init.zeros_(self.dense.weight)
                # Use buffer (not parameter) to satisfy transformers init but not save to state_dict
                self.register_buffer("weight", torch.ones(dim), persistent=False)
            else:
                # Standard RMSNorm weight parameter
                self.weight = nn.Parameter(torch.zeros(dim))
                self.dense = None

        def _norm(self, x):
            var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
            normed_inputs = x * torch.rsqrt(var + self.eps)
            return normed_inputs

        def forward(self, x, cond=None):
            dtype = x.dtype
            normed_inputs = self._norm(x)

            if not self._use_adarms:
                # regular RMSNorm (standard Gemma layers)
                normed_inputs = normed_inputs * (1.0 + self.weight.float())
                return normed_inputs.to(dtype), None

            if cond is None:
                # adaRMS layer but no conditioning provided - just return normalized
                return normed_inputs.to(dtype), None

            # adaptive RMSNorm with conditioning
            if cond.shape[-1] != self.cond_dim:
                raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

            # Convert cond to dense weight dtype for computation
            weight_dtype = self.dense.weight.dtype
            modulation = self.dense(cond.to(weight_dtype))
            if len(x.shape) == 3:
                modulation = modulation.unsqueeze(1)

            scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
            normed_inputs = normed_inputs * (1 + scale) + shift

            return normed_inputs.to(dtype), gate.to(dtype)

        def extra_repr(self):
            repr_str = f"dim={self.dim}, eps={self.eps}"
            if self._use_adarms:
                repr_str += f", adaptive=True, cond_dim={self.cond_dim}"
            return repr_str

    # Replace the class in the module
    modeling_gemma.GemmaRMSNorm = GemmaRMSNorm
    logger.debug("Patched GemmaRMSNorm")


def _patch_gated_residual():
    """Add _gated_residual function to modeling_gemma."""
    from transformers.models.gemma import modeling_gemma

    def _gated_residual(x, y, gate):
        """Gated residual connection.

        Note: gate is used directly without sigmoid, matching JAX implementation.
        The gate values come from the adaRMS dense layer output which is already
        in the appropriate range from training.
        """
        if gate is None:
            return x + y
        return x + y * gate

    modeling_gemma._gated_residual = _gated_residual
    logger.debug("Patched _gated_residual")


def _patch_gemma_decoder_layer():
    """Patch GemmaDecoderLayer to support adarms_cond."""
    import torch
    from transformers.models.gemma import modeling_gemma

    original_layer_init = modeling_gemma.GemmaDecoderLayer.__init__

    def patched_layer_init(self, config, layer_idx: int):
        original_layer_init(self, config, layer_idx)

        # Replace norms with our custom version if adarms is enabled
        cond_dim = getattr(config, 'adarms_cond_dim', None) if getattr(config, 'use_adarms', False) else None

        if cond_dim is not None:
            self.input_layernorm = modeling_gemma.GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )
            self.post_attention_layernorm = modeling_gemma.GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )

    modeling_gemma.GemmaDecoderLayer.__init__ = patched_layer_init

    # Patch forward method
    original_layer_forward = modeling_gemma.GemmaDecoderLayer.forward

    def patched_layer_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        adarms_cond=None,
        **kwargs,
    ):
        residual = hidden_states

        # Input layernorm with optional adaptive normalization
        norm_result = self.input_layernorm(hidden_states, adarms_cond if hasattr(self.input_layernorm, 'cond_dim') else None)
        if isinstance(norm_result, tuple):
            hidden_states, gate = norm_result
        else:
            hidden_states, gate = norm_result, None

        # Self Attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if isinstance(attn_output, tuple):
            hidden_states = attn_output[0]
            self_attn_weights = attn_output[1] if len(attn_output) > 1 else None
        else:
            hidden_states = attn_output
            self_attn_weights = None

        hidden_states = modeling_gemma._gated_residual(residual, hidden_states, gate)

        # MLP
        residual = hidden_states
        norm_result = self.post_attention_layernorm(hidden_states, adarms_cond if hasattr(self.post_attention_layernorm, 'cond_dim') else None)
        if isinstance(norm_result, tuple):
            hidden_states, gate = norm_result
        else:
            hidden_states, gate = norm_result, None

        hidden_states = self.mlp(hidden_states)
        hidden_states = modeling_gemma._gated_residual(residual, hidden_states, gate)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

    modeling_gemma.GemmaDecoderLayer.forward = patched_layer_forward
    logger.debug("Patched GemmaDecoderLayer")


def _patch_gemma_model():
    """Patch GemmaModel to support adarms_cond."""
    from transformers.models.gemma import modeling_gemma

    original_model_init = modeling_gemma.GemmaModel.__init__

    def patched_model_init(self, config):
        original_model_init(self, config)

        # Replace final norm with our custom version if adarms is enabled
        cond_dim = getattr(config, 'adarms_cond_dim', None) if getattr(config, 'use_adarms', False) else None

        if cond_dim is not None:
            self.norm = modeling_gemma.GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )

    modeling_gemma.GemmaModel.__init__ = patched_model_init

    # Patch forward method to pass adarms_cond to layers
    original_model_forward = modeling_gemma.GemmaModel.forward

    def patched_model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        adarms_cond=None,
        **kwargs,
    ):
        # If adarms_cond is provided, use our custom forward logic
        if adarms_cond is not None:
            return _custom_gemma_model_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                adarms_cond=adarms_cond,
                **kwargs,
            )

        # Otherwise use original forward
        return original_model_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

    modeling_gemma.GemmaModel.forward = patched_model_forward
    logger.debug("Patched GemmaModel")


def _create_causal_mask(batch_size, seq_len, device, dtype):
    """Create a causal attention mask for decoder models."""
    import torch
    # Create causal mask: (batch, 1, seq_len, seq_len)
    # Lower triangular matrix with -inf for masked positions
    mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
        diagonal=1
    )
    # Expand to 4D: (batch, 1, seq, seq)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    return mask


def _custom_gemma_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    adarms_cond=None,
    **kwargs,
):
    """Custom forward pass for GemmaModel with adarms_cond support."""
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.gemma import modeling_gemma

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # Normalize embeddings
    hidden_states = inputs_embeds * (self.config.hidden_size ** 0.5)

    if cache_position is None:
        cache_position = torch.arange(
            0, hidden_states.shape[1], device=hidden_states.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # Use provided attention_mask if available, otherwise create causal mask
    # The caller (Pi0.5) provides the correct attention mask for cross-attention patterns
    if attention_mask is None:
        batch_size, seq_len = hidden_states.shape[:2]
        causal_mask = _create_causal_mask(batch_size, seq_len, hidden_states.device, hidden_states.dtype)
    else:
        # Use the provided mask directly
        causal_mask = attention_mask

    # Get position embeddings
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # Iterate through layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for layer in self.layers:
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_outputs = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            adarms_cond=adarms_cond,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns = all_self_attns + (layer_outputs[1],)

    # Final norm with optional adaptive normalization
    norm_result = self.norm(hidden_states, adarms_cond if hasattr(self.norm, 'cond_dim') else None)
    if isinstance(norm_result, tuple):
        hidden_states = norm_result[0]
    else:
        hidden_states = norm_result

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# Auto-patch when module is imported
_patched = False

def ensure_patched():
    """Ensure transformers is patched (idempotent)."""
    global _patched
    if not _patched:
        patch_transformers()
        _patched = True
