"""Zero-Python Fast Sampler with CUDA Graphs.

This module provides ultra-fast inference by minimizing Python intervention:
1. Vision encoding (eager or TRT)
2. VLM Prefill (eager, computed once)
3. VLM Decode (CUDA Graph captured)
4. Denoising Loop (CUDA Graph captured)

Target: 36 Hz (27.5ms) vs TRT FP8 baseline 8.3 Hz (120ms)

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import time

from openpi.modules.static_vlm import (
    StaticKVCache,
    StaticKVCacheConfig,
)
from openpi.modules.static_denoise import (
    StaticDenoiseLoop,
)


@dataclass
class FastSamplerConfig:
    """Configuration for fast sampler."""
    num_denoise_steps: int = 3
    max_seq_len: int = 1024
    batch_size: int = 1
    dtype: torch.dtype = torch.bfloat16
    use_vision_graph: bool = False  # Optional: capture vision encoder too
    warmup_iters: int = 3


class FastSampler(nn.Module):
    """Ultra-fast action sampler with CUDA Graphs.

    This class orchestrates the complete inference pipeline with minimal
    Python overhead:

    Workflow:
    1. Vision Encoder: Process images (eager for now, optionally graphed)
    2. VLM Prefill: Compute prefix embeddings and KV cache (once per frame)
    3. Denoising Loop: N steps of denoise (CUDA Graph captured)

    The key optimization is capturing the denoise loop as a single CUDA Graph,
    eliminating the 168ms Python dispatch overhead observed in eager mode.
    """

    def __init__(
        self,
        model,  # PI0Pytorch with W4A16 quantization applied
        device: torch.device,
        config: FastSamplerConfig = None,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.config = config or FastSamplerConfig()

        # Get model configs
        paligemma_lm = model.paligemma_with_expert.paligemma.language_model
        expert = model.paligemma_with_expert.gemma_expert.model
        lm_config = paligemma_lm.config

        # Create static KV cache
        cache_config = StaticKVCacheConfig(
            num_layers=lm_config.num_hidden_layers,
            num_kv_heads=lm_config.num_key_value_heads,
            head_dim=lm_config.head_dim,
            max_seq_len=self.config.max_seq_len,
            batch_size=self.config.batch_size,
            dtype=self.config.dtype,
        )
        self.static_kv_cache = StaticKVCache(cache_config, device)

        # Graph components (initialized during warm_up)
        self._graphed_denoise: Optional[StaticDenoiseLoop] = None
        self._prefix_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._prefix_pad_masks: Optional[torch.Tensor] = None

        # State
        self._warmed_up = False

        # Timing stats
        self._timing_enabled = True
        self._timing_stats = {
            'vision_ms': 0.0,
            'prefill_ms': 0.0,
            'denoise_ms': 0.0,
            'total_ms': 0.0,
        }

    def _preprocess_and_embed_prefix(self, observation) -> Tuple[
        torch.Tensor,  # prefix_embs
        torch.Tensor,  # prefix_pad_masks
        torch.Tensor,  # prefix_att_masks
        torch.Tensor,  # state
    ]:
        """Process observation and embed prefix (vision + language).

        This is the "VLM Prefill" step - run once per frame.
        """
        images, img_masks, lang_tokens, lang_masks, state = \
            self.model._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = \
            self.model.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        return prefix_embs, prefix_pad_masks, prefix_att_masks, state

    def warm_up(self, sample_observation):
        """Warm up and capture CUDA Graphs.

        This must be called once with a sample observation before inference.
        The observation should have the same structure as production inputs.

        Args:
            sample_observation: Sample Observation for shape inference
        """
        print("FastSampler: Starting warm-up...")

        with torch.no_grad():
            # 1. Embed prefix
            print("  1. Embedding prefix...")
            prefix_embs, prefix_pad_masks, prefix_att_masks, state = \
                self._preprocess_and_embed_prefix(sample_observation)

            # 2. Compute prefix KV cache
            print("  2. Computing prefix KV cache...")
            self._prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )
            self._prefix_pad_masks = prefix_pad_masks

            # Write to static cache
            self.static_kv_cache.write_prefix_cache(self._prefix_kv_cache)

            # 3. Create static denoise loop
            print("  3. Creating static denoise loop...")
            self._graphed_denoise = StaticDenoiseLoop(
                model=self.model,
                prefix_kv_cache=self._prefix_kv_cache,
                prefix_pad_masks=self._prefix_pad_masks,
                num_steps=self.config.num_denoise_steps,
                batch_size=self.config.batch_size,
                device=self.device,
                dtype=self.config.dtype,
            )

            # 4. Capture graph
            print("  4. Capturing CUDA Graph...")
            self._graphed_denoise.capture_graph(warmup_iters=self.config.warmup_iters)

        self._warmed_up = True
        print("FastSampler: Warm-up complete!")

    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fast action sampling with CUDA Graphs.

        This is the main inference entry point. Flow:
        1. Vision + Prefill (eager, ~15ms)
        2. Denoise Loop (graphed, ~12ms)
        Total: ~27ms = 37 Hz

        Args:
            observation: Input Observation
            noise: Optional initial noise

        Returns:
            Denoised actions (batch, action_horizon, action_dim)
        """
        if not self._warmed_up:
            raise RuntimeError("Sampler not warmed up. Call warm_up() first.")

        start_event = torch.cuda.Event(enable_timing=True)
        vision_event = torch.cuda.Event(enable_timing=True)
        prefill_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        if self._timing_enabled:
            start_event.record()

        # === Step 1: Vision + Language Embedding ===
        prefix_embs, prefix_pad_masks, prefix_att_masks, state = \
            self._preprocess_and_embed_prefix(observation)

        if self._timing_enabled:
            vision_event.record()

        # === Step 2: Compute KV Cache ===
        prefix_kv_cache = self.model.compute_prefix_kv_cache(
            prefix_embs, prefix_pad_masks, prefix_att_masks
        )

        # Update KV cache in-place (required for CUDA Graph)
        # The graph was captured with references to _denoise_step.prefix_kv_cache
        # We must copy new values into those same buffers
        for layer_idx in range(len(prefix_kv_cache)):
            new_k, new_v = prefix_kv_cache[layer_idx]
            old_k, old_v = self._graphed_denoise._denoise_step.prefix_kv_cache[layer_idx]
            old_k.copy_(new_k)
            old_v.copy_(new_v)

        if self._timing_enabled:
            prefill_event.record()

        # === Step 3: Initialize noise ===
        batch_size = observation.state.shape[0]
        if noise is None:
            actions_shape = (batch_size, self.model.config.action_horizon, self.model.config.action_dim)
            noise = self.model.sample_noise(actions_shape, self.device)

        model_dtype = next(self.model.parameters()).dtype
        noise = noise.to(model_dtype)

        # === Step 4: Denoising Loop (CUDA Graph) ===
        actions = self._graphed_denoise(state, noise)

        if self._timing_enabled:
            end_event.record()
            torch.cuda.synchronize()

            self._timing_stats['vision_ms'] = start_event.elapsed_time(vision_event)
            self._timing_stats['prefill_ms'] = vision_event.elapsed_time(prefill_event)
            self._timing_stats['denoise_ms'] = prefill_event.elapsed_time(end_event)
            self._timing_stats['total_ms'] = start_event.elapsed_time(end_event)

        return actions

    @torch.no_grad()
    def sample_actions_with_prefix_reuse(
        self,
        state: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Even faster inference by reusing prefix from previous frame.

        If the scene hasn't changed significantly, we can skip vision encoding
        and prefix computation, reusing the cached KV from warm_up().

        This achieves maximum speed: only denoising loop runs.

        Args:
            state: (batch, state_dim) - robot state
            noise: Optional initial noise

        Returns:
            Denoised actions
        """
        if not self._warmed_up:
            raise RuntimeError("Sampler not warmed up. Call warm_up() first.")

        batch_size = state.shape[0]
        if noise is None:
            actions_shape = (batch_size, self.model.config.action_horizon, self.model.config.action_dim)
            noise = self.model.sample_noise(actions_shape, self.device)

        model_dtype = next(self.model.parameters()).dtype
        noise = noise.to(model_dtype)

        # Only run denoising (skip vision + prefill)
        return self._graphed_denoise(state, noise)

    def get_timing_stats(self) -> dict:
        """Get timing statistics from last inference."""
        return self._timing_stats.copy()

    def enable_timing(self, enabled: bool = True):
        """Enable/disable timing instrumentation."""
        self._timing_enabled = enabled


class EagerFastSampler(nn.Module):
    """Eager-mode fast sampler for comparison/debugging.

    This uses the same structure as FastSampler but without CUDA Graphs.
    Useful for:
    - Validating correctness before graph capture
    - Debugging issues with graph capture
    - Measuring baseline eager performance
    """

    def __init__(
        self,
        model,
        device: torch.device,
        num_denoise_steps: int = 3,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.num_denoise_steps = num_denoise_steps

    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eager-mode inference for comparison."""
        return self.model.sample_actions(
            self.device,
            observation,
            noise=noise,
            num_steps=self.num_denoise_steps,
            use_kv_cache=True,
        )


def create_fast_sampler(
    model,
    device: torch.device,
    num_denoise_steps: int = 3,
) -> FastSampler:
    """Factory function to create FastSampler with default config.

    Args:
        model: PI0Pytorch model (with W4A16 quantization applied)
        device: CUDA device
        num_denoise_steps: Number of denoising steps

    Returns:
        Configured FastSampler instance
    """
    config = FastSamplerConfig(
        num_denoise_steps=num_denoise_steps,
        batch_size=1,
        dtype=torch.bfloat16,
    )
    return FastSampler(model, device, config)
