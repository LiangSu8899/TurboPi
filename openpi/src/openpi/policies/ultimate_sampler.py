"""Ultimate Sampler: Full Pipeline with Minimum Latency.

This module provides the fastest possible end-to-end inference by combining:
1. TRT Vision Encoder (~5ms)
2. Static Prefill with CUDA Graph (~15ms)
3. Static Denoise Loop with CUDA Graph (~29ms)

Target: Total < 60ms (> 15 Hz) for full inference
        Denoise-only: ~29ms (34 Hz) with KV cache reuse

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
import time
import logging

from openpi.modules.vision_trt import VisionEncoderWrapper, get_default_engine_path
from openpi.modules.static_prefill import StaticPrefill, StaticEmbedPrefix
from openpi.modules.static_denoise import StaticDenoiseLoop
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

logger = logging.getLogger(__name__)


@dataclass
class UltimateSamplerConfig:
    """Configuration for ultimate sampler."""
    num_denoise_steps: int = 3
    max_prefix_len: int = 1024
    batch_size: int = 1
    dtype: torch.dtype = torch.bfloat16
    use_trt_vision: bool = True
    use_prefill_graph: bool = True
    use_denoise_graph: bool = True
    warmup_iters: int = 3
    vision_engine_path: Optional[str] = None


class UltimateSampler(nn.Module):
    """Ultimate fast sampler with full CUDA Graph pipeline.

    This class achieves maximum inference speed by:
    1. Using TRT for Vision Encoding (~5ms)
    2. Using CUDA Graph for Prefill (~15ms)
    3. Using CUDA Graph for Denoise (~29ms)

    Total: ~50ms = 20 Hz (vs baseline 120ms = 8.3 Hz)

    Usage:
        sampler = UltimateSampler(model, device, config)
        sampler.warm_up(sample_observation)  # One-time setup

        # Full inference (20 Hz)
        actions = sampler.sample_actions(observation)

        # Denoise-only with KV reuse (34 Hz)
        actions = sampler.sample_actions_denoise_only(state)
    """

    def __init__(
        self,
        model,  # PI0Pytorch with W4A16 quantization applied
        device: torch.device,
        config: UltimateSamplerConfig = None,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.config = config or UltimateSamplerConfig()

        # Get model configs
        paligemma_lm = model.paligemma_with_expert.paligemma.language_model
        expert = model.paligemma_with_expert.gemma_expert.model
        lm_config = paligemma_lm.config

        # =========================================================================
        # Component 1: TRT Vision Encoder
        # =========================================================================
        engine_path = self.config.vision_engine_path or get_default_engine_path()
        self._vision_encoder: Optional[VisionEncoderWrapper] = None

        if self.config.use_trt_vision and engine_path:
            try:
                self._vision_encoder = VisionEncoderWrapper(
                    model=model,
                    engine_path=engine_path,
                    device=device,
                    use_trt=True,
                )
                logger.info(f"Using TRT Vision Encoder: {engine_path}")
            except Exception as e:
                logger.warning(f"TRT Vision Encoder failed: {e}")
                self._vision_encoder = None

        # =========================================================================
        # Component 2: Static Prefix Embedding
        # =========================================================================
        self._embed_prefix = StaticEmbedPrefix(
            model=model,
            max_num_images=3,  # base, left_wrist, right_wrist
            num_img_tokens=256,  # 16x16 patches
            max_lang_tokens=200,
            batch_size=self.config.batch_size,
            device=device,
            dtype=self.config.dtype,
        )

        # =========================================================================
        # Component 3: Static Prefill (CUDA Graph)
        # =========================================================================
        self._static_prefill: Optional[StaticPrefill] = None
        if self.config.use_prefill_graph:
            self._static_prefill = StaticPrefill(
                model=model,
                max_prefix_len=self.config.max_prefix_len,
                batch_size=self.config.batch_size,
                device=device,
                dtype=self.config.dtype,
            )

        # =========================================================================
        # Component 4: Static Denoise Loop (CUDA Graph)
        # =========================================================================
        self._graphed_denoise: Optional[StaticDenoiseLoop] = None
        self._prefix_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._prefix_pad_masks: Optional[torch.Tensor] = None

        # State
        self._warmed_up = False
        self._captured_prefix_len: int = 0

        # Timing stats
        self._timing_enabled = True
        self._timing_stats = {
            'vision_ms': 0.0,
            'embed_ms': 0.0,
            'prefill_ms': 0.0,
            'denoise_ms': 0.0,
            'total_ms': 0.0,
        }

    def _preprocess_observation(self, observation):
        """Preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def _encode_vision(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        """Encode images using TRT or eager mode."""
        embeddings = []
        for img in images:
            if self._vision_encoder is not None:
                emb = self._vision_encoder.forward(img)
            else:
                # Fallback to eager mode
                emb = self.model.paligemma_with_expert.embed_image(img)
            embeddings.append(emb)
        return embeddings

    def warm_up(self, sample_observation):
        """Warm up and capture all CUDA Graphs.

        This must be called once before inference with a representative observation.
        """
        logger.info("UltimateSampler: Starting warm-up...")

        with torch.no_grad():
            # ===================================================================
            # Step 1: Preprocess observation
            # ===================================================================
            logger.info("  1. Preprocessing observation...")
            images, img_masks, lang_tokens, lang_masks, state = \
                self._preprocess_observation(sample_observation)

            # ===================================================================
            # Step 2: Encode vision (TRT or eager)
            # ===================================================================
            logger.info("  2. Encoding vision...")
            if self._vision_encoder is not None:
                self._vision_encoder.capture_graph(batch_size=self.config.batch_size)

            image_embeddings = self._encode_vision(images)

            # ===================================================================
            # Step 3: Embed prefix
            # ===================================================================
            logger.info("  3. Embedding prefix...")
            prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_len = \
                self._embed_prefix(image_embeddings, img_masks, lang_tokens, lang_masks)

            self._captured_prefix_len = prefix_len

            # ===================================================================
            # Step 4: Capture Prefill Graph
            # ===================================================================
            if self._static_prefill is not None and self.config.use_prefill_graph:
                logger.info(f"  4. Capturing Prefill Graph (len={prefix_len})...")

                # Pad to captured length if needed
                padded_embs = torch.zeros(
                    self.config.batch_size, prefix_len, prefix_embs.shape[-1],
                    dtype=self.config.dtype, device=self.device
                )
                padded_embs[:, :prefix_embs.shape[1]].copy_(prefix_embs)

                self._static_prefill.capture_graph(
                    seq_len=prefix_len,
                    warmup_iters=self.config.warmup_iters
                )

                # Compute initial KV cache
                self._prefix_kv_cache = self._static_prefill(padded_embs)
            else:
                # Eager prefill
                logger.info("  4. Computing Prefill (eager)...")
                self._prefix_kv_cache = self.model.compute_prefix_kv_cache(
                    prefix_embs, prefix_pad_masks, prefix_att_masks
                )

            self._prefix_pad_masks = prefix_pad_masks

            # ===================================================================
            # Step 5: Capture Denoise Graph
            # ===================================================================
            if self.config.use_denoise_graph:
                logger.info("  5. Capturing Denoise Graph...")
                self._graphed_denoise = StaticDenoiseLoop(
                    model=self.model,
                    prefix_kv_cache=self._prefix_kv_cache,
                    prefix_pad_masks=self._prefix_pad_masks,
                    num_steps=self.config.num_denoise_steps,
                    batch_size=self.config.batch_size,
                    device=self.device,
                    dtype=self.config.dtype,
                )
                self._graphed_denoise.capture_graph(warmup_iters=self.config.warmup_iters)

        self._warmed_up = True
        logger.info("UltimateSampler: Warm-up complete!")

    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full inference with TRT Vision + Graph Prefill + Graph Denoise.

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
        embed_event = torch.cuda.Event(enable_timing=True)
        prefill_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        if self._timing_enabled:
            start_event.record()

        # =====================================================================
        # Step 1: Vision Encoding (TRT ~5ms or Eager ~30ms)
        # =====================================================================
        images, img_masks, lang_tokens, lang_masks, state = \
            self._preprocess_observation(observation)
        image_embeddings = self._encode_vision(images)

        if self._timing_enabled:
            vision_event.record()

        # =====================================================================
        # Step 2: Embed Prefix
        # =====================================================================
        prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_len = \
            self._embed_prefix(image_embeddings, img_masks, lang_tokens, lang_masks)

        if self._timing_enabled:
            embed_event.record()

        # =====================================================================
        # Step 3: Prefill (Graph ~15ms or Eager ~138ms)
        # =====================================================================
        if self._static_prefill is not None and self._static_prefill._graph_captured:
            # Pad to captured length
            if prefix_len != self._captured_prefix_len:
                # Sequence length changed - need to use eager or recapture
                logger.warning(f"Prefix length changed: {prefix_len} vs {self._captured_prefix_len}, using eager")
                prefix_kv_cache = self.model.compute_prefix_kv_cache(
                    prefix_embs, prefix_pad_masks, prefix_att_masks
                )
            else:
                prefix_kv_cache = self._static_prefill(prefix_embs)
        else:
            prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )

        # Update KV cache in denoise module (required for CUDA Graph)
        for layer_idx in range(len(prefix_kv_cache)):
            new_k, new_v = prefix_kv_cache[layer_idx]
            old_k, old_v = self._graphed_denoise._denoise_step.prefix_kv_cache[layer_idx]
            old_k.copy_(new_k)
            old_v.copy_(new_v)

        if self._timing_enabled:
            prefill_event.record()

        # =====================================================================
        # Step 4: Initialize Noise
        # =====================================================================
        batch_size = observation.state.shape[0]
        if noise is None:
            actions_shape = (batch_size, self.model.config.action_horizon, self.model.config.action_dim)
            noise = self.model.sample_noise(actions_shape, self.device)

        model_dtype = next(self.model.parameters()).dtype
        noise = noise.to(model_dtype)

        # =====================================================================
        # Step 5: Denoise Loop (Graph ~29ms)
        # =====================================================================
        actions = self._graphed_denoise(state, noise)

        if self._timing_enabled:
            end_event.record()
            torch.cuda.synchronize()

            self._timing_stats['vision_ms'] = start_event.elapsed_time(vision_event)
            self._timing_stats['embed_ms'] = vision_event.elapsed_time(embed_event)
            self._timing_stats['prefill_ms'] = embed_event.elapsed_time(prefill_event)
            self._timing_stats['denoise_ms'] = prefill_event.elapsed_time(end_event)
            self._timing_stats['total_ms'] = start_event.elapsed_time(end_event)

        return actions

    @torch.no_grad()
    def sample_actions_denoise_only(
        self,
        state: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fast inference with KV cache reuse (denoise only).

        Use this when the scene hasn't changed significantly.
        Achieves maximum speed: ~29ms = 34 Hz.

        Args:
            state: (batch, state_dim) robot state
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

        return self._graphed_denoise(state, noise)

    def get_timing_stats(self) -> dict:
        """Get timing statistics from last inference."""
        return self._timing_stats.copy()

    def enable_timing(self, enabled: bool = True):
        """Enable/disable timing instrumentation."""
        self._timing_enabled = enabled


def create_ultimate_sampler(
    model,
    device: torch.device,
    num_denoise_steps: int = 3,
    use_trt_vision: bool = True,
) -> UltimateSampler:
    """Factory function to create UltimateSampler.

    Args:
        model: PI0Pytorch model (with W4A16 quantization applied)
        device: CUDA device
        num_denoise_steps: Number of denoising steps
        use_trt_vision: Whether to use TRT for vision encoding

    Returns:
        Configured UltimateSampler instance
    """
    config = UltimateSamplerConfig(
        num_denoise_steps=num_denoise_steps,
        batch_size=1,
        dtype=torch.bfloat16,
        use_trt_vision=use_trt_vision,
        use_prefill_graph=True,
        use_denoise_graph=True,
    )
    return UltimateSampler(model, device, config)
