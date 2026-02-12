"""Hybrid Sampler: TRT FP8 Prefill + W4A16 TVM Decode.

This module implements the "Frankenstein" architecture that stitches together:
1. TRT Vision Encoder (FP16) - ~5ms
2. TRT FP8 Prefill (compute-bound) - ~40ms
3. W4A16 Linear + CUDA Graph Denoise (memory-bound) - ~30ms

Total Target: < 80ms (> 12 Hz) single-frame inference

Architecture:
```
                    ┌─────────────────────────────────────────────────┐
                    │              TensorRT Runtime                   │
                    │  ┌─────────────┐     ┌─────────────────────┐   │
   Image ──────────►│  │ TRT Vision  │────►│   TRT FP8 Prefill   │───┼──► KV Cache (FP16)
                    │  │  (SigLIP)   │     │  (2.9x vs BF16)     │   │
                    │  └─────────────┘     └─────────────────────┘   │
                    └─────────────────────────────────────────────────┘
                                                 │
                                                 │ (dtype convert: FP16 → BF16)
                                                 ▼
                    ┌─────────────────────────────────────────────────┐
                    │             PyTorch + TVM Runtime               │
                    │  ┌─────────────────────────────────────────┐   │
   State ──────────►│  │    W4A16 Decoder (StaticDenoiseLoop)    │   │
                    │  │  - W4A16 MLP (TVM kernel, 0.125ms each) │   │
                    │  │  - CUDA Graph (eliminate dispatch)       │───┼──► Actions
                    │  │  - KV Cache Reuse (prefix from TRT)     │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
```

Critical Design Decisions:
1. **Single-Frame Pipeline**: Pi0.5 requires full context update every frame.
   No split-frequency control; every frame runs the complete pipeline.

2. **Data Bridging**: TRT outputs FP16, W4A16 expects BF16. We perform
   efficient dtype conversion during KV cache handover.

3. **Graph Reuse**: StaticDenoiseLoop's CUDA Graph is captured once and
   replayed with updated KV cache pointers for each frame.

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import time

import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

logger = logging.getLogger(__name__)


@dataclass
class HybridSamplerConfig:
    """Configuration for Hybrid Sampler."""
    num_denoise_steps: int = 3
    batch_size: int = 1
    dtype: torch.dtype = torch.bfloat16

    # TRT components
    use_trt_vision: bool = True
    use_trt_prefill: bool = True
    compile_trt_on_init: bool = False  # Expensive, defer to explicit call

    # W4A16 components
    use_w4a16_decode: bool = True
    use_cuda_graph: bool = True

    # Paths
    checkpoint_dir: Optional[str] = None
    vision_engine_path: Optional[str] = None

    # Warmup
    warmup_iters: int = 3


class HybridSampler(nn.Module):
    """Hybrid Sampler with TRT FP8 Prefill + W4A16 TVM Decode.

    This class achieves maximum single-frame inference speed by combining:
    1. TRT FP8 for compute-bound prefill (2.9x faster than BF16)
    2. W4A16 TVM for memory-bound decode (optimized for batch=1)

    Usage:
        sampler = HybridSampler(model, device, config)
        sampler.compile_trt()  # One-time, ~6 min
        sampler.warm_up(sample_observation)

        # Single-frame inference
        actions = sampler.sample_actions(observation)
    """

    def __init__(
        self,
        model,  # PI0Pytorch (with W4A16 patching applied)
        device: torch.device,
        config: HybridSamplerConfig = None,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.config = config or HybridSamplerConfig()

        # Get checkpoint directory
        checkpoint_dir = self.config.checkpoint_dir
        if checkpoint_dir is None:
            from pathlib import Path
            checkpoint_dir = str(Path.home() / ".cache/openpi/checkpoints/pi05_libero")
        self.checkpoint_dir = checkpoint_dir

        # =========================================================================
        # Component 1: TRT Vision Encoder
        # =========================================================================
        self._trt_vision = None
        if self.config.use_trt_vision:
            try:
                from openpi.modules.vision_trt import VisionEncoderWrapper, get_default_engine_path
                engine_path = self.config.vision_engine_path or get_default_engine_path()
                if engine_path:
                    self._trt_vision = VisionEncoderWrapper(
                        model=model,
                        engine_path=engine_path,
                        device=device,
                        use_trt=True,
                    )
                    logger.info(f"TRT Vision Encoder loaded: {engine_path}")
            except Exception as e:
                logger.warning(f"TRT Vision Encoder not available: {e}")

        # =========================================================================
        # Component 2: TRT FP8 Prefill
        # =========================================================================
        self._trt_prefill = None
        if self.config.use_trt_prefill:
            try:
                from openpi.modules.trt_prefill import TRTPrefillWrapper
                self._trt_prefill = TRTPrefillWrapper(
                    checkpoint_dir=checkpoint_dir,
                    device=str(device),
                    compile_on_init=self.config.compile_trt_on_init,
                )
                logger.info("TRT FP8 Prefill Wrapper created")
            except Exception as e:
                logger.warning(f"TRT FP8 Prefill not available: {e}")

        # =========================================================================
        # Component 3: Static Denoise Loop (W4A16 + CUDA Graph)
        # =========================================================================
        self._graphed_denoise = None
        self._prefix_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._prefix_pad_masks: Optional[torch.Tensor] = None
        self._cached_prefix_len: int = 0

        # =========================================================================
        # Static Prefix Embedding (eager, shared)
        # =========================================================================
        from openpi.modules.static_prefill import StaticEmbedPrefix
        self._embed_prefix = StaticEmbedPrefix(
            model=model,
            max_num_images=3,
            num_img_tokens=256,
            max_lang_tokens=200,
            batch_size=self.config.batch_size,
            device=device,
            dtype=self.config.dtype,
        )

        # State
        self._warmed_up = False
        self._trt_compiled = False

        # Timing stats
        self._timing_stats = {
            'vision_ms': 0.0,
            'embed_ms': 0.0,
            'prefill_ms': 0.0,
            'denoise_ms': 0.0,
            'total_ms': 0.0,
        }

        logger.info("HybridSampler initialized")

    def compile_trt(self):
        """Compile TRT FP8 engines (expensive, ~6 min).

        This must be called before warm_up() if use_trt_prefill=True.
        """
        if self._trt_prefill is None:
            logger.warning("TRT Prefill not available")
            return False

        logger.info("Compiling TRT FP8 Prefill engines...")
        success = self._trt_prefill.compile_trt()

        if success:
            self._trt_compiled = True
            logger.info("TRT FP8 compilation complete")
        else:
            logger.warning("TRT FP8 compilation failed, will use fallback")

        return success

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
            if self._trt_vision is not None:
                emb = self._trt_vision.forward(img)
            else:
                emb = self.model.paligemma_with_expert.embed_image(img)
            embeddings.append(emb)
        return embeddings

    def warm_up(self, sample_observation):
        """Warm up all components and capture CUDA Graphs.

        Args:
            sample_observation: Representative observation for shape inference
        """
        logger.info("HybridSampler: Starting warm-up...")

        with torch.no_grad():
            # ===================================================================
            # Step 1: Preprocess observation
            # ===================================================================
            logger.info("  1. Preprocessing observation...")
            images, img_masks, lang_tokens, lang_masks, state = \
                self._preprocess_observation(sample_observation)

            # ===================================================================
            # Step 2: Warm up Vision Encoder
            # ===================================================================
            logger.info("  2. Warming up Vision Encoder...")
            if self._trt_vision is not None:
                self._trt_vision.capture_graph(batch_size=self.config.batch_size)
            image_embeddings = self._encode_vision(images)

            # ===================================================================
            # Step 3: Embed Prefix
            # ===================================================================
            logger.info("  3. Embedding prefix...")
            prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_len = \
                self._embed_prefix(image_embeddings, img_masks, lang_tokens, lang_masks)

            self._cached_prefix_len = prefix_len

            # ===================================================================
            # Step 4: Compute KV Cache (TRT FP8 or eager)
            # ===================================================================
            if self._trt_prefill is not None and self._trt_compiled:
                logger.info(f"  4. Computing KV Cache via TRT FP8 (seq_len={prefix_len})...")
                self._trt_prefill.warmup(seq_len=prefix_len)
                self._prefix_kv_cache = self._trt_prefill.forward(prefix_embs, None, None)
            else:
                logger.info(f"  4. Computing KV Cache via eager mode (seq_len={prefix_len})...")
                self._prefix_kv_cache = self.model.compute_prefix_kv_cache(
                    prefix_embs, prefix_pad_masks, prefix_att_masks
                )

            self._prefix_pad_masks = prefix_pad_masks

            # ===================================================================
            # Step 5: Create and capture Denoise Graph
            # ===================================================================
            if self.config.use_cuda_graph:
                logger.info("  5. Creating and capturing Denoise Graph...")
                from openpi.modules.static_denoise import StaticDenoiseLoop

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
        logger.info("HybridSampler: Warm-up complete!")

    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample actions using hybrid TRT + W4A16 pipeline.

        This is the main inference method that executes the full single-frame pipeline:
        1. Vision (TRT) -> Image embeddings
        2. Embed Prefix -> Prefix embeddings
        3. Prefill (TRT FP8) -> KV Cache
        4. Denoise (W4A16 + CUDA Graph) -> Actions

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

        start_event.record()

        # =====================================================================
        # Step 1: Vision Encoding
        # =====================================================================
        images, img_masks, lang_tokens, lang_masks, state = \
            self._preprocess_observation(observation)
        image_embeddings = self._encode_vision(images)

        vision_event.record()

        # =====================================================================
        # Step 2: Embed Prefix
        # =====================================================================
        prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_len = \
            self._embed_prefix(image_embeddings, img_masks, lang_tokens, lang_masks)

        embed_event.record()

        # =====================================================================
        # Step 3: Compute KV Cache (TRT FP8 or eager)
        # =====================================================================
        if self._trt_prefill is not None and self._trt_compiled:
            prefix_kv_cache = self._trt_prefill.forward(prefix_embs, None, None)
        else:
            prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )

        # Update KV cache in denoise module (critical for CUDA Graph!)
        # The graph references the original buffers; we must update in-place
        if self._graphed_denoise is not None:
            for layer_idx in range(len(prefix_kv_cache)):
                new_k, new_v = prefix_kv_cache[layer_idx]
                old_k, old_v = self._graphed_denoise._denoise_step.prefix_kv_cache[layer_idx]
                old_k.copy_(new_k)
                old_v.copy_(new_v)

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
        # Step 5: Denoise Loop (W4A16 + CUDA Graph)
        # =====================================================================
        if self._graphed_denoise is not None:
            actions = self._graphed_denoise(state, noise)
        else:
            # Fallback to eager denoise
            actions = self._sample_actions_eager(
                prefix_kv_cache, prefix_pad_masks, state, noise
            )

        end_event.record()
        torch.cuda.synchronize()

        # Record timing stats
        self._timing_stats['vision_ms'] = start_event.elapsed_time(vision_event)
        self._timing_stats['embed_ms'] = vision_event.elapsed_time(embed_event)
        self._timing_stats['prefill_ms'] = embed_event.elapsed_time(prefill_event)
        self._timing_stats['denoise_ms'] = prefill_event.elapsed_time(end_event)
        self._timing_stats['total_ms'] = start_event.elapsed_time(end_event)

        return actions

    def _sample_actions_eager(
        self,
        prefix_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback eager denoise loop."""
        # Use model's built-in denoise
        actions = noise
        for step in range(self.config.num_denoise_steps):
            timestep = torch.tensor(
                [step / self.config.num_denoise_steps],
                dtype=torch.float32, device=self.device
            ).expand(actions.shape[0])

            actions = self.model.denoise_step_with_cache(
                prefix_kv_cache,
                prefix_pad_masks,
                state,
                actions,
                timestep,
            )
        return actions

    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics from last inference."""
        return self._timing_stats.copy()

    def benchmark(
        self,
        observation,
        num_iterations: int = 50,
        warmup_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Benchmark hybrid sampler performance."""
        if not self._warmed_up:
            raise RuntimeError("Sampler not warmed up. Call warm_up() first.")

        # Warmup
        for _ in range(warmup_iterations):
            _ = self.sample_actions(observation)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = self.sample_actions(observation)
            end.record()
            torch.cuda.synchronize()

            times.append(start.elapsed_time(end))

        import numpy as np
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        hz = 1000 / mean_ms

        return {
            'mean_ms': mean_ms,
            'std_ms': std_ms,
            'hz': hz,
            'timing_breakdown': self._timing_stats.copy(),
            'trt_vision_enabled': self._trt_vision is not None,
            'trt_prefill_enabled': self._trt_prefill is not None and self._trt_compiled,
            'cuda_graph_enabled': self._graphed_denoise is not None,
        }


def create_hybrid_sampler(
    model,
    device: torch.device,
    checkpoint_dir: str = None,
    num_denoise_steps: int = 3,
    compile_trt: bool = True,
) -> HybridSampler:
    """Factory function to create HybridSampler.

    Args:
        model: PI0Pytorch model (with W4A16 patching applied)
        device: CUDA device
        checkpoint_dir: Path to model checkpoint
        num_denoise_steps: Number of denoising steps
        compile_trt: Whether to compile TRT engines

    Returns:
        Configured HybridSampler instance
    """
    config = HybridSamplerConfig(
        num_denoise_steps=num_denoise_steps,
        batch_size=1,
        dtype=torch.bfloat16,
        use_trt_vision=True,
        use_trt_prefill=True,
        use_w4a16_decode=True,
        use_cuda_graph=True,
        checkpoint_dir=checkpoint_dir,
        compile_trt_on_init=False,
    )

    sampler = HybridSampler(model, device, config)

    if compile_trt:
        sampler.compile_trt()

    return sampler
