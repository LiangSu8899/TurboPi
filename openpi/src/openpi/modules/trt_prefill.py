"""TRT FP8 Prefill Wrapper for Hybrid Pipeline.

This module provides a wrapper for the Torch-TRT FP8 KV Cache Engine,
enabling seamless integration with the W4A16 decode path.

Architecture:
1. Input: Prefix embeddings (from vision + language tokens)
2. Process: TRT FP8 compiled transformer layers (2.9x faster than BF16)
3. Output: KV Cache for W4A16 decoder consumption

Data Bridging:
- TRT outputs FP16 tensors
- W4A16 expects BF16 tensors
- Zero-copy is not possible across runtimes; efficient copy is used

Performance Target:
- TRT FP8 Prefill: ~40ms (vs 120ms BF16)
- Combined with W4A16 Decode: <50ms total

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import TRT FP8 engine
try:
    from openpi.inference.torch_trt_fp8_kv_cache import (
        TorchTRTFP8KVCacheEngine,
        TorchTRTFP8KVCacheModel,
        load_weights,
        compile_trt_fp8_mlps,
        HIDDEN_SIZE,
        NUM_LAYERS,
        SEQ_LEN,
        HAS_TORCH_TRT,
        HAS_MODELOPT,
    )
    HAS_TRT_PREFILL = HAS_TORCH_TRT and HAS_MODELOPT
except ImportError:
    HAS_TRT_PREFILL = False
    logger.warning("TRT FP8 Prefill not available")


class TRTPrefillWrapper(nn.Module):
    """TRT FP8 Prefill Engine Wrapper.

    This class wraps the TRT FP8 KV Cache Engine to provide a clean interface
    for the hybrid pipeline. It handles:

    1. Input preparation (embeddings -> TRT format)
    2. TRT FP8 execution
    3. Output conversion (TRT FP16 -> PyTorch BF16 KV Cache)

    Usage:
        wrapper = TRTPrefillWrapper(checkpoint_dir)
        wrapper.compile_trt()  # One-time compilation (~6 min)
        wrapper.warmup()

        kv_cache = wrapper.forward(prefix_embeddings)  # Fast inference
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        max_seq_len: int = 1024,
        dtype: torch.dtype = torch.bfloat16,
        compile_on_init: bool = False,
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        if not HAS_TRT_PREFILL:
            logger.warning("TRT FP8 not available, will use fallback")
            self._engine = None
            self._trt_available = False
            return

        self._trt_available = True

        # Create the TRT engine
        # Note: compile_trt=False to defer compilation (expensive)
        if compile_on_init:
            logger.info("Creating TRT FP8 Prefill Engine with compilation...")
            self._engine = TorchTRTFP8KVCacheEngine(
                checkpoint_dir=checkpoint_dir,
                device=device,
                compile_trt=True,
            )
        else:
            logger.info("Creating TRT FP8 Prefill Engine (deferred compilation)...")
            self._engine = TorchTRTFP8KVCacheEngine(
                checkpoint_dir=checkpoint_dir,
                device=device,
                compile_trt=False,  # Defer to compile_trt() call
            )

        # Pre-allocate output buffers for efficient copy
        self._kv_cache_buffer: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._cached_seq_len: int = 0

        logger.info("TRT Prefill Wrapper initialized")

    def compile_trt(self):
        """Compile TRT FP8 engines for all layers.

        This is expensive (~20s per layer, ~6 minutes for 18 layers).
        Should be called once during setup.
        """
        if not self._trt_available:
            logger.warning("TRT not available, skipping compilation")
            return False

        if self._engine._trt_compiled_count > 0:
            logger.info("TRT already compiled")
            return True

        logger.info("Compiling TRT FP8 MLPs for all layers...")
        try:
            trt_mlps = compile_trt_fp8_mlps(self._engine.model, self.device)
            if trt_mlps is not None:
                for i, layer in enumerate(self._engine.model.layers):
                    if trt_mlps[i] is not None:
                        layer._trt_mlp = trt_mlps[i]
                        self._engine._trt_compiled_count += 1
            logger.info(f"TRT compilation complete: {self._engine._trt_compiled_count}/{NUM_LAYERS} layers")
            return True
        except Exception as e:
            logger.error(f"TRT compilation failed: {e}")
            return False

    def warmup(self, seq_len: int = None, num_iterations: int = 5):
        """Warmup the TRT engine."""
        if not self._trt_available:
            return

        seq_len = seq_len or SEQ_LEN
        logger.info(f"Warming up TRT Prefill (seq_len={seq_len})...")

        dummy_hidden = torch.randn(
            1, seq_len, HIDDEN_SIZE,
            device=self.device, dtype=torch.float16
        )
        dummy_position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self._engine.infer(dummy_hidden, None, dummy_position_ids)

        torch.cuda.synchronize()
        logger.info("TRT Prefill warmup complete")

    def _allocate_kv_cache_buffer(self, seq_len: int, batch_size: int = 1):
        """Pre-allocate KV cache buffers for efficient copy."""
        if self._cached_seq_len == seq_len and self._kv_cache_buffer is not None:
            return

        logger.info(f"Allocating KV cache buffer (seq_len={seq_len})")

        # Get dimensions from engine
        num_kv_heads = 1  # GQA with 1 KV head
        head_dim = 256

        self._kv_cache_buffer = []
        for _ in range(NUM_LAYERS):
            k = torch.zeros(
                batch_size, num_kv_heads, seq_len, head_dim,
                dtype=self.dtype, device=self.device
            )
            v = torch.zeros(
                batch_size, num_kv_heads, seq_len, head_dim,
                dtype=self.dtype, device=self.device
            )
            self._kv_cache_buffer.append((k, v))

        self._cached_seq_len = seq_len

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Run TRT FP8 Prefill and return KV cache.

        Args:
            hidden_states: (batch, seq_len, hidden_size) prefix embeddings
            attention_mask: Optional attention mask
            position_ids: Optional position IDs

        Returns:
            List of (K, V) tuples for each layer, in BF16 format for W4A16 decoder
        """
        if not self._trt_available:
            raise RuntimeError("TRT not available. Use fallback method.")

        batch_size, seq_len, _ = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        # Allocate buffers if needed
        self._allocate_kv_cache_buffer(seq_len, batch_size)

        # Run TRT inference
        # Engine expects FP16 input
        hidden_fp16 = hidden_states.to(torch.float16)
        keys, values, _ = self._engine.infer(hidden_fp16, attention_mask, position_ids)

        # Convert to BF16 and copy to pre-allocated buffers
        # Keys shape: (batch, num_layers, num_kv_heads, seq_len, head_dim)
        # Values shape: (batch, num_layers, num_kv_heads, seq_len, head_dim)
        for i in range(NUM_LAYERS):
            # Extract layer KV and convert to target dtype
            k_layer = keys[:, i].to(self.dtype)
            v_layer = values[:, i].to(self.dtype)

            # Copy to pre-allocated buffers (efficient in-place)
            self._kv_cache_buffer[i][0].copy_(k_layer)
            self._kv_cache_buffer[i][1].copy_(v_layer)

        return self._kv_cache_buffer

    def get_last_hidden_state(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run TRT FP8 Prefill and return last hidden state.

        This is useful when you need the final hidden state for action generation.
        """
        if not self._trt_available:
            raise RuntimeError("TRT not available")

        batch_size, seq_len, _ = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        hidden_fp16 = hidden_states.to(torch.float16)
        _, _, last_hidden = self._engine.infer(hidden_fp16, attention_mask, position_ids)

        return last_hidden.to(self.dtype)

    def benchmark(self, seq_len: int = None, num_iters: int = 50):
        """Benchmark TRT Prefill performance."""
        if not self._trt_available:
            logger.warning("TRT not available for benchmarking")
            return None

        seq_len = seq_len or SEQ_LEN

        logger.info(f"Benchmarking TRT Prefill (seq_len={seq_len})...")

        hidden = torch.randn(
            1, seq_len, HIDDEN_SIZE,
            device=self.device, dtype=torch.float16
        )
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        # Warmup
        for _ in range(10):
            _ = self.forward(hidden.to(self.dtype), None, position_ids)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times = []
        for _ in range(num_iters):
            start.record()
            _ = self.forward(hidden.to(self.dtype), None, position_ids)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        import numpy as np
        mean_ms = np.mean(times)
        std_ms = np.std(times)

        logger.info(f"TRT Prefill: {mean_ms:.2f} ± {std_ms:.2f} ms ({1000/mean_ms:.1f} Hz)")

        return {
            'mean_ms': mean_ms,
            'std_ms': std_ms,
            'hz': 1000 / mean_ms,
            'seq_len': seq_len,
            'trt_layers': self._engine._trt_compiled_count if self._engine else 0,
        }


def get_default_checkpoint_dir() -> str:
    """Get default checkpoint directory."""
    return str(Path.home() / ".cache/openpi/checkpoints/pi05_libero")


def create_trt_prefill(
    checkpoint_dir: str = None,
    device: str = "cuda",
    compile_on_init: bool = False,
) -> TRTPrefillWrapper:
    """Factory function to create TRT Prefill Wrapper.

    Args:
        checkpoint_dir: Path to model checkpoint
        device: CUDA device
        compile_on_init: Whether to compile TRT on initialization

    Returns:
        TRTPrefillWrapper instance
    """
    checkpoint_dir = checkpoint_dir or get_default_checkpoint_dir()
    return TRTPrefillWrapper(
        checkpoint_dir=checkpoint_dir,
        device=device,
        compile_on_init=compile_on_init,
    )
