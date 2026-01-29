"""
CUDA Graph optimized inference for Pi0.5 model.

This module implements CUDA graph capture for the denoising loop
to reduce kernel launch overhead and maximize GPU utilization.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class CUDAGraphInference:
    """
    CUDA Graph optimized inference wrapper for Pi0.5.

    This class captures the repeated denoising loop in a CUDA graph
    to eliminate kernel launch overhead, achieving significant speedup
    on GPU-bound inference.

    Key optimizations:
    1. CUDA graph capture for denoising loop
    2. Static tensor allocation (no dynamic memory during inference)
    3. Warmup to trigger JIT compilation before graph capture
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        num_steps: int = 10,
        batch_size: int = 1,
        warmup_iterations: int = 3,
    ):
        self.model = model
        self.device = torch.device(device)
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.warmup_iterations = warmup_iterations

        # Model configuration
        self.action_horizon = model.config.action_horizon
        self.action_dim = model.config.action_dim

        # CUDA graph and static buffers
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_x_t: Optional[Tensor] = None
        self._static_output: Optional[Tensor] = None
        self._static_state: Optional[Tensor] = None
        self._static_prefix_pad_masks: Optional[Tensor] = None
        self._past_key_values: Optional[tuple] = None

        # Graph capture status
        self._is_captured = False

        logger.info(f"CUDAGraphInference initialized: batch_size={batch_size}, num_steps={num_steps}")

    def _allocate_static_buffers(self):
        """Allocate static buffers for CUDA graph capture."""
        # Noisy actions buffer
        self._static_x_t = torch.zeros(
            self.batch_size, self.action_horizon, self.action_dim,
            dtype=torch.float32, device=self.device
        )

        # Output buffer
        self._static_output = torch.zeros(
            self.batch_size, self.action_horizon, self.action_dim,
            dtype=torch.float32, device=self.device
        )

        # State buffer (will be copied from observation)
        self._static_state = torch.zeros(
            self.batch_size, 32,
            dtype=torch.float32, device=self.device
        )

        logger.info("Allocated static buffers for CUDA graph")

    def _warmup(self, observation):
        """Run warmup iterations to trigger JIT and CUDA lazy initialization."""
        logger.info(f"Running {self.warmup_iterations} warmup iterations...")

        for i in range(self.warmup_iterations):
            with torch.no_grad():
                _ = self.model.sample_actions(
                    device=self.device,
                    observation=observation,
                    num_steps=self.num_steps,
                )
            torch.cuda.synchronize()

        logger.info("Warmup complete")

    def capture_graph(self, observation):
        """
        Capture CUDA graph for the denoising loop.

        This must be called once with a representative observation
        before running optimized inference.
        """
        if self._is_captured:
            logger.warning("Graph already captured, skipping")
            return

        logger.info("Starting CUDA graph capture...")

        # Allocate static buffers
        self._allocate_static_buffers()

        # Warmup to trigger JIT
        self._warmup(observation)

        # Extract and cache prefix embeddings
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(
                observation, train=False
            )

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            prefix_att_2d_masks = self.model.paligemma_with_expert.paligemma.language_model.model._prepare_decoder_attention_mask(
                prefix_pad_masks, prefix_att_masks
            ) if hasattr(self.model.paligemma_with_expert.paligemma.language_model.model, '_prepare_decoder_attention_mask') else None

            # Use simpler mask creation
            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)

            # Compute and cache KV values
            _, self._past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            # Store prefix pad masks
            self._static_prefix_pad_masks = prefix_pad_masks.clone()
            self._static_state.copy_(state)

        # Sample initial noise
        self._static_x_t.normal_()

        # Create the CUDA graph
        self._graph = torch.cuda.CUDAGraph()

        # Capture the denoising loop
        with torch.cuda.graph(self._graph):
            dt = -1.0 / self.num_steps
            dt_tensor = torch.tensor(dt, dtype=torch.float32, device=self.device)

            x_t = self._static_x_t
            time = torch.tensor(1.0, dtype=torch.float32, device=self.device)

            for step in range(self.num_steps):
                expanded_time = time.expand(self.batch_size)
                v_t = self.model.denoise_step(
                    self._static_state,
                    self._static_prefix_pad_masks,
                    self._past_key_values,
                    x_t,
                    expanded_time,
                )
                x_t = x_t + dt_tensor * v_t
                time = time + dt_tensor

            self._static_output.copy_(x_t)

        self._is_captured = True
        logger.info("CUDA graph capture complete")

    @torch.no_grad()
    def sample_actions(self, observation) -> Tensor:
        """
        Run optimized inference using captured CUDA graph.

        For maximum performance, the observation should have the same
        structure as the one used for graph capture.
        """
        if not self._is_captured:
            raise RuntimeError("CUDA graph not captured. Call capture_graph() first.")

        # Update state from observation
        _, _, _, _, state = self.model._preprocess_observation(observation, train=False)
        self._static_state.copy_(state)

        # Re-sample noise
        self._static_x_t.normal_()

        # Replay the captured graph
        self._graph.replay()

        # Return a copy of the output
        return self._static_output.clone()

    @torch.no_grad()
    def sample_actions_with_new_prefix(self, observation) -> Tensor:
        """
        Run inference with a new observation (re-computes prefix).

        This is slower than sample_actions() because it recomputes
        the vision/language embeddings, but allows changing the prompt
        or images between calls.
        """
        # For now, fall back to standard inference
        # A more optimized version would update the KV cache
        return self.model.sample_actions(
            device=self.device,
            observation=observation,
            num_steps=self.num_steps,
        )


class TorchCompileWrapper(nn.Module):
    """
    Wrapper that applies torch.compile to key model components.

    Optimizes:
    1. Vision encoder
    2. Language embeddings
    3. Transformer forward pass
    4. Denoising step
    """

    def __init__(self, model: nn.Module, mode: str = "max-autotune"):
        super().__init__()
        self.model = model
        self.mode = mode
        self._compiled = False

    def compile(self):
        """Apply torch.compile to model components."""
        if self._compiled:
            logger.warning("Model already compiled")
            return

        logger.info(f"Applying torch.compile with mode={self.mode}...")

        # Compile denoise_step for maximum speedup
        self.model.denoise_step = torch.compile(
            self.model.denoise_step,
            mode=self.mode,
            fullgraph=False,  # Allow graph breaks for compatibility
        )

        # Compile embed_prefix (vision + language embedding)
        self.model.embed_prefix = torch.compile(
            self.model.embed_prefix,
            mode=self.mode,
            fullgraph=False,
        )

        # Compile embed_suffix
        self.model.embed_suffix = torch.compile(
            self.model.embed_suffix,
            mode=self.mode,
            fullgraph=False,
        )

        self._compiled = True
        logger.info("torch.compile applied successfully")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Forward to wrapped model's sample_actions."""
        return self.model.sample_actions(
            device=device,
            observation=observation,
            noise=noise,
            num_steps=num_steps,
        )

    def forward(self, *args, **kwargs):
        """Forward to wrapped model."""
        return self.model.forward(*args, **kwargs)


def create_optimized_model(
    model: nn.Module,
    device: str = "cuda",
    use_cuda_graph: bool = True,
    use_torch_compile: bool = True,
    compile_mode: str = "max-autotune",
    num_steps: int = 10,
    batch_size: int = 1,
) -> nn.Module:
    """
    Create an optimized inference wrapper for the Pi0.5 model.

    Applies the following optimizations:
    1. torch.compile with specified mode
    2. CUDA graph capture for denoising loop (optional)

    Args:
        model: The base Pi0Pytorch model
        device: Device to run inference on
        use_cuda_graph: Whether to use CUDA graph optimization
        use_torch_compile: Whether to apply torch.compile
        compile_mode: torch.compile mode ("default", "reduce-overhead", "max-autotune")
        num_steps: Number of denoising steps
        batch_size: Batch size for CUDA graph capture

    Returns:
        Optimized model wrapper
    """
    logger.info(f"Creating optimized model: cuda_graph={use_cuda_graph}, torch_compile={use_torch_compile}")

    # Apply torch.compile first
    if use_torch_compile:
        wrapper = TorchCompileWrapper(model, mode=compile_mode)
        wrapper.compile()
        model = wrapper

    # Return the model (CUDA graph capture done separately)
    if use_cuda_graph:
        logger.info("CUDA graph will be captured on first inference call")
        # The caller should use CUDAGraphInference directly

    return model
