"""TensorRT Vision Encoder Wrapper.

This module provides a TRT-accelerated Vision Encoder (SigLIP) with:
1. Zero-copy PyTorch<->TRT data transfer
2. Optional CUDA Graph capture
3. Fallback to PyTorch eager mode

Expected performance:
- TRT FP16: ~5ms per image
- PyTorch BF16: ~30ms per image

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# TensorRT imports (optional)
try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    trt = None


class TRTVisionEncoder(nn.Module):
    """TensorRT-accelerated Vision Encoder for SigLIP.

    This wrapper loads a pre-built TRT engine and provides a PyTorch-compatible
    interface with zero-copy tensor transfer.

    Input: (batch, 3, 224, 224) RGB image in FP32 or FP16
    Output: (batch, 256, 1152) image embeddings in FP16

    Note: After SigLIP, the output goes through multi_modal_projector to get
    the final image embeddings for the LM.
    """

    def __init__(
        self,
        engine_path: str,
        device: torch.device = None,
        use_cuda_graph: bool = False,
    ):
        super().__init__()
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT not available. Install with: pip install tensorrt")

        self.device = device or torch.device('cuda')
        self.use_cuda_graph = use_cuda_graph
        self.engine_path = engine_path

        # Initialize TensorRT
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        # Load engine
        logger.info(f"Loading TRT Vision Encoder: {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get IO info
        self._setup_io()

        # Pre-allocate static buffers for CUDA Graph
        self._static_input: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None

        # CUDA Graph state
        self._graph_captured = False
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None

        logger.info(f"TRT Vision Encoder loaded: input={self.input_shape}, output={self.output_shape}")

    def _setup_io(self):
        """Setup input/output tensor info."""
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)

            # Handle dynamic dimensions
            shape = list(shape)
            for j, s in enumerate(shape):
                if s == -1:
                    shape[j] = 1  # Default batch size
            shape = tuple(shape)

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = shape
            else:
                self.output_name = name
                self.output_shape = shape

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Failed to find input/output tensors in TRT engine")

    def _allocate_static_buffers(self, batch_size: int = 1):
        """Allocate static buffers for CUDA Graph capture."""
        # Input: (batch, 3, 224, 224)
        input_shape = (batch_size,) + self.input_shape[1:]
        self._static_input = torch.zeros(
            input_shape, dtype=torch.float16, device=self.device
        )

        # Output: (batch, num_patches, hidden_size)
        output_shape = (batch_size,) + self.output_shape[1:]
        self._static_output = torch.zeros(
            output_shape, dtype=torch.float16, device=self.device
        )

        # Set tensor addresses in context
        self.context.set_tensor_address(self.input_name, self._static_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self._static_output.data_ptr())

    def _infer_trt(self, stream: torch.cuda.Stream = None):
        """Execute TRT inference."""
        stream_handle = stream.cuda_stream if stream else torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle)

    def capture_graph(self, batch_size: int = 1, warmup_iters: int = 3):
        """Capture CUDA Graph for inference."""
        if not self.use_cuda_graph:
            logger.warning("CUDA Graph disabled, skipping capture")
            return

        if self._graph_captured:
            return

        # Allocate static buffers
        self._allocate_static_buffers(batch_size)

        # Warmup
        stream = torch.cuda.Stream()
        for _ in range(warmup_iters):
            with torch.cuda.stream(stream):
                self._infer_trt(stream)
        stream.synchronize()

        # Capture
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph, stream=stream):
            self._infer_trt(stream)

        stream.synchronize()
        self._graph_captured = True
        logger.info("TRT Vision Encoder: CUDA Graph captured")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through TRT Vision Encoder.

        Args:
            pixel_values: (batch, 3, H, W) input images

        Returns:
            (batch, num_patches, hidden_size) image embeddings
        """
        batch_size = pixel_values.shape[0]

        # Convert to FP16 for TRT
        if pixel_values.dtype != torch.float16:
            pixel_values = pixel_values.to(torch.float16)

        if self._graph_captured:
            # Use captured CUDA Graph
            if batch_size != self._static_input.shape[0]:
                raise RuntimeError(f"Batch size mismatch: got {batch_size}, expected {self._static_input.shape[0]}")

            self._static_input.copy_(pixel_values)
            self._cuda_graph.replay()
            return self._static_output.clone()
        else:
            # Direct TRT inference (no graph)
            self._allocate_static_buffers(batch_size)
            self._static_input.copy_(pixel_values)
            self._infer_trt()
            torch.cuda.synchronize()
            return self._static_output.clone()


class VisionEncoderWrapper(nn.Module):
    """High-level Vision Encoder wrapper with TRT fallback.

    This class wraps both SigLIP vision encoder and multi-modal projector,
    providing a unified interface that matches the original model.

    If TRT engine is available, uses TRT acceleration.
    Otherwise, falls back to PyTorch eager mode.
    """

    def __init__(
        self,
        model,  # PI0Pytorch
        engine_path: Optional[str] = None,
        device: torch.device = None,
        use_trt: bool = True,
    ):
        super().__init__()
        self.model = model
        self.device = device or torch.device('cuda')
        self.use_trt = use_trt and engine_path is not None and HAS_TENSORRT

        # Get original components
        self.paligemma = model.paligemma_with_expert.paligemma
        self.multi_modal_projector = self.paligemma.multi_modal_projector

        # TRT engine (if available)
        self._trt_encoder: Optional[TRTVisionEncoder] = None

        if self.use_trt:
            try:
                self._trt_encoder = TRTVisionEncoder(
                    engine_path=engine_path,
                    device=self.device,
                    use_cuda_graph=True,
                )
                logger.info("Using TRT Vision Encoder")
            except Exception as e:
                logger.warning(f"Failed to load TRT engine, falling back to PyTorch: {e}")
                self.use_trt = False

    def capture_graph(self, batch_size: int = 1):
        """Capture CUDA Graph for TRT encoder."""
        if self._trt_encoder is not None:
            self._trt_encoder.capture_graph(batch_size)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through Vision Encoder.

        Args:
            pixel_values: (batch, 3, 224, 224) input images

        Returns:
            (batch, num_patches, hidden_size) projected image embeddings
        """
        if self._trt_encoder is not None:
            # TRT path: SigLIP (TRT) + Projector (PyTorch)
            # TRT output is FP16, projector expects BF16
            vision_outputs = self._trt_encoder(pixel_values)
            vision_outputs = vision_outputs.to(self.multi_modal_projector.linear.weight.dtype)
        else:
            # PyTorch path: Full eager mode
            vision_outputs = self.paligemma.vision_tower(pixel_values).last_hidden_state

        # Project to LM hidden size
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    def encode_images(
        self,
        images: List[torch.Tensor],
        image_masks: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode multiple images with masks.

        This matches the interface used by embed_prefix().

        Args:
            images: List of image tensors (batch, 3, 224, 224)
            image_masks: List of mask tensors (batch,)

        Returns:
            image_embeddings: List of (batch, num_patches, hidden_size)
            expanded_masks: List of (batch, num_patches) bool masks
        """
        embeddings = []
        expanded_masks = []

        for img, mask in zip(images, image_masks):
            img_emb = self.forward(img)
            batch_size, num_patches = img_emb.shape[:2]

            embeddings.append(img_emb)
            expanded_masks.append(mask[:, None].expand(batch_size, num_patches))

        return embeddings, expanded_masks


def get_default_engine_path() -> Optional[str]:
    """Get default TRT engine path."""
    candidates = [
        Path.home() / ".cache/openpi/checkpoints/pi05_libero/onnx_exports/siglip_vision_encoder.engine",
        Path.home() / "suliang/Turbo-Pi/upload_staging/tensorrt_engines/siglip_vision_encoder.engine",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return None
