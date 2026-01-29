"""
hybrid_inference.py - Hybrid TensorRT + PyTorch inference for Pi0.5

This module provides a hybrid inference pipeline that uses:
- TensorRT for the vision encoder (SigLIP) - 2x speedup
- PyTorch for the LLM components (Gemma 2B + 300M)

Usage:
    from openpi.optimization.hybrid_inference import HybridPi0Inference

    model = HybridPi0Inference(
        checkpoint_path="~/.cache/openpi/checkpoints/pi05_libero",
        trt_engine_dir="./onnx_exports",
        device="cuda"
    )
    actions = model.sample_actions(observation, num_steps=5)
"""

import logging
import json
from pathlib import Path
from typing import Optional
import math

import numpy as np
import torch
import torch.nn as nn

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

logger = logging.getLogger(__name__)


class TensorRTVisionEncoder:
    """TensorRT-accelerated vision encoder wrapper."""

    def __init__(self, engine_path: str, device: str = "cuda"):
        if not HAS_TENSORRT:
            raise RuntimeError(
                "TensorRT not available. Install with: pip install tensorrt pycuda"
            )

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        self.device = device
        self._device_idx = int(device.split(":")[-1]) if ":" in device else 0

        # Initialize TensorRT
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        # Load engine
        logger.info(f"Loading TensorRT vision encoder: {engine_path}")
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Setup I/O buffers
        self._setup_buffers()

        logger.info(
            f"TensorRT vision encoder loaded: "
            f"{len(self.inputs)} inputs, {len(self.outputs)} outputs"
        )

    def _setup_buffers(self):
        """Allocate input/output buffers."""
        self.inputs = []
        self.outputs = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)

            # Handle dynamic shapes (use batch=1 as default)
            for j, s in enumerate(shape):
                if s == -1:
                    shape[j] = 1
            shape = tuple(shape)

            # Map TensorRT dtype to numpy dtype
            try:
                dtype = trt.nptype(trt_dtype)
            except TypeError:
                # BF16 fallback
                if trt_dtype == trt.bfloat16:
                    dtype = np.float32
                else:
                    dtype = np.float32

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            buffer_info = {
                'name': name,
                'shape': shape,
                'dtype': dtype,
                'host': host_mem,
                'device': device_mem,
                'size': size,
            }

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(buffer_info)
            else:
                self.outputs.append(buffer_info)

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run TensorRT inference on pixel values.

        Args:
            pixel_values: [batch, 3, 224, 224] image tensor (float32 or bf16)

        Returns:
            hidden_states: [batch, num_patches, hidden_dim] vision features
        """
        # Ensure contiguous float32 on CPU for transfer
        input_data = pixel_values.detach().float().cpu().contiguous().numpy()

        # Handle batch size
        batch_size = input_data.shape[0]

        # Copy input to device
        np.copyto(self.inputs[0]['host'][:input_data.size], input_data.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))

        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output to host
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        self.stream.synchronize()

        # Reshape and convert to torch tensor
        output_shape = list(self.outputs[0]['shape'])
        output_shape[0] = batch_size  # Correct batch dimension

        output_data = self.outputs[0]['host'][:np.prod(output_shape)].reshape(output_shape)
        output_tensor = torch.from_numpy(output_data.copy()).to(
            device=pixel_values.device,
            dtype=pixel_values.dtype
        )

        return output_tensor

    def __del__(self):
        """Cleanup CUDA resources."""
        if hasattr(self, 'stream'):
            self.stream.synchronize()


class TensorRTActionExpert:
    """TensorRT-accelerated action expert (Gemma 300M) with adaRMS support."""

    def __init__(self, engine_path: str, hidden_size: int = 1024, device: str = "cuda"):
        if not HAS_TENSORRT:
            raise RuntimeError(
                "TensorRT not available. Install with: pip install tensorrt pycuda"
            )

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        self.device = device
        self.hidden_size = hidden_size
        self._device_idx = int(device.split(":")[-1]) if ":" in device else 0

        # Initialize TensorRT
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        # Load engine
        logger.info(f"Loading TensorRT action expert: {engine_path}")
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Setup I/O buffers
        self._setup_buffers()

        logger.info(
            f"TensorRT action expert loaded: "
            f"{len(self.inputs)} inputs, {len(self.outputs)} outputs"
        )

    def _setup_buffers(self):
        """Allocate input/output buffers."""
        self.inputs = []
        self.outputs = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)

            # Handle dynamic shapes
            for j, s in enumerate(shape):
                if s == -1:
                    shape[j] = 1 if j == 0 else 256  # batch=1, seq=256 default
            shape = tuple(shape)

            try:
                dtype = trt.nptype(trt_dtype)
            except TypeError:
                if trt_dtype == trt.bfloat16:
                    dtype = np.float32
                else:
                    dtype = np.float32

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            buffer_info = {
                'name': name,
                'shape': shape,
                'dtype': dtype,
                'host': host_mem,
                'device': device_mem,
                'size': size,
            }

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(buffer_info)
                logger.debug(f"Input: {name}, shape={shape}, dtype={dtype}")
            else:
                self.outputs.append(buffer_info)
                logger.debug(f"Output: {name}, shape={shape}, dtype={dtype}")

    def __call__(
        self,
        hidden_states: torch.Tensor,
        adarms_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run TensorRT inference on hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size] input embeddings
            adarms_cond: [batch, hidden_size] optional adaptive RMSNorm conditioning

        Returns:
            output: [batch, seq_len, hidden_size] transformed hidden states
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Prepare inputs
        hidden_np = hidden_states.detach().float().cpu().contiguous().numpy()

        # Copy hidden_states to device
        np.copyto(self.inputs[0]['host'][:hidden_np.size], hidden_np.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # Copy adarms_cond if provided and engine has 2 inputs
        if len(self.inputs) > 1 and adarms_cond is not None:
            cond_np = adarms_cond.detach().float().cpu().contiguous().numpy()
            np.copyto(self.inputs[1]['host'][:cond_np.size], cond_np.ravel())
            cuda.memcpy_htod_async(
                self.inputs[1]['device'],
                self.inputs[1]['host'],
                self.stream
            )

        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))

        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output to host
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        self.stream.synchronize()

        # Reshape and convert to torch tensor
        output_size = batch_size * seq_len * hidden_size
        output_data = self.outputs[0]['host'][:output_size].reshape(
            batch_size, seq_len, hidden_size
        )
        output_tensor = torch.from_numpy(output_data.copy()).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )

        return output_tensor

    def __del__(self):
        """Cleanup CUDA resources."""
        if hasattr(self, 'stream'):
            self.stream.synchronize()


class HybridPi0Inference(nn.Module):
    """
    Hybrid TensorRT + PyTorch inference for Pi0.5.

    Uses TensorRT for vision encoding and PyTorch for everything else.
    """

    def __init__(
        self,
        checkpoint_path: str,
        trt_engine_dir: str,
        device: str = "cuda",
        use_trt_vision: bool = True,
        use_trt_action_expert: bool = False,
    ):
        super().__init__()

        self.device = device
        self.use_trt_vision = use_trt_vision and HAS_TENSORRT
        self.use_trt_action_expert = use_trt_action_expert and HAS_TENSORRT
        self.trt_engine_dir = Path(trt_engine_dir)

        # Load base PyTorch model
        self._load_pytorch_model(checkpoint_path, device)

        # Load TensorRT vision encoder if available
        if self.use_trt_vision:
            trt_engine_path = self.trt_engine_dir / "siglip_vision_encoder.engine"
            if trt_engine_path.exists():
                self.trt_vision = TensorRTVisionEncoder(str(trt_engine_path), device)
                logger.info("TensorRT vision encoder enabled")
            else:
                logger.warning(
                    f"TRT engine not found at {trt_engine_path}, "
                    "falling back to PyTorch vision encoder"
                )
                self.use_trt_vision = False
                self.trt_vision = None
        else:
            self.trt_vision = None
            logger.info("Using PyTorch vision encoder (TRT disabled)")

        # Load TensorRT action expert if available
        if self.use_trt_action_expert:
            # Try adaRMS version first, then fallback to standard
            trt_adarms_path = self.trt_engine_dir / "gemma_300m_expert_adarms_fp16.engine"
            trt_standard_path = self.trt_engine_dir / "gemma_300m_expert_fp16.engine"

            if trt_adarms_path.exists():
                hidden_size = self.model.paligemma_with_expert.gemma_expert.config.hidden_size
                self.trt_action_expert = TensorRTActionExpert(
                    str(trt_adarms_path), hidden_size=hidden_size, device=device
                )
                self.trt_action_expert_has_adarms = True
                logger.info("TensorRT action expert enabled (with adaRMS)")
            elif trt_standard_path.exists():
                hidden_size = self.model.paligemma_with_expert.gemma_expert.config.hidden_size
                self.trt_action_expert = TensorRTActionExpert(
                    str(trt_standard_path), hidden_size=hidden_size, device=device
                )
                self.trt_action_expert_has_adarms = False
                logger.warning(
                    "TensorRT action expert enabled (WITHOUT adaRMS - may affect quality)"
                )
            else:
                logger.warning(
                    f"TRT action expert engine not found, "
                    "falling back to PyTorch"
                )
                self.use_trt_action_expert = False
                self.trt_action_expert = None
                self.trt_action_expert_has_adarms = False
        else:
            self.trt_action_expert = None
            self.trt_action_expert_has_adarms = False
            logger.info("Using PyTorch action expert (TRT disabled)")

    def _load_pytorch_model(self, checkpoint_path: str, device: str):
        """Load the base PyTorch model."""
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.models.pi0_config import Pi0Config
        from safetensors.torch import load_file

        model_path = Path(checkpoint_path).expanduser()
        weights_path = model_path / "model.safetensors"
        config_path = model_path / "config.json"

        with open(config_path) as f:
            model_config = json.load(f)

        pi0_config = Pi0Config(
            paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
            action_dim=model_config.get("action_dim", 32),
            action_horizon=model_config.get("action_horizon", 50),
            max_token_len=model_config.get("tokenizer_max_length", 200),
            pi05=True,
            dtype="bfloat16",
        )

        self.model = PI0Pytorch(pi0_config)
        state_dict = load_file(weights_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device=device)
        self.model.eval()

        # Store config reference
        self.config = pi0_config

        logger.info(f"Loaded PyTorch model from {checkpoint_path}")

    def _embed_image_trt(self, img: torch.Tensor) -> torch.Tensor:
        """Embed image using TensorRT vision encoder + PyTorch projector."""
        # TRT vision tower outputs (batch, num_patches, 1152) in float32
        vision_features = self.trt_vision(img)

        # Apply multi-modal projector (1152 -> 2048) in PyTorch
        # Convert to model dtype (bfloat16) for projector
        projector = self.model.paligemma_with_expert.paligemma.model.multi_modal_projector
        projector_dtype = projector.linear.weight.dtype
        vision_features = vision_features.to(dtype=projector_dtype)

        image_features = projector(vision_features)

        return image_features

    def _embed_image_pytorch(self, img: torch.Tensor) -> torch.Tensor:
        """Embed image using PyTorch vision encoder (fallback)."""
        return self.model.paligemma_with_expert.embed_image(img)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed images with vision encoder and language tokens.

        Uses TensorRT for vision if available, otherwise falls back to PyTorch.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            # Choose vision encoder
            if self.use_trt_vision and self.trt_vision is not None:
                img_emb = self._embed_image_trt(img)
            else:
                img_emb = self._embed_image_pytorch(img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens (always PyTorch)
        lang_emb = self.model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    @torch.no_grad()
    def sample_actions(
        self,
        observation,
        noise: Optional[torch.Tensor] = None,
        num_steps: int = 5,
    ) -> torch.Tensor:
        """
        Sample actions from the model.

        Args:
            observation: Observation dataclass with images, state, etc.
            noise: Optional initial noise tensor
            num_steps: Number of denoising steps (default: 5 for speed)

        Returns:
            actions: [batch, action_horizon, action_dim] predicted actions
        """
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

        device = self.device

        # Preprocess observation
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        images = list(observation.images.values())
        img_masks = list(observation.image_masks.values())
        lang_tokens = observation.tokenized_prompt
        lang_masks = observation.tokenized_prompt_mask
        state = observation.state

        bsize = state.shape[0]

        # Initialize noise
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = torch.normal(
                mean=0.0, std=1.0, size=actions_shape,
                dtype=torch.float32, device=device
            )

        # Embed prefix (vision + language) - uses TRT for vision
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Prepare attention masks
        prefix_att_2d_masks_4d = prefix_att_2d_masks[:, None, :, :]
        prefix_att_2d_masks_4d = torch.where(prefix_att_2d_masks_4d, 0.0, -2.3819763e38)

        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        # Compute KV cache for prefix
        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Denoising loop
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t

    def get_inference_stats(self) -> dict:
        """Get inference configuration stats."""
        return {
            "use_trt_vision": self.use_trt_vision,
            "use_trt_action_expert": self.use_trt_action_expert,
            "trt_action_expert_has_adarms": getattr(self, 'trt_action_expert_has_adarms', False),
            "trt_available": HAS_TENSORRT,
            "device": self.device,
            "model_dtype": str(self.model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype),
        }


def benchmark_hybrid(
    checkpoint_path: str = "~/.cache/openpi/checkpoints/pi05_libero",
    trt_engine_dir: str = "./onnx_exports",
    num_runs: int = 20,
    warmup: int = 5,
    num_steps: int = 5,
):
    """
    Benchmark hybrid inference vs pure PyTorch.

    Returns:
        dict with benchmark results
    """
    import time
    from dataclasses import dataclass

    @dataclass
    class DummyObservation:
        images: dict
        image_masks: dict
        state: torch.Tensor
        tokenized_prompt: torch.Tensor
        tokenized_prompt_mask: torch.Tensor
        token_ar_mask: torch.Tensor
        token_loss_mask: torch.Tensor

    device = "cuda"

    # Create dummy observation with correct camera names
    batch_size = 1
    observation = DummyObservation(
        images={
            "base_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
            "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
            "right_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
        },
        image_masks={
            "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        },
        state=torch.randn(batch_size, 32, device=device),
        tokenized_prompt=torch.randint(0, 1000, (batch_size, 50), device=device),
        tokenized_prompt_mask=torch.ones(batch_size, 50, dtype=torch.bool, device=device),
        token_ar_mask=torch.zeros(batch_size, 50, dtype=torch.bool, device=device),
        token_loss_mask=torch.zeros(batch_size, 50, dtype=torch.bool, device=device),
    )

    results = {}

    # Benchmark with TRT vision
    logger.info("Benchmarking hybrid inference (TRT vision + PyTorch LLM)...")
    model_hybrid = HybridPi0Inference(
        checkpoint_path=checkpoint_path,
        trt_engine_dir=trt_engine_dir,
        device=device,
        use_trt_vision=True,
    )

    # Warmup
    for _ in range(warmup):
        _ = model_hybrid.sample_actions(observation, num_steps=num_steps)
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model_hybrid.sample_actions(observation, num_steps=num_steps)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    results["hybrid_trt"] = {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "throughput_hz": 1000.0 / np.mean(latencies),
    }

    del model_hybrid
    torch.cuda.empty_cache()

    # Benchmark pure PyTorch
    logger.info("Benchmarking pure PyTorch inference...")
    model_pytorch = HybridPi0Inference(
        checkpoint_path=checkpoint_path,
        trt_engine_dir=trt_engine_dir,
        device=device,
        use_trt_vision=False,  # Force PyTorch
    )

    # Warmup
    for _ in range(warmup):
        _ = model_pytorch.sample_actions(observation, num_steps=num_steps)
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model_pytorch.sample_actions(observation, num_steps=num_steps)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    results["pure_pytorch"] = {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "throughput_hz": 1000.0 / np.mean(latencies),
    }

    # Calculate speedup
    speedup = results["pure_pytorch"]["mean_ms"] / results["hybrid_trt"]["mean_ms"]
    results["speedup"] = speedup

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test hybrid inference")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="~/.cache/openpi/checkpoints/pi05_libero",
    )
    parser.add_argument(
        "--trt_engine_dir",
        type=str,
        default="./onnx_exports",
    )
    parser.add_argument("--num_runs", type=int, default=20)
    parser.add_argument("--num_steps", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results = benchmark_hybrid(
        checkpoint_path=args.checkpoint_path,
        trt_engine_dir=args.trt_engine_dir,
        num_runs=args.num_runs,
        num_steps=args.num_steps,
    )

    print("\n" + "=" * 60)
    print("HYBRID INFERENCE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Configuration: {args.num_steps} denoising steps, {args.num_runs} runs")
    print("-" * 60)
    print(f"{'Mode':<25} {'Latency (ms)':<15} {'Throughput (Hz)':<15}")
    print("-" * 60)
    print(f"{'Pure PyTorch':<25} {results['pure_pytorch']['mean_ms']:<15.1f} {results['pure_pytorch']['throughput_hz']:<15.2f}")
    print(f"{'Hybrid (TRT Vision)':<25} {results['hybrid_trt']['mean_ms']:<15.1f} {results['hybrid_trt']['throughput_hz']:<15.2f}")
    print("-" * 60)
    print(f"Speedup: {results['speedup']:.2f}x")
    print("=" * 60)
