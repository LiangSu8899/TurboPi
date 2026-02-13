"""TensorRT wrapper for Denoise module.

This module provides a drop-in replacement for CUDA Graph-based denoising
using TensorRT FP8 quantization.

Expected performance improvement:
- CUDA Graph BF16: 109 ms
- TensorRT FP8: ~40 ms (estimated, 2.7x speedup from FP8)

Author: Turbo-Pi Team
Date: 2026-02-13
"""

import ctypes
import atexit
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

# TensorRT imports
try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    print("Warning: tensorrt not available. TRT inference disabled.")


def destroy(engine):
    """Cleanup function for TRT engine."""
    if hasattr(engine, 'execution_context'):
        del engine.execution_context
    if hasattr(engine, 'engine'):
        del engine.engine
    if hasattr(engine, 'runtime'):
        del engine.runtime


class TRTEngine:
    """TensorRT Engine wrapper for Denoise module.

    Similar to NVIDIA's trt_torch.py but specialized for Denoise.
    """

    def __init__(self, engine_path: str, plugins: List[str] = None):
        """Load TensorRT engine from file.

        Args:
            engine_path: Path to .engine file
            plugins: List of plugin library paths to load
        """
        if not HAS_TRT:
            raise RuntimeError("TensorRT not available")

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        # Load plugins
        self.plugins = []
        if plugins:
            for plugin in plugins:
                self.plugins.append(ctypes.CDLL(plugin, ctypes.RTLD_GLOBAL))

        # Load engine
        self.load(engine_path)

        # Register cleanup
        atexit.register(destroy, self)

    def load(self, engine_path: str):
        """Load engine from file."""
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.execution_context = self.engine.create_execution_context()

        # Get input/output metadata
        self.in_meta = []
        self.out_meta = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            dtype = torch.from_numpy(dtype.__class__.__bases__[0]()).dtype

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_meta.append((name, shape, dtype))
            else:
                self.out_meta.append((name, shape, dtype))

        print(f"Loaded TRT engine: {engine_path}")
        print(f"  Inputs: {[(n, s) for n, s, _ in self.in_meta]}")
        print(f"  Outputs: {[(n, s) for n, s, _ in self.out_meta]}")

    def set_runtime_tensor_shape(self, name: str, shape: tuple):
        """Set runtime shape for dynamic input."""
        self.execution_context.set_input_shape(name, shape)

    def __call__(self, *args, **kwargs):
        """Run inference."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, return_list: bool = False):
        """Run TRT inference.

        Args:
            *args: Input tensors in order (noise, suffix_position_ids, adarms_conds_stacked, cached_keys, cached_values)
            return_list: If True, return list of outputs; else return dict

        Returns:
            Output tensor(s)
        """
        stream = torch.cuda.current_stream()

        # Set input tensors
        for iarg, x in enumerate(args):
            name, expected_shape, expected_dtype = self.in_meta[iarg]

            # Validate
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Input {iarg} ({name}) must be torch.Tensor")

            runtime_shape = list(self.execution_context.get_tensor_shape(name))

            # Handle dynamic shapes
            if -1 in runtime_shape:
                self.set_runtime_tensor_shape(name, tuple(x.shape))
                runtime_shape = list(x.shape)

            if list(x.shape) != runtime_shape:
                raise ValueError(
                    f"Input {name} shape mismatch: expected {runtime_shape}, got {list(x.shape)}"
                )

            if x.dtype != expected_dtype:
                x = x.to(expected_dtype)

            if not x.is_cuda:
                x = x.cuda()

            x = x.contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())

        # Allocate outputs
        outputs = {}
        for name, shape, dtype in self.out_meta:
            runtime_shape = self.execution_context.get_tensor_shape(name)
            out = torch.empty(
                tuple(runtime_shape), dtype=dtype, device="cuda"
            ).contiguous()
            outputs[name] = out
            self.execution_context.set_tensor_address(name, out.data_ptr())

        # Execute
        self.execution_context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        if return_list:
            return list(outputs.values())
        return outputs


class DenoiseTRT(nn.Module):
    """TensorRT-based Denoise module.

    Drop-in replacement for ChainedDenoiseGraphs.
    """

    def __init__(
        self,
        model,
        engine_path: str,
        num_steps: int = 10,
        device: torch.device = torch.device("cuda"),
    ):
        """Initialize TRT Denoise.

        Args:
            model: PI0Pytorch model (for precomputing adarms_conds)
            engine_path: Path to Denoise TRT engine
            num_steps: Number of denoise steps (must match engine)
            device: CUDA device
        """
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.device = device

        # Load TRT engine
        self.engine = TRTEngine(engine_path)

        # State
        self._initialized = False
        self._static_adarms_conds = None
        self._static_suffix_position_ids = None

    def initialize(
        self,
        batch_size: int,
        prefix_len: int,
        prefix_pad_masks: torch.Tensor,
    ):
        """Pre-compute static tensors.

        Args:
            batch_size: Batch size
            prefix_len: Length of prefix (for position IDs)
            prefix_pad_masks: (B, prefix_len) - for computing position offset
        """
        import torch.nn.functional as F
        from openpi.models_pytorch.embedding import create_sinusoidal_pos_embedding

        action_horizon = self.model.config.action_horizon

        # Compute suffix position IDs
        prefix_offset = torch.sum(prefix_pad_masks.long(), dim=-1, keepdim=True)
        self._static_suffix_position_ids = prefix_offset + torch.arange(
            action_horizon, device=self.device, dtype=torch.long
        )

        # Pre-compute adarms_conds for all timesteps
        model_dtype = self.model.action_in_proj.weight.dtype
        dt = -1.0 / self.num_steps

        adarms_conds = []
        for i in range(self.num_steps):
            timestep_val = 1.0 + i * dt
            timestep = torch.tensor(
                [timestep_val] * batch_size,
                dtype=torch.float32,
                device=self.device
            )

            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.model.action_in_proj.out_features,
                min_period=4e-3,
                max_period=4.0,
                device=self.device
            )
            time_emb = time_emb.to(dtype=model_dtype)

            with torch.no_grad():
                x = self.model.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.model.time_mlp_out(x)
                adarms_cond = F.silu(x)

            adarms_conds.append(adarms_cond)

        # Stack: (num_steps, B, hidden_size)
        self._static_adarms_conds = torch.stack(adarms_conds, dim=0)

        self._initialized = True
        print(f"DenoiseTRT initialized: batch_size={batch_size}, prefix_len={prefix_len}")

    def forward(
        self,
        noise: torch.Tensor,
        prefix_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Run TRT denoise inference.

        Args:
            noise: (B, action_horizon, action_dim) - initial noise
            prefix_kv_cache: List of (K, V) tuples for each layer

        Returns:
            actions: (B, action_horizon, action_dim) - denoised actions
        """
        if not self._initialized:
            raise RuntimeError("DenoiseTRT not initialized. Call initialize() first.")

        # Stack KV cache: (num_layers, B, num_kv_heads, prefix_len, head_dim)
        cached_keys = torch.stack([kv[0] for kv in prefix_kv_cache], dim=0)
        cached_values = torch.stack([kv[1] for kv in prefix_kv_cache], dim=0)

        # Convert to FP16 for TRT
        dtype = torch.float16
        noise_fp16 = noise.to(dtype).contiguous()
        adarms_fp16 = self._static_adarms_conds.to(dtype).contiguous()
        keys_fp16 = cached_keys.to(dtype).contiguous()
        values_fp16 = cached_values.to(dtype).contiguous()

        # Run TRT inference
        outputs = self.engine(
            noise_fp16,
            self._static_suffix_position_ids,
            adarms_fp16,
            keys_fp16,
            values_fp16,
        )

        # Get output (convert back to model dtype if needed)
        actions = outputs["actions"]

        return actions.to(noise.dtype)

    def update_kv_cache(
        self,
        prefix_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
    ):
        """Update position IDs when prefix changes.

        Args:
            prefix_kv_cache: Not used directly (passed to forward)
            prefix_pad_masks: (B, prefix_len) - for computing position offset
        """
        action_horizon = self.model.config.action_horizon

        # Update position IDs
        prefix_offset = torch.sum(prefix_pad_masks.long(), dim=-1, keepdim=True)
        self._static_suffix_position_ids = prefix_offset + torch.arange(
            action_horizon, device=self.device, dtype=torch.long
        )


def setup_denoise_trt(model, engine_path: str, num_steps: int = 10) -> DenoiseTRT:
    """Factory function to create DenoiseTRT.

    Args:
        model: PI0Pytorch model
        engine_path: Path to Denoise TRT engine
        num_steps: Number of denoise steps

    Returns:
        DenoiseTRT module
    """
    device = next(model.parameters()).device
    denoise_trt = DenoiseTRT(
        model=model,
        engine_path=engine_path,
        num_steps=num_steps,
        device=device,
    )
    return denoise_trt
