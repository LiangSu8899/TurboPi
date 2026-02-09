#!/usr/bin/env python3
"""
NVFP4 PyTorch Wrapper for Thor SM110

This module provides PyTorch-friendly wrappers for NVFP4 GEMM operations.
Handles batch size padding, weight pre-quantization, and model integration.

Key features:
- Automatic batch padding to CUTLASS-supported sizes
- Pre-quantized weight storage for inference efficiency
- Drop-in replacement for nn.Linear and GemmaMLP
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import warnings

# Try to import the compiled extension
try:
    import nvfp4_gemm as _nvfp4_ext
    NVFP4_EXTENSION_AVAILABLE = True
except ImportError:
    NVFP4_EXTENSION_AVAILABLE = False
    warnings.warn(
        "NVFP4 extension not compiled. Build with: "
        "cd nvfp4_extension && python setup.py install"
    )

# NVFP4 supported batch sizes (M dimension)
# CUTLASS requires M aligned to certain multiples
# Based on testing, these sizes work reliably
SUPPORTED_M_SIZES = [64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024]

# Block size for scale factors
NVFP4_BLOCK_SIZE = 32


def is_nvfp4_available() -> bool:
    """Check if NVFP4 extension is available."""
    return NVFP4_EXTENSION_AVAILABLE


def find_padded_size(batch_size: int) -> int:
    """Find the smallest supported size >= batch_size."""
    for size in SUPPORTED_M_SIZES:
        if size >= batch_size:
            return size
    # If larger than all supported sizes, pad to nearest 256 multiple
    return ((batch_size + 255) // 256) * 256


def pad_tensor(x: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, int]:
    """Pad tensor's first dimension to target size."""
    original_size = x.shape[0]
    if original_size == target_size:
        return x, original_size

    if original_size > target_size:
        raise ValueError(f"Original size {original_size} > target size {target_size}")

    pad_size = target_size - original_size
    padding = torch.zeros(pad_size, *x.shape[1:], dtype=x.dtype, device=x.device)
    return torch.cat([x, padding], dim=0), original_size


def quantize_to_nvfp4(
    tensor: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize BF16/FP16 tensor to NVFP4 format with block scaling.

    NVFP4 (e2m1) representable values: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

    Args:
        tensor: [M, K] input tensor
        block_size: Size of each scaling block (default 32)

    Returns:
        Tuple of (quantized_data, scale_factors)
        - quantized_data: [M, K] in simulated NVFP4 (stored as BF16 for now)
        - scale_factors: [M, K/block_size] scale factors
    """
    M, K = tensor.shape
    assert K % block_size == 0, f"K ({K}) must be divisible by block_size ({block_size})"

    device = tensor.device
    dtype = tensor.dtype

    # NVFP4 max value
    nvfp4_max = 6.0

    # NVFP4 representable absolute values
    nvfp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=device,
        dtype=torch.float32
    )

    # Reshape to blocks: [M, num_blocks, block_size]
    num_blocks = K // block_size
    tensor_blocked = tensor.view(M, num_blocks, block_size).float()

    # Compute per-block scale factors
    block_max = tensor_blocked.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale_factors = block_max.squeeze(-1) / nvfp4_max  # [M, num_blocks]

    # Scale to FP4 range
    scaled = tensor_blocked / block_max * nvfp4_max

    # Quantize to nearest FP4 value
    signs = scaled.sign()
    abs_scaled = scaled.abs()

    # Find nearest quantization level
    distances = (abs_scaled.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1)
    quantized_abs = nvfp4_values[indices]
    quantized = signs * quantized_abs

    # Reshape back
    quantized = quantized.view(M, K).to(dtype)

    return quantized, scale_factors.to(torch.float32)


def nvfp4_gemm(
    input_quantized: torch.Tensor,
    weight_quantized: torch.Tensor,
    input_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    NVFP4 Block-Scaled GEMM.

    Computes: output = (input @ weight.T) * input_scales * weight_scales + bias

    For now, this is a simulation using dequantized BF16. The actual CUTLASS
    kernel will be called when the extension is compiled.

    Args:
        input_quantized: [M, K] quantized input
        weight_quantized: [N, K] quantized weight
        input_scales: [M, K/block_size] input scale factors
        weight_scales: [N, K/block_size] weight scale factors
        bias: Optional [N] bias

    Returns:
        output: [M, N] BF16 tensor
    """
    if NVFP4_EXTENSION_AVAILABLE:
        # Use compiled extension
        return _nvfp4_ext.gemm(
            input_quantized, weight_quantized,
            input_scales, weight_scales,
            bias
        )
    else:
        # Fallback: simulate with BF16
        # Dequantize by applying scales
        M, K = input_quantized.shape
        N = weight_quantized.shape[0]
        block_size = K // input_scales.shape[1]

        # Reshape and apply scales
        input_blocked = input_quantized.view(M, -1, block_size)
        weight_blocked = weight_quantized.view(N, -1, block_size)

        input_dequant = (input_blocked * input_scales.unsqueeze(-1)).view(M, K)
        weight_dequant = (weight_blocked * weight_scales.unsqueeze(-1)).view(N, K)

        # Standard matmul
        output = torch.matmul(input_dequant, weight_dequant.T)

        if bias is not None:
            output = output + bias

        return output.to(input_quantized.dtype)


class NVFP4Linear(nn.Module):
    """
    NVFP4 quantized Linear layer.

    Drop-in replacement for nn.Linear with NVFP4 quantization.
    Supports automatic batch padding for CUTLASS compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: int = NVFP4_BLOCK_SIZE,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Original weights for initialization
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Pre-quantized weight buffers
        self.register_buffer('weight_quantized', None)
        self.register_buffer('weight_scales', None)
        self._quantized = False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = NVFP4_BLOCK_SIZE) -> 'NVFP4Linear':
        """Create NVFP4Linear from existing nn.Linear."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)

        # Pre-quantize weights
        layer.quantize_weights()
        return layer

    def quantize_weights(self):
        """Pre-quantize weights for inference."""
        with torch.no_grad():
            self.weight_quantized, self.weight_scales = quantize_to_nvfp4(
                self.weight, self.block_size
            )
            self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic batch padding.

        Args:
            x: [batch_size, in_features] input tensor

        Returns:
            [batch_size, out_features] output tensor
        """
        if not self._quantized:
            self.quantize_weights()

        original_batch = x.shape[0]
        target_batch = find_padded_size(original_batch)

        # Pad input if needed
        if original_batch != target_batch:
            x_padded, _ = pad_tensor(x, target_batch)
        else:
            x_padded = x

        # Quantize input
        input_quantized, input_scales = quantize_to_nvfp4(x_padded, self.block_size)

        # NVFP4 GEMM
        output = nvfp4_gemm(
            input_quantized,
            self.weight_quantized,
            input_scales,
            self.weight_scales,
            self.bias,
        )

        # Remove padding
        if original_batch != target_batch:
            output = output[:original_batch]

        return output


class NVFP4MLP(nn.Module):
    """
    NVFP4 quantized MLP (GeLU activation).

    Structure: gate_proj -> act -> * up_proj -> down_proj
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu_tanh",
        block_size: int = NVFP4_BLOCK_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.block_size = block_size

        self.gate_proj = NVFP4Linear(hidden_size, intermediate_size, bias=False, block_size=block_size)
        self.up_proj = NVFP4Linear(hidden_size, intermediate_size, bias=False, block_size=block_size)
        self.down_proj = NVFP4Linear(intermediate_size, hidden_size, bias=False, block_size=block_size)

        if activation == "gelu_tanh":
            self.act_fn = lambda x: torch.nn.functional.gelu(x, approximate='tanh')
        elif activation == "silu":
            self.act_fn = torch.nn.functional.silu
        else:
            self.act_fn = torch.nn.functional.gelu

    @classmethod
    def from_gemma_mlp(cls, mlp: nn.Module, block_size: int = NVFP4_BLOCK_SIZE) -> 'NVFP4MLP':
        """Create NVFP4MLP from existing GemmaMLP."""
        hidden_size = mlp.gate_proj.in_features
        intermediate_size = mlp.gate_proj.out_features

        layer = cls(hidden_size, intermediate_size, block_size=block_size)

        # Copy weights
        layer.gate_proj = NVFP4Linear.from_linear(mlp.gate_proj, block_size)
        layer.up_proj = NVFP4Linear.from_linear(mlp.up_proj, block_size)
        layer.down_proj = NVFP4Linear.from_linear(mlp.down_proj, block_size)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def replace_mlp_with_nvfp4(
    model: nn.Module,
    model_name: str = "model",
    block_size: int = NVFP4_BLOCK_SIZE,
) -> int:
    """
    Replace all GemmaMLP layers in model with NVFP4 versions.

    Args:
        model: The language model (e.g., paligemma.language_model)
        model_name: Identifier for logging
        block_size: NVFP4 block size

    Returns:
        Number of MLP layers replaced
    """
    replaced_count = 0

    if hasattr(model, 'layers'):
        for layer_idx, layer in enumerate(model.layers):
            if hasattr(layer, 'mlp') and layer.mlp is not None:
                original_mlp = layer.mlp

                # Create NVFP4 MLP
                nvfp4_mlp = NVFP4MLP.from_gemma_mlp(original_mlp, block_size)

                # Move to same device
                device = next(original_mlp.parameters()).device
                nvfp4_mlp = nvfp4_mlp.to(device)

                # Replace
                layer.mlp = nvfp4_mlp
                replaced_count += 1

    print(f"[{model_name}] Replaced {replaced_count} MLP layers with NVFP4 versions")
    return replaced_count


def replace_pi0_mlp_with_nvfp4(
    model,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> Tuple[int, int]:
    """
    Replace MLP layers in PI0Pytorch model with NVFP4 versions.

    Args:
        model: PI0Pytorch model instance
        block_size: NVFP4 block size

    Returns:
        Tuple of (paligemma_replaced, gemma_replaced) counts
    """
    # Replace PaliGemma MLP (KV Cache)
    paligemma_lm = model.paligemma_with_expert.paligemma.language_model
    pali_count = replace_mlp_with_nvfp4(paligemma_lm, "paligemma", block_size)

    # Replace Gemma Expert MLP (Denoise) - Optional, may want to keep FP8
    # gemma_expert = model.paligemma_with_expert.gemma_expert.model
    # gemma_count = replace_mlp_with_nvfp4(gemma_expert, "gemma_expert", block_size)
    gemma_count = 0  # Keep Gemma Expert at FP8 (already optimized with TRT)

    return pali_count, gemma_count
