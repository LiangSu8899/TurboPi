"""
NVFP4 GEMM Extension for Thor SM110

This module provides NVFP4 (4-bit floating point) quantized GEMM operations
using CUTLASS SM110a kernels for the NVIDIA Thor GPU.

Performance: 2.8-7.8x speedup vs cuBLAS BF16

Usage:
    from openpi.models_pytorch.nvfp4_extension import NVFP4Linear, replace_mlp_with_nvfp4

    # Replace model MLP layers with NVFP4
    replace_mlp_with_nvfp4(model)
"""

from .nvfp4_wrapper import (
    NVFP4Linear,
    NVFP4MLP,
    quantize_to_nvfp4,
    nvfp4_gemm,
    replace_mlp_with_nvfp4,
    is_nvfp4_available,
)

__all__ = [
    "NVFP4Linear",
    "NVFP4MLP",
    "quantize_to_nvfp4",
    "nvfp4_gemm",
    "replace_mlp_with_nvfp4",
    "is_nvfp4_available",
]
