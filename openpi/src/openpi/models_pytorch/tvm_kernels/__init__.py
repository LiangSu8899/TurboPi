"""
TVM TensorIR Kernels for nvFP4/FP8 Quantization

This module provides pure TVM TensorIR implementations for:
1. W4A4: Block-scaled nvFP4 quantization + GEMM
2. W4A8: FP8 activation + nvFP4 weight GEMM (software bypass for mxf8f6f4)
3. W4A16: nvFP4 weight dequantization to BF16

Target: Thor SM110, breaking through FP8 12Hz baseline
"""

from .nvfp4_quantize import (
    build_nvfp4_quantize_kernel,
    quantize_to_nvfp4_tvm,
)

from .nvfp4_gemm import (
    build_nvfp4_gemm_kernel,
    nvfp4_gemm_tvm,
)

from .w4a8_gemm import (
    build_w4a8_gemm_kernel,
    w4a8_gemm_tvm,
)

from .w4a16_dequant import (
    build_w4a16_dequant_kernel,
    dequant_nvfp4_to_bf16_tvm,
)

__all__ = [
    # W4A4
    "build_nvfp4_quantize_kernel",
    "quantize_to_nvfp4_tvm",
    "build_nvfp4_gemm_kernel",
    "nvfp4_gemm_tvm",
    # W4A8
    "build_w4a8_gemm_kernel",
    "w4a8_gemm_tvm",
    # W4A16
    "build_w4a16_dequant_kernel",
    "dequant_nvfp4_to_bf16_tvm",
]
