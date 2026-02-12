"""Neural network modules for OpenPI.

This package contains optimized modules for high-performance inference:
- w4a16_linear: W4A16 INT4 quantized linear layer with TVM kernel
- static_vlm: Static KV cache and VLM wrappers for CUDA Graph
- static_diffusion: Unrolled denoising loops for CUDA Graph
"""

from .w4a16_linear import W4A16Linear
from .static_vlm import (
    StaticKVCache,
    StaticKVCacheConfig,
    StaticVLMDecode,
    GraphedDenoiseStep,
    GraphedPI0,
)
from .static_diffusion import (
    DiffusionConfig,
    UnrolledDenoiseLoop,
    CapturedDenoiseChain,
    FullGraphedDenoise,
)

__all__ = [
    # W4A16 Linear
    "W4A16Linear",
    # Static VLM
    "StaticKVCache",
    "StaticKVCacheConfig",
    "StaticVLMDecode",
    "GraphedDenoiseStep",
    "GraphedPI0",
    # Static Diffusion
    "DiffusionConfig",
    "UnrolledDenoiseLoop",
    "CapturedDenoiseChain",
    "FullGraphedDenoise",
]
