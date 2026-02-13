"""Neural network modules for OpenPI.

This package contains optimized modules for high-performance inference:
- w4a16_linear: W4A16 INT4 quantized linear layer with TVM kernel
- static_vlm: Static KV cache and VLM wrappers for CUDA Graph
- static_diffusion: Unrolled denoising loops for CUDA Graph
"""

import os

# Lazy import W4A16Linear to avoid TVM dependency when not needed
def __getattr__(name):
    if name == "W4A16Linear":
        from .w4a16_linear import W4A16Linear
        return W4A16Linear
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Only import W4A16Linear if TVM is available
_W4A16Linear = None
if os.environ.get("OPENPI_SKIP_TVM") != "1":
    try:
        from .w4a16_linear import W4A16Linear as _W4A16Linear
    except (ImportError, OSError):
        pass  # TVM not available

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
from .graphed_denoise import (
    GraphedDenoiseConfig,
    GraphedDenoiseLoop,
    ChainedDenoiseGraphs,
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
    # Graphed Denoise
    "GraphedDenoiseConfig",
    "GraphedDenoiseLoop",
    "ChainedDenoiseGraphs",
]
