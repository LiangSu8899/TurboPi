"""
Optimization module for Pi0.5 model inference on NVIDIA Jetson Thor.

Provides:
- CUDAGraphInference: CUDA graph optimization (Phase 3)
- HybridPi0Inference: TensorRT vision + PyTorch LLM (Phase 4/5)
"""

from openpi.optimization.cuda_graph_inference import (
    CUDAGraphInference,
    create_optimized_model,
)
from openpi.optimization.hybrid_inference import (
    HybridPi0Inference,
    TensorRTVisionEncoder,
    TensorRTActionExpert,
    benchmark_hybrid,
)

__all__ = [
    "CUDAGraphInference",
    "create_optimized_model",
    "HybridPi0Inference",
    "TensorRTVisionEncoder",
    "TensorRTActionExpert",
    "benchmark_hybrid",
]
