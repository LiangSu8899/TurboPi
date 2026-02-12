"""
Inference Pipeline for Pi0.5 VLA Model.

Provides:
- UnifiedPolicy: Consistent API across PyTorch/TensorRT/Pipelined backends
- TensorRT pipelines: High-performance inference with CUDA stream parallelization

Recommended Usage:
    from openpi.inference import UnifiedPolicy

    policy = UnifiedPolicy(
        checkpoint_dir="~/.cache/openpi/checkpoints/pi05_libero",
        backend="tensorrt_pipelined",  # or "pytorch", "pytorch_pipelined"
        num_denoising_steps=3,
    )

    result = policy.infer({
        "observation/image": image,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": "pick up the black bowl",
    })
"""

# All exports
__all__ = []

# TensorRT pipeline components
try:
    from openpi.inference.trt_pipeline import (
        PipelineConfig,
        PipelineStats,
        TensorRTEngineAsync,
        TensorRTPipeline,
        DoubleBuffer,
        run_pipeline_benchmark,
        print_comparison,
    )
    __all__.extend([
        "PipelineConfig",
        "PipelineStats",
        "TensorRTEngineAsync",
        "TensorRTPipeline",
        "DoubleBuffer",
        "run_pipeline_benchmark",
        "print_comparison",
    ])
except ImportError:
    pass

# Unified policy interface (recommended for most use cases)
try:
    from openpi.inference.unified_policy import (
        UnifiedPolicy,
        PolicyConfig,
        PyTorchBackend,
    )
    __all__.extend([
        "UnifiedPolicy",
        "PolicyConfig",
        "PyTorchBackend",
    ])

    # Optional backends in unified_policy
    try:
        from openpi.inference.unified_policy import PyTorchPipelinedBackend
        __all__.append("PyTorchPipelinedBackend")
    except ImportError:
        pass

    try:
        from openpi.inference.unified_policy import HybridTensorRTBackend
        __all__.append("HybridTensorRTBackend")
    except ImportError:
        pass

    try:
        from openpi.inference.unified_policy import TripleStreamBackend
        __all__.append("TripleStreamBackend")
    except ImportError:
        pass

    try:
        from openpi.inference.unified_policy import TorchTRTBackend
        __all__.append("TorchTRTBackend")
    except ImportError:
        pass

    try:
        from openpi.inference.unified_policy import FP8MLPBackend
        __all__.append("FP8MLPBackend")
    except ImportError:
        pass

    try:
        from openpi.inference.unified_policy import FlashFP8Backend
        __all__.append("FlashFP8Backend")
    except ImportError:
        pass

except ImportError:
    pass

# W4A16 TVM Backend (FASTEST MLP - 2.37-2.62x vs TRT FP8)
try:
    from openpi.inference.w4a16_backend import (
        W4A16TVMBackend,
    )
    __all__.append("W4A16TVMBackend")
except ImportError:
    W4A16TVMBackend = None

# Torch-TensorRT FP8 KV Cache Engine (TRT FP8 MLP + Flash Attention)
try:
    from openpi.inference.torch_trt_fp8_kv_cache import (
        TorchTRTFP8KVCacheEngine,
        TorchTRTFP8TransformerBlock,
        TorchTRTFP8KVCacheModel,
        SEQ_LEN as TORCH_TRT_FP8_SEQ_LEN,
        HIDDEN_SIZE as TORCH_TRT_FP8_HIDDEN_SIZE,
    )
    __all__.extend([
        "TorchTRTFP8KVCacheEngine",
        "TorchTRTFP8TransformerBlock",
        "TorchTRTFP8KVCacheModel",
        "TORCH_TRT_FP8_SEQ_LEN",
        "TORCH_TRT_FP8_HIDDEN_SIZE",
    ])
except ImportError:
    pass

# W4A16 TVM KV Cache
try:
    from openpi.inference.w4a16_tvm_kv_cache import (
        W4A16TVMKVCacheEngine,
    )
    __all__.append("W4A16TVMKVCacheEngine")
except Exception:
    # TVM may fail to load due to missing libraries
    pass
