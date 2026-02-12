"""
Inference Pipeline for Pi0.5 VLA Model.

Provides:
- UnifiedPolicy: Consistent API across PyTorch/TensorRT/Pipelined backends
- HybridTensorRTPipeline: Complete inference with optional TensorRT acceleration
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
except Exception:
    pass

# Hybrid TensorRT + PyTorch pipeline (complete inference flow)
try:
    from openpi.inference.hybrid_trt_pipeline import (
        HybridTensorRTPipeline,
        HybridPipelineConfig,
        PipelineStats as HybridPipelineStats,
    )
except Exception:
    pass

# Async PyTorch pipeline
try:
    from openpi.inference.async_pipeline import (
        AsyncVLAPipeline,
    )
except Exception:
    pass

# Unified policy interface (recommended for most use cases)
try:
    from openpi.inference.unified_policy import (
        UnifiedPolicy,
        PolicyConfig,
        PyTorchBackend,
        PyTorchPipelinedBackend,
        HybridTensorRTBackend,
        TripleStreamBackend,
        TorchTRTBackend,
        FP8MLPBackend,
        FlashFP8Backend,  # RECOMMENDED: Flash Attention + FP8 MLP
    )
except Exception:
    pass

# Flash Attention + FP8 KV Cache (FASTEST on Thor)
try:
    from openpi.inference.flash_fp8_kv_cache import (
        FlashFP8KVCacheModel,
        FlashFP8KVCacheEngine,
        FlashFP8TransformerBlock,
        FlashGQAAttention,
        load_flash_fp8_weights,
        benchmark_all_variants,
    )
except Exception:
    pass

# FP8 MLP (FP16 Attention + FP8 MLP using torch._scaled_mm)
try:
    from openpi.inference.fp8_mlp import (
        FP8MLP,
        FP8Linear,
        FP8HybridMLP,  # Recommended: FP8 gate/up, FP16 down
        FP8MLPWithStaticScale,
        quantize_weight_to_fp8,
        quantize_activation_to_fp8,
        benchmark_fp8_mlp,
    )
except Exception:
    pass

# FP8 KV Cache (FP16 Attention + FP8 MLP)
try:
    from openpi.inference.fp8_kv_cache import (
        FP8KVCacheModel,
        FP8KVCacheEngine,
        FP8TransformerBlock,
        load_fp8_weights,
    )
except Exception:
    pass

# Torch-TensorRT KV Cache Engine
try:
    from openpi.inference.torch_trt_kv_cache import (
        TorchTRTKVCacheEngine,
        create_torch_trt_engine,
    )
except Exception:
    pass

# Torch-TensorRT FP8 KV Cache Engine (TRT FP8 MLP + Flash Attention)
try:
    from openpi.inference.torch_trt_fp8_kv_cache import (
        TorchTRTFP8KVCacheEngine,
        TorchTRTFP8TransformerBlock,
        TorchTRTFP8KVCacheModel,
        SEQ_LEN as TORCH_TRT_FP8_SEQ_LEN,
        HIDDEN_SIZE as TORCH_TRT_FP8_HIDDEN_SIZE,
    )
except Exception:
    pass

# Triple Stream Pipeline (26+ Hz)
try:
    from openpi.inference.triple_stream_pipeline import (
        TripleStreamPipeline,
        TripleStreamConfig,
        PipelineStats as TriplePipelineStats,
    )
except Exception:
    pass

# Triple Buffer
try:
    from openpi.inference.triple_buffer import (
        TripleBuffer,
        TripleKVCacheBuffer,
        KVCacheBuffer,
    )
except Exception:
    pass

# KV Cache TensorRT
try:
    from openpi.inference.kv_cache_trt import (
        TensorRTKVCacheEngine,
        PyTorchKVCacheEngine,
        create_kv_cache_engine,
    )
except Exception:
    pass

__all__ = [
    # Recommended API
    "UnifiedPolicy",
    "PolicyConfig",
    # Hybrid Pipeline (complete flow with optional TensorRT)
    "HybridTensorRTPipeline",
    "HybridPipelineConfig",
    "HybridTensorRTBackend",
    # Triple Stream Pipeline (26+ Hz)
    "TripleStreamPipeline",
    "TripleStreamConfig",
    "TripleStreamBackend",
    "TriplePipelineStats",
    # Triple Buffer
    "TripleBuffer",
    "TripleKVCacheBuffer",
    "KVCacheBuffer",
    # KV Cache TensorRT
    "TensorRTKVCacheEngine",
    "PyTorchKVCacheEngine",
    "create_kv_cache_engine",
    # Torch-TensorRT KV Cache (recommended for Thor)
    "TorchTRTBackend",
    "TorchTRTKVCacheEngine",
    "create_torch_trt_engine",
    # Torch-TensorRT FP8 KV Cache (TRT FP8 MLP + Flash Attention)
    "TorchTRTFP8KVCacheEngine",
    "TorchTRTFP8TransformerBlock",
    "TorchTRTFP8KVCacheModel",
    "TORCH_TRT_FP8_SEQ_LEN",
    "TORCH_TRT_FP8_HIDDEN_SIZE",
    # FP8 MLP Backend (FP16 Attention + FP8 MLP)
    "FP8MLPBackend",
    "FP8MLP",
    "FP8Linear",
    "FP8HybridMLP",  # Recommended: 1.12x speedup
    "FP8MLPWithStaticScale",
    "FP8KVCacheModel",
    "FP8KVCacheEngine",
    "FP8TransformerBlock",
    "quantize_weight_to_fp8",
    "quantize_activation_to_fp8",
    "load_fp8_weights",
    "benchmark_fp8_mlp",
    # Flash Attention + FP8 Backend (RECOMMENDED - FASTEST on Thor)
    "FlashFP8Backend",
    "FlashFP8KVCacheModel",
    "FlashFP8KVCacheEngine",
    "FlashFP8TransformerBlock",
    "FlashGQAAttention",
    "load_flash_fp8_weights",
    "benchmark_all_variants",
    # Async PyTorch Pipeline
    "AsyncVLAPipeline",
    # Backend classes
    "PyTorchBackend",
    "PyTorchPipelinedBackend",
    # TensorRT internals
    "PipelineConfig",
    "PipelineStats",
    "HybridPipelineStats",
    "TensorRTEngineAsync",
    "TensorRTPipeline",
    "DoubleBuffer",
    "run_pipeline_benchmark",
    "print_comparison",
]
