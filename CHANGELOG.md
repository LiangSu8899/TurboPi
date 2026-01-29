# Changelog

All notable changes to this project compared to the original [OpenPi](https://github.com/Physical-Intelligence/openpi) repository.

## [1.0.0] - 2026-01-29

### Added Files

#### Core Model Optimizations
| File | Description |
|------|-------------|
| `openpi/src/openpi/models_pytorch/pi0_pytorch.py` | PyTorch implementation with KV Cache support |
| `openpi/src/openpi/models_pytorch/transformers_replace/` | Custom Gemma model with adaRMSNorm and KV Cache |
| `openpi/src/openpi/inference/trt_pipeline.py` | TensorRT inference pipeline |

#### Benchmark Scripts
| File | Description |
|------|-------------|
| `openpi/scripts/benchmark_thor.py` | Main benchmark script for Jetson Thor |
| `openpi/scripts/benchmark_baseline.py` | Baseline PyTorch benchmark |
| `openpi/scripts/benchmark_kv_cache.py` | KV Cache performance comparison |
| `openpi/scripts/benchmark_trt_e2e.py` | TensorRT end-to-end benchmark |
| `openpi/scripts/benchmark_tensorrt.py` | TensorRT engine benchmark |
| `openpi/scripts/benchmark_pipeline.py` | Pipeline benchmark with dual streams |
| `openpi/scripts/benchmark_steps_accuracy.py` | Denoising steps vs accuracy analysis |
| `openpi/scripts/benchmark_quick.py` | Quick performance test |
| `openpi/scripts/benchmark_quantized.py` | Quantized model benchmark |
| `openpi/scripts/benchmark_fp8_native.py` | FP8 native benchmark |

#### Utility Scripts
| File | Description |
|------|-------------|
| `openpi/scripts/check_adarms_weights.py` | Verify adaRMSNorm weight loading |
| `openpi/scripts/check_weights.py` | General weight verification |
| `openpi/scripts/compare_full_inference.py` | Compare JAX vs PyTorch inference |
| `openpi/scripts/compare_jax_pytorch.py` | JAX/PyTorch output comparison |
| `openpi/scripts/export_onnx_components.py` | ONNX export for TensorRT |
| `openpi/scripts/rebuild_trt_engines.py` | TensorRT engine rebuilder |
| `openpi/scripts/run_libero_pytorch.sh` | LIBERO evaluation runner |

#### Quantization (Experimental)
| File | Description |
|------|-------------|
| `openpi/src/openpi/quantization/` | FP4/FP8 quantization framework |
| `openpi/scripts/quantize_model.py` | Model quantization entry point |
| `openpi/scripts/validate_quantization.py` | Quantization accuracy validation |

#### Docker
| File | Description |
|------|-------------|
| `openpi/Dockerfile.libero_eval` | LIBERO evaluation container |
| `Dockerfile` | Main deployment container |

### Modified Files

#### Model Implementation
| File | Changes |
|------|---------|
| `openpi/src/openpi/models/pi0.py` | Added PyTorch model loading support |
| `openpi/src/openpi/policies/pi0_policy.py` | KV Cache integration |

#### Configuration
| File | Changes |
|------|---------|
| `openpi/pyproject.toml` | Added nvidia-modelopt, tensorrt dependencies |

### Key Optimizations

1. **KV Cache Implementation**
   - Files: `pi0_pytorch.py`, `modeling_gemma.py`
   - Caches prefix K,V across denoising steps
   - 3.66x speedup for cached vs non-cached inference

2. **TensorRT Export**
   - Files: `export_onnx_components.py`, `trt_pipeline.py`
   - ONNX export for SigLIP, Gemma components
   - FP16 TensorRT engine building

3. **adaRMSNorm Custom Implementation**
   - Files: `modeling_gemma.py`
   - Fused adaptive RMSNorm for action expert
   - Compatible with TensorRT export

4. **Reduced Denoising Steps**
   - Default: 10 → 3 steps
   - MSE increase: <1%
   - Throughput: 7.6 Hz → 20.6 Hz

### Performance Results

| Metric | Original | Optimized |
|--------|----------|-----------|
| Throughput | 1.4 Hz | 20.6 Hz |
| Latency | 714 ms | 48.5 ms |
| Memory | ~8 GB | ~7.65 GB |
| LIBERO Spatial | - | 98% |

### Directory Structure

```
TurboPi/
├── README.md              # User documentation
├── CHANGELOG.md           # This file
├── Dockerfile             # Deployment container
├── .gitignore
└── openpi/                # Modified OpenPi codebase
    ├── src/openpi/
    │   ├── models_pytorch/    # [NEW] PyTorch implementations
    │   ├── inference/         # [NEW] TensorRT pipeline
    │   └── quantization/      # [NEW] Quantization framework
    ├── scripts/               # [MODIFIED] Added benchmarks
    ├── examples/libero/       # LIBERO evaluation
    └── Dockerfile.libero_eval # [NEW] Eval container
```
