# Changelog

All notable changes to this project compared to the original [OpenPi](https://github.com/Physical-Intelligence/openpi) repository.

## [1.3.1] - 2026-02-14

### Fixed - TRT FP8 Denoise Attention Mask Bug

Critical bug fix for TRT FP8 Denoise module that was causing 0% accuracy on LIBERO benchmark.

#### Root Cause

The original model's `denoise_step_with_cache` uses `F.scaled_dot_product_attention` **without any attention mask**:
```python
# Original model comment: "Since suffix attention mask is ALL TRUE (bidirectional),
# we can skip the mask entirely."
att_output = F.scaled_dot_product_attention(query_states, full_key_states, full_value_states)
```

The TRT implementation incorrectly applied an attention mask with `-10000` for padding positions, causing output divergence.

#### Impact

| Test | Before Fix | After Fix |
|------|------------|-----------|
| Single step cos_sim | 0.999 | 0.999 |
| 10-step loop cos_sim | **-0.18** | **0.995** |
| Step 9 (worst) | 0.54 | 0.997 |

#### Changes

- **`denoise_torch_trt_static.py`**: Removed `attn_mask` parameter from `SimpleAttention`, `SimpleDenoiseLayer`, `StaticDenoiseStep`, `StaticDenoiseLoop`; switched to SDPA without mask
- **Test files updated**: `debug_trt_loop_step.py`, `debug_trt_step_by_step.py`, `test_trt_vs_original.py`, `test_trt_10step_loop.py`, `libero_eval_trt_fp8_full.py`

#### Additional Fixes (from previous session)

1. **Action projection bias**: Changed `bias=False` to `bias=True` for `action_in_proj` and `action_out_proj`
2. **RoPE inv_freq precision**: Compute in BF16 to match original model precision

#### Documentation

- Added: `docs/debug-10-trt-denoise-fix.md`

---

## [1.3.0] - 2026-02-07

### Changed - Code Organization and Documentation

Major cleanup and documentation update for project maintainability.

#### README Updates
- Added comprehensive project structure documentation
- Added scripts categorization (Production, Benchmark, Build/Export, Test, Debug, Research)
- Added inference backends documentation (Production vs Experimental)
- Added optimization approaches summary with status indicators
- Added KV Reuse research archive section

#### Research Archives
- **KV Cache Reuse**: Archived as research (not production ready)
  - Full report: `docs/cliff_report.md`
  - Conclusion: Not viable for Diffusion Policy due to modal consistency requirements
  - Key finding: Vision-State temporal alignment is critical
  - Scripts archived: `experiment_kv_reuse_modality.py`, `experiment_synchronized_reuse.py`

---

## [1.2.0] - 2026-02-04

### Added - Torch-TRT FP8 Static Graph Optimization (NO ONNX)

**12.0 Hz achieved** with full Torch-TRT FP8 mixed precision pipeline, **bypassing ONNX** and **eliminating reformat operators**.

#### Key Innovation: Bypass ONNX, Use torch_tensorrt.compile()

This approach directly compiles PyTorch models to TensorRT without ONNX intermediate representation:
- **Static graph optimization** at compile time
- **No reformat operators** - TRT automatically optimizes tensor layouts
- **FP8 mixed precision** - MLP layers use FP8, Attention uses BF16

#### Performance Results (Verified on LIBERO)

| Denoising Steps | Total Latency | Hz | Vision | KV Cache | Denoise | Accuracy |
|-----------------|---------------|-----|--------|----------|---------|----------|
| 1 | **83.50 ms** | **12.0** | 17.26 ms | 52.14 ms | 10.10 ms | ✅ 100% |
| 2 | 93.07 ms | 10.7 | 17.12 ms | 52.13 ms | 19.70 ms | ✅ 100% |
| 3 | 102.81 ms | 9.7 | 17.22 ms | 52.14 ms | 29.38 ms | ✅ 100% |
| 5 | 122.21 ms | 8.2 | 17.22 ms | 52.23 ms | 48.87 ms | ✅ 100% |
| 10 | 171.36 ms | 5.8 | 17.26 ms | 52.13 ms | 97.69 ms | ✅ 100% |

#### Component Optimization Details

##### 1. Vision Encoder (torch_tensorrt FP16) - 2.03x speedup
```python
# Direct TRT compilation, NO ONNX export needed
import torch_tensorrt

vision_trt = torch_tensorrt.compile(
    VisionWrapper(vision_tower).half(),
    inputs=[torch_tensorrt.Input(shape=(1, 3, 224, 224), dtype=torch.float16)],
    enabled_precisions={torch.float16},
    workspace_size=4 << 30,
    min_block_size=1,
)
# 35ms → 17.2ms (2.03x speedup)
```

##### 2. KV Cache MLP (ModelOpt FP8 + torch_tensorrt) - 2.94x speedup
```python
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.utils import export_torch_mode

# FP8 quantization with calibration
mlp_fp8 = mtq.quantize(mlp, mtq.FP8_DEFAULT_CFG, forward_loop=calibrate)

# Compile to TRT (bypasses ONNX entirely, no reformat ops)
with export_torch_mode():
    trt_mlp = torch_tensorrt.compile(
        mlp_fp8,
        inputs=[x],
        enabled_precisions={torch.float16, torch.float8_e4m3fn},
        workspace_size=8 << 30,
    )
# Per layer: 3.33ms → 1.13ms (2.94x speedup)
# 18 layers: 59.89ms → 20.39ms
```

##### 3. Denoise Loop (CUDA Graph) - 2.59x speedup
```python
# Pre-allocate ALL tensors BEFORE graph capture
# (Critical: Cannot allocate tensors during capture)
self.static_timesteps = [torch.tensor([1.0], device=device), ...]

# Capture CUDA Graph
self.graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(self.graph):
    for step in range(self.num_steps):
        v_t = wrapper(..., self.static_timesteps[step])
        x_t = x_t + self.static_dt * v_t
    self.static_output = x_t

# Inference: replay captured graph
def infer(self, x_t_init):
    self.static_inputs['x_t'].copy_(x_t_init)
    self.graph.replay()
    return self.static_output.clone()
# 77.6ms → 30.0ms for 3 steps (2.59x speedup)
```

#### Compiled Operators (Torch-TRT FP8 Path)

| Operator | Precision | Implementation | Speedup |
|----------|-----------|----------------|---------|
| SigLIP Vision Encoder | FP16 | torch_tensorrt.compile() | 2.03x |
| MLP gate_proj | FP8 | ModelOpt + torch_tensorrt | 2.94x |
| MLP up_proj | FP8 | ModelOpt + torch_tensorrt | 2.94x |
| MLP down_proj | FP8 | ModelOpt + torch_tensorrt | 2.94x |
| GELU activation | FP16 | TRT fused | - |
| Attention QKV | BF16 | PyTorch SDPA | - |
| Attention Softmax | BF16 | PyTorch SDPA | - |
| RMSNorm | BF16 | torch.compile | - |
| Denoise Loop | BF16 | CUDA Graph | 2.59x |

#### Why This Approach Works

1. **Torch-TRT vs TRT Python API**:
   - TRT Python API causes Myelin segfault on Thor with FP8/FP4
   - Torch-TRT correctly handles FP8 quantization scale factors
   - No ONNX step = fewer precision conversion issues

2. **FP8 for MLP only**:
   - MLP is memory bandwidth bound (2048×16384 weights)
   - FP8 reduces memory transfer by 50%
   - Attention uses PyTorch SDPA for numerical stability

3. **CUDA Graph for Denoise**:
   - Small MLP dimension (1024×4096) limits TRT benefit
   - CUDA Graph eliminates Python overhead completely
   - Pre-allocated tensors avoid runtime memory allocation

#### Bug Fixes in This Version

1. **SEQ_LEN mismatch** (debug-06):
   - Fixed: 970 → 968 (256 image + 512 language + 200 state)
   - Root cause: TRT shape check fails silently, falls back to slow path

2. **Vision TRT dtype** (debug-07):
   - Fixed: Add `.to(torch.bfloat16)` after Vision TRT output
   - Root cause: Vision TRT outputs FP16, multi_modal_projector expects BF16

3. **Layer weight sharing** (debug-mix-05):
   - Fixed: Compile separate TRT MLP for each of 18 layers
   - Root cause: Initially all layers incorrectly shared layer 0's weights

#### New Files

| File | Description |
|------|-------------|
| `openpi/src/openpi/inference/torch_trt_fp8_kv_cache.py` | Torch-TRT FP8 KV Cache engine |
| `openpi/scripts/libero_eval_full_optimized.py` | Full optimized pipeline evaluation |
| `openpi/scripts/benchmark_torch_trt_fp8_libero.py` | FP8 benchmark comparison |
| `docs/debug-06.md` | SEQ_LEN fix + Vision/Denoise optimization |
| `docs/debug-07.md` | Vision TRT dtype fix + component profiling |
| `docs/debug-08.md` | Full Coverage TRT validation |
| `docs/debug-mix-05.md` | NVFP4 investigation (not viable on Thor) |
| `docs/todo-01.md` | Optimization status summary |
| `docs/record2.6.md` | Final optimization summary |

---

## [1.1.2] - 2026-02-01

### Fixed - TRT KV Cache Precision

Identified and documented TensorRT SDPA precision issues.

#### Root Cause
TensorRT's SDPA (Scaled Dot Product Attention) implementation differs numerically from PyTorch BF16, causing error accumulation across 18 transformer layers.

#### Solution
Use **explicit attention** (matmul + softmax) instead of `F.scaled_dot_product_attention` with **FP32 precision**.

#### Files Changed
| File | Changes |
|------|---------|
| `openpi/scripts/export_kv_cache_explicit_attn.py` | NEW: Explicit attention ONNX export |
| `openpi/docs/TRT_KV_CACHE_PRECISION_REPORT.md` | NEW: Root cause analysis |

#### Precision Comparison
| Method | Layer 17 Error | LIBERO Accuracy |
|--------|----------------|-----------------|
| SDPA TRT FP16 | max=16.9 | 0% |
| SDPA TRT BF16 | max=15.5 | 0% |
| Explicit Attention FP16 | max=42.0 | 0% |
| **Explicit Attention FP32** | **max=0.02** | **100%** |

---

## [1.1.1] - 2026-01-30

### Fixed - Quantile Normalization for UnifiedPolicy

Critical fix for UnifiedPolicy to match Pi0.5 training normalization scheme.

#### Problem
Initial UnifiedPolicy implementation had 0% success rate on LIBERO benchmarks due to missing normalization transforms.

#### Root Cause
Pi0.5 models use **quantile normalization** (q01/q99), not z-score normalization (mean/std).

#### Solution
Added quantile normalization to `unified_policy.py`:
```python
# State normalization: map [q01, q99] to [-1, 1]
normalized = (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

# Action unnormalization: map [-1, 1] back to [q01, q99]
unnormalized = (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
```

#### Benchmark Results
| Benchmark | Before Fix | After Fix |
|-----------|------------|-----------|
| LIBERO Spatial | 0% | **98%** |
| LIBERO 10 | 0% | **91%** |

---

## [1.1.0] - 2026-01-29

### Added - Unified Policy Interface

#### New Files
| File | Description |
|------|-------------|
| `openpi/src/openpi/inference/unified_policy.py` | Unified policy API for all backends |
| `openpi/scripts/libero_eval_unified.py` | Unified LIBERO evaluation script |

#### Usage
```python
from openpi.inference import UnifiedPolicy

policy = UnifiedPolicy(
    checkpoint_dir="/path/to/checkpoint",
    backend="tensorrt_pipelined",  # or "pytorch", "tensorrt"
    num_denoising_steps=3,
)
result = policy.infer(observation)
```

---

## [1.0.0] - 2026-01-29

### Initial Release

**4x speedup** over original JAX baseline.

#### Core Model Optimizations
| File | Description |
|------|-------------|
| `openpi/src/openpi/models_pytorch/pi0_pytorch.py` | PyTorch implementation with KV Cache |
| `openpi/src/openpi/models_pytorch/transformers_replace/` | Custom Gemma with adaRMSNorm |
| `openpi/src/openpi/inference/trt_pipeline.py` | TensorRT inference pipeline |

#### Key Optimizations
1. **KV Cache Implementation** (3.66x speedup)
   - Caches prefix K,V across denoising steps
   - Files: `pi0_pytorch.py`, `modeling_gemma.py`

2. **TensorRT Export** (5.8x for Vision, 2.6x for Denoise)
   - ONNX export for SigLIP, Gemma components
   - FP16 TensorRT engine building
   - Files: `export_onnx_components.py`, `trt_pipeline.py`

3. **adaRMSNorm Implementation**
   - Fused adaptive RMSNorm for action expert
   - Compatible with TensorRT export
   - Files: `modeling_gemma.py`

4. **Reduced Denoising Steps** (10 -> 3)
   - MSE increase: <1%
   - Throughput improvement: significant

#### Performance Results
| Metric | Original (JAX) | v1.0.0 (PyTorch) |
|--------|----------------|------------------|
| Throughput | 1.4 Hz | 5.6 Hz |
| Latency | 714 ms | 178 ms |
| LIBERO Spatial | - | 98% |
| LIBERO 10 | - | 91% |

#### Added Files
- `openpi/Dockerfile.libero_eval` - LIBERO evaluation container
- `Dockerfile` - Main deployment container
- 70+ benchmark scripts
- 50+ debug/validation scripts

---

## Experimental Features (Not Production Ready)

### NVFP4 Quantization (Archived 2026-02-04)

**Status**: Not viable on Thor platform.

**Investigation Summary** (from `docs/debug-mix-05.md`):

| Method | Latency | Precision | Issue |
|--------|---------|-----------|-------|
| TRT Python API FP8 | - | - | Myelin segfault |
| TRT Python API FP4 | - | - | Myelin segfault |
| Torch-TRT FP8 | 1.38ms | cos=0.9996 | ✅ Working |
| **Torch-TRT NVFP4** | 0.58ms | **cos=0.0004** | Scale ignored |
| PyTorch NVFP4 | 10.18ms | cos=-0.0005 | Scale ignored |
| W4A8 (FP4+FP8) | 9.64ms | cos=-0.0008 | Scale ignored |

**Conclusion**: NVFP4 cannot be used on Thor due to TRT scale factor handling issues. Use Torch-TRT FP8 instead.

**Related GitHub Issues**:
- [#4590](https://github.com/NVIDIA/TensorRT/issues/4590): Thor FP8/FP4 silent fallback
- [#4599](https://github.com/NVIDIA/TensorRT/issues/4599): Thor ViT FP8 low performance
- [#8974](https://github.com/NVIDIA/TensorRT-LLM/issues/8974): FP8/NVFP4 kernel not replaced

### KV Cache Reuse (Archived 2026-02-07)

**Status**: Research complete, not recommended for production.

**Files:**
- `openpi/scripts/experiment_kv_reuse_modality.py`
- `openpi/scripts/experiment_synchronized_reuse.py`
- `openpi/src/openpi/inference/async_kv_reuse_pipeline.py`
- `docs/cliff_report.md` (full research report)

**Key Findings:**
- Vision-State temporal alignment is critical for Diffusion Policy
- Full KV reuse (vision + state together) performs better than partial
- Safe threshold (0.985) only provides 1.14x speedup
- Aggressive thresholds cause severe accuracy drops (100% -> 36-57%)

### TRT ONNX-based Engines (Deprecated)

**Status**: Replaced by Torch-TRT approach in v1.2.0.

**Issue**: TRT ONNX path has precision issues with SDPA and requires complex export handling.

**Files (legacy)**:
- `openpi/src/openpi/inference/trt_mixed_precision.py`
- `openpi/scripts/build_kv_cache_int8.py`
- `openpi/scripts/export_kv_cache_trt.py`

**Recommendation**: Use `torch_trt_fp8_kv_cache.py` instead for better precision and no ONNX requirement.

### Flash Attention Integration (Experimental)

**Status**: Implemented but not production-validated.

**Files:**
- `openpi/src/openpi/inference/flash_fp8_kv_cache.py`
- `openpi/src/openpi/inference/flashattn_kv_cache.py`
- `openpi/scripts/benchmark_flashattn.py`
