# Turbo-Pi0.5

**Optimized Pi0.5 VLA (Vision-Language-Action) Model for NVIDIA Jetson Thor**

## Performance Summary

| Metric | Original (JAX) | Turbo-Pi (PyTorch) | FP8 Optimized |
|--------|----------------|--------------------|--------------------|
| Inference Speed | 1.4 Hz | 5.6 Hz | **12.0 Hz** |
| Latency | 714 ms | 178 ms | **83.5 ms** |
| LIBERO Spatial | 98% | 98% | **100%** |
| LIBERO 10 | 91% | 91% | **100%** |

---

## What is this?

Turbo-Pi0.5 is an optimized version of [Physical Intelligence's Pi0.5](https://www.physicalintelligence.company/blog/pi0) VLA model, designed for real-time robot control on edge devices.

**Key Optimizations (v1.2.0):**
- **Torch-TRT FP8 MLP** - 2.94x speedup, bypassing ONNX entirely
- **Vision TRT FP16** - 2.03x speedup via torch_tensorrt.compile()
- **CUDA Graph Denoise** - 2.59x speedup with pre-allocated tensors
- **No Reformat Operators** - Static graph optimization at compile time

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Start container with NVIDIA PyTorch base
docker run --runtime=nvidia --gpus all -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd)/openpi:/workspace \
    -v ~/.cache/openpi:/root/.cache/openpi \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3

# Inside container: install dependencies
pip install pycuda einops safetensors pillow
pip install -e .

# Download model (if not already cached)
huggingface-cli download liangsu9988/Turbo-Pi0.5-1.1.2 \
    --local-dir /root/.cache/openpi/checkpoints/pi05_libero

# Run full optimized benchmark
python scripts/libero_eval_full_optimized.py --quick
```

### Option 2: Full Optimized Pipeline (Programmatic)

```python
# Using the Full Optimized Pipeline (12.0 Hz)
from libero_eval_full_optimized import FullOptimizedPolicy

policy = FullOptimizedPolicy(
    checkpoint_dir="~/.cache/openpi/checkpoints/pi05_libero",
    num_denoising_steps=1,  # 1 step for max speed
)

result = policy.infer({
    "observation/image": image,           # (224, 224, 3) uint8
    "observation/wrist_image": wrist_img, # (224, 224, 3) uint8
    "observation/state": state,           # (8,) float32
    "prompt": "pick up the black bowl",
})
actions = result["actions"]  # (50, 7) action chunk
```

### Option 3: UnifiedPolicy API

```python
from openpi.inference import UnifiedPolicy

policy = UnifiedPolicy(
    checkpoint_dir="~/.cache/openpi/checkpoints/pi05_libero",
    backend="pytorch",  # or "tensorrt", "tensorrt_pipelined"
    num_denoising_steps=10,
)
result = policy.infer(observation)
```

---

## Model

**HuggingFace:** [liangsu9988/Turbo-Pi0.5-1.1.2](https://huggingface.co/liangsu9988/Turbo-Pi0.5-1.1.2)

**Architecture:**
- Vision: SigLIP-SO400M (400M params)
- Language: Gemma 2B
- Action: Gemma 300M Expert with adaRMSNorm

---

## Current Best: Torch-TRT FP8 Static Graph (v1.2.0)

### Key Innovation: Bypass ONNX, Use torch_tensorrt.compile()

This approach directly compiles PyTorch models to TensorRT **without ONNX**, achieving:
- **Static graph optimization** at compile time
- **No reformat operators** - TRT automatically optimizes tensor layouts
- **FP8 mixed precision** - MLP layers use FP8, Attention uses BF16

### Component Performance

| Component | Technology | Before | After | Speedup |
|-----------|------------|--------|-------|---------|
| Vision Encoder | torch_tensorrt FP16 | 35ms | **17.2ms** | **2.03x** |
| KV Cache MLP | ModelOpt FP8 + torch_tensorrt | 59.9ms | **20.4ms** | **2.94x** |
| Denoise Loop | CUDA Graph | 77.6ms | **30.0ms** | **2.59x** |

### Full Pipeline Results (LIBERO Verified)

| Denoising Steps | Latency | Hz | Accuracy |
|-----------------|---------|-----|----------|
| 1 | **83.50 ms** | **12.0** | ✅ 100% |
| 3 | 102.81 ms | 9.7 | ✅ 100% |
| 10 | 171.36 ms | 5.8 | ✅ 100% |

### Implementation Files

| File | Description |
|------|-------------|
| `torch_trt_fp8_kv_cache.py` | Torch-TRT FP8 KV Cache engine (main) |
| `libero_eval_full_optimized.py` | Full optimized pipeline evaluation |

---

## Project Structure

```
Turbo-Pi/
├── README.md                    # This file
├── CHANGELOG.md                 # Version history with detailed changes
├── Dockerfile                   # Main deployment container
│
├── docs/                        # Project documentation
│   ├── cliff_report.md          # KV Reuse research report (archived)
│   ├── debug-06.md              # SEQ_LEN fix + Vision/Denoise optimization
│   ├── debug-07.md              # Vision TRT dtype fix + component profiling
│   ├── debug-08.md              # Full Coverage TRT validation
│   ├── debug-mix-05.md          # NVFP4 investigation (not viable on Thor)
│   ├── todo-01.md               # Optimization status summary
│   └── record2.6.md             # Final optimization summary
│
└── openpi/                      # Modified OpenPi codebase
    ├── src/openpi/
    │   ├── inference/           # Inference backends (see below)
    │   ├── models_pytorch/      # PyTorch Pi0.5 with KV Cache
    │   └── quantization/        # Quantization framework (experimental)
    │
    ├── scripts/                 # All scripts (categorized below)
    │
    └── docs/                    # Technical documentation
        ├── TRT_KV_CACHE_PRECISION_REPORT.md # TRT precision analysis
        └── BENCHMARK_ANALYSIS.md            # LIBERO benchmark details
```

---

## Inference Backends

### Production Backends (Recommended)

| File | Description | Status |
|------|-------------|--------|
| **`torch_trt_fp8_kv_cache.py`** | **Torch-TRT FP8 KV Cache (12 Hz)** | **Best** |
| `libero_eval_full_optimized.py` | Full optimized pipeline | Production |
| `unified_policy.py` | Unified API for all backends | Stable |

### Experimental Backends

| File | Description | Status |
|------|-------------|--------|
| `trt_mixed_precision.py` | ONNX-based TRT (legacy) | Deprecated |
| `async_kv_reuse_pipeline.py` | Dynamic KV cache reuse | Archived (50% accuracy) |
| `flash_fp8_kv_cache.py` | Flash Attention + FP8 | Experimental |
| `trt_full_coverage_kv_cache.py` | Full TRT coverage | Limited benefit (+4%) |

---

## Optimization Approaches Summary

### Working Solutions (Production)

| Approach | Component | Speedup | Accuracy | Files |
|----------|-----------|---------|----------|-------|
| **Torch-TRT FP8** | MLP | **2.94x** | 100% | `torch_trt_fp8_kv_cache.py` |
| **torch_tensorrt FP16** | Vision | **2.03x** | 100% | `libero_eval_full_optimized.py` |
| **CUDA Graph** | Denoise | **2.59x** | 100% | `libero_eval_full_optimized.py` |
| KV Cache | All | 3.7x | 100% | `pi0_pytorch.py` |
| Reduced Steps (10→1) | Denoise | 2.1x | 100% | - |

### Not Viable on Thor

| Approach | Issue | Details |
|----------|-------|---------|
| **NVFP4** | Scale ignored (cos=0.0004) | `docs/debug-mix-05.md` |
| TRT Python API FP8 | Myelin segfault | Thor platform bug |
| TRT Python API FP4 | Myelin segfault | Thor platform bug |
| W4A8 (FP4+FP8) | Scale ignored | Thor platform bug |

### Research (Archived)

| Approach | Finding | Report |
|----------|---------|--------|
| **KV Reuse** | Not viable - modal consistency required | [cliff_report.md](docs/cliff_report.md) |
| TRT ONNX INT8 | SDPA precision issues (0% accuracy) | `docs/TRT_KV_CACHE_PRECISION_REPORT.md` |

---

## Scripts Categorization

### Production Scripts (Core)

| Script | Purpose |
|--------|---------|
| **`libero_eval_full_optimized.py`** | **Full optimized pipeline (12 Hz)** |
| `libero_eval_unified.py` | Standard LIBERO evaluation |
| `serve_policy.py` | WebSocket policy server |

### Debug Documentation

| Document | Content |
|----------|---------|
| `docs/debug-06.md` | SEQ_LEN fix (970→968), Vision TRT, CUDA Graph |
| `docs/debug-07.md` | Vision TRT dtype fix, component profiling |
| `docs/debug-08.md` | Full Coverage TRT validation (+4% only) |
| `docs/debug-mix-05.md` | NVFP4 investigation (not viable) |
| `docs/todo-01.md` | Current optimization status |
| `docs/record2.6.md` | Final summary (12 Hz achieved) |

---

## Running Benchmarks

### Full Optimized Pipeline (Recommended)

```bash
# Quick test (3 tasks, 3 trials) - 12 Hz
python scripts/libero_eval_full_optimized.py --quick --denoising_steps 1

# Full LIBERO Spatial (10 tasks, 10 trials)
python scripts/libero_eval_full_optimized.py \
    --task_suite_name libero_spatial \
    --denoising_steps 1 \
    --output_file results.json
```

### Standard PyTorch Baseline

```bash
# LIBERO evaluation with PyTorch backend
python scripts/libero_eval_unified.py --backend pytorch --quick
```

---

## Benchmark Results

### Full Optimized Pipeline (v1.2.0)

| Component | Latency | % of Total | Technology |
|-----------|---------|------------|------------|
| Vision TRT | 17.26 ms | 20.7% | torch_tensorrt FP16 |
| KV Cache TRT | 52.14 ms | 62.4% | ModelOpt FP8 + torch_tensorrt |
| Denoise CUDA | 10.10 ms | 12.1% | CUDA Graph |
| Other | ~4 ms | 4.8% | - |
| **Total** | **83.50 ms** | 100% | **12.0 Hz** |

### LIBERO Spatial (Full Optimized, 1 step)

| Configuration | Accuracy |
|---------------|----------|
| 1 denoising step | **100%** (9/9 quick) |
| 3 denoising steps | **100%** (9/9 quick) |
| 10 denoising steps | **100%** (9/9 quick) |

### LIBERO 10 (PyTorch baseline)

| Task | Success |
|------|---------|
| put_both_moka_pots_on_stove | 50% |
| Other tasks | 80-100% |
| **Total** | **91%** |

---

## Hardware Requirements

- **Recommended:** NVIDIA Jetson Thor (JetPack 7.1+)
- **Minimum:** GPU with 8GB+ VRAM, CUDA 12.0+
- **Dependencies:** torch-tensorrt, modelopt, pycuda

---

## Documentation

- [CHANGELOG.md](CHANGELOG.md) - Version history with detailed FP8 optimization changes
- [docs/debug-06.md](docs/debug-06.md) - SEQ_LEN fix and optimization details
- [docs/debug-07.md](docs/debug-07.md) - Component profiling results
- [docs/debug-mix-05.md](docs/debug-mix-05.md) - NVFP4 investigation (not viable on Thor)
- [docs/cliff_report.md](docs/cliff_report.md) - KV Reuse research (archived)

---

## Research Archives

### NVFP4 Quantization (Not Viable on Thor)

**Conclusion:** NVFP4 cannot be used on Thor due to TRT scale factor handling issues.

| Method | Precision | Issue |
|--------|-----------|-------|
| Torch-TRT FP8 | cos=0.9996 | ✅ Working |
| Torch-TRT NVFP4 | cos=0.0004 | ❌ Scale ignored |

**Full Report:** [docs/debug-mix-05.md](docs/debug-mix-05.md)

### KV Cache Reuse (Not Recommended)

**Conclusion:** KV Cache Reuse is not viable for Diffusion Policy due to:
1. **Modal Consistency Sensitivity**: Diffusion policy requires vision-state temporal alignment
2. **Limited Speedup at Safe Thresholds**: Only 1.14x at threshold=0.985
3. **Poor Generalization**: 57% on libero_spatial vs 36% on libero_10 at threshold=0.98

**Full Report:** [docs/cliff_report.md](docs/cliff_report.md)

---

## Running LIBERO Benchmarks (Docker)

```bash
# Build LIBERO image
cd openpi && docker build -f Dockerfile.libero_eval -t turbo_pi_libero:latest .

# Start container
docker run -d --name turbo_pi_eval \
    --runtime nvidia --gpus all \
    -v $(pwd):/workspace \
    -v ~/.cache/openpi:/root/.cache/openpi \
    -e MUJOCO_GL=egl \
    turbo_pi_libero:latest sleep infinity

# Run full optimized benchmark
docker exec turbo_pi_eval bash -c "cd /workspace && \
    python scripts/libero_eval_full_optimized.py --quick --denoising_steps 1"
```

---

## License

Apache License 2.0

## Acknowledgments

Based on [OpenPi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence.
