# Pi0.5 TRT FP8 Full Optimized Pipeline - Performance Baseline

**Source**: debug-07.md (LIBERO验证通过)
**Tag**: v1.2.0_full_optimized
**Platform**: NVIDIA Jetson Thor (SM110)
**Accuracy**: 100% (所有配置)

## Backend Configuration

| Component | Backend | Technology | Status |
|-----------|---------|------------|--------|
| Vision Encoder | TRT FP16 | torch_tensorrt.compile + dtype conversion | ✅ 2.03x speedup |
| KV Cache | TRT FP8 MLP + Flash Attention | ModelOpt FP8 + torch_tensorrt | ✅ 1.7x speedup |
| Denoise Loop | CUDA Graph | torch.cuda.CUDAGraph (静态图) | ✅ 2.6x speedup |

## Executive Summary

| Denoising Steps | Total Latency | Frequency | Vision | KV Cache | Denoise |
|-----------------|---------------|-----------|--------|----------|---------|
| **1** | **83.50 ms** | **12.0 Hz** | 17.26 ms | 52.14 ms | 10.10 ms |
| 2 | 93.07 ms | 10.7 Hz | 17.12 ms | 52.13 ms | 19.70 ms |
| 3 | 102.81 ms | 9.7 Hz | 17.22 ms | 52.14 ms | 29.38 ms |
| 5 | 122.21 ms | 8.2 Hz | 17.22 ms | 52.23 ms | 48.87 ms |
| 10 | 171.36 ms | 5.8 Hz | 17.26 ms | 52.13 ms | 97.69 ms |

## Performance Analysis

### 1 Denoising Step (最快配置)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Vision TRT FP16 | 17.26 | 20.7% |
| KV Cache TRT FP8 | **52.14** | **62.4%** |
| Denoise CUDA Graph | 10.10 | 12.1% |
| **E2E Total** | **83.50** | 100% |

### 3 Denoising Steps (平衡配置)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Vision TRT FP16 | 17.22 | 16.7% |
| KV Cache TRT FP8 | **52.14** | **50.7%** |
| Denoise CUDA Graph (3x) | 29.38 | 28.6% |
| **E2E Total** | **102.81** | 100% |

### 10 Denoising Steps (最高质量)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Vision TRT FP16 | 17.26 | 10.1% |
| KV Cache TRT FP8 | 52.13 | 30.4% |
| Denoise CUDA Graph (10x) | **97.69** | **57.0%** |
| **E2E Total** | **171.36** | 100% |

## Denoise 线性模型

```
Denoise Time (ms) ≈ 10.1 × steps
```

| Steps | 预测 (ms) | 实测 (ms) | 误差 |
|-------|-----------|-----------|------|
| 1 | 10.1 | 10.10 | 0% |
| 2 | 20.2 | 19.70 | -2.5% |
| 3 | 30.3 | 29.38 | -3.0% |
| 5 | 50.5 | 48.87 | -3.2% |
| 10 | 101.0 | 97.69 | -3.3% |

## Bottleneck Analysis

### Primary Bottleneck: KV Cache (~52 ms, 50-62%)

KV Cache TRT FP8 包含:
- **Embed Prefix**: ~5-8 ms (image + language token fusion)
- **TRT FP8 Prefill**: ~47 ms (18层 transformer, MLP + Flash Attention)

KV Cache 细分:
- MLP FP8 (18层): ~35 ms (每层 ~2 ms)
- Flash Attention (18层): ~12 ms (每层 ~0.7 ms)

### Vision Encoder (~17 ms, 10-21%)

- TRT FP16 编译的 SigLIP Vision Tower
- 处理 2 张图像 (base + wrist) × 224×224
- 包含 FP16→BF16 dtype conversion

### Denoise CUDA Graph (~10 ms/step)

- Action Expert 18层 transformer
- 每步 ~10.1 ms (CUDA Graph replay 开销极小)
- 线性扩展，适合减少 steps 提升帧率

## Comparison with Baseline

| Metric | PyTorch BF16 | TRT FP8 Optimized | Speedup |
|--------|--------------|-------------------|---------|
| Total (3 steps) | 176 ms | **102.81 ms** | **1.71x** |
| Vision | 35 ms | **17.22 ms** | **2.03x** |
| KV Cache | 88 ms | **52.14 ms** | **1.69x** |
| Denoise/step | 18 ms | **9.8 ms** | **1.84x** |
| Hz (3 steps) | 5.7 | **9.7** | **1.70x** |

## Target Performance

| Config | Current | Target | Gap |
|--------|---------|--------|-----|
| 1-step | 83.50 ms (12.0 Hz) | 50 ms (20 Hz) | **33.5 ms** |
| 3-step | 102.81 ms (9.7 Hz) | 80 ms (12.5 Hz) | **22.8 ms** |

### Optimization Opportunities

| 优先级 | 组件 | 当前 | 优化方案 | 预期收益 |
|--------|------|------|----------|----------|
| **1** | KV Cache MLP | 35 ms | W4A16 INT4 (2.5x) | **-21 ms** |
| **2** | Vision | 17 ms | INT8 量化 | ~-5 ms |
| **3** | Embed | 5 ms | Kernel fusion | ~-2 ms |

## Raw Data (JSON)

```json
{
  "1_steps": {
    "e2e_total_ms": 83.50,
    "vision_ms": 17.26,
    "kv_cache_ms": 52.14,
    "denoise_ms": 10.10,
    "hz": 12.0
  },
  "3_steps": {
    "e2e_total_ms": 102.81,
    "vision_ms": 17.22,
    "kv_cache_ms": 52.14,
    "denoise_ms": 29.38,
    "hz": 9.7
  },
  "10_steps": {
    "e2e_total_ms": 171.36,
    "vision_ms": 17.26,
    "kv_cache_ms": 52.13,
    "denoise_ms": 97.69,
    "hz": 5.8
  }
}
```

## Test Environment

- **Platform**: NVIDIA Jetson Thor (SM110)
- **GPU Memory**: 128 GB unified
- **Docker Image**: turbo_pi_libero:latest
- **Checkpoint**: pi05_libero
- **Model**: Pi0.5 VLA (3B parameters)
- **Validation**: LIBERO spatial (3 tasks × 3 trials, 100% accuracy)
