# Pi0.5 TRT FP8 Mixed Quantization - Detailed Performance Breakdown

**Generated**: 2026-02-12 08:49:18
**Tag**: v1.2.0_baseline
**Platform**: NVIDIA Jetson Thor (SM110)
**Backend**: PyTorch BF16 Baseline (TRT FP8 compilation not enabled in this run)

## Executive Summary

| Denoising Steps | Total Latency | Frequency | Vision | Embed | KV Cache | Denoise |
|-----------------|---------------|-----------|--------|-------|----------|---------|
| **1** | **142.6 ms** | **7.0 Hz** | 23.4 ms | 35.1 ms | 88.4 ms | 18.0 ms |
| **3** | **179.3 ms** | **5.6 Hz** | 23.4 ms | 35.2 ms | 88.5 ms | 54.0 ms |
| **10** | **302.4 ms** | **3.3 Hz** | 23.3 ms | 34.9 ms | 88.3 ms | 179.7 ms |

## Performance Breakdown Overview

```
Pipeline Latency Distribution (1 denoising step = 142.6 ms)

┌──────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  Vision (16.4%)          Embed (24.6%)      KV Cache (62.0%)         │
│  ████████                 ████████████       ██████████████████████  │
│  23.4 ms                  35.1 ms            88.4 ms                 │
│                                                                       │
│  + Denoise (12.6%)                                                   │
│    ██████                                                            │
│    18.0 ms                                                           │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## 1 Denoising Step - Detailed Breakdown

| Component | Time (ms) | Std (ms) | % of Total | Notes |
|-----------|-----------|----------|------------|-------|
| Vision (base) | 11.77 | 0.23 | 8.3% | SigLIP encoder (base camera) |
| Vision (wrist) | 11.59 | 0.16 | 8.1% | SigLIP encoder (wrist camera) |
| **Vision Total** | **23.35** | - | **16.4%** | 2x SigLIP forward passes |
| Embed Prefix | 35.12 | 0.23 | 24.6% | Image + Language token fusion |
| KV Cache Prefill | 88.45 | 0.13 | **62.0%** | PaliGemma 18 layers (**瓶颈**) |
| Denoise Loop (1x) | 18.00 | 0.32 | 12.6% | Action Expert 18 layers |
| **E2E Total** | **142.61** | 0.61 | 100% | **7.0 Hz** |

### KV Cache 分析 (主要优化目标)

KV Cache Prefill 占用 **62.0%** 的总延迟，是最大的瓶颈:
- PaliGemma 18层 Transformer
- 每层: Attention (SDPA) + MLP
- MLP: gate_proj (2048→16384) + up_proj (2048→16384) + down_proj (16384→2048)
- 预计 MLP 占 KV Cache 的 ~70% (约 62 ms)
- 预计 Attention 占 ~30% (约 26 ms)

### Denoise Analysis

- Single step: 17.90 ms
- 1 step total: 18.00 ms
- Loop overhead: 0.11 ms (0.6%)

---

## 3 Denoising Steps - Detailed Breakdown

| Component | Time (ms) | Std (ms) | % of Total | Notes |
|-----------|-----------|----------|------------|-------|
| Vision (base) | 11.72 | 0.19 | 6.5% | SigLIP encoder |
| Vision (wrist) | 11.64 | 0.17 | 6.5% | SigLIP encoder |
| **Vision Total** | **23.36** | - | **13.0%** | |
| Embed Prefix | 35.18 | 0.24 | 19.6% | Image + Language fusion |
| KV Cache Prefill | 88.48 | 0.17 | **49.3%** | PaliGemma 18 layers |
| Denoise Loop (3x) | 54.01 | 1.35 | **30.1%** | Action Expert |
| **E2E Total** | **179.32** | 1.59 | 100% | **5.6 Hz** |

### Denoise Analysis

- Single step: 17.81 ms
- 3 steps total: 54.01 ms
- Per-step average: 18.00 ms
- Loop overhead: 0.57 ms (1.1%)

---

## 10 Denoising Steps - Detailed Breakdown

| Component | Time (ms) | Std (ms) | % of Total | Notes |
|-----------|-----------|----------|------------|-------|
| Vision (base) | 11.69 | 0.25 | 3.9% | SigLIP encoder |
| Vision (wrist) | 11.61 | 0.14 | 3.8% | SigLIP encoder |
| **Vision Total** | **23.30** | - | **7.7%** | |
| Embed Prefix | 34.87 | 0.20 | 11.5% | Image + Language fusion |
| KV Cache Prefill | 88.27 | 0.22 | 29.2% | PaliGemma 18 layers |
| Denoise Loop (10x) | 179.69 | 0.81 | **59.4%** | Action Expert (**瓶颈**) |
| **E2E Total** | **302.40** | 0.79 | 100% | **3.3 Hz** |

### Denoise Analysis

- Single step: 18.00 ms
- 10 steps total: 179.69 ms
- Per-step average: 17.97 ms
- Loop overhead: -0.34 ms (negligible)

---

## 优化机会分析

### 当前瓶颈排序 (1 denoising step)

| 优先级 | 组件 | 当前耗时 | 占比 | 优化方案 | 预期收益 |
|--------|------|----------|------|----------|----------|
| **1** | KV Cache MLP | ~62 ms | ~43% | TRT FP8 (2.94x) | **-41 ms** |
| **2** | KV Cache Attn | ~26 ms | ~18% | Flash Attention | ~-10 ms |
| **3** | Embed Prefix | 35 ms | 25% | Kernel Fusion | ~-5 ms |
| **4** | Vision | 23 ms | 16% | TRT INT8 | ~-8 ms |
| **5** | Denoise | 18 ms | 13% | CUDA Graph | ~-3 ms |

### 预期优化后性能

```
当前 (BF16 baseline):     142.6 ms → 7.0 Hz
优化后 (TRT FP8 + Flash): ~75 ms  → 13.3 Hz (预期)
```

### 详细优化路线图

#### Phase 1: KV Cache MLP TRT FP8 (最高优先级)
- **目标**: 62 ms → 21 ms (2.94x speedup)
- **方法**: ModelOpt FP8 量化 + Torch-TensorRT 编译
- **预期收益**: -41 ms

#### Phase 2: KV Cache Attention 优化
- **目标**: 26 ms → 16 ms
- **方法**: Flash Attention 2 集成
- **预期收益**: -10 ms

#### Phase 3: Embed Prefix 融合
- **目标**: 35 ms → 30 ms
- **方法**: Vision embedding + Language embedding 融合
- **预期收益**: -5 ms

#### Phase 4: Vision Encoder TRT
- **目标**: 23 ms → 15 ms
- **方法**: TRT FP16/INT8 编译
- **预期收益**: -8 ms

#### Phase 5: Denoise Loop CUDA Graph
- **目标**: 18 ms → 15 ms
- **方法**: CUDA Graph capture
- **预期收益**: -3 ms

---

## 模型架构参考

```
PI0Pytorch (Pi0.5 VLA)
├── Vision Encoder (SigLIP-SO400M)
│   ├── Input: 224×224×3
│   ├── Output: 256 patch tokens
│   └── Time: ~11.7 ms per image
│
├── Embed Prefix
│   ├── Vision tokens: 2×256 = 512
│   ├── Language tokens: ~200
│   ├── Total: ~712 tokens
│   └── Time: ~35 ms
│
├── KV Cache Prefill (PaliGemma 2B)
│   ├── 18 Transformer Layers
│   │   ├── Self-Attention (GQA: 8 heads, 1 KV head)
│   │   │   ├── Q: 2048→2048, K: 2048→256, V: 2048→256
│   │   │   ├── SDPA / Flash Attention
│   │   │   └── O: 2048→2048
│   │   └── MLP (主要优化目标)
│   │       ├── gate_proj: 2048→16384
│   │       ├── GELU activation
│   │       ├── up_proj: 2048→16384
│   │       └── down_proj: 16384→2048
│   └── Time: ~88 ms (MLP ~62ms, Attn ~26ms)
│
└── Denoise Loop (Action Expert 300M)
    ├── 18 Transformer Layers
    │   ├── Self-Attention
    │   ├── Cross-Attention (to PaliGemma)
    │   └── MLP: 1024→4096→1024
    └── Time: ~18 ms per step
```

---

## 基准数据 (JSON)

```json
{
  "1_steps": {
    "e2e_total": {"mean": 142.61, "std": 0.61},
    "vision_base": {"mean": 11.77, "std": 0.23},
    "vision_wrist": {"mean": 11.59, "std": 0.16},
    "embed_prefix": {"mean": 35.12, "std": 0.23},
    "kv_cache_prefill": {"mean": 88.45, "std": 0.13},
    "denoise_loop": {"mean": 18.00, "std": 0.32},
    "denoise_single_step": {"mean": 17.90, "std": 0.21}
  },
  "3_steps": {
    "e2e_total": {"mean": 179.32, "std": 1.59},
    "kv_cache_prefill": {"mean": 88.48, "std": 0.17},
    "denoise_loop": {"mean": 54.01, "std": 1.35}
  },
  "10_steps": {
    "e2e_total": {"mean": 302.40, "std": 0.79},
    "kv_cache_prefill": {"mean": 88.27, "std": 0.22},
    "denoise_loop": {"mean": 179.69, "std": 0.81}
  }
}
```

---

## 下一步行动

1. **运行 TRT FP8 MLP 编译测试** - 验证 2.94x MLP 加速
2. **集成 Flash Attention 2** - 加速 KV Cache Attention
3. **构建完整优化图** - Vision + Embed + KV Cache + Denoise
4. **运行 LIBERO 精度验证** - 确保优化不影响任务成功率
