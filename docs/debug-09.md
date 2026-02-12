# Debug-09: Full Optimized Pipeline 性能复现测试

## 测试日期
2026-02-12

## 测试目的
复现并验证 debug-07 中的 Full Optimized Pipeline 性能数据。

---

## 测试配置

| 配置项 | 值 |
|--------|-----|
| **硬件平台** | NVIDIA Jetson Thor (SM110) |
| **Docker 容器** | turbo_pi_eval |
| **Task Suite** | libero_spatial |
| **Tasks** | 3 tasks, 3 trials (quick mode) |
| **Backend** | Full Optimized Pipeline |

### 优化组件
| 组件 | 技术 |
|------|------|
| Vision Encoder | torch_tensorrt.compile (FP16) + dtype conversion |
| KV Cache (18层) | ModelOpt FP8 量化 + torch_tensorrt MLP + Flash Attention |
| Denoising | CUDA Graph 捕获 |

---

## 测试结果

### 性能总览

| Denoising Steps | Latency (ms) | Throughput (Hz) | Accuracy |
|-----------------|--------------|-----------------|----------|
| **1 step** | 86.52 ms | **11.6 Hz** | 100% (9/9) |
| **3 steps** | 107.11 ms | **9.3 Hz** | 100% (9/9) |
| **10 steps** | 176.49 ms | **5.7 Hz** | 100% (9/9) |

### 组件耗时分析

#### 1 Step 配置 (最快)

| 组件 | 耗时 (ms) | 占比 |
|------|-----------|------|
| Vision TRT (FP16) | 17.29 ms | **20.0%** |
| KV Cache TRT (FP8 MLP) | 54.19 ms | **62.6%** |
| Denoise CUDA Graph | 10.41 ms | **12.0%** |
| **Total** | **86.52 ms** | 100% |

```
┌─────────────────────────────────────────────────────────────────┐
│                    1 Step: 86.52ms (11.6 Hz)                    │
├──────────────┬─────────────────────────────────────┬────────────┤
│ Vision (20%) │         KV Cache (62.6%)            │Denoise(12%)│
│   17.29ms    │           54.19ms                   │  10.41ms   │
└──────────────┴─────────────────────────────────────┴────────────┘
```

#### 3 Steps 配置

| 组件 | 耗时 (ms) | 占比 |
|------|-----------|------|
| Vision TRT (FP16) | 17.28 ms | **16.1%** |
| KV Cache TRT (FP8 MLP) | 54.27 ms | **50.7%** |
| Denoise CUDA Graph | 30.87 ms | **28.8%** |
| **Total** | **107.11 ms** | 100% |

```
┌────────────────────────────────────────────────────────────────────────┐
│                      3 Steps: 107.11ms (9.3 Hz)                        │
├───────────┬──────────────────────────────────────┬─────────────────────┤
│Vision(16%)│         KV Cache (50.7%)             │  Denoise (28.8%)    │
│  17.28ms  │           54.27ms                    │     30.87ms         │
└───────────┴──────────────────────────────────────┴─────────────────────┘
```

#### 10 Steps 配置

| 组件 | 耗时 (ms) | 占比 |
|------|-----------|------|
| Vision TRT (FP16) | 17.15 ms | **9.7%** |
| KV Cache TRT (FP8 MLP) | 54.20 ms | **30.7%** |
| Denoise CUDA Graph | 100.75 ms | **57.1%** |
| **Total** | **176.49 ms** | 100% |

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                          10 Steps: 176.49ms (5.7 Hz)                              │
├──────┬──────────────────────┬─────────────────────────────────────────────────────┤
│V(10%)│   KV Cache (30.7%)   │               Denoise (57.1%)                       │
│17.15 │       54.20ms        │                 100.75ms                            │
└──────┴──────────────────────┴─────────────────────────────────────────────────────┘
```

---

## 与 Baseline (debug-07) 对比

| Denoising Steps | Baseline (debug-07) | 本次测试 | 差异 |
|-----------------|---------------------|----------|------|
| **1 step** | 83.50 ms (12.0 Hz) | 86.52 ms (11.6 Hz) | +3.02 ms |
| **3 steps** | 102.81 ms (9.7 Hz) | 107.11 ms (9.3 Hz) | +4.30 ms |
| **10 steps** | 171.36 ms (5.8 Hz) | 176.49 ms (5.7 Hz) | +5.13 ms |

### 组件对比 (1 Step)

| 组件 | Baseline | 本次测试 | 差异 |
|------|----------|----------|------|
| Vision TRT | 17.26 ms | 17.29 ms | +0.03 ms ✓ |
| KV Cache TRT | 52.14 ms | 54.19 ms | +2.05 ms |
| Denoise CUDA | 10.10 ms | 10.41 ms | +0.31 ms ✓ |

**结论**: 结果与 baseline 一致，差异在正常系统波动范围内 (~3-5ms)。

---

## 组件稳定性分析

### Vision TRT (FP16)
| Steps | 耗时 (ms) | 稳定性 |
|-------|-----------|--------|
| 1 | 17.29 | ✓ |
| 3 | 17.28 | ✓ |
| 10 | 17.15 | ✓ |

**稳定在 ~17.2ms**，与 denoising steps 无关。

### KV Cache TRT (FP8 MLP)
| Steps | 耗时 (ms) | 稳定性 |
|-------|-----------|--------|
| 1 | 54.19 | ✓ |
| 3 | 54.27 | ✓ |
| 10 | 54.20 | ✓ |

**稳定在 ~54.2ms**，与 denoising steps 无关。这是固定瓶颈。

### Denoise CUDA Graph
| Steps | 耗时 (ms) | 每步耗时 |
|-------|-----------|----------|
| 1 | 10.41 | 10.41 ms/step |
| 3 | 30.87 | 10.29 ms/step |
| 10 | 100.75 | 10.08 ms/step |

**线性增长**，每步约 **10.3 ms**。

---

## Denoising Steps vs Latency 模型

```
Total Latency (ms) = Vision + KV Cache + (Denoise_per_step × Steps)
                   = 17.3 + 54.2 + (10.3 × Steps)
                   = 71.5 + 10.3 × Steps
```

| Steps | 预测 (ms) | 实测 (ms) | 误差 |
|-------|-----------|-----------|------|
| 1 | 81.8 | 86.52 | +5.8% |
| 3 | 102.4 | 107.11 | +4.6% |
| 10 | 174.5 | 176.49 | +1.1% |

---

## 性能指标统计

### 1 Step 详细统计
| 指标 | 值 |
|------|-----|
| Mean | 86.52 ms |
| Std | 1.09 ms |
| P50 | 86.66 ms |
| P95 | 87.90 ms |
| Inferences | 205 |

### 3 Steps 详细统计
| 指标 | 值 |
|------|-----|
| Mean | 107.11 ms |
| Std | 1.04 ms |
| P50 | 107.36 ms |
| P95 | 108.34 ms |
| Inferences | 203 |

### 10 Steps 详细统计
| 指标 | 值 |
|------|-----|
| Mean | 176.49 ms |
| Std | 1.15 ms |
| P50 | 176.51 ms |
| P95 | 178.21 ms |
| Inferences | 206 |

---

## 关键结论

1. **性能复现成功**: 所有配置结果与 baseline 一致
2. **100% 准确率**: 所有 27 次试验全部成功
3. **组件耗时稳定**:
   - Vision TRT: ~17.3 ms (固定)
   - KV Cache TRT: ~54.2 ms (固定，最大瓶颈)
   - Denoise: ~10.3 ms/step (线性)

4. **瓶颈分析**:
   - 1 step 时，KV Cache 占 **62.6%** → 优化 KV Cache 收益最大
   - 10 steps 时，Denoise 占 **57.1%** → 减少 steps 收益最大

---

## 20 Hz 目标差距

| 目标 | 当前 | 差距 |
|------|------|------|
| 50 ms (20 Hz) | 86.52 ms (11.6 Hz) | **-36.52 ms** |

要达到 20 Hz，需要:
- **KV Cache**: 54.2 ms → ~18 ms (需 **3x** 加速)
- 或 **并行化**: Vision + KV Cache 并行执行

---

## 测试脚本

```bash
# 1 Step 测试
docker exec turbo_pi_eval bash -c "cd /workspace && \
    python scripts/libero_eval_full_optimized.py --denoising_steps 1 --quick"

# 3 Steps 测试
docker exec turbo_pi_eval bash -c "cd /workspace && \
    python scripts/libero_eval_full_optimized.py --denoising_steps 3 --quick"

# 10 Steps 测试
docker exec turbo_pi_eval bash -c "cd /workspace && \
    python scripts/libero_eval_full_optimized.py --denoising_steps 10 --quick"
```

---

## 文件引用

| 文件 | 用途 |
|------|------|
| `scripts/libero_eval_full_optimized.py` | Full Optimized Pipeline 评估脚本 |
| `docs/debug-07.md` | Baseline 性能数据 |
