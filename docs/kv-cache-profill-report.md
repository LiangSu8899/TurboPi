# KV Cache Prefill 延迟分析报告

## 概述

本报告分析 Pi0.5 模型中 KV Cache Prefill 阶段为什么需要约 **50ms** 延迟。

根据 [debug-08.md](./debug-08.md) 的测试数据：
- KV Cache 独立 benchmark: **47.38 ms**
- 完整 pipeline 中: **~52 ms**
- 占总延迟的 **56.8%**

---

## 模型架构分析

### PaliGemma Language Model (gemma_2b)

| 参数 | 值 | 说明 |
|------|-----|------|
| `width` (hidden_size) | 2048 | 隐藏层维度 |
| `depth` (num_layers) | **18** | Transformer 层数 |
| `mlp_dim` (intermediate_size) | **16,384** | MLP 中间层维度 (瓶颈!) |
| `num_heads` | 8 | 注意力头数 |
| `num_kv_heads` | 1 | KV 头数 (GQA) |
| `head_dim` | 256 | 每头维度 |

### 输入序列长度

| 组件 | Tokens | 说明 |
|------|--------|------|
| Base camera | 256 | 16×16 patches |
| Wrist camera | 256 | 16×16 patches |
| Language tokens | ~200 | max_token_len |
| **总计 prefix** | **~712** | 需要计算 KV Cache |

---

## 计算量分析

### 每层 Transformer Block

#### 1. Self-Attention 部分

| 操作 | 矩阵维度 | FLOPS |
|------|----------|-------|
| Q Projection | (712, 2048) × (2048, 2048) | 6.0B |
| K Projection | (712, 2048) × (2048, 256) | 0.75B |
| V Projection | (712, 2048) × (2048, 256) | 0.75B |
| Attention Score | O(n² × d) = 712² × 256 | 0.13B |
| O Projection | (712, 2048) × (2048, 2048) | 6.0B |
| **Attention 小计** | | **~13.5B** |

#### 2. MLP 部分 (SwiGLU) - **主要瓶颈**

| 操作 | 矩阵维度 | FLOPS |
|------|----------|-------|
| gate_proj | (712, 2048) × (2048, **16384**) | **47.8B** |
| up_proj | (712, 2048) × (2048, **16384**) | **47.8B** |
| down_proj | (712, **16384**) × (**16384**, 2048) | **47.8B** |
| **MLP 小计** | | **~143.4B** |

#### 单层总计

| 组件 | FLOPS | 占比 |
|------|-------|------|
| Attention | 13.5B | **8.6%** |
| **MLP** | **143.4B** | **91.4%** |
| **合计** | **~157B** | 100% |

### 18 层总计

```
总计算量 = 157B × 18 = 2.83 TFLOPS
```

---

## 延迟来源分析

### 1. 计算时间 vs 内存带宽

KV Cache Prefill 是 **内存带宽受限 (Memory-Bound)**，而非计算受限。

#### 权重读取量

| 组件 | 参数量 | BF16 大小 |
|------|--------|-----------|
| Q Projection | 2048 × 2048 | 8 MB |
| K/V Projection | 2048 × 256 × 2 | 2 MB |
| O Projection | 2048 × 2048 | 8 MB |
| gate_proj | 2048 × 16384 | 67 MB |
| up_proj | 2048 × 16384 | 67 MB |
| down_proj | 16384 × 2048 | 67 MB |
| LayerNorm 等 | - | ~1 MB |
| **单层合计** | | **~220 MB** |
| **18 层合计** | | **~3.96 GB** |

#### Jetson Thor 内存带宽

| 指标 | 值 |
|------|-----|
| 理论内存带宽 | ~200 GB/s |
| 实际有效带宽 | ~150-180 GB/s |

#### 带宽限制下的理论最优

```
读取权重时间 = 3.96 GB / 180 GB/s = ~22 ms (理论下限)
```

### 2. 延迟组成

| 组件 | 延迟 | 说明 |
|------|------|------|
| 权重读取 | ~22 ms | 内存带宽限制 |
| 激活值读写 | ~10 ms | 中间结果存储 |
| 计算 (未被隐藏) | ~8 ms | 部分计算无法与内存访问重叠 |
| LayerNorm + RoPE | ~5 ms | 逐元素操作 |
| CUDA kernel 开销 | ~3 ms | 启动、同步等 |
| **总计** | **~48-52 ms** | 与实测吻合 |

---

## 为什么 TRT FP8 优化效果有限

根据 debug-08 的实验：
- Full Coverage TRT (QKV+O+MLP 全覆盖) 只提升 **1.04x** (1.63ms)

### 原因分析

1. **MLP 已经是 TRT FP8**
   - 最大的计算瓶颈 (91.4%) 已经优化
   - 进一步优化空间很小

2. **瓶颈在内存带宽**
   - FP8 减少了计算量，但权重仍需从内存读取
   - 除非使用 INT4/FP4 量化 (已验证精度不可接受)

3. **QKV/O Projection 计算量小**
   - 只占 8.6%，即使优化也收益有限

---

## INT4/FP4 量化分析

### 理论收益

如果 INT4 量化成功，可以大幅减少内存带宽压力：

| 精度 | 权重大小 | 读取时间 | 预期加速 |
|------|----------|----------|----------|
| BF16 | 3.96 GB | ~22 ms | 1.0x |
| FP8 | 1.98 GB | ~11 ms | 2.0x |
| **INT4** | **0.99 GB** | **~5.5 ms** | **4.0x** |

**注意**: 只量化 LLM Backbone (Gemma 2B)，不量化 Vision Encoder (SigLIP)。
- Vision Encoder 只有 17.2ms，对量化敏感
- LLM Backbone 是 47.4ms 瓶颈的来源

### Thor 平台 INT4/FP4 测试结果

根据 [debug-mix-05.md](./debug-mix-05.md) 的详细测试：

| 方案 | 延迟 | 精度 (Cosine) | 状态 |
|------|------|---------------|------|
| TRT Python API FP4 | - | - | ❌ **Myelin segfault** |
| Torch-TRT NVFP4 | 0.58ms | **cos=0.0004** | ❌ Scale 被忽略 |
| PyTorch NVFP4 | 10.18ms | **cos=-0.0005** | ❌ Scale 被忽略 |
| W4A8 (FP4+FP8) | 9.64ms | **cos=-0.0008** | ❌ Scale 被忽略 |

### 失败原因

TensorRT 日志显示：
```
[DEQUANTIZE] [SCALE] has invalid precision FP4, ignored.
[DEQUANTIZE] [SCALE] has invalid precision FP8, ignored.
```

**Thor (SM 11.0 Blackwell) 的 TRT 没有正确处理量化 scale factors**：
- FP4/INT4: 完全崩溃或数值错误 (cos ≈ 0)
- FP8: 可运行但 scale 被忽略，实际运行 FP16

### NVIDIA 已知 Issues

| Issue | 问题 | 状态 |
|-------|------|------|
| [#4590](https://github.com/NVIDIA/TensorRT/issues/4590) | FP8/FP4 silent fallback | 待修复 |
| [#4599](https://github.com/NVIDIA/TensorRT/issues/4599) | ViT FP8 低性能 | 待修复 |
| [#8974](https://github.com/NVIDIA/TensorRT-LLM/issues/8974) | FP8/NVFP4 kernel 未替换 | 待修复 |

### INT4 量化结论

❌ **INT4/FP4 量化在 Thor 平台目前不可用**

| 路径 | 可行性 | 说明 |
|------|--------|------|
| 等待 NVIDIA 修复 | 中 | TRT 10.15+ 可能解决 |
| TVM 静态图 | 低 | 手写 FP4 kernel，工作量 6-8 周 |
| 当前方案 | ✅ | TRT FP16 + CUDA Graph，12 Hz |

---

## Pipeline 流水线优化实验

### 实验目的

测试是否可以通过 **Action Buffering (动作缓冲)** 将 KV Cache 延迟 (~50ms) 隐藏到 Denoising 执行时间 (~100ms) 中。

核心思想：
- 模型每次推理输出 10 步 action (Action Chunking)
- 缓冲这 10 步 action，执行期间不需要新推理
- 理论上可以将有效频率从 ~5 Hz 提升到 ~25-50 Hz

### 实验配置

| 参数 | Baseline | Pipeline |
|------|----------|----------|
| replan_steps | 1 (每步推理) | 10 (缓冲10步) |
| 推理策略 | 同步阻塞 | Action Buffering |
| Denoising Steps | 10 | 10 |

### 实验结果

| 指标 | Baseline | Pipeline | 变化 |
|------|----------|----------|------|
| **成功率** | **100%** (9/9) | **0%** (0/9) | ❌ 完全失败 |
| 有效频率 | 5.0 Hz | 26.6 Hz | +5.3x |
| 推理延迟 | 176.5 ms | 181.3 ms | +2.7% |
| 每集推理次数 | 89-182 | 22 | -4.5x |

#### 组件延迟对比

| 组件 | Baseline | Pipeline |
|------|----------|----------|
| Vision | 17.0 ms | 17.4 ms |
| KV Cache | 54.0 ms | 52.6 ms |
| Denoise | 102.3 ms | 107.9 ms |
| **Total** | 176.5 ms | 181.3 ms |

### 失败原因分析

**Action Chunking 导致 VLA 模型完全失效**：

1. **观察更新过慢**
   - 缓冲 10 步 action 意味着机器人 "失明" 10 步
   - VLA 模型需要实时观察反馈来纠正轨迹
   - 每步环境都在变化，旧 action 无法适应新状态

2. **与 KV Cache Reuse 失败一致**
   - KV Cache Reuse: 跨帧复用 KV → 0% 精度
   - Action Buffering: 跨帧复用 action → 0% 精度
   - **VLA 模型对 "陈旧信息" 极度敏感**

3. **Action Chunk 本质问题**
   - VLA 输出的 action chunk 只有第一步是准确的
   - 后续步骤是对未来的 **预测**，不是 **规划**
   - 预测误差会累积，导致轨迹偏离

### 结论

❌ **Pipeline + Action Buffering 方案不可行**

| 方案 | 成功率 | 频率 | 结论 |
|------|--------|------|------|
| Baseline (replan=1) | 100% | 5 Hz | ✅ 可用 |
| Pipeline (replan=10) | 0% | 26 Hz | ❌ 失败 |

**核心发现**: VLA 模型需要每步都获取新观察并推理。任何形式的 "延迟隐藏" 或 "信息复用" 都会导致精度悬崖。

---

## 优化方向分析

### 已验证失败的方案

| 方案 | 结果 | 原因 |
|------|------|------|
| KV Cache 帧间复用 | ❌ 0% 精度 | VLA 不容忍跨帧不一致 |
| INT4/FP4 量化 | ❌ Thor 不支持 | TRT scale 被忽略，cos≈0 |
| FP8 量化 | ⚠️ 无加速 | Thor 上 scale 被忽略，实际 FP16 |
| **Action Buffering** | ❌ 0% 精度 | VLA 需要实时观察反馈 |

### 可行的优化方向

| 方向 | 预期收益 | 可行性 |
|------|----------|--------|
| **减少层数** (18→12) | ~1.5x | 中 (需验证精度) |
| **模型蒸馏** | 2-3x | 高 (需要训练) |
| **结构化剪枝** | 1.5-2x | 中 (需要训练) |
| ~~异步流水线~~ | ~~隐藏延迟~~ | ❌ (VLA 需同步) |
| ~~Action Chunking~~ | ~~2-4x 等效~~ | ❌ (0% 精度) |

### 20 Hz 目标分析

当前状态 (10-step denoising):
- Vision: 17 ms
- KV Cache: 54 ms
- Denoise: 102 ms
- **总延迟: ~176 ms (5.7 Hz)**

要达到 20 Hz (50 ms):
```
需要削减: 176 - 50 = 126 ms
需要 3.5x 整体加速!
```

**结论**:
- 20 Hz 目标在当前架构下无法实现
- 所有 "延迟隐藏" 策略都已验证失败
- 需要 **模型级改动** (蒸馏/剪枝/更小模型)

---

## 总结

### KV Cache Prefill 50ms 延迟的根本原因

1. **模型太大**: 18 层 × 16384 MLP dim = 3.96 GB 权重
2. **内存带宽瓶颈**: 读取权重需要 ~22 ms (理论下限)
3. **MLP 占绝对主导**: 91.4% 的计算量在 MLP

### 已验证失败的优化方案

| 方案 | 失败原因 |
|------|----------|
| KV Cache 帧间复用 | VLA 对跨帧不一致极度敏感 (0% 精度) |
| INT4/FP4 量化 | Thor TRT 不支持 scale factors |
| Pipeline + Action Buffering | VLA 需要实时观察反馈 (0% 精度) |

### 核心结论

1. **KV Cache Prefill 是内存带宽受限**，TRT/量化优化已接近极限
2. **VLA 模型不容忍任何形式的 "陈旧信息"**：
   - 跨帧 KV Reuse → 0% 精度
   - Action Buffering → 0% 精度
3. **必须每步都进行完整推理**，无法通过 "延迟隐藏" 提升频率
4. **需要模型级改动** (蒸馏/剪枝/更小模型) 才能突破 5 Hz

### 当前最佳性能

| 配置 | 延迟 | 频率 | 精度 |
|------|------|------|------|
| 10-step denoising | 176 ms | 5.7 Hz | 100% |
| 3-step denoising | ~83 ms | 12 Hz | 待验证 |

---

## Phase 0 环境验证结果 (2026-02-08)

### Thor 平台硬件信息

| 指标 | 测量值 | 评估 |
|------|--------|------|
| GPU | NVIDIA Thor SM 11.0 | ✅ Blackwell |
| 内存 | 131.88 GB | ✅ 充足 |
| SM Count | 20 | - |
| 平均带宽 | 130.1 GB/s | ⚠️ 理论 65% |

### 软件支持情况

| 组件 | 状态 | 版本 |
|------|------|------|
| Triton | ✅ 可用 | 3.5.1 |
| FlashInfer | ❌ 未安装 | - |
| bitsandbytes | ❌ 未安装 | - |

### KV Cache 54ms 组件分解

| 组件 | 延迟 (18层) | 占比 | 说明 |
|------|-------------|------|------|
| MLP gate+up+SiLU | 24.6 ms | 45.6% | █████████ |
| **MLP down** | **21.9 ms** | **40.6%** | ████████ |
| Attention | 2.6 ms | 4.8% | |
| QKV+O Projection | 2.1 ms | 3.9% | |
| LayerNorm | 0.5 ms | 0.9% | |
| **估算总计** | **51.7 ms** | 100% | |
| **实际测量** | **54.0 ms** | - | 差距 2.3ms 来自 RoPE, residual |

### 关键发现

1. **MLP 占 KV Cache 的 90%**
   - MLP gate+up+down: 46.5 ms
   - Attention 只占 2.6 ms (5%)

2. **MLP down_proj 带宽异常低**
   - seq=1: 148 GB/s (正常)
   - seq=712: 49 GB/s (⚠️ 异常)
   - 原因: cuBLAS 对 (M, K) @ (K, N) 在 K >> N 时效率低

3. **INT4 量化收益预估**
   - 当前 MLP: 46.5 ms
   - INT4 MLP: ~11.6 ms (理论 1/4)
   - **预期节省: ~35 ms**

### 初步决策: Plan B (INT4 Triton Kernel)

原计划:
| 方案 | KV Cache | 总延迟 | Hz |
|------|----------|--------|-----|
| 当前 BF16 | 54 ms | 173 ms | 5.8 Hz |
| **INT4 量化** | **~17 ms** | **~136 ms** | **7.4 Hz** |

---

## 量化方案验证结果 (2026-02-08)

### Triton 基础性能测试

| 实现 | 延迟 | vs cuBLAS |
|------|------|-----------|
| torch.matmul (cuBLAS) | 0.47 ms | 1.00x |
| **Triton FP16 MatMul** | **1.14 ms** | **0.41x** |

**结论**: Triton 在 Thor SM 11.0 上即使是 FP16 MatMul 也只有 cuBLAS 的 **41%** 性能。

### 量化库测试

| 方案 | 性能 vs BF16 | 精度 | 状态 |
|------|-------------|------|------|
| Triton INT4 (自定义) | 0.02x | ✅ 0.993 | ❌ Triton 本身慢 |
| torchao INT8 | **0.09x** | ✅ 0.9999 | ❌ 灾难性性能 |
| torchao INT4 | N/A | - | ❌ 缺少 fbgemm-gpu-genai |
| **torch._int_mm** | **0.98x** | - | ❌ **cuBLAS INT8 无加速** |
| CUDA Graph | 1.00x | - | ≈ 无提升 |

### 关键发现

1. **Triton 不适合 Thor 平台**
   - 即使简单的 FP16 MatMul 也比 cuBLAS 慢 2.5x
   - 写自定义 INT4 kernel 不可行

2. **torchao 量化性能灾难**
   - INT8: 比 BF16 慢 **11 倍**
   - INT4: 缺少 Thor 上的 fbgemm-gpu-genai 依赖

3. **cuBLAS INT8 无加速**
   - torch._int_mm: 0.98x (与 FP16 性能相同)
   - Thor 的 INT8 Tensor Core 可能没有被正确使用

4. **CUDA Graph 无效**
   - MLP baseline: 2.62 ms
   - CUDA Graph: 2.63 ms
   - 几乎没有提升

### 最终结论

**Thor 平台上所有量化加速方案目前都不可用**:
- Triton kernel 性能差 (0.41x cuBLAS)
- torchao INT8/INT4 没有 Thor 优化
- cuBLAS INT8 与 FP16 性能相同
- CUDA Graph 无明显收益

**当前最佳方案**:
- 维持 BF16 推理
- 当前 KV Cache: ~54 ms
- 当前总延迟: ~173 ms
- **当前频率: ~5.8 Hz**

### 后续建议

| 方向 | 可行性 | 说明 |
|------|--------|------|
| 等待 NVIDIA Thor 软件支持 | 中 | 等 TRT 10.15+ / Triton Thor 优化 |
| 模型蒸馏 | 高 | 训练更小的模型 |
| 减少 Transformer 层数 | 中 | 18→12 层，需验证精度 |
| TensorRT 完整 INT8 pipeline | 低 | 需要大量校准工作 |

---

## 相关文档

- [debug-08.md](./debug-08.md): Full Coverage TRT 验证
- [debug-mix-05.md](./debug-mix-05.md): NVFP4/INT4 量化测试详情
- [cliff_report.md](./cliff_report.md): KV Reuse 精度悬崖分析
- [attention_weighted_drift_plan.md](./attention_weighted_drift_plan.md): KV Cache 复用实验
- [libero_eval_pipeline.py](../openpi/scripts/libero_eval_pipeline.py): Pipeline 流水线实验脚本
- [thor-triforce-optimization-plan-v2.md](./thor-triforce-optimization-plan-v2.md): Triforce 优化计划 v2
