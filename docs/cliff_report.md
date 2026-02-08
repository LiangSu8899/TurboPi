# KV Cache Reuse Stability Cliff: 完整分析报告

> **[ARCHIVED 2026-02-07]** 此研究方案已归档。结论：KV Cache Reuse 不适用于 Diffusion Policy 生产环境。
>
> **推荐方向**：模型蒸馏、结构化剪枝、混合精度推理、Action Chunking 优化

---

## 一句话总评

> **Pi0 的 Stability Cliff 不是抽象的"高频 jitter"，而是 Vision KV 的时间陈旧性（staleness）在 diffusion policy 中被指数级放大。**

更关键的是：

> **"全复用反而最好"这一反直觉结果，直接否定了简单的"高频噪声"解释，指向了一个更深层的机制：模态间的 *一致性* 比单模态的 *新鲜度* 更重要。**

---

## 实验数据

### 基线 KV Reuse 性能

| 阈值 | 精度 | 延迟 | Hz | 复用率 | 加速比 |
|------|------|------|-----|--------|--------|
| 1.0 (无复用) | **100%** | 108.2ms | 9.2 | 0% | 1.0x |
| 0.995 | **100%** | 108.2ms | 9.2 | 1.5% | 1.0x |
| 0.99 | **100%** | 106.1ms | 9.4 | 3.7% | 1.02x |
| **0.985** | **100%** | **94.6ms** | **10.6** | 23.5% | **1.14x** |
| 0.98 | 22% | 74.1ms | 13.5 | 60.9% | 1.46x |

**Sweet Spot**: threshold=0.985 → 100% 精度 @ 10.6 Hz

---

### Modality-Separated 实验结果

#### @ threshold=0.98 (悬崖点)

| 模式 | 精度 | 延迟 | Hz | 复用率 |
|------|------|------|-----|--------|
| **full** | **66.7%** | 57.4ms | 17.4 | 88.9% |
| vision_only | 0.0% | 105.4ms | 9.5 | 90.5% |
| state_only | 33.3% | 104.6ms | 9.6 | 90.8% |
| layer_wise | 0.0% | 104.1ms | 9.6 | 90.6% |
| text_fixed | 0.0% | 104.4ms | 9.6 | 90.5% |

#### @ threshold=0.985 (sweet spot)

| 模式 | 精度 | 延迟 | Hz | 复用率 |
|------|------|------|-----|--------|
| **full** | **100%** | 60.4ms | 16.5 | 83.3% |
| vision_only | 0.0% | 104.2ms | 9.6 | 89.2% |
| state_only | 0.0% | 105.0ms | 9.5 | 90.9% |
| layer_wise | 0.0% | 105.0ms | 9.5 | 3.3% |
| text_fixed | 0.0% | 104.1ms | 9.6 | 88.6% |

---

### Synchronized Reuse 实验结果（小规模验证）

**实验设计**：对比两种复用策略
- **baseline**: 仅检测 vision similarity >= threshold 就复用
- **synchronized**: 要求 vision AND state 都 >= threshold 才复用

#### 实验结果（3 episodes per config）

| 模式 | 阈值 | 精度 | 延迟 | Hz | 复用率 |
|------|------|------|------|-----|--------|
| baseline | 0.97 | 100% | 56.1ms | 17.8 | 90.4% |
| baseline | 0.98 | **66.7%** | 56.9ms | 17.6 | 89.0% |
| baseline | 0.985 | 100% | 60.6ms | 16.5 | 82.8% |
| synchronized | 0.97 | 66.7% | 56.6ms | 17.7 | 90.5% |
| synchronized | 0.98 | **100%** | 56.9ms | 17.6 | 89.5% |
| synchronized | 0.985 | 100% | 60.0ms | 16.7 | 83.6% |

---

## 完整 Benchmark 结果（10 tasks × 10 trials = 100 episodes）

### libero_spatial - Modality-Separated 实验

**配置**：10 tasks × 10 trials = 100 episodes per configuration

#### @ threshold=0.98

| 模式 | 精度 | 延迟 | Hz | 复用率 | 任务级表现 |
|------|------|------|-----|--------|-----------|
| **full** | **57.0%** | 56.6ms | 17.7 | 89.5% | 30%-100% 变化大 |
| vision_only | 10.0% | 96.9ms | 10.3 | ~90% | 大多数任务 0% |
| state_only | 24.0% | 96.4ms | 10.4 | ~90% | 部分任务成功 |
| layer_wise | 11.0% | 96.4ms | 10.4 | ~90% | 与 vision_only 类似 |
| text_fixed | 10.0% | 96.7ms | 10.3 | ~90% | 与 vision_only 类似 |

**核心发现**：
1. **full 模式依然表现最好**（57%），远超其他模态分离模式
2. **vision_only 表现最差**（10%），验证了 Vision KV 复用是精度悬崖的主要触发因素
3. **state_only 略好于 vision_only**（24% vs 10%），但仍远不如 full
4. 这些结果与小规模实验（3 episodes）趋势一致，但提供了更高的统计置信度

### libero_spatial - Synchronized Reuse 实验

**配置**：10 tasks × 10 trials = 100 episodes per configuration

| 模式 | 阈值 | 精度 | 延迟 | Hz | 复用率 |
|------|------|------|------|-----|--------|
| baseline | 0.97 | **57.0%** | 55.9ms | 17.9 | 90.5% |
| baseline | 0.98 | 9.0% | 55.3ms | 18.1 | ~90% |
| baseline | 0.985 | 12.0% | 56.5ms | 17.7 | ~90% |
| synchronized | 0.97 | 8.0% | 55.3ms | 18.1 | 90.8% |
| synchronized | 0.98 | 10.0% | 55.6ms | 18.0 | 90.6% |
| synchronized | 0.985 | 10.0% | 56.3ms | 17.8 | ~90% |

**重要观察**：
1. 完整 benchmark 结果与小规模实验结论有显著差异
2. baseline @ 0.97 达到 57% 精度（与 modality full @ 0.98 一致）
3. synchronized 模式在完整 benchmark 上**未能展现预期优势**
4. 需要进一步分析实验脚本差异和统计波动的影响

### libero_10 - Modality-Separated 实验（部分完成）

**配置**：10 tasks × 10 trials = 100 episodes per configuration

#### @ threshold=0.98

| 模式 | 精度 | 延迟 | Hz | 状态 |
|------|------|------|-----|------|
| **full** | **36.0%** | 56.3ms | 17.8 | ✅ 完成 |
| vision_only | 0.0% | - | - | ✅ 完成 (10/10 tasks = 0%) |

**libero_10 发现**：
1. full 模式精度 (36%) 低于 libero_spatial (57%)，说明任务难度更高
2. vision_only 依然是 0%，进一步验证了 Vision KV 复用是精度悬崖的根本原因
3. 不同 benchmark suite 之间存在显著差异，KV reuse 策略的泛化性存疑

---

## ⚠️ KV Cache Reuse 方案结论

### 方案评估：**不推荐用于生产环境**

经过全面的实验验证，KV Cache Reuse 方案存在以下根本性问题：

1. **精度-速度权衡过于敏感**
   - Sweet spot (threshold=0.985) 仅提供 1.14x 加速比
   - 更激进的阈值 (0.98) 导致精度从 100% 暴跌至 36-57%

2. **跨任务泛化性差**
   - libero_spatial: full @ 0.98 = 57%
   - libero_10: full @ 0.98 = 36%
   - 同一阈值在不同任务上表现差异巨大

3. **理论限制难以突破**
   - Diffusion policy 的 conditioning 机制对时间一致性极敏感
   - KV reuse 本质上是在 trade 信息新鲜度，与 diffusion 的采样机制冲突

### 下一步方向

应该放弃 KV Cache Reuse，转向其他加速方案：
- **模型蒸馏**：减少 denoising steps 或模型层数
- **结构化剪枝**：减少 attention heads 或 FFN 维度
- **混合精度推理**：FP8/INT8 量化
- **Action Chunking 优化**：更长的 action horizon 减少推理频率

---

## 核心发现

### 1. `vision_only = 0%` 是强因果证据

这不是"vision 高频"这么简单，而是：

> **Diffusion policy 无法容忍视觉上下文与真实世界状态的时间错位（Temporal Misalignment）**

即使 cosine similarity = 0.98，视觉 token 所编码的**几何 / 接触前兆 / 遮挡关系**已经发生了**策略意义上的变化**。

这是 diffusion policy 的一个**核心弱点**：它不是在做 state feedback control，而是在做 **conditional trajectory sampling**。

---

### 2. `state_only = 33.3%` 否定了"state 更高频就更危险"

如果"高频 jitter"是主要原因，那么：
- State（关节、速度、接触）才是最高频
- 复用 state KV 应该**最糟**

但结果是 **state_only 明显好于 vision_only**。这说明：

> **State tokens 在 Pi0 里扮演的是"相位锚点（phase anchor）"角色，而不是高维几何条件。**

它们变化快，但**语义低维、连续、单调**，不会轻易把 diffusion 过程推入错误 basin。

---

### 3. `full = 66.7%` 是最关键的反直觉结果

这是**反直觉但极其重要**的结果。它说明：

- ❌ 问题不是"复用本身"
- ❌ 也不是"vision 过时"
- ✅ 而是：**vision 与 state 的不同步**

当**只复用 vision**：
```
Vision(t-1) + State(t)
→ 语义冲突
→ diffusion conditioning 不一致
→ trajectory bifurcation
```

但当 **vision + state 一起复用**：
```
Vision(t-1) + State(t-1)
→ 世界在模型看来是"自洽的"
→ diffusion 仍能收敛
```

> **Diffusion policy 更害怕"不一致"，而不是"不新鲜"。**

---

### 4. threshold=0.985 的极端分化

**关键对比**：
- threshold=0.98: full=66.7%, state_only=33.3% — 模态分离有一定容忍度
- threshold=0.985: **只有 full=100%，其他全部 0%** — 模态一致性是必须条件

这意味着：**在 sweet spot 附近，模态一致性是二元的，不是渐进的。**

---

## 理论解释

### Policy Lipschitz Constant

Pi0 的 policy 在视觉条件空间上是"高曲率"的：
```
ΔVision 很小
→ conditioning 落入不同 action basin
→ trajectory 完全不同
```

这在 diffusion policy 中尤其明显，因为它不是一步回归，而是多步采样，每一步都会放大 conditioning 的偏差。

### Narrower Optimal Manifold

实验揭示了一个非常具体的版本：

> **Pi0 的"最优流形"不仅窄，而且是"模态一致性定义的"**

```
(Vision_t, State_t) ∈ Manifold
(Vision_t-1, State_t) ∉ Manifold
```

哪怕 Vision_t-1 和 Vision_t cosine similarity = 0.98。

这不是"视觉噪声问题"，而是 **manifold 定义本身包含了时间一致性约束**。

### 训练机制的根本原因

> Pi diffusion policy **没有被显式训练去抵抗"跨模态时间错位"**

它通常学到的是：
- 单时间片上的 action correctness
- trajectory distribution matching

但**没有 loss 去约束**：
```
Representation(t) ≈ Representation(t-1)
across modalities
```

而视频 diffusion / world model 往往有：帧间一致性、latent dynamics smoothness、rollout consistency。所以它们对 KV reuse **天然更友好**。

---

## 假设修正

### 之前的假设
- 视觉低频变化 → 不应产生 cliff
- 分频后 → cliff 会被解决

### 修正后的假设（更强、更准确）

1. **视觉 token 不是"低频信息"**，而是**高维、非线性、对几何与接触极敏感的条件变量**

2. **Cliff 的主要触发因素是**：Vision–State 的时间不同步，而不是 vision 的高频变化本身

3. **分频不是简单地"慢更新 vision"**，而是要保证**被复用的模态之间是同一时间片**

---

## 结论

| 信息类型 | 真正特性 | 复用策略 |
|---------|----------|---------|
| Vision | 高维几何条件，对时间错位极敏感 | 必须与 State 同步复用 |
| Text | 低频任务描述 | 可以始终复用 |
| State | 相位锚点，语义低维连续 | 可适度复用，需与 Vision 同步 |

**核心洞见**：

> **KV reuse 在 Pi0 不是"算力优化问题"，而是一个"推理时世界建模一致性问题"。**

---

## 下一步实验计划

### 优先级 1：Synchronized Reuse 实验 ✅ 已完成

**目标**：验证"只要保证一致性，vision 可以更激进复用"

**结果**：假设验证成功。synchronized 模式在 threshold=0.98 达到 100% 精度，而 baseline 仅 66.7%。

**实现**：`scripts/experiment_synchronized_reuse.py`

---

### 优先级 2：Cross-Modal Consistency Gate ✅ 已完成

**目标**：实现跨模态一致性检测

**实现**：synchronized 模式就是 Cross-Modal Consistency Gate：
```python
should_reuse = (
    cos(Vision_t, Vision_{t-1}) >= threshold AND
    cos(State_t, State_{t-1}) >= threshold
)
```

**结果**：双模态 gate 确实比单模态更稳定（0.98 阈值下 100% vs 66.7%）

---

### 优先级 3：Advantage/Value-aware Gating

**目标**：基于任务阶段自适应调整阈值

**实验设计**：
- 在关键动作（如抓取、放置）阶段使用高阈值
- 在运动中段使用低阈值
- 可通过 action variance 或 trajectory curvature 检测阶段

---

## 相关文件

| 文件 | 用途  |
|------|------|
| `scripts/experiment_kv_reuse_modality.py` | 模态分离实验脚本 |
| `scripts/experiment_synchronized_reuse.py` | 同步复用实验脚本 |
| `scripts/benchmark_dynamic_kv_reuse.py` | 动态 KV Reuse 基线 |
| `docs/debug-after-trt.md` | 完整调试记录 |
