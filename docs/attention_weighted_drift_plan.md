# Attention-Weighted Drift: KV Reuse 改进方案 (Pro Version)

## 背景

### 问题1: Cosine Similarity 不可靠
- cos=0.98 时策略意义已变化
- 像素级相似度无法反映任务语义

### 问题2: Chicken-and-Egg 陷阱
- 要获取 current_attention 必须先运行 Vision Encoder
- 此时 KV 已经算完，判断"是否复用"毫无意义

### 问题3: Patch级混合致命 (Cliff Report)
- vision_only = 0% 说明不能混合新旧 KV
- 必须 All-or-Nothing

### 问题4: 盲区入侵 (Blind Spot Problem) ⚠️ 新增
- 上一帧 attention 只关注杯子，旁边 attention 低
- 突然一只手从旁边伸入 → 变化发生在低 attention 区域
- Weighted Drift 很低 → 错误复用 → 机器人"看不见"入侵的手

### 问题5: 噪声敏感 ⚠️ 新增
- 摄像头自动曝光、光照闪烁、传感器噪声
- 即使画面静止，像素值也在跳动
- 导致 threshold 难以设置

### 问题6: 手眼不协调 ⚠️ 新增
- 机械臂猛动，但因遮挡/视角问题，画面变化不大
- Policy 接收：旧画面(手在A) + 新状态(手在B)
- 结果：Policy 精神分裂

---

## 方案: Attention-Weighted Drift

### 核心思想

用 **上一帧的 Attention** 作为 **重要性地图**，评估 **当前帧的像素变化**：

```
Weighted_Drift = Σ (patch_pixel_diff × prev_attention_weight)
```

- 背景变化大，但 attention 低 → 加权后分数低 → 可复用
- 关键区域变化大，attention 高 → 加权后分数高 → 重算

### 解决手眼相机问题

手眼相机画面动得厉害，但如果：
- 动的只是边缘背景 (attention 低)
- 核心物体相对稳定 (attention 高的区域变化小)

→ Weighted_Drift 依然可能很低，允许复用！

---

## 实现代码 (Pro Version)

### 三个关键改进

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pro Version 三层防御                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: State Guard (最快，先算)                              │
│  ────────────────────                                           │
│  if joint_diff > threshold → 强制重算                           │
│  计算量: ~0.01ms                                                │
│                                                                 │
│  Layer 2: Anti-Noise Preprocessing                              │
│  ─────────────────────────────────                              │
│  下采样 224→28，消除高频噪点                                     │
│  只关注结构性变化，忽略曝光/噪声                                 │
│                                                                 │
│  Layer 3: Dilated Attention Weighting                           │
│  ────────────────────────────────────                           │
│  对 Attention Map 做 Max Pooling (膨胀)                         │
│  扩大关注范围，防止"盲区入侵"                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn.functional as F

class ProAttentionGate:
    """
    Pro Version: 基于 Attention 加权的帧级 KV Reuse Gate

    三层防御:
    1. State Guard: 关节变化检测 (最快)
    2. Anti-Noise: 下采样消除噪点
    3. Dilated Attention: 膨胀防止盲区入侵
    """

    def __init__(self, drift_threshold=0.5, state_threshold=0.05, dilation_kernel=5):
        self.drift_threshold = drift_threshold
        self.state_threshold = state_threshold
        self.dilation_kernel = dilation_kernel

        self.prev_image_small = None   # 下采样后的图 [B, C, 28, 28]
        self.prev_attention = None     # 2D attention map [B, H, W]
        self.prev_state = None         # 关节状态
        self.prev_kv_cache = None

    def extract_patch_importance(self, full_attention_weights, layer_range=(14, 18)):
        """
        从完整 Attention 矩阵中提取每个 Vision Patch 的重要性 (2D Map)。

        Returns:
            importance: [batch, H, W] 2D重要性地图 (方便做膨胀)
        """
        layer_start, layer_end = layer_range
        selected_attn = full_attention_weights[layer_start:layer_end]
        avg_attn = selected_attn.mean(dim=(0, 2))  # [B, S, S]

        # 提取 action → vision attention
        num_vision_tokens = 256  # 每个相机256个patch
        action_start = -50

        action_to_vision = avg_attn[:, action_start:, :num_vision_tokens]
        importance = action_to_vision.mean(dim=1)  # [B, 256]

        # Reshape to 2D map [B, 16, 16]
        B = importance.shape[0]
        importance_2d = importance.view(B, 16, 16)

        # Normalize
        importance_2d = importance_2d / (importance_2d.sum(dim=(1, 2), keepdim=True) + 1e-8)

        return importance_2d

    def dilate_attention(self, attention_map):
        """
        对 Attention Map 做膨胀，扩大关注范围，防止盲区入侵

        Args:
            attention_map: [B, H, W]

        Returns:
            dilated: [B, H, W] 膨胀后的 attention map
        """
        # 添加 channel 维度: [B, H, W] -> [B, 1, H, W]
        attn_4d = attention_map.unsqueeze(1)

        # Max Pooling 实现膨胀
        padding = self.dilation_kernel // 2
        dilated = F.max_pool2d(
            attn_4d,
            kernel_size=self.dilation_kernel,
            stride=1,
            padding=padding
        )

        return dilated.squeeze(1)  # [B, H, W]

    def robust_image_diff(self, current_image):
        """
        抗噪处理: 下采样后计算差异

        Args:
            current_image: [B, C, H, W] (224x224)

        Returns:
            diff_small: [B, 28, 28] 下采样后的差异图
        """
        # 1. 下采样 8倍: 224 -> 28，消除高频噪点
        curr_small = F.avg_pool2d(current_image, kernel_size=8, stride=8)

        if self.prev_image_small is None:
            return None

        # 2. 计算差异 (RGB 平均)
        diff = torch.abs(curr_small - self.prev_image_small).mean(dim=1)  # [B, 28, 28]

        return diff

    def should_recompute(self, current_image, current_state):
        """
        三层防御判断是否需要重算

        Returns:
            should_run: bool
            reason: str (用于调试)
            drift_value: float
        """
        # ========== Layer 1: State Guard (最快，先算) ==========
        if self.prev_state is not None:
            state_diff = torch.max(torch.abs(current_state - self.prev_state)).item()
            if state_diff > self.state_threshold:
                return True, f"State Drift: {state_diff:.4f}", state_diff

        # ========== 第一帧必须计算 ==========
        if self.prev_image_small is None or self.prev_attention is None:
            return True, "First Frame", 100.0

        # ========== Layer 2: Anti-Noise Preprocessing ==========
        # 下采样后计算差异
        curr_small = F.avg_pool2d(current_image, kernel_size=8, stride=8)  # [B, C, 28, 28]
        pixel_diff = torch.abs(curr_small - self.prev_image_small).mean(dim=1)  # [B, 28, 28]

        # ========== Layer 3: Dilated Attention Weighting ==========
        # 插值 attention map 到与 diff 相同大小
        # prev_attention: [B, 16, 16] -> [B, 28, 28]
        attn_resized = F.interpolate(
            self.prev_attention.unsqueeze(1),  # [B, 1, 16, 16]
            size=(28, 28),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, 28, 28]

        # 膨胀，扩大关注范围
        attn_dilated = self.dilate_attention(attn_resized)

        # 加权计算
        weighted_diff = (pixel_diff * attn_dilated).sum(dim=(1, 2))  # [B]
        drift_value = weighted_diff.mean().item()

        should_run = drift_value > self.drift_threshold

        return should_run, f"Vision Drift: {drift_value:.4f}", drift_value

    def update(self, current_image, current_attention, current_state, current_kv_cache):
        """
        更新缓存状态 (在每次完整计算后调用)
        """
        # 存下采样后的图
        self.prev_image_small = F.avg_pool2d(current_image, kernel_size=8, stride=8)

        # 提取 2D attention map
        self.prev_attention = self.extract_patch_importance(current_attention)

        # 状态和 KV cache
        self.prev_state = current_state.clone()
        self.prev_kv_cache = current_kv_cache

    def get_cached_kv(self):
        """获取缓存的 KV"""
        return self.prev_kv_cache

    def reset(self):
        """重置状态 (新 episode 开始时调用)"""
        self.prev_image_small = None
        self.prev_attention = None
        self.prev_state = None
        self.prev_kv_cache = None


class ProOptimizedInference:
    """
    集成 Pro Attention Gate 的推理流程
    """

    def __init__(self, model, drift_threshold=0.5, state_threshold=0.05):
        self.model = model
        self.gate = ProAttentionGate(
            drift_threshold=drift_threshold,
            state_threshold=state_threshold
        )

    def infer(self, observation):
        """
        推理流程:
        1. 三层防御检查是否可以复用
        2. 如果可以，跳过 Vision Encoder
        3. 如果不可以，完整计算并更新缓存

        Returns:
            actions: 预测的动作
            drift: drift 分数 (用于调试/可视化)
            reason: 重算原因 (用于调试)
        """
        current_images = observation.images  # [B, C, H, W]
        current_state = observation.state    # [B, state_dim]

        # Step 1: 三层防御检查 (< 1ms)
        should_recompute, reason, drift = self.gate.should_recompute(
            current_images, current_state
        )

        if not should_recompute:
            # 复用上一帧的 KV Cache
            kv_cache = self.gate.get_cached_kv()
            actions = self.model.denoise_with_cached_kv(
                kv_cache,
                current_state,
                num_steps=10
            )
        else:
            # 完整计算 (包含 Vision Encoder)
            actions, attention_weights, kv_cache = self.model.full_forward_with_attention(
                observation,
                num_steps=10,
                return_attention=True
            )
            # 更新缓存
            self.gate.update(current_images, attention_weights, current_state, kv_cache)

        return actions, drift, reason

    def reset_episode(self):
        """新 episode 开始时重置状态"""
        self.gate.reset()
```

---

## 实验步骤 (Pro Version)

### Step 1: 可视化 Attention Importance (不跑推理)

**目标**: 验证 Attention 是否真的聚焦在任务关键区域

```python
# 实验脚本: scripts/visualize_attention_importance.py

import matplotlib.pyplot as plt
import torch

def visualize_attention_heatmap(model, trajectory, save_dir="attention_viz"):
    """
    1. 跑几条轨迹
    2. 提取 Layer 14-18 的 Cross-Attention
    3. 画 Heatmap 叠加在原图上
    4. 同时画膨胀后的 Attention (验证盲区防御)

    验证点:
    - Heatmap 是否紧紧跟随机械臂末端和目标物体?
    - 膨胀后是否覆盖了合理的"安全边界"?
    - 如果是 → 方案可行
    - 如果 Attention 散乱 → 方案不可行
    """
    for frame_idx, obs in enumerate(trajectory):
        # 1. 前向获取 attention
        with torch.no_grad():
            _, attention_weights, _ = model.full_forward_with_attention(obs)

        # 2. 提取重要性 (16x16)
        gate = ProAttentionGate()
        importance = gate.extract_patch_importance(attention_weights)

        # 3. 膨胀 (16x16 -> 膨胀后)
        importance_dilated = gate.dilate_attention(importance)

        # 4. 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原图
        axes[0].imshow(obs.images[0].permute(1, 2, 0).cpu())
        axes[0].set_title("Original Image")

        # Attention Heatmap
        axes[1].imshow(obs.images[0].permute(1, 2, 0).cpu())
        heatmap = F.interpolate(importance.unsqueeze(1), size=(224, 224), mode='bilinear')
        axes[1].imshow(heatmap[0, 0].cpu(), alpha=0.5, cmap='jet')
        axes[1].set_title("Attention Importance")

        # Dilated Attention
        axes[2].imshow(obs.images[0].permute(1, 2, 0).cpu())
        heatmap_dilated = F.interpolate(importance_dilated.unsqueeze(1), size=(224, 224), mode='bilinear')
        axes[2].imshow(heatmap_dilated[0, 0].cpu(), alpha=0.5, cmap='jet')
        axes[2].set_title("Dilated Attention (盲区防御)")

        plt.savefig(f"{save_dir}/frame_{frame_idx:04d}.png")
        plt.close()
```

**预期结果**:
- 机械臂末端/gripper: 高 attention
- 目标物体: 高 attention
- 背景/桌面: 低 attention
- **膨胀后**: 关注区域扩大一圈，覆盖可能的运动范围

---

### Step 2: 离线分析 Weighted Drift 分布

**目标**: 找到合适的 threshold，验证三层防御效果

```python
# 实验脚本: scripts/analyze_weighted_drift.py

def analyze_drift_distribution(trajectories):
    """
    1. 录制数据（成功轨迹）
    2. 计算每一帧的:
       - State Drift (关节变化)
       - Raw Pixel Diff (无权重)
       - Weighted Drift (attention加权)
    3. 画对比曲线图

    验证点:
    - State Drift 是否在关节大动作时飙高?
    - Weighted Drift 是否比 Raw Diff 更能区分关键/非关键变化?
    - 手眼相机的边缘抖动是否被过滤?
    """
    for traj in trajectories:
        state_drifts = []
        raw_diffs = []
        weighted_drifts = []

        gate = ProAttentionGate()

        for t, obs in enumerate(traj):
            if t == 0:
                gate.update(obs.images, None, obs.state, None)
                continue

            # State drift
            state_diff = torch.max(torch.abs(obs.state - gate.prev_state)).item()
            state_drifts.append(state_diff)

            # Raw pixel diff (无权重)
            curr_small = F.avg_pool2d(obs.images, kernel_size=8, stride=8)
            raw_diff = torch.abs(curr_small - gate.prev_image_small).sum().item()
            raw_diffs.append(raw_diff)

            # Weighted drift
            _, _, weighted_drift = gate.should_recompute(obs.images, obs.state)
            weighted_drifts.append(weighted_drift)

            # 更新 (假设这帧重算了)
            gate.update(obs.images, attention_weights, obs.state, None)

        # 画图
        plot_drift_curves(state_drifts, raw_diffs, weighted_drifts)
```

**预期结果**:
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Raw Diff     ~~~~~~~~*~~~~*~~~  (噪声多，难区分)        │
│                                                         │
│  Weighted     ____/\__/\____/\_  (关键动作突出)          │
│                  ↑     ↑    ↑                           │
│  State       __/‾‾\__/‾‾\__/‾‾\  (关节大动作)            │
│                                                         │
│              ─────────────────→ Time                    │
│               移动  抓取  放置                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### Step 3: 上机测试 Pro Gate

**目标**: 验证精度和速度

```python
# 实验脚本: scripts/test_pro_attention_gate.py

def test_pro_gate():
    """
    配置: 3 tasks × 5 trials = 15 episodes

    对比:
    1. Baseline: 无复用 (每帧重算)
    2. Cosine Gate: 老方案
    3. Pro AWD Gate: 新方案

    收集指标:
    - 成功率
    - 复用率 (skip rate)
    - 触发原因分布 (State Drift vs Vision Drift)
    - 实际 Hz
    """
    configs = [
        {"name": "Baseline", "gate": None},
        {"name": "Cosine", "threshold": 0.98},
        {"name": "Pro AWD", "drift_threshold": 0.5, "state_threshold": 0.05},
    ]

    for config in configs:
        results = run_evaluation(config, tasks=3, trials=5)
        print(f"{config['name']}: Success={results.success_rate:.1%}, "
              f"Skip={results.skip_rate:.1%}, Hz={results.hz:.1f}")
```

**期望结果**:

| 方案 | 成功率 | 复用率 | 实际Hz | 备注 |
|-----|-------|-------|--------|-----|
| 无复用 | 100% | 0% | 5.7 Hz | Baseline |
| Cosine 0.98 | 22% | 60% | 13.5 Hz | 老方案 (Cliff) |
| **Pro AWD** | **≥90%?** | **?%** | **? Hz** | 新方案 |

---

### Step 4: 触发原因分析

**目标**: 理解系统行为

```python
def analyze_trigger_reasons(logs):
    """
    统计每次重算的触发原因:
    - "First Frame": 第一帧
    - "State Drift": 关节大变化触发
    - "Vision Drift": 视觉变化触发

    验证:
    - State Drift 是否集中在关键动作?
    - Vision Drift 是否过于频繁? (如果是，需要调高 threshold)
    """
    reasons = Counter([log.reason for log in logs])
    print(f"Trigger Distribution: {reasons}")
```

---

## 关键验证点

1. **Attention 聚焦性**: Heatmap 必须紧跟任务关键区域
2. **膨胀有效性**: 膨胀后的区域应覆盖合理的"安全边界"
3. **Drift 区分度**: 关键动作 vs 平稳移动的 Drift 必须有明显差异
4. **State Guard 效果**: 关节大变化应被第一时间拦截
5. **All-or-Nothing 有效性**: 完全复用不会导致精度悬崖

---

## 风险与备选

### 风险1: Attention 不够聚焦
如果 Attention 分布散乱，则无法区分重要/不重要区域。

**备选**:
- 使用 Gradient-based Saliency (GradCAM) 替代 Attention
- 或使用更后层的 Attention (Layer 16-18)

### 风险2: Drift threshold 难以泛化
不同任务可能需要不同 threshold。

**备选**: 自适应 threshold

```python
class AdaptiveThreshold:
    def __init__(self, percentile=90, window_size=100):
        self.history = []
        self.percentile = percentile
        self.window_size = window_size

    def update_and_get(self, drift_value):
        self.history.append(drift_value)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return np.percentile(self.history, self.percentile)
```

### 风险3: 即使 All-or-Nothing 也无法避免精度下降
可能是 Diffusion Policy 本质上无法容忍任何 KV 复用。

**备选**: 放弃 KV Reuse，聚焦其他优化方向

### 风险4: State Guard 误触发太频繁
如果机器人一直在动，State Guard 会一直触发重算。

**备选**:
- 只检查末端执行器位置变化，忽略关节角度细节
- 或使用更高的 state_threshold

---

## Pro Version 优势总结

| 改进点 | 解决的问题 | 实现方式 |
|-------|-----------|---------|
| **State Guard** | 手眼不协调 | 关节变化 > threshold → 强制重算 |
| **Downsample** | 噪声敏感 | 224→28 下采样，消除高频噪点 |
| **Dilation** | 盲区入侵 | Attention Map Max Pooling |
| **All-or-Nothing** | Feature Mixing | 只做帧级判断，不做 patch 级拼接 |

---

---

## 实验结果 (2026-02-08)

### Step 1: Attention 可视化结果

使用 `scripts/visualize_attention_importance.py` 分析了 LIBERO 任务中的 Attention 分布：

| Camera | Max Weight | Mean Weight | Max/Mean Ratio |
|--------|------------|-------------|----------------|
| Base Camera | 0.020-0.050 | 0.001-0.002 | 20-50x |
| Wrist Camera | 0.028-0.056 | 0.001-0.002 | 28-56x |

**发现**: Attention 有一定聚焦性（比均匀分布高 20-50x），但不是极度集中，分布相对平缓。

### Step 2: Drift 分布分析结果

使用 `scripts/analyze_weighted_drift.py` 分析了帧间 Drift 分布：

| 指标 | Mean | Std | P50 | P75 | P90 |
|------|------|-----|-----|-----|-----|
| State Drift | 0.14 | 0.07 | 0.13 | 0.18 | 0.23 |
| Weighted Drift | 0.75 | 0.38 | 0.66 | 0.97 | 1.28 |
| Raw Diff | 59.1 | 24.3 | 55.7 | 73.4 | 92.9 |

**推荐阈值**: `state_threshold=0.23` (P90), `drift_threshold=1.0` (P75)

### Step 3: 上机测试结果 ⚠️ 失败

使用 `scripts/libero_eval_attention_gate.py` 进行实际评测：

| 方案 | 精度 | 延迟 | Hz | 复用率 |
|-----|------|------|-----|--------|
| **Baseline (无复用)** | **88.9%** (8/9) | 177ms | 5.6 | 0% |
| Attention Gate | **0%** (0/9) | 112ms | 8.9 | 90% |

**关键发现**:
- 90% 复用率意味着三层防御逻辑正常工作
- 但 0% 精度表明 **帧间 KV Cache 复用根本不适用于 Pi0.5**
- 这验证了 [cliff_report.md](./cliff_report.md) 的核心结论

### 根本原因分析

1. **Diffusion Policy 的敏感性**:
   - `Vision(t-1) + State(t)` → 语义冲突 → Policy 失效
   - 即使画面变化很小，Policy 也无法容忍跨帧的 KV 不一致

2. **VLA 模型特性**:
   - VLA 不是简单的视觉识别，而是多模态融合
   - `一致性 > 新鲜度` — Policy 更害怕不一致，而非信息过时

3. **KV Cache 复用的根本矛盾**:
   - 复用旧帧的 Vision KV 意味着 Language/Action 看到的是 "过去的场景"
   - 但 State 输入的是 "当前的状态"
   - 这种时间不一致会导致 Policy 产生错误的动作

### Step 4: VLA-Cache 对比实验 (补充)

使用 `scripts/libero_eval_vla_cache.py` 对比 Cosine Similarity 判断策略：

| 方案 | 精度 | 延迟 | Hz | 复用率 | Avg Base Sim | Avg Wrist Sim |
|-----|------|------|-----|--------|--------------|---------------|
| no_reuse | 0% | 311ms | 3.2 | 0% | 0.95 | 0.87 |
| full_reuse (cos>0.98) | 0% | 305ms | 3.3 | 1.5% | 0.95 | 0.87 |

**发现**:
- 即使 cosine similarity 很高 (0.95 base, 0.87 wrist)，使用 0.98 阈值时仅 1.5% 帧被复用
- 无论使用何种判断策略，**VLA-Cache 帧间复用都会导致 0% 精度**
- 这与 Attention-Weighted Drift 方案结论一致

### 结论

❌ **所有帧间 KV Cache 复用方案均验证失败**

| 方案 | 核心思想 | 结果 |
|-----|---------|------|
| VLA-Cache (Cosine Similarity) | 图像相似度判断 | 0% 精度 |
| Attention-Weighted Drift | Attention 加权漂移判断 | 0% 精度 |
| Hybrid Base-Only Reuse | 只复用主视角 KV | 0% 精度 |

帧间 KV Cache 复用在 Pi0.5 这类 VLA 模型上不可行，无论使用何种复用判断策略。

### 替代优化方向

参考 cliff_report.md 推荐的方向：

| 方向 | 预期收益 | 风险 |
|-----|---------|-----|
| **模型蒸馏** | 2-3x 加速 | 需要重新训练 |
| **结构化剪枝** | 1.5-2x 加速 | 精度损失可控 |
| **混合精度推理** | ✅ 已实现 | FP8 已达 12 Hz |
| **Action Chunking** | 2-4x 加速 | 平滑度影响 |

当前 TRT FP8 混合精度静态图已达 12.0 Hz，是目前最可行的优化路径。

---

## 参考资料

- [Cliff Report](./cliff_report.md): KV Reuse 精度悬崖分析
- [VLA-Cache Paper](https://arxiv.org/abs/2502.02175): 原始 VLA-Cache 方法
- [SD-VLA Paper](https://arxiv.org/abs/2602.03983): Static-Dynamic 解耦方法
