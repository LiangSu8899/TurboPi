# Turbo-Pi 与官方 OpenPi 对齐分析

本文档详细分析 Turbo-Pi 与官方 [OpenPi](https://github.com/Physical-Intelligence/openpi) 的对齐程度，明确差异来源，为后续贡献和协作提供参考。

---

## 1. 重要澄清：官方已有 PyTorch 支持

**官方 OpenPi 已经提供了完整的 PyTorch 实现**，包括：

| 功能 | 官方支持状态 | 说明 |
|------|-------------|------|
| Pi0 PyTorch 模型 | ✓ 已支持 | `models_pytorch/pi0_pytorch.py` |
| Pi0.5 (adaRMS) | ✓ 已支持 | `use_adarms=[False, True]` |
| 混合精度 (bfloat16) | ✓ 已支持 | 训练和推理均支持 |
| PyTorch 训练 | ✓ 已支持 | `train_pytorch.py` |
| JAX ↔ PyTorch 转换 | ✓ 已支持 | `load_pytorch()` 方法 |

> 官方文档明确说明：*"openpi now provides PyTorch implementations of π₀ and π₀.₅ models alongside the original JAX versions!"*

因此，**PyTorch 模型实现不是 Turbo-Pi 的独特贡献**。

---

## 2. 总体对齐情况

| 维度 | 对齐状态 | 说明 |
|------|----------|------|
| 模型架构 | **官方已有** | PyTorch + JAX 双实现 |
| 数值精度 | **完全对齐** | LIBERO Spatial 98%, LIBERO 10 91% |
| 归一化方式 | **完全对齐** | Quantile normalization (q01/q99) |
| 推理接口 | **向后兼容** | 支持原有 `serve_policy.py` |
| **TensorRT 加速** | **Turbo-Pi 独有** | 官方没有 |
| **异步流水线** | **Turbo-Pi 独有** | 官方没有 |
| **UnifiedPolicy** | **Turbo-Pi 独有** | 官方没有 `inference/` 目录 |

---

## 3. 官方 OpenPi 目录结构

```
src/openpi/
├── models/              # JAX 模型实现
├── models_pytorch/      # PyTorch 模型实现 ← 官方已有！
│   ├── pi0_pytorch.py   # Pi0/Pi0.5 PyTorch 模型
│   ├── gemma_pytorch.py # Gemma 架构
│   └── transformers_replace/  # HuggingFace 补丁
├── policies/            # 策略接口
├── serving/             # WebSocket 服务
├── shared/              # 共享工具（归一化等）
└── training/            # 训练代码
```

**注意**：官方**没有** `inference/` 目录。

---

## 4. Turbo-Pi 独有贡献

### 4.1 `inference/` 目录（完全独有）

官方 OpenPi 没有 `inference/` 目录，这是 Turbo-Pi 的核心贡献：

| 文件 | 功能 | 独有性 |
|------|------|--------|
| `inference/unified_policy.py` | 统一推理接口 | **Turbo-Pi 独有** |
| `inference/trt_pipeline.py` | TensorRT 推理管线 | **Turbo-Pi 独有** |
| `inference/async_pipeline.py` | 异步流水线执行 | **Turbo-Pi 独有** |

### 4.2 `transformers_replace` 修改（ONNX 兼容性）

官方和 Turbo-Pi 都有 `transformers_replace/` 目录，但我们做了以下修改：

| 修改 | 原因 | 文件 |
|------|------|------|
| 添加 ONNX 导出注释 | 说明 weight 参数必须存在 | `modeling_gemma.py` |
| 创建 `__init__.py` | 运行时 patching 备选方案 | `transformers_replace/__init__.py` |
| `ensure_patched()` 函数 | 动态 patching transformers | `__init__.py` |

**关键代码差异**（`modeling_gemma.py` 第 56-59 行）：
```python
# Turbo-Pi 添加的注释：
# Always define weight for ONNX export compatibility
# In non-adaptive mode: used for scaling
# In adaptive mode: not used directly but required for ONNX export
self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
```

**原因**：ONNX 导出时需要 weight 属性存在，否则会报错。官方没有这个需求因为不做 ONNX 导出。

**官方状态**：官方没有 `__init__.py`，只有 `models/` 目录。

### 4.3 TensorRT 加速

官方不提供任何 TensorRT 相关功能：

| 组件 | 文件 | 功能 |
|------|------|------|
| ONNX 导出 | `scripts/export_onnx_components.py` | 模型转换 |
| TRT Engine 构建 | `trt_pipeline.py` | 自动构建引擎 |
| FP16 推理 | 内置 | 2x 速度提升 |

### 4.3 异步流水线

双 CUDA Stream 并行执行架构：

```
Vision Stream:  [Vision(n+1)] ────────────────────>
                                    ↘ 重叠执行
Action Stream:              [Denoise(n) x3] ────────>

结果: 37.2ms 延迟, 26.9 Hz 吞吐量
```

### 4.4 UnifiedPolicy 接口

提供跨后端的统一 API：

```python
from openpi.inference import UnifiedPolicy

policy = UnifiedPolicy(
    checkpoint_dir="...",
    backend="tensorrt_pipelined",  # 官方没有这个选项
    num_denoising_steps=3,
)
result = policy.infer(observation)
```

官方推理方式需要通过 `policies/` 模块，接口不同。

---

## 5. 对齐的组件（官方已有）

### 5.1 模型架构

| 组件 | 官方状态 | 说明 |
|------|----------|------|
| SigLIP Vision Encoder | ✓ 官方已有 | JAX + PyTorch |
| PaliGemma (2B) | ✓ 官方已有 | JAX + PyTorch |
| Gemma Expert (300M) | ✓ 官方已有 | 包括 adaRMS |
| Shared Attention | ✓ 官方已有 | 前缀+后缀联合计算 |

### 5.2 归一化方式

官方和 Turbo-Pi 使用完全相同的 quantile normalization：

```python
# 两者完全一致
normalized = (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
unnormalized = (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
```

### 5.3 WebSocket 服务

```bash
# 官方接口完全支持
python scripts/serve_policy.py --env=LIBERO --port=8000 \
    policy:checkpoint --policy.config=pi05_libero
```

---

## 6. 依赖差异

### 6.1 核心依赖对比

| 依赖 | 官方 OpenPi | Turbo-Pi | 差异说明 |
|------|-------------|----------|----------|
| `jax[cuda12]` | 必需 | 可选 | Turbo-Pi 可纯 PyTorch |
| `flax` | 必需 | 可选 | JAX 模型需要 |
| `torch` | 必需 (PyTorch 后端) | 必需 | 两者都需要 |
| `transformers` | 需要 | 需要 + 补丁 | 自定义 Gemma 层 |
| `tensorrt` | **无** | **必需** | Turbo-Pi 独有 |
| `pycuda` | **无** | **必需** | CUDA 流管理 |

### 6.2 Turbo-Pi 额外依赖

```
tensorrt>=8.6
pycuda
onnx
onnxruntime-gpu  # 可选，用于验证
```

---

## 7. 差异分类汇总

### 7.1 按来源分类

```
┌─────────────────────────────────────────────────────────┐
│                    官方 OpenPi                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ JAX 模型 + PyTorch 模型 + 训练 + 服务           │   │
│  │ (models/ + models_pytorch/ + training/ + serving/)│   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          ↓ 完全兼容
┌─────────────────────────────────────────────────────────┐
│                    Turbo-Pi 独有                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │ inference/ 目录（全部）                          │   │
│  │ - unified_policy.py (统一接口)                  │   │
│  │ - trt_pipeline.py (TensorRT 推理)              │   │
│  │ - async_pipeline.py (异步流水线)               │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ONNX 导出                                        │   │
│  │ - export_onnx_components.py                     │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Docker 配置 (Thor 专属)                          │   │
│  │ - Dockerfile                                    │   │
│  │ - Dockerfile.libero_eval                        │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 7.2 按可贡献性分类

| 分类 | 组件 | 建议 |
|------|------|------|
| ~~可直接贡献~~ | ~~PyTorch 模型实现~~ | ❌ 官方已有 |
| **可贡献（新功能）** | `inference/` 目录 | 讨论是否需要 |
| **可贡献（新功能）** | ONNX 导出脚本 | 作为可选工具 |
| **保持独立** | TensorRT 流水线 | 硬件依赖性强 |
| **Thor 专属** | 异步流水线 + Docker | 不适合上游 |

---

## 8. 硬件专属 vs 通用优化

### 8.1 通用优化（可移植到其他 GPU）

| 组件 | 适用范围 | 说明 |
|------|----------|------|
| ONNX 导出 | 所有 CUDA GPU | 模型格式转换 |
| TensorRT 顺序推理 | 所有支持 TRT 的 GPU | ~20 Hz |
| UnifiedPolicy | 所有 PyTorch 环境 | 纯 Python 接口 |

### 8.2 Thor 专属优化（硬件绑定）

| 组件 | 绑定原因 | 其他 GPU 情况 |
|------|----------|---------------|
| 异步流水线 | SM 分区 + 内存带宽 | 效果可能不同 |
| Docker 基础镜像 | sm_110 + JetPack 7.1 | 需要更换镜像 |
| 双流并行 | TensorRT 10.x | 版本依赖 |
| 26.9 Hz 吞吐量 | Thor 硬件特性 | 其他 GPU 数值不同 |

---

## 9. 性能对比

| 后端 | 吞吐量 | 来源 |
|------|--------|------|
| 官方 JAX | 1.4 Hz | OpenPi 原始 |
| 官方 PyTorch | ~2-3 Hz | OpenPi models_pytorch |
| Turbo-Pi TensorRT | ~20 Hz | **Turbo-Pi 独有** |
| Turbo-Pi Pipelined | 26.9 Hz | **Turbo-Pi 独有 (Thor)** |

**核心价值**：Turbo-Pi 的 TensorRT 优化带来 **10-19x 加速**。

---

## 10. 上游贡献建议（修订版）

### 10.1 不需要贡献

| 组件 | 原因 |
|------|------|
| PyTorch 模型实现 | 官方已有 `models_pytorch/` |
| Pi0.5 (adaRMS) 支持 | 官方已有 |
| 混合精度训练 | 官方已有 |

### 10.2 可讨论贡献

| 组件 | 价值 | 复杂度 |
|------|------|--------|
| ONNX 导出脚本 | 便于模型转换 | 低 |
| TensorRT 文档 | 性能优化指南 | 低 |
| UnifiedPolicy 接口 | 统一推理 API | 中 |

### 10.3 保持独立

| 组件 | 原因 |
|------|------|
| 异步流水线 | 硬件依赖性强 |
| Thor Docker | 特定平台 |
| TensorRT 引擎 | 需要 NVIDIA 工具链 |

---

## 11. 总结

### 关键结论

1. **官方 OpenPi 已有完整 PyTorch 支持**（Pi0 + Pi0.5）
2. **Turbo-Pi 的核心贡献是 TensorRT 加速层**（`inference/` 目录）
3. **19x 性能提升**来自 TensorRT + 异步流水线
4. **部分优化是 Thor 专属**，需要特定硬件

### 差异来源分类（修订版）

```
Turbo-Pi 独有工作 100%
├── TensorRT 加速 (45%)
│   ├── ONNX 导出脚本 (15%)
│   ├── TRT 引擎构建 (15%)
│   └── trt_pipeline.py (15%)
│
├── 推理接口 (20%)
│   ├── unified_policy.py (15%)
│   └── 后端抽象 (5%)
│
├── 异步流水线 (15%)
│   └── async_pipeline.py (Thor 专属)
│
├── transformers_replace 修改 (10%)
│   ├── ONNX 兼容性注释 (5%)
│   └── __init__.py patching (5%)
│
└── 部署配置 (10%)
    ├── Dockerfile (5%)
    └── Dockerfile.libero_eval (5%)
```

**注意**：`transformers_replace/models/` 目录中的模型文件与官方基本相同，我们只添加了 ONNX 导出相关的注释和运行时 patching 机制。

### 与官方对齐的建议行动

1. **认可官方 PyTorch 实现**：不重复造轮子
2. **专注 TensorRT 优化**：这是 Turbo-Pi 的核心价值
3. **可选贡献 ONNX 导出**：作为官方可选工具
4. **维护 Turbo-Pi 作为性能优化分支**：专注边缘设备部署

---

*最后更新: 2026-01-31*
*版本: v1.1.2*
