# OpenPi 对齐会议准备

本文档为与 Physical Intelligence 官方团队对齐会议准备的参考建议。

---

## 1. 开源优化社区的成功模式参考

### 1.1 业界案例分析

#### vLLM 与 TensorRT-LLM 的生态位分工

| 项目 | 定位 | 维护方 | 特点 |
|------|------|--------|------|
| [vLLM](https://github.com/vllm-project/vllm) | 社区驱动的通用推理框架 | UC Berkeley → 社区 | 硬件无关、易用性优先 |
| [TensorRT-LLM](https://developer.nvidia.com/tensorrt) | NVIDIA 官方极致优化 | NVIDIA | 硬件绑定、性能优先 |
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) | 模型压缩统一库 | NVIDIA | 连接上游模型与下游部署框架 |

**关键洞察**: 即使是同一领域（LLM 推理），也形成了明确的分层生态：
- **上游模型层**: Hugging Face Transformers (模型定义)
- **中间优化层**: Model Optimizer (量化、剪枝)
- **下游部署层**: vLLM/TensorRT-LLM (推理服务)

#### PyTorch 生态系统工作组模式

根据 [PyTorch Ecosystem Working Group](https://pytorch.org/blog/introducing-the-pytorch-ecosystem-working-group-and-project-spotlights/) 的模式：

- **入选标准**: 测试完善、易于上手、社区活跃
- **关系定位**: 独立项目，非官方子项目
- **协作方式**: 官方推荐但不负责维护

**成功案例**:
- [SGLang](https://github.com/sgl-project/sglang) - 社区驱动的推理优化
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 微软的训练优化
- Hugging Face Transformers - 模型库生态

#### Hugging Face TGI 的定位

[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) 的演进路径值得参考：

1. **初期**: HuggingFace 内部优化
2. **中期**: 开源社区参与，推动 KV Cache 等标准化
3. **现在**: 明确表示推荐下游框架 (vLLM, SGLang)

> *"TGI has initiated the movement for optimized inference engines to rely on a transformers model architectures. This approach is now adopted by downstream inference engines, which Hugging Face contributes to and recommends using going forward."*

---

### 1.2 上下游协作最佳实践

根据 [Fedora Upstream First 原则](https://docs.fedoraproject.org/en-US/project/upstream-first/) 和 [Linux Foundation 指南](https://www.linuxfoundation.org/resources/open-source-guides/improving-your-open-source-development-impact)：

#### "Upstream First" 原则

| 原则 | 说明 | Turbo-Pi 应用 |
|------|------|---------------|
| **减少分叉维护** | 上游演进时下游补丁难以维护 | 通用 Bug 修复应贡献上游 |
| **质量保证** | 上游测试覆盖更广 | dtype 修复等应接受上游 CI |
| **社区信任** | 小改动开始，逐步建立信任 | 先贡献 Bug fix，再讨论优化 |

#### 沟通策略

> *"If you are the first collaborator from your company, listen and observe the behavior of existing maintainers. Try to learn more about the existing collaboration process from them."*

**建议**:
1. 先了解官方的路线图和设计理念
2. 询问官方对边缘设备部署的看法
3. 展示具体数据（LIBERO 98%/91%）而非仅代码

---

## 2. Turbo-Pi 工作定位分析

### 2.1 我们做了什么

```
Turbo-Pi 工作分布 (约 2,700 行代码)

┌─────────────────────────────────────────────────────────────┐
│  TensorRT 加速层 (45%)                                       │
│  ├── ONNX 导出 + TRT 引擎构建                               │
│  ├── trt_pipeline.py (582 行)                               │
│  └── 目标: 3-4x 额外加速                                    │
├─────────────────────────────────────────────────────────────┤
│  KV Cache 优化 (20%)                                        │
│  ├── compute_prefix_kv_cache()                              │
│  ├── denoise_step_with_cache()                              │
│  └── 目标: 7-8x 推理加速 (可贡献上游)                       │
├─────────────────────────────────────────────────────────────┤
│  统一推理接口 (15%)                                         │
│  ├── unified_policy.py (557 行)                             │
│  └── 目标: 屏蔽后端差异                                     │
├─────────────────────────────────────────────────────────────┤
│  异步流水线 (15%)                                           │
│  ├── async_pipeline.py (570 行)                             │
│  └── 目标: Thor 专属 26.9 Hz                                │
├─────────────────────────────────────────────────────────────┤
│  兼容性修复 (5%)                                            │
│  ├── dtype 转换修复 (Bug fix)                               │
│  ├── weight tying 修复                                      │
│  └── ONNX 兼容性                                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 我们的技术栈偏向

| 维度 | 我们的偏向 | 官方偏向 |
|------|-----------|----------|
| **硬件** | Jetson Thor (边缘 GPU) | 云端 GPU (A100/H100) |
| **框架** | PyTorch + TensorRT | JAX + PyTorch |
| **优先级** | 延迟 (实时控制) | 吞吐量 (批量推理) |
| **部署** | 嵌入式容器化 | 服务器集群 |
| **用户** | 机器人开发者 | 研究人员 |

### 2.3 验证的正确性

| 基准 | 结果 | 预期 | 状态 |
|------|------|------|------|
| LIBERO Spatial | 98% | 98% | :white_check_mark: 完全对齐 |
| LIBERO 10 | 91% | 91% | :white_check_mark: 完全对齐 |
| 数值精度 | bfloat16 | bfloat16 | :white_check_mark: 一致 |

---

## 3. 官方 OpenPi 工作定位分析

### 3.1 官方已完成

| 功能 | 状态 | 说明 |
|------|------|------|
| JAX 模型实现 | :white_check_mark: | 原始参考实现 |
| PyTorch 模型实现 | :white_check_mark: | Pi0 + Pi0.5 (adaRMS) |
| 训练管线 | :white_check_mark: | JAX + PyTorch |
| WebSocket 服务 | :white_check_mark: | serve_policy.py |
| LIBERO 基准 | :white_check_mark: | 官方验证数据 |

### 3.2 官方可能关注的方向

基于 Physical Intelligence 的定位（机器人基础模型公司），推测其路线图：

| 方向 | 可能性 | Turbo-Pi 相关度 |
|------|--------|-----------------|
| 新模型架构 (Pi1.0?) | 高 | 低 - 需要重新适配 |
| 更多机器人平台支持 | 高 | 中 - 边缘部署需求 |
| 云端推理服务 | 中 | 低 - 不同场景 |
| 边缘设备优化 | 中 | **高 - 核心价值** |
| 社区生态建设 | 中 | **高 - 可贡献** |

### 3.3 官方可能的顾虑

| 顾虑 | 说明 | 应对策略 |
|------|------|----------|
| **维护负担** | 不想维护硬件相关代码 | 明确 Turbo-Pi 独立维护 |
| **质量保证** | 担心优化引入 Bug | 展示完整测试覆盖 |
| **架构分歧** | 设计理念可能不同 | 征求意见，适配官方风格 |
| **依赖复杂性** | TensorRT 依赖增加门槛 | 保持可选，不影响核心 |

---

## 4. 互补合作方案建议

### 4.1 合作模式选项

#### 选项 A: 官方认可的独立生态项目

**模式**: 类似 DeepSpeed 之于 PyTorch

```
Physical Intelligence (官方)
    └── OpenPi (核心项目)
            ↓ 认可 & 推荐
        Turbo-Pi (独立项目)
            └── 边缘设备优化分支
```

**优点**:
- 最小化官方维护负担
- Turbo-Pi 保持独立演进
- 官方文档可推荐

**缺点**:
- 需要持续跟进上游变化
- 品牌关联度较低

#### 选项 B: 上游贡献 + 下游扩展

**模式**: 类似 TGI 贡献 vLLM 再推荐 vLLM

```
贡献通用优化:
    - KV Cache 优化 → 合入 OpenPi
    - dtype 修复 → 合入 OpenPi
    - SDPA attention → 合入 OpenPi

保持独立:
    - TensorRT 管线 → Turbo-Pi
    - 异步流水线 → Turbo-Pi
    - Thor 专属优化 → Turbo-Pi
```

**优点**:
- 通用优化惠及所有用户
- 减少下游维护负担
- 建立上游信任

**缺点**:
- 需要适配官方代码风格
- PR 审核周期不确定

#### 选项 C: 官方边缘部署子项目

**模式**: 类似 Hugging Face Optimum

```
OpenPi
├── openpi-core (官方维护)
├── openpi-edge (联合维护)  ← Turbo-Pi 演进
│   ├── tensorrt/
│   ├── jetson/
│   └── ...
└── openpi-cloud (官方维护)
```

**优点**:
- 最紧密的官方关系
- 共享测试基础设施
- 品牌统一

**缺点**:
- 需要官方大量投入
- 决策流程可能复杂

### 4.2 推荐方案: 选项 B (上游贡献 + 下游扩展)

**理由**:
1. **务实**: 不需要官方改变项目结构
2. **渐进**: 先建立信任，再深化合作
3. **灵活**: 保持 Turbo-Pi 独立演进能力

---

## 5. 稳定性、可维护性、可扩展性考量

### 5.1 稳定性策略

| 策略 | 实施方式 |
|------|----------|
| **版本锁定** | 明确标注兼容的 OpenPi 版本 |
| **回归测试** | LIBERO Spatial/10 作为 CI 基准 |
| **数值验证** | JAX vs PyTorch vs TensorRT 输出对比 |
| **渐进发布** | alpha → beta → stable 阶段 |

```yaml
# 建议的版本兼容矩阵
turbo-pi: "1.1.x"
openpi: ">=0.2.0,<0.3.0"  # 明确兼容范围
tensorrt: ">=10.0"
pytorch: ">=2.0"
```

### 5.2 可维护性策略

| 策略 | 实施方式 |
|------|----------|
| **最小化分叉** | 仅在必要处修改上游代码 |
| **清晰边界** | `inference/` 目录完全独立 |
| **自动同步** | CI 检测上游更新并警告 |
| **文档维护** | 每次上游更新时更新 align_detail.md |

```python
# 代码组织原则
src/openpi/
├── [官方代码]          # 尽量不动，或提交 PR
├── inference/          # Turbo-Pi 独有，完全自维护
└── [适配层]            # 最小化，有注释说明原因
```

### 5.3 可扩展性策略

| 策略 | 实施方式 |
|------|----------|
| **后端插件化** | UnifiedPolicy 抽象不同后端 |
| **配置驱动** | 通过配置而非代码切换优化 |
| **模块解耦** | TensorRT 可选，不影响 PyTorch 路径 |

```python
# 扩展性示例
policy = UnifiedPolicy(
    backend="pytorch",           # 或 "tensorrt", "tensorrt_pipelined"
    num_denoising_steps=3,       # 可配置
    enable_kv_cache=True,        # 可选优化
)
```

### 5.4 上游同步计划

```
每月检查:
├── 拉取最新 OpenPi main 分支
├── 运行 LIBERO 回归测试
├── 更新 align_detail.md 差异分析
└── 评估是否需要适配

每季度评估:
├── 是否有通用优化可以贡献上游
├── 上游 API 变化是否需要重构
└── 社区反馈收集
```

---

## 6. 会议议程建议

### 6.1 开场 (5 分钟)

- 自我介绍
- 感谢邀请对齐
- 说明 Turbo-Pi 的起源和目标

### 6.2 技术展示 (15 分钟)

**展示重点**:
1. LIBERO 98%/91% 验证结果 (证明正确性)
2. 26.9 Hz 性能数据 (证明价值)
3. 代码结构差异图 (证明最小侵入)

**可准备的 Demo**:
```bash
# 实时在 Jetson Thor 上运行 LIBERO 任务
python scripts/libero_eval_unified.py --backend tensorrt_pipelined --quick
```

### 6.3 合作讨论 (20 分钟)

**需要询问官方的问题**:

1. **路线图**: "OpenPi 未来是否有边缘设备部署的计划？"
2. **设计理念**: "对于推理优化，官方更倾向于什么方式？"
3. **贡献意愿**: "对于 KV Cache、SDPA 等通用优化，是否欢迎 PR？"
4. **合作模式**: "您建议我们以什么方式协作最合适？"

**我们可以提供的承诺**:

1. 独立维护 TensorRT/Thor 相关代码
2. 贡献通用 Bug 修复和性能优化
3. 保持与上游版本的兼容性测试
4. 在文档中明确说明与官方的关系

### 6.4 后续计划 (5 分钟)

- 确定下一步行动项
- 建立沟通渠道 (Slack/Discord/GitHub)
- 约定下次同步时间

---

## 7. 关键信息准备

### 7.1 一句话定位

> "Turbo-Pi 是 OpenPi 的边缘设备优化分支，专注于 Jetson Thor 等嵌入式 GPU 上的高性能部署，在保持与官方完全数值对齐的前提下，实现 19x 推理加速。"

### 7.2 核心数据

| 指标 | 数值 | 说明 |
|------|------|------|
| 性能提升 | 19x | 1.4 Hz → 26.9 Hz |
| 精度对齐 | 100% | LIBERO 98%/91% 与预期一致 |
| 代码改动 | ~400 行 | 在官方 ~20,000 行基础上 |
| 新增代码 | ~2,300 行 | 主要在 `inference/` 目录 |
| 平台专属 | 25% | 仅 async_pipeline + Thor 脚本 |

### 7.3 技术差异化

```
官方 OpenPi 强项:
├── JAX 训练生态
├── 多机器人平台支持
├── 研究可复现性
└── 模型架构创新

Turbo-Pi 强项:
├── TensorRT 推理优化
├── 边缘设备部署
├── 实时控制延迟
└── 嵌入式容器化
```

---

## 8. 总结

### 8.1 我们的核心价值

1. **不重复造轮子**: 认可并依赖官方 PyTorch 实现
2. **专注边缘优化**: 填补官方尚未覆盖的领域
3. **保持向后兼容**: 官方 checkpoint 直接可用
4. **愿意贡献上游**: 通用优化可以惠及所有人

### 8.2 希望达成的合作

1. **官方认可**: 在文档中提及 Turbo-Pi 作为边缘部署选项
2. **技术协作**: 接受通用优化 PR (KV Cache, dtype fix)
3. **沟通渠道**: 建立定期同步机制
4. **长期关系**: 作为生态系统的一部分共同发展

### 8.3 会议成功标准

- [ ] 明确官方对边缘部署的态度
- [ ] 确定哪些优化可以贡献上游
- [ ] 商定合作模式 (A/B/C 或其他)
- [ ] 建立后续沟通渠道

---

## 参考资料

- [PyTorch Ecosystem Working Group](https://pytorch.org/blog/introducing-the-pytorch-ecosystem-working-group-and-project-spotlights/)
- [Fedora Upstream First](https://docs.fedoraproject.org/en-US/project/upstream-first/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- [Hugging Face TGI](https://github.com/huggingface/text-generation-inference)
- [Linux Foundation - Improving Open Source Impact](https://www.linuxfoundation.org/resources/open-source-guides/improving-your-open-source-development-impact)

---

*准备日期: 2026-01-30*
*版本: v1.0*
