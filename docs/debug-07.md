# Debug-07: Vision TRT dtype 修复 + 完整优化 Pipeline 性能测试

## 问题描述

在 debug-06 中，Vision TRT 被禁用，原因是：
- Vision TRT 输出 **FP16 (half)**
- `multi_modal_projector` 期望 **BFloat16** 输入
- 导致 RuntimeError: "mat1 and mat2 must have the same dtype, but got Half and BFloat16"

## 修复方案

### 1. dtype 转换修复

**修改文件**: `openpi/scripts/libero_eval_full_optimized.py`

**核心修复**: 在 Vision TRT 输出后添加 dtype 转换

```python
def _setup_vision_trt(self):
    """Setup Vision TRT with dtype conversion fix."""
    import torch_tensorrt

    # Get vision tower and multi_modal_projector
    vision_tower = self.model.paligemma_with_expert.paligemma.vision_tower
    self.multi_modal_projector = self.model.paligemma_with_expert.paligemma.model.multi_modal_projector

    # Compile vision tower with TRT FP16
    wrapper = VisionWrapper(vision_tower).to(self.device).half()
    wrapper.eval()

    self.vision_trt = torch_tensorrt.compile(
        wrapper,
        inputs=[torch_tensorrt.Input(shape=(1, 3, 224, 224), dtype=torch.float16)],
        enabled_precisions={torch.float16},
        workspace_size=4 << 30,
        min_block_size=1,
    )

    # Warmup with dtype conversion
    for _ in range(5):
        img_fp16 = torch.randn(1, 3, 224, 224, device=self.device, dtype=torch.float16)
        vision_out = self.vision_trt(img_fp16)
        # Convert FP16 → BFloat16 for multi_modal_projector
        vision_out_bf16 = vision_out.to(torch.bfloat16)
        _ = self.multi_modal_projector(vision_out_bf16)

    self.use_vision_trt = True
```

**推理时的数据流**:
```
Image (FP16) → Vision TRT (FP16) → .to(bfloat16) → multi_modal_projector (BF16) → prefix_embs
```

### 2. 组件耗时统计

新增 `component_latencies` 字典记录每个组件的耗时：

```python
self.component_latencies = {
    'vision': [],
    'kv_cache': [],
    'denoise': [],
    'total': [],
}
```

在 `infer()` 方法中分段计时：

```python
# Vision timing
torch.cuda.synchronize()
vision_start = time.perf_counter()
# ... vision processing ...
torch.cuda.synchronize()
vision_time = (time.perf_counter() - vision_start) * 1000
self.component_latencies['vision'].append(vision_time)

# KV Cache timing
torch.cuda.synchronize()
kv_start = time.perf_counter()
# ... kv cache processing ...
torch.cuda.synchronize()
kv_time = (time.perf_counter() - kv_start) * 1000
self.component_latencies['kv_cache'].append(kv_time)

# Denoise timing
torch.cuda.synchronize()
denoise_start = time.perf_counter()
# ... denoise processing ...
torch.cuda.synchronize()
denoise_time = (time.perf_counter() - denoise_start) * 1000
self.component_latencies['denoise'].append(denoise_time)
```

---

## LIBERO 验证结果

### 测试配置
- Task Suite: libero_spatial (3 tasks, 3 trials per task - quick mode)
- Backend: Full Optimized (Vision TRT + KV Cache TRT FP8 + Denoise CUDA Graph)
- Vision: TRT FP16 + dtype conversion (修复后)

### 不同 Denoising Steps 性能对比

| Denoising Steps | Accuracy | Mean Latency (ms) | Hz | Vision (ms) | KV Cache (ms) | Denoise (ms) |
|-----------------|----------|-------------------|-----|-------------|---------------|--------------|
| 10 | **100%** | 171.36 | 5.8 | 17.26 (10.1%) | 52.13 (30.4%) | 97.69 (57.0%) |
| 5 | **100%** | 122.21 | 8.2 | 17.22 (14.1%) | 52.23 (42.7%) | 48.87 (40.0%) |
| 3 | **100%** | 102.81 | 9.7 | 17.22 (16.7%) | 52.14 (50.7%) | 29.38 (28.6%) |
| 2 | **100%** | 93.07 | 10.7 | 17.12 (18.4%) | 52.13 (56.0%) | 19.70 (21.2%) |
| 1 | **100%** | **83.50** | **12.0** | 17.26 (20.7%) | 52.14 (62.4%) | 10.10 (12.1%) |

### 组件性能分析 (3 steps 配置)

| 组件 | 耗时 (ms) | 占比 | 技术 |
|------|-----------|------|------|
| **Vision** | 17.22 | 16.7% | torch_tensorrt FP16 + dtype conversion |
| **KV Cache** | 52.14 | 50.7% | TRT FP8 MLP + Flash Attention |
| **Denoise** | 29.38 | 28.6% | CUDA Graph (3 steps) |
| **Total** | 102.81 | 100% | - |

---

## 性能对比 (相比 debug-06)

| 配置 | debug-06 (无 Vision TRT) | debug-07 (完整优化) | 提升 |
|------|--------------------------|---------------------|------|
| Total Latency (3 steps) | 120.6 ms | **102.81 ms** | **-17.8 ms** |
| Hz | 8.3 Hz | **9.7 Hz** | **+1.4 Hz** |
| Vision | ~35 ms (PyTorch) | **17.22 ms** (TRT) | **2.03x** |
| Accuracy | 100% | **100%** | 保持 |

---

## 关键发现

1. **所有配置 100% 准确率**: 1/2/3/5/10 步全部保持 100% 成功率！

2. **1 步配置达到 12.0 Hz**: 最快配置，83.50ms 延迟

3. **Vision TRT 稳定**: ~17.2ms，不随 denoising steps 变化

4. **KV Cache 是固定瓶颈**: 稳定在 ~52.1ms，占比随 steps 减少而增加

5. **Denoise 线性增长**: ~10ms per step (CUDA Graph 开销低)

6. **相比 baseline 提升 2.1x**: 从 ~176ms (5.7 Hz) 到 83.5ms (12.0 Hz)

---

## 优化技术总结

| 组件 | 优化技术 | 加速比 |
|------|----------|--------|
| Vision Encoder | torch_tensorrt.compile(FP16) + dtype conversion | ~2.03x |
| KV Cache (18层) | ModelOpt FP8 量化 + torch_tensorrt MLP + Flash Attention | ~1.7x |
| Denoising | CUDA Graph 捕获 (预分配 tensor, 静态 timestep) | ~2.6x |

---

## 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `scripts/libero_eval_full_optimized.py` | 1. 修复 Vision TRT dtype 问题 (FP16→BF16 转换)<br>2. 添加组件级耗时统计<br>3. 启用 Vision TRT |

---

## 评估脚本

```bash
# 运行完整优化 pipeline 评估
python scripts/libero_eval_full_optimized.py \
    --task_suite_name libero_spatial \
    --denoising_steps 3 \
    --quick \
    --output_file results.json
```

---

## Denoising Steps 分析

### Denoise 耗时线性模型

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

**结论**: CUDA Graph 开销稳定，每步约 10ms，适合减少 denoising steps 提升帧率。

---

## 20Hz 目标分析

当前最快配置 (1 step): **83.50ms (12.0 Hz)**

要达到 **20 Hz (50ms)**，需要：
- Vision: 17.2ms → 保持
- KV Cache: 52.1ms → **需优化到 ~23ms** (2.3x 加速)
- Denoise: 10.1ms → 保持

**KV Cache 是达成 20Hz 目标的关键瓶颈！**

---

## 待办

- [x] Vision TRT dtype 兼容性修复
- [x] 组件级耗时统计
- [x] 不同 denoising steps 测试 (1/2/3/5/10)
- [ ] 完整 LIBERO-spatial 验证 (10 tasks, 10 trials)
- [ ] 进一步优化 KV Cache (占比 62.4% @ 1 step)

---

## 经验教训

1. **dtype 转换成本低**: FP16 → BFloat16 转换开销可忽略不计，不影响 TRT 加速收益

2. **组件级 profiling 很重要**: 分段计时能准确定位瓶颈

3. **CUDA synchronize 保证准确计时**: 每个组件前后都需要同步

4. **Vision TRT 编译稳定**: 27 层 SigLIP 完整编译到 TRT，无 fallback
