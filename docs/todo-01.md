# TODO-01: TRT 优化项目状态总结

## 目标
- **目标延迟**: 45ms (22 Hz) - Zhiyuan 标准
- **当前最佳**: 83.50ms (12.0 Hz) @ 1 step - debug-07 已验证

---

## 已验证的优化 (来自 debug 日志)

### Debug-07: 完整优化 Pipeline (已验证 ✅)

| 组件 | 技术 | 延迟 | 加速比 | 精度 |
|------|------|------|--------|------|
| **Vision** | torch_tensorrt FP16 | 17.22 ms | 2.03x | ✅ 100% |
| **KV Cache** | ModelOpt FP8 + TRT MLP | 52.14 ms | 1.7x | ✅ 100% |
| **Denoise** | CUDA Graph | 10.1 ms/step | 2.6x | ✅ 100% |

### 不同 Denoising Steps 性能 (已验证 ✅)

| Steps | 总延迟 | Hz | Vision | KV Cache | Denoise | 精度 |
|-------|--------|-----|--------|----------|---------|------|
| 1 | **83.50 ms** | **12.0** | 17.26 ms | 52.14 ms | 10.10 ms | ✅ 100% |
| 2 | 93.07 ms | 10.7 | 17.12 ms | 52.13 ms | 19.70 ms | ✅ 100% |
| 3 | 102.81 ms | 9.7 | 17.22 ms | 52.14 ms | 29.38 ms | ✅ 100% |
| 5 | 122.21 ms | 8.2 | 17.22 ms | 52.23 ms | 48.87 ms | ✅ 100% |
| 10 | 171.36 ms | 5.8 | 17.26 ms | 52.13 ms | 97.69 ms | ✅ 100% |

### Debug-06: TRT FP8 MLP 修复 (已验证 ✅)

- **问题**: SEQ_LEN 硬编码为 970，实际为 968
- **修复后**: KV Cache 89ms → 51.95ms (1.71x)
- **TRT MLP 调用**: 0 → 360 (全部使用 TRT)

### Debug-03: Flash Attention + FP8 精度修复 (已验证 ✅)

- **问题1**: Flash Attention 不支持任意 attention mask → 改用 SDPA
- **问题2**: 错误使用 SiLU → 改用 GELU(approximate='tanh')
- **问题3**: FP8 dtype 不匹配 → 统一为 bfloat16

### Debug-mix-05: NVFP4 调研 (已验证 ❌ 不可用)

- TRT Python API FP4: Myelin segfault
- Torch-TRT NVFP4: cos=0.0004 (数值错误)
- **结论**: NVFP4 在 Thor 上目前不可用

### Debug-08: Full Coverage TRT 验证 (已验证 ⚠️ 提升有限)

| 方案 | 延迟 | Hz | 提升 |
|------|------|-----|------|
| torch_trt_fp8 (当前最佳) | 47.38 ms | 21.1 Hz | baseline |
| Full Coverage TRT | 45.75 ms | 21.9 Hz | **1.04x** |

- **TRT 覆盖率**: QKV 18/18, O 18/18, MLP 18/18 (全部编译成功)
- **精度**: Key cos=0.996, Value cos=0.996 (✅ PASSED)
- **结论**: QKV/O 额外 TRT 编译仅提升 4%，瓶颈在 MLP 和 Attention

---

## 已有实现文件 (按验证状态分类)

### ✅ 已验证可用

| 文件 | 功能 | 验证来源 |
|------|------|----------|
| `scripts/libero_eval_full_optimized.py` | 完整优化 Pipeline | debug-07 |
| `torch_trt_fp8_kv_cache.py` | FP8 KV Cache (SEQ_LEN=968) | debug-06 |
| `flash_fp8_kv_cache.py` | Flash Attention + SDPA | debug-03 |
| `fp8_mlp.py` | FP8 MLP (GELU 修复) | debug-03 |
| `trt_full_coverage_kv_cache.py` | 完整层 TRT (QKV+O+MLP) | debug-08 (⚠️ 仅 1.04x) |

### ⚠️ 待验证

| 文件 | 功能 | 备注 |
|------|------|------|
| `trt_mixed_precision.py` | 混合精度 TRT | 未找到验证记录 |
| `turbo_titan_pipeline.py` | Titan + KV Reuse | 未找到验证记录 |
| `turbo_trt_engine.py` | TurboAttention Plugin | 未找到验证记录 |

### ❌ 已知问题

| 文件/方案 | 问题 | 来源 |
|-----------|------|------|
| TRT Python API FP4 | Myelin segfault | debug-mix-05 |
| NVFP4 | Scale 被忽略，cos≈0 | debug-mix-05 |
| TRT INT8 KV Cache | 精度问题 | OPTIMIZATION_FINAL_REPORT |

---

## 当前瓶颈分析

### 组件延迟 (debug-08 更新)

| 组件 | Pipeline 中 | 独立 benchmark | 说明 |
|------|-------------|----------------|------|
| Vision TRT | 17.2 ms | - | 已优化 |
| **KV Cache** | **52.1 ms** | **47.4 ms** | 有 ~5ms 同步开销 |
| Denoise (1 step) | 10.1 ms | - | 已优化 |
| 其他开销 | ~4 ms | - | 数据传输等 |
| **Total** | **83.5 ms** | - | **12.0 Hz** |

### 达到 20Hz (50ms) 需要的优化

| 组件 | 当前 | 目标 | 需要加速 |
|------|------|------|----------|
| Vision | 17.2 ms | 17.2 ms | 保持 |
| **KV Cache** | **47.4 ms** | **~14 ms** | **3.4x** |
| Denoise | 10.1 ms | 10.1 ms | 保持 |
| 其他 | 8.8 ms | 8.8 ms | - |
| **总计** | **83.5 ms** | **50.1 ms** | - |

**KV Cache 需要 3.4x 加速才能达到 20Hz！单靠 TRT 优化已接近极限。**

### KV Cache 占比分析

- @ 1 step: 47.4ms / 83.5ms = **56.8%** (最大瓶颈)
- 独立 benchmark 47.4ms vs pipeline 52.1ms 有 ~5ms 差异

---

## 待办事项

### 高优先级 (算法级优化)
- [x] 验证 `trt_full_coverage_kv_cache.py` → **仅 1.04x，不显著** (debug-08)
- [ ] **探索 KV Cache Reuse 策略**:
  - [ ] 验证 `turbo_titan_pipeline.py` 的帧间 KV 复用
  - [ ] 动态相似度检测 + 部分复用
- [ ] **探索模型简化**:
  - [ ] 减少层数 (18层 → 12层)
  - [ ] Action Chunking (每次输出多步)

### 中优先级 (架构级优化)
- [ ] 完整 LIBERO-spatial 验证 (10 tasks, 10 trials)
- [ ] 异步三流水线 (Vision | KV Cache | Denoise 并行)
- [ ] KV Cache 预计算 + 后台更新

### 低优先级
- [ ] 清理重复/过时的实现文件
- [ ] 统一接口和命名规范

### 长期 (需要训练)
- [ ] 模型蒸馏 (训练更小的学生模型)

---

## 参考文档

| 文档 | 内容 |
|------|------|
| `debug-08.md` | Full Coverage TRT 验证 (1.04x, 提升有限) |
| `debug-07.md` | Vision TRT dtype 修复 + 完整优化结果 |
| `debug-06.md` | TRT FP8 SEQ_LEN 修复 + Vision/Denoise 优化 |
| `debug-03.md` | Flash Attention + FP8 精度问题修复 |
| `debug-mix-05.md` | NVFP4/混合量化调研 (不可用) |
| `OPTIMIZATION_FINAL_REPORT.md` | TRT INT8 性能报告 (有精度问题) |

---

## 最佳配置

```bash
# 运行完整优化 Pipeline (debug-07 验证过)
docker exec turbo_pi_eval python scripts/libero_eval_full_optimized.py \
    --task_suite_name libero_spatial \
    --denoising_steps 1 \
    --quick \
    --output_file results.json
```

**预期结果**: 83.50ms (12.0 Hz), 100% 准确率
