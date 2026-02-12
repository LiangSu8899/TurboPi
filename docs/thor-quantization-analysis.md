# Thor SM110 量化分析报告

**日期**: 2026-02-09
**目标**: 评估 W4A4/W4A8/W4A16 量化在 Jetson Thor (SM110) 上的可行性

## 执行摘要

经过全面测试，发现 **Thor SM110 目前的量化支持有限**：

| 方案 | 状态 | 性能 | 备注 |
|------|------|------|------|
| **PyTorch FP16/BF16 (cuBLAS)** | ✅ 最佳 | 89.88 TFLOPS (batch=256) | 当前推荐方案 |
| TorchAO INT4/INT8 | ❌ | 比BF16慢 | 需要 fbgemm-gpu-genai |
| TorchAO FP8 | ⚠️ 工作但慢 | 0.48 TFLOPS | 无优化kernel |
| TensorRT FP16 | ✅ | 2.16 TFLOPS | 与PyTorch相当 |
| TensorRT INT8 | ❌ | 需要calibrator | 未测试 |
| TensorRT FP8 | ⚠️ | 可能fallback | 未真正加速 |
| Triton自定义kernel | ❌ | 比cuBLAS慢3x | 需要Thor特定优化 |
| CUTLASS W4A16 | ⚠️ | 编译问题 | SM100a/SM110a PTX限制 |

## 详细测试结果

### 1. PyTorch cuBLAS 基准性能

Thor SM110 上原生 cuBLAS 性能（FP16/BF16）：

| Batch Size | 矩阵大小 | FP16 TFLOPS | BF16 TFLOPS |
|------------|----------|-------------|-------------|
| 1 | 1536×6144 | 0.59 | 0.63 |
| 8 | 1536×6144 | 5.06 | 5.11 |
| 64 | 1536×6144 | 38.74 | 38.80 |
| 256 | 1536×6144 | 89.15 | 89.88 |

**理论峰值**: 258 TFLOPS (FP16/BF16)
**实测效率**: ~35% (batch=256)

### 2. TorchAO 量化测试

所有 TorchAO 量化方法在 Thor 上都 **比 BF16 更慢**：

```
W8A16 (Int8WeightOnly):     0.66 TFLOPS, Speedup: 0.16x
W8A8 (Int8DynActInt8Weight): 0.06 TFLOPS, Speedup: 0.01x
FP8 Dynamic (W8A8-FP8):     0.48 TFLOPS, Speedup: 0.11x
W4A8-FP8:                   ❌ Requires fbgemm-gpu-genai >= 1.2.0
```

**原因**: TorchAO 的量化后端没有针对 SM110 优化的 kernel，fallback 到通用实现。

### 3. TensorRT 量化测试

TensorRT 10.14 在 Thor 上：

- **FP16**: 工作正常，与 PyTorch 性能相当
- **INT8**: 需要 calibrator，无法直接使用
- **FP8**: 编译成功但可能未真正使用 FP8 kernel（输出与 FP16 完全相同）

### 4. CUTLASS 测试

参见 [w4a8-plugin.md](./w4a8-plugin.md)

- **W4A4 (NVFP4×NVFP4)**: 基础 kernel 可工作 (141 TFLOPS)，但高性能 2SM kernel 不支持
- **W4A8 (FP8×NVFP4)**: SM110 不支持 `tcgen05.mma.kind::mxf8f6f4`
- **W4A16 (INT4×BF16)**: 编译时 PTX target 问题 (sm_100 vs sm_100a)

SM110 不支持的 PTX 特性：
- `.cta_group::2` (2SM 协作)
- `.block_scale` (块级缩放)
- `.kind::mxf8f6f4` (FP8×FP4 混合类型)
- `tcgen05.mma` 的大部分高级功能

### 5. Triton 测试

自定义 Triton kernel 在 Thor 上比 cuBLAS 慢约 3x：

```
Size          cuBLAS    Triton    Speedup
8x1536x6144   5.08T     1.37T     0.27x
8x6144x1536   4.78T     1.65T     0.34x
```

**原因**: Triton 的 autotune 没有针对 SM110 优化的配置。

## Thor SM110 硬件限制分析

Thor 是 Blackwell 架构的 **edge/automotive 变体**，相比 datacenter GPU (B100/B200)：

1. **SM 数量**: 20 SMs (vs B100 的 160+ SMs)
2. **功能裁剪**: 为降低功耗/成本，移除了部分 datacenter 特性
3. **PTX 限制**: 不支持某些 `tcgen05` 指令变体

### 社区报告对比

| 设备 | 架构 | W4A4 性能 | 来源 |
|------|------|-----------|------|
| B100/B200 | SM100 | 878 TFLOPS | NVIDIA Forum |
| Thor | SM110 | 141 TFLOPS | 我们测试 |

差距 6x 的原因是 Thor 不支持优化的 2SM 调度。

## 建议方案

### 短期 (当前可用)

1. **继续使用 BF16/FP16**
   - cuBLAS 已经很优化
   - Batch=1 时 ~5 TFLOPS，足够实时推理

2. **使用 TensorRT FP16 静态图**
   - 参见 [tvm-trt.md](./tvm-trt.md) 中的成功案例
   - 已验证: 12.0 Hz @ FP8 静态图

### 中期 (等待软件更新)

1. **等待 fbgemm-gpu-genai 支持 SM110**
   - 可解锁 TorchAO 的 INT4/INT8 量化

2. **等待 NVIDIA 更新 TensorRT for Thor**
   - 可能添加真正的 FP8/INT8 优化

### 长期 (硬件限制)

某些功能 (如 W4A8) 可能永远不会在 Thor 上可用，因为硬件不支持。

## 测试环境

```
Device: NVIDIA Thor
Compute Capability: 11.0
SMs: 20
Memory: 131.88 GB

TensorRT: 10.14.1.48
PyTorch: 2.10.0a0+b4e4ee81d3.nv25.12
TorchAO: 0.15.0+git01374eb5
Triton: 3.5.1
CUDA: 13.x
```

## 相关资源

- [w4a8-plugin.md](./w4a8-plugin.md) - W4A8 开发记录和 SM110 限制分析
- [tvm-trt.md](./tvm-trt.md) - TVM+TRT 集成成功案例
- [GitHub Issue #4590](https://github.com/NVIDIA/tensorrt/issues/4590) - Thor FP8/FP4 silent fallback

---

*最后更新: 2026-02-09*
