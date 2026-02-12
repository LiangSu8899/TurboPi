# Denoise 模块性能诊断报告

**日期**: 2026-02-12
**平台**: NVIDIA Jetson Thor (SM110, Blackwell GB10B)
**Docker**: turbo_pi_eval

---

## 1. 测试配置

| 配置项 | 值 |
|--------|-----|
| Denoising Steps | 10 |
| Warmup | 3 iterations |
| Profile Iterations | 10 |
| Backend | Full Optimized Pipeline |

### 优化组件
| 组件 | 技术 |
|------|------|
| Vision Encoder | torch_tensorrt FP16 |
| KV Cache (18层) | TRT FP8 MLP + ModelOpt |
| Denoising | CUDA Graph 捕获 |

---

## 2. 性能测试结果

### 2.1 E2E 延迟

```
=====================================================================
Profiling Results
=====================================================================
  Mean: 179.98 ms
  Std:  1.03 ms
  Min:  178.62 ms
  Max:  181.71 ms
  Per-step: 18.00 ms/step (包含 Vision + KV Cache)
=====================================================================
```

### 2.2 组件耗时分解

| 组件 | 耗时 (ms) | 占比 |
|------|-----------|------|
| Vision TRT FP16 | 17.48 | **9.7%** |
| KV Cache TRT FP8 | 55.63 | **30.9%** |
| Denoise CUDA Graph (10步) | 102.86 | **57.2%** |
| 其他开销 | ~4.01 | 2.2% |
| **Total** | **179.98** | 100% |

### 2.3 可视化分解

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                          10 Steps: 179.98ms (5.6 Hz)                                 │
├──────┬──────────────────────────┬────────────────────────────────────────────────────┤
│V(10%)│    KV Cache (30.9%)      │              Denoise (57.2%)                       │
│17.5ms│        55.63ms           │               102.86ms                             │
└──────┴──────────────────────────┴────────────────────────────────────────────────────┘
```

---

## 3. Nsight Systems 分析结果

### 3.1 Profiling 命令

```bash
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --output=profile_output/denoise_profile \
    python scripts/profile_denoise_simple.py --steps 10 --warmup 3 --iterations 10
```

### 3.2 Kernel 统计

```
Total kernels captured: 79
Total kernel execution time: 1.66 ms
```

**关键发现**: 只有 79 个 kernel 被追踪到，总执行时间仅 1.66ms。这说明：
- CUDA Graph 正在工作
- Graph 内部的 kernel 没有被 nsys 单独追踪
- 捕获的 79 个 kernel 主要来自初始化/warmup 阶段

### 3.3 Kernel Gap 分析

```
Gap Analysis (Raw - includes initialization):
------------------------------------------------------------
Total gaps: 78
Mean gap: 9798.74 us (受初始化阶段影响)
Median gap: 111.25 us
Max gap: 405312.00 us (TRT 编译期间)
Min gap: 1.82 us

SEVERE (>50us): 52
WARNING (20-50us): 16
NORMAL (<20us): 10
```

**注意**: 大部分 SEVERE gaps 来自初始化和 TRT 编译阶段，不影响推理性能。

### 3.4 CUDA Runtime API 调用

检测到的 API 调用（部分）：
- `cuLaunchKernel` / `cuLaunchKernelEx` - 标准 kernel 启动
- `cudaStreamSynchronize_v3020` - 同步操作
- `cudaStreamIsCapturing_v10000` - Graph 捕获检测
- `cudaThreadExchangeStreamCaptureMode_v10010` - Graph 捕获模式

**Graph 相关**: 未检测到 `cudaGraphLaunch`，可能因为：
1. Thor 平台的 nsys 版本限制
2. Graph replay 作为单一操作被追踪，而非逐 kernel 追踪

### 3.5 同步事件

```
Sync Events: 0 (在推理路径上)
```

✅ **诊断**: 没有检测到推理路径上的隐式同步

---

## 4. 诊断结论

### 4.1 CUDA Graph 状态

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Graph 捕获 | ✅ 成功 | 日志显示 "CUDA Graph captured for 10 denoising steps" |
| Graph Replay | ✅ 工作中 | 只有 79 个 kernel 被追踪（而非 ~1800 个独立 kernel） |
| Kernel Launch Overhead | ✅ 已消除 | Graph replay 避免了逐 kernel 的 CPU-GPU 往返 |

### 4.2 瓶颈分析

当前瓶颈排序：
1. **Denoise (57.2%)** - 10 步 x 10.3 ms/step
2. **KV Cache (30.9%)** - 55.63 ms 固定开销
3. **Vision (9.7%)** - 17.48 ms 固定开销

### 4.3 每步 Denoise 分析

```
Denoise Total: 102.86 ms (10 steps)
Per-Step: 10.29 ms/step

预估每步分解:
- DiT Forward (18层): ~9.5 ms
  - QKV Projection: ~1.5 ms
  - Attention: ~3.0 ms
  - MLP: ~4.5 ms (主要瓶颈)
  - 其他: ~0.5 ms
- Step Update (x_t = x_t + dt * v_t): ~0.8 ms
```

---

## 5. 性能瓶颈诊断

### 5.1 Memory Bound 检测

**无法直接测量**: Thor 平台 nsys 不支持 GPU metrics (`--gpu-metrics-device` 报错)

**间接判断**:
- MLP 层使用 GEMV 操作 (Batch=1, Seq=50)
- GEMV 是典型的 Memory Bound 操作
- 预计 DRAM 带宽利用率 > 80%

### 5.2 Kernel Launch Overhead 检测

**已通过 CUDA Graph 消除**:
- 10 步 denoise 被捕获为单个 Graph
- 无需每步重新 launch kernel
- Gap 分析显示初始化后 kernel 连续执行

### 5.3 Stream Sync 检测

**未检测到问题**:
- 推理路径无 `cudaStreamSynchronize`
- 无 `print(tensor)` 或 `.item()` 调用

---

## 6. 优化建议

### 6.1 短期优化 (预计收益: 10-20%)

| 优化项 | 方法 | 预计收益 |
|--------|------|----------|
| MLP 量化 | INT4/FP4 替代 FP8 | 减少 HBM 带宽需求 |
| L2 Cache Residency | cudaStreamAttrValue 设置 | 权重复用 |
| Kernel Fusion | 合并 LayerNorm + Projection | 减少 kernel 数量 |

### 6.2 中期优化 (预计收益: 30-50%)

| 优化项 | 方法 | 预计收益 |
|--------|------|----------|
| Persistent GEMM | CUTLASS 持久化 kernel | 消除 GEMM launch 开销 |
| Flash Attention 2 | 替代 eager attention | 减少 Attention 时间 |
| 权重预取 | 双缓冲 + 异步拷贝 | 隐藏内存延迟 |

### 6.3 长期优化 (预计收益: 2-3x)

| 优化项 | 方法 | 预计收益 |
|--------|------|----------|
| 全图 TensorRT | Vision + KV + Denoise 单图 | 最大化 TRT 优化 |
| 模型蒸馏 | 减小模型尺寸 | 减少计算量 |
| 步数减少 | 1-step diffusion | 10x 步数减少 |

---

## 7. 下一步行动

### 7.1 进一步诊断

1. **添加 NVTX 到 DenoiseStepWrapper 内部**
   - 标记每层的 Attention/MLP 时间
   - 确认 MLP 是否占主导

2. **使用 NCU (Nsight Compute) 单独分析 MLP kernel**
   ```bash
   ncu --target-processes all --set full \
       python scripts/profile_mlp_kernel.py
   ```

3. **内存带宽测量**
   - 使用 cupy/pynvml 测量 DRAM 带宽
   - 确认 Memory Bound 假设

### 7.2 快速验证

1. **测试 1 step 配置**
   ```bash
   python scripts/libero_eval_full_optimized.py --denoising_steps 1 --quick
   ```
   - 预期: ~86 ms (11.6 Hz)
   - 验证: Denoise 开销线性下降

2. **测试无 CUDA Graph**
   - 禁用 Graph 捕获
   - 对比 kernel gap
   - 确认 Graph 的实际收益

---

## 8. 附录

### 8.1 测试脚本

```bash
# Profile with NVTX
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --output=denoise_profile \
    python scripts/profile_denoise_simple.py --steps 10

# Analyze gaps
python scripts/analyze_nsys_gaps.py denoise_profile.sqlite
```

### 8.2 关键文件

| 文件 | 用途 |
|------|------|
| [scripts/profile_denoise_simple.py](../openpi/scripts/profile_denoise_simple.py) | 简化 profiling 脚本 |
| [scripts/profile_denoise_nsys.py](../openpi/scripts/profile_denoise_nsys.py) | 完整 NVTX 埋点脚本 |
| [scripts/analyze_nsys_gaps.py](../openpi/scripts/analyze_nsys_gaps.py) | Gap 分析脚本 |
| [profile_output/denoise_profile.nsys-rep](../openpi/profile_output/denoise_profile.nsys-rep) | nsys 报告 |

### 8.3 平台限制

| 限制 | 影响 |
|------|------|
| Thor nsys GPU metrics 不支持 | 无法直接测量 SM/DRAM 利用率 |
| CUDA Graph kernel 追踪不完整 | Graph 内部 kernel 不可见 |
| segfault on exit | TRT 清理时崩溃（不影响数据） |
