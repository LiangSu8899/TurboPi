# Denoise Module Deep Profiling Guide

## 目标

精密诊断 Denoise 模块 100ms (10 steps) 延迟的根因：
1. **Kernel Launch Overhead** - CPU 到 GPU 的启动延迟
2. **Memory Bandwidth** - HBM 带宽瓶颈
3. **Stream Synchronization** - 隐式同步问题

---

## 诊断判据

### 1. Gap Analysis (Kernel Launch Latency)

| Gap 时间 | 诊断结论 | 行动 |
|---------|---------|------|
| **5-10us** | ✅ NORMAL | CUDA Graph 正常工作 |
| **20-50us** | ⚠️ WARNING | 存在轻微 CPU 干预 |
| **50us-1ms** | 🚨 SEVERE | 严重 CPU Launch Bound |

**异常现象**: Step 与 Step 之间存在 >50us 的"气泡"

**Root Cause**:
- Python for-loop 开销
- 中间有 tensor 的 CPU/GPU 同步
- 动态 shape 导致无法图捕获

### 2. Memory Bandwidth Analysis

| SM Util | DRAM BW | 诊断结论 |
|---------|---------|---------|
| > 60% | < 50% | Compute Bound |
| < 30% | > 80% | 🚨 Memory Bound |
| < 30% | < 50% | Launch Bound |

**异常现象**: MLP/Linear 层运行时，SM 利用率低但 DRAM 带宽爆满

**Root Cause**:
- 权重矩阵太大，L2 Cache 放不下
- 每次 GEMM 都要从 HBM 读取权重

### 3. Stream Synchronization

| 检查项 | 正常 | 异常 |
|--------|------|------|
| `cudaStreamSynchronize` | 0 次 | 🚨 有调用 |
| `cudaDeviceSynchronize` | 仅在末尾 | 🚨 循环内有 |
| `print(tensor)` | 无 | 🚨 有打印 |
| `tensor.item()` | 无 | 🚨 有调用 |

---

## 使用方法

### Step 1: 运行 NVTX 标记的 Profiling

```bash
# 在 Docker 容器内运行
docker exec -it turbo_pi_eval bash

cd /workspace

# 完整 profiling
./scripts/run_denoise_profiling.sh

# 或快速模式 (较少迭代)
./scripts/run_denoise_profiling.sh --quick
```

### Step 2: 查看 Gap 分析报告

```bash
# 文本报告
cat profile_output/denoise_profile.analysis.txt

# JSON 摘要
cat profile_output/denoise_profile.analysis.json
```

### Step 3: 在 Nsight Systems GUI 中查看

```bash
# 打开可视化 Timeline
nsys-ui profile_output/denoise_profile.nsys-rep
```

在 GUI 中检查：
1. **Timeline View**: 看 kernel 之间的间隙
2. **GPU Metrics Row**: 看 SM Utilization 和 DRAM Bandwidth
3. **NVTX Markers**: 定位到具体的 Step 和 Layer

---

## nsys 命令详解

```bash
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \    # 追踪 CUDA API + NVTX + 系统调用
    --cuda-memory-usage=true \                # 内存使用统计
    --gpu-metrics-device=all \                # GPU 指标采样 (SM%, DRAM BW)
    --sample=cpu \                            # CPU 采样
    --cpuctxsw=process-tree \                 # 进程上下文切换
    --output=denoise_profile \                # 输出文件名
    --force-overwrite=true \                  # 覆盖已有文件
    --stats=true \                            # 输出统计摘要
    python scripts/profile_denoise_nsys.py --steps 10
```

---

## NVTX 标记结构

```
Denoise_Full_Loop
├── Denoise_Step_0
│   ├── Step_0/Time_Embed
│   ├── Step_0/Action_Proj_In
│   ├── Step_0/Mask_Prep
│   ├── Step_0/Layer_0
│   │   ├── Step_0/Layer_0/LN
│   │   ├── Step_0/Layer_0/QKV_Proj
│   │   ├── Step_0/Layer_0/RoPE
│   │   ├── Step_0/Layer_0/KV_Concat
│   │   ├── Step_0/Layer_0/Attn
│   │   ├── Step_0/Layer_0/O_Proj
│   │   ├── Step_0/Layer_0/Res1
│   │   ├── Step_0/Layer_0/PostLN
│   │   ├── Step_0/Layer_0/MLP       ← 重点关注
│   │   └── Step_0/Layer_0/Res2
│   ├── Step_0/Layer_1
│   │   └── ...
│   ├── Step_0/Final_Norm
│   └── Step_0/Action_Proj_Out
├── Denoise_Step_1
│   └── ...
└── Denoise_Step_9
```

---

## 预期输出示例

### 正常情况 (CUDA Graph 工作良好)

```
Gap Analysis:
  SEVERE Gaps (>50us): 0
  WARNING Gaps (20-50us): 5
  NORMAL Gaps (<20us): 1800
  Average Gap: 8.5 us

✅ DIAGNOSIS: KERNEL LAUNCH EFFICIENT
```

### 异常情况 (CPU Launch Bound)

```
Gap Analysis:
  SEVERE Gaps (>50us): 180  ← 每个 step 之间都有
  Average Gap: 85.3 us
  Max Gap: 1200.5 us

🚨 DIAGNOSIS: CPU LAUNCH BOUND
   - Python for-loop overhead is significant
   - Recommend: CUDA Graph capture or kernel fusion
```

---

## 下一步行动

### 如果诊断为 CPU Launch Bound

1. **确认 CUDA Graph 捕获失败的原因**
   ```python
   # 检查是否有动态 shape
   # 检查是否有 data-dependent control flow
   ```

2. **改用 Persistent Kernel**
   - 在 kernel 内部实现 grid-level 循环
   - 避免重复 launch overhead

### 如果诊断为 Memory Bound

1. **启用 L2 Cache Residency**
   ```c
   cudaStreamAttrValue stream_attribute;
   stream_attribute.accessPolicyWindow.base_ptr = weights;
   stream_attribute.accessPolicyWindow.num_bytes = weight_size;
   stream_attribute.accessPolicyWindow.hitRatio = 1.0;
   stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
   ```

2. **权重压缩**
   - INT4/FP4 量化减少 HBM 带宽需求
   - 使用 CUTLASS Mixed-Precision GEMM

### 如果发现隐式同步

1. **删除所有 `print(tensor)` 语句**
2. **避免 `tensor.item()` 和 `tensor.cpu()`**
3. **用 NVTX marker 替代打印调试**

---

## 文件清单

| 文件 | 用途 |
|------|------|
| [scripts/profile_denoise_nsys.py](../openpi/scripts/profile_denoise_nsys.py) | NVTX 埋点的 Denoise 执行脚本 |
| [scripts/analyze_nsys_gaps.py](../openpi/scripts/analyze_nsys_gaps.py) | SQLite 分析脚本 |
| [scripts/run_denoise_profiling.sh](../openpi/scripts/run_denoise_profiling.sh) | 一键运行脚本 |

---

## 参考资料

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [CUDA Best Practices: Kernel Launch Overhead](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [L2 Cache Residency Control](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-cache-management)
