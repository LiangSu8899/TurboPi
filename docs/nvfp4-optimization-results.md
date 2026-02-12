# nvFP4 优化突破报告

**日期**: 2026-02-10
**作者**: Claude Code

---

## 执行摘要

成功实现了 **比 TRT FP8 快 1.46x** 的 nvFP4 GEMV kernel！

| 指标 | TRT FP8 | nvFP4 Packed | 提升 |
|------|---------|--------------|------|
| **延迟** | 0.53 ms | 0.36 ms | **1.46x** |
| **内存读取** | 18.8 MB | 4.72 MB | **4x** |
| **带宽利用** | ~35 GB/s | ~16 GB/s | - |

---

## 问题诊断

### 原始问题

TVM 生成的 kernel 比 TRT FP8 慢 1.6x：

```
TVM Naive:  ~0.93 ms
TRT FP8:    ~0.53 ms
差距:       1.75x 慢
```

### 根本原因分析

1. **错误的数据格式**: TVM kernel 使用 `float32` 模拟 FP4 值
   - 内存读取: 36.7 MB
   - 这比 FP8 (18.8 MB) 还多 2x！

2. **Global memory 累加**: TVM 编译器没有将累加提升到寄存器
   ```cuda
   // TVM 生成的代码
   for (k = 0; k < 3072; ++k) {
       C[idx] = C[idx] + a_val * w_val;  // 每次都读写 global memory!
   }
   ```

3. **缺乏向量化**: 标量访问，无 float4/uint4 向量化

---

## 解决方案

### 关键优化：使用真正的 4-bit Packed 格式

```
权重格式对比:
┌─────────────────────────────────────────────────────┐
│ float32 模拟:  [f32][f32][f32][f32]... = 36.7 MB   │
│ packed uint8:  [u8] = 2 个 FP4 值    ... = 4.72 MB  │
└─────────────────────────────────────────────────────┘
带宽节省: 8x
```

### 优化版本演进

| 版本 | 优化内容 | 延迟 | vs FP8 |
|------|---------|------|--------|
| V0 | TVM Naive (float32) | 0.93 ms | 0.57x |
| V1 | Packed + 寄存器累加 | 0.44 ms | 1.22x |
| V2 | + Shared memory A | 0.54 ms | 0.98x |
| V3 | + 向量化 (uint32) | 0.42 ms | 1.26x |
| **V4** | + **Warp reduce** | **0.36 ms** | **1.46x** |

### 最佳实现：V4 Warp Reduce

```cuda
// 核心思想: 每个 warp (32 threads) 协作计算一个输出元素
// K 维度被 32 个线程并行处理，最后用 warp shuffle 规约

__global__ void nvfp4_gemv_packed_v4_warp_reduce(...) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float local_sum = 0.0f;

    // 每个 lane 处理 K/32 的部分
    for (int k = lane_id * (K/32); k < (lane_id+1) * (K/32); k += 2) {
        uint8_t w_packed = W_packed[j * (K/2) + k/2];
        float2 w_vals = unpack_nvfp4_pair(w_packed);
        local_sum += A[k] * scale_A * w_vals.x * scale_W + ...;
    }

    // Warp reduce (使用 __shfl_down_sync)
    float total = warpReduceSum(local_sum);

    if (lane_id == 0) C[j] = total;
}
```

---

## 性能分析

### 带宽分析

```
理论最小读取时间 (假设 200 GB/s 带宽):
- float32 格式: 36.7 MB / 200 GB/s = 0.18 ms
- packed uint8:  4.72 MB / 200 GB/s = 0.024 ms

实测:
- Packed V4: 0.36 ms → 实际带宽 ~16 GB/s
- 带宽利用率: 16 / 200 = 8%
```

### 为什么带宽利用率低？

1. **小规模 GEMV**: M=1 意味着计算量小，无法充分利用 GPU
2. **Kernel launch overhead**: 每次调用的固定开销
3. **Memory access pattern**: 每个线程访问不连续内存

### 进一步优化空间

1. **Persistent Kernel**: 保持 kernel 常驻，避免 launch overhead
2. **Batched GEMV**: 合并多层的 GEMV 调用
3. **更激进的向量化**: 使用 uint128 / float4

---

## 下一步行动

### 短期 (1-2 天)

1. **集成 Packed Kernel 到 TRT Plugin**
   - 位置: `openpi/tvm_trt_plugin/nvfp4_gemm/`
   - 任务: 创建 IPluginV3 包装器

2. **端到端验证**
   - 测试精度: cosine similarity > 0.999
   - 测试稳定性: 1000 次推理无异常

### 中期 (1 周)

3. **集成到 Pi0 推理流程**
   - 替换当前的 MLP 层
   - 测量整体 Hz 提升

4. **预处理权重 Packing**
   - 离线将权重 pack 为 uint8 格式
   - 避免运行时 packing 开销

### 长期

5. **扩展到其他量化方案**
   - W4A8: 权重 packed FP4, 激活 FP8
   - W4A16: 权重 packed FP4, 激活 FP16/BF16

---

## 代码位置

### 新创建的文件

```
openpi/tvm_trt_plugin/nvfp4_gemm/
├── nvfp4_gemm_packed.cu      # ✅ Packed FP4 kernels (比 FP8 快 1.46x)
├── nvfp4_gemm_optimized.cu   # Float32 优化 kernels (验证)
├── nvfp4_gemm_kernel.cu      # TVM 生成的 naive kernel

openpi/src/openpi/models_pytorch/tvm_kernels/
├── diagnose_bottleneck.py    # 瓶颈诊断脚本
├── nvfp4_gemm_thor_optimized.py  # TVM TensorIR 优化尝试
```

### 运行命令

```bash
# 编译并测试 packed kernel
cd openpi/tvm_trt_plugin/nvfp4_gemm
nvcc -O3 -arch=sm_110 nvfp4_gemm_packed.cu -o test_packed
./test_packed 3072 3072 50 200

# 运行 TVM 诊断
source /path/to/tvm/venv/bin/activate
python openpi/src/openpi/models_pytorch/tvm_kernels/diagnose_bottleneck.py
```

---

## 结论

✅ **目标达成**: nvFP4 Packed kernel 超越 TRT FP8 **1.46x**

关键 insight:
1. **数据格式是决定性因素** - 使用真正的 4-bit packed 格式
2. **TVM 的限制** - TensorIR 难以生成高效的 packed 格式处理
3. **手写 CUDA 更可控** - 对于这种特殊场景

下一步: 集成到 TRT Plugin，端到端验证后部署到 Pi0 推理流程。
