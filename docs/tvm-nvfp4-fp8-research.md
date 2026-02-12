# TVM 原生 nvFP4/FP8 量化研究

**日期**: 2026-02-10
**目标**: 使用原生 TVM 解决之前的限制，突破 FP8 的 12 Hz 性能

## 1. 背景：之前验证的结果

### 精度验证 (已完成，都 OK)
| 方案 | 精度 (Cosine Sim) | 状态 |
|------|------------------|------|
| W4A4 (nvFP4 × nvFP4) | 0.9998 | ✅ |
| W4A8 (nvFP4 × FP8) | 0.9997 | ✅ |
| W4A16 (nvFP4 × BF16) | 0.9997 | ✅ |

### 之前的问题 (需要 TVM 解决)
| 问题 | 描述 | TVM 解决方案 |
|------|------|-------------|
| W4A4 激活量化慢 | Python 7.6ms/layer | TVM 生成优化的量化 kernel |
| W4A8 硬件限制 | CUTLASS mxf8f6f4 指令不支持 SM110 | TVM 软件实现绕过限制 |
| W4A16 缺少 dequant | 需要 nvFP4→BF16 反量化 kernel | TVM 生成高效 dequant kernel |

## 2. TVM 原生支持调研

### 2.1 数据类型支持

**FP8 支持** (从 TVM v0.13.0 开始):
- `float8_e4m3` - 4位指数，3位尾数，max=448
- `float8_e5m2` - 5位指数，2位尾数，max=57344
- 支持 SM 89+ (Ada/Hopper/Blackwell)
- 来源: [PR #14863](https://github.com/apache/tvm/pull/14863)

**FP4 支持**:
- `Float4_e2m1fn` - 2位指数，1位尾数
- 这正是 nvFP4 的 E2M1 格式！
- 支持 SM 80+ (Ampere+)
- 来源: [data_type.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/data_type.h)

### 2.2 GPU 架构支持

TVM `codegen_cuda.cc` 中的架构检查:
```cpp
// SM 53+: FP16
"#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)"

// SM 61+: INT8
"#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)"

// SM 80+: BF16, FP4
"#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)"

// SM 89+: FP8
"#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)"

// SM 100+: FP6 (Blackwell)
"#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)"
```

**结论**: TVM 支持 SM110 (Thor) 通过 `__CUDA_ARCH__ >= 1000` 条件！

### 2.3 量化算子

TVM Relax 提供的量化算子:
- `relax.quantize(data, scale, zero_point, axis, out_dtype)`
- `relax.dequantize(data, scale, zero_point, axis, out_dtype)`

**当前限制**:
- 默认只支持 per-channel 量化 (通过 `axis` 参数)
- Block scaling (32-element groups) 需要自定义 TensorIR 实现

## 3. TVM 解决方案设计

### 3.1 W4A4 解决方案：优化的激活量化 Kernel

**问题**: Python 量化 7.6ms/layer，需要 <1ms

**TVM 方案**:
```python
# TensorIR 实现 block-scaled nvFP4 量化
@T.prim_func
def quantize_nvfp4_block(
    x: T.Buffer((M, K), "bfloat16"),
    q: T.Buffer((M, K), "float4_e2m1fn"),
    scales: T.Buffer((M, K // 32), "float8_e4m3"),
):
    for i, j in T.grid(M, K // 32):
        # 计算 block max
        block_max = T.float32(0)
        for k in range(32):
            block_max = T.max(block_max, T.abs(x[i, j * 32 + k]))

        # 计算 scale (FP8)
        scale = block_max / 6.0  # nvFP4 max = 6
        scales[i, j] = T.cast(scale, "float8_e4m3")

        # 量化每个元素
        for k in range(32):
            scaled = x[i, j * 32 + k] / scale
            q[i, j * 32 + k] = T.cast(T.round(scaled), "float4_e2m1fn")
```

**优化策略**:
1. 使用 TVM auto-scheduler 优化 kernel
2. 利用 shared memory 缓存 block 数据
3. 向量化 load/store 操作
4. 并行化 block 处理

### 3.2 W4A8 解决方案：软件实现绕过硬件限制

**问题**: CUTLASS `mxf8f6f4` 指令在 SM110 不支持

**TVM 方案**: 不依赖特定 PTX 指令，用 TensorIR 实现混合精度 GEMM
```python
@T.prim_func
def gemm_w4a8(
    A: T.Buffer((M, K), "float8_e4m3"),      # Activation (FP8)
    B: T.Buffer((N, K), "float4_e2m1fn"),    # Weight (nvFP4)
    scale_A: T.Buffer((M, K // 32), "float8_e4m3"),
    scale_B: T.Buffer((N, K // 32), "float8_e4m3"),
    C: T.Buffer((M, N), "bfloat16"),
):
    # TVM 会自动生成兼容 SM110 的 CUDA 代码
    # 不依赖特定的 tensor core 指令
    for i, j, k in T.grid(M, N, K):
        # 反量化并累加
        a_val = T.cast(A[i, k], "float32") * scale_A[i, k // 32]
        b_val = T.cast(B[j, k], "float32") * scale_B[j, k // 32]
        C[i, j] += a_val * b_val
```

**优化策略**:
1. TVM 可以选择使用或不使用 Tensor Core
2. 对于 SM110，使用 CUDA Core 计算也能获得不错性能
3. 利用 TVM 的 auto-tuning 找到最优 schedule

### 3.3 W4A16 解决方案：高效 Dequant Kernel

**问题**: 需要 nvFP4→BF16 反量化 kernel

**TVM 方案**:
```python
@T.prim_func
def dequant_nvfp4_to_bf16(
    q: T.Buffer((N, K), "float4_e2m1fn"),
    scales: T.Buffer((N, K // 32), "float8_e4m3"),
    out: T.Buffer((N, K), "bfloat16"),
):
    for i, j in T.grid(N, K):
        scale = T.cast(scales[i, j // 32], "float32")
        val = T.cast(q[i, j], "float32") * scale
        out[i, j] = T.cast(val, "bfloat16")
```

**然后使用标准 BF16 GEMM**:
```python
# 反量化后的权重 + BF16 激活 = cuBLAS GEMM
C = matmul(A_bf16, dequant(W_nvfp4, scales))
```

## 4. 实现计划

### Phase 1: 环境搭建 (Day 1)
- [ ] 安装 Apache TVM (latest main branch)
- [ ] 验证 TVM 对 SM110 的支持
- [ ] 测试 FP4/FP8 数据类型

### Phase 2: W4A4 实现 (Day 2-3)
- [ ] 实现 block-scaled nvFP4 量化 kernel
- [ ] 使用 TVM auto-scheduler 优化
- [ ] Benchmark: 目标 <1ms/layer (vs Python 7.6ms)
- [ ] 集成到 Pi0 推理流程

### Phase 3: W4A8 实现 (Day 4-5)
- [ ] 实现软件 W4A8 GEMM (不依赖 mxf8f6f4)
- [ ] 优化 kernel schedule
- [ ] Benchmark 对比 CUTLASS W4A4

### Phase 4: W4A16 实现 (Day 6)
- [ ] 实现 nvFP4→BF16 dequant kernel
- [ ] 集成 cuBLAS GEMM
- [ ] Benchmark 对比其他方案

### Phase 5: 综合评估 (Day 7)
- [ ] 对比 W4A4, W4A8, W4A16 性能
- [ ] 选择最优方案
- [ ] 对比 FP8 baseline (12 Hz)

## 5. 预期性能

| 方案 | 预期性能 | 加速比 (vs FP8) | 备注 |
|------|---------|----------------|------|
| W4A4 | 16-18 Hz | 1.3-1.5x | 权重带宽减半 + 激活量化 |
| W4A8 | 14-16 Hz | 1.2-1.3x | 软件实现有开销 |
| W4A16 | 13-14 Hz | 1.1-1.2x | dequant 有开销 |

**目标**: 突破 FP8 的 12 Hz，达到 14-18 Hz

## 6. MLC-LLM 现有实现分析

### 6.1 Block-Scale FP8 量化 (已有)

MLC-LLM 在 `mlc_llm/quantization/block_scale_quantization.py` 已实现:

```python
# 核心: rowwise_group_quant_fp8 函数
# 支持后端: CUTLASS + Triton
def rowwise_group_quant_fp8(x, group_size, dtype, transpose_scale):
    # 1. 计算每个 group 的 max_abs
    # 2. 计算 scale = max_abs / fp8_max (448 for e4m3)
    # 3. 量化: x_quantized = clamp(x / scale, -fp8_max, fp8_max)
```

### 6.2 CUTLASS nvFP4 Examples (MLC-LLM 3rdparty)

| Example | 架构 | 类型 | 说明 |
|---------|-----|------|------|
| 72a/72b | SM100 | W4A4 | Datacenter Blackwell |
| 79a/79b | SM120 | W4A4 | GeForce RTX 50 |

**SM110 适配**: 修改 `ArchTag = Sm110`，Cluster shape = 1x1x1

### 6.3 HuggingFace MXFP4 实现

`transformers/integrations/mxfp4.py` 包含完整的 MXFP4 量化:
- `FP4_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, ...]` (E2M1 格式)
- `quantize_to_mxfp4()` - Triton kernel 量化
- `swizzle_mxfp4()` - Scale factor 布局转换

## 7. Thor SM110 特殊考虑

| 特性 | SM100 | SM110 (Thor) | SM120 |
|------|-------|--------------|-------|
| TMA Multicast | ✅ | ❌ | ❌ |
| 2SM Cooperation | ✅ | ❌ | ❌ |
| Block-Scale MMA | ✅ | ✅ | ✅ |

**TVM 优势**: 生成标准 CUDA 代码，不依赖特定 PTX 指令

## 8. 实现路径总结

### 方案 A: 基于 MLC-LLM 扩展 (推荐)
1. 复用 `block_scale_quantization.py` 框架
2. 添加 `float4_e2m1fn` 数据类型支持
3. 复用 CUTLASS/Triton 后端

### 方案 B: 纯 TVM TensorIR
1. 用 `@T.prim_func` 编写量化/GEMM kernel
2. 使用 auto-scheduler 优化
3. 完全不依赖外部库

### 方案 C: HuggingFace MXFP4 移植
1. 移植 `mxfp4.py` 的 Triton kernel
2. 适配 Pi0 模型结构
3. 验证 Thor SM110 兼容性

## 9. 实现进展 (2026-02-10)

### 9.1 TVM TensorIR Kernel 实现 ✅

已创建完整的 TVM TensorIR 实现：

```
openpi/src/openpi/models_pytorch/tvm_kernels/
├── __init__.py
├── nvfp4_quantize.py   # W4A4 激活量化 kernel
├── nvfp4_gemm.py       # W4A4 GEMM kernel
├── w4a8_gemm.py        # W4A8 GEMM (绕过 mxf8f6f4)
├── w4a16_dequant.py    # W4A16 反量化 kernel
└── benchmark_all.py    # 综合 benchmark
```

### 9.2 TVM 环境验证

**TVM 0.24.dev0 验证结果**:
- ✅ FP8 E4M3/E5M2 数据类型支持
- ✅ FP4 E2M1fn (nvFP4) 数据类型支持
- ✅ SM110 Target 创建成功
- ✅ CUDA Kernel 编译成功 (`tvm.build()`)
- ❌ 运行时 segfault (TVM 0.24.dev0 开发版 bug)

### 9.3 环境问题

**问题 1: TVM 0.24.dev0 CUDA Runtime Bug**
```
RuntimeError: Assert fail: not T.isnullptr(var_A)
# 或
Segmentation fault in runtime.Tensor()
```
- 原因: TVM 0.24.dev0 的 CUDA tensor 创建有 bug
- 解决方案: 等待 TVM 稳定版本，或使用 TVM + PyTorch DLPack 接口

**问题 2: openpi venv PyTorch 缺少库**
```
ImportError: libucc.so.1: cannot open shared object file
ImportError: libcusparseLt.so.0: cannot open shared object file
```
- 原因: NVIDIA PyTorch (nv25.9) 需要额外的 NVIDIA 库
- 库已存在位置:
  - `/home/heima-thor/suliang/Turbo-Pi/openpi/.venv/lib/python3.12/site-packages/nvidia/*/lib/`
- 缺少: `libucc.so.1` (NVIDIA Unified Collective Communication)

### 9.4 解决方案

**方案 A: 修复 TVM 运行时 (推荐)**
1. 使用 PyTorch + TVM DLPack 接口
2. 或等待 TVM 稳定版本

**方案 B: 使用 Triton 替代 (可行)**
- Triton 3.5.1 在当前环境可用
- 现有 `nvfp4_triton.py` 已实现量化 kernel
- 可继续使用 Triton 实现 GEMM

**方案 C: 修复 PyTorch 环境**
```bash
# 安装缺少的 NVIDIA 库
pip install nvidia-ucc  # 如果可用
# 或设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/nvidia/libs:$LD_LIBRARY_PATH
```

### 9.5 下一步

1. **短期**: 使用 Triton 验证 nvFP4 性能
2. **中期**: 修复 TVM 环境或等待稳定版
3. **长期**: 完整 TVM 集成

## 10. 参考资料

- [TVM FP8 PR #14863](https://github.com/apache/tvm/pull/14863)
- [TVM data_type.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/data_type.h)
- [MLC-LLM block_scale_quantization.py](Robot-llm/mlc-llm_tvm/mlc-llm/python/mlc_llm/quantization/)
- [HuggingFace MXFP4](transformers/integrations/mxfp4.py)
- [CUTLASS Example 79](cutlass/examples/79_blackwell_geforce_gemm/)

## 11. TVM Benchmark 结果 (2026-02-10)

### 11.1 环境修复

成功在 Docker 容器中运行 TVM + PyTorch:
```bash
docker run --rm --runtime=nvidia -v /tmp/tvm_test:/test nvcr.io/nvidia/pytorch:25.12-py3 ...
```

**关键发现**:
- TVM 0.24.dev0 中 `T.serial()` 循环内的局部变量累加器无法正常工作
- 解决方案: 直接累加到输出 buffer 而非使用局部变量

### 11.2 Benchmark 结果

| Kernel | N=3072 (ms) | N=12288 (ms) | vs FP8 cuBLAS |
|--------|-------------|--------------|---------------|
| **FP8 Baseline (cuBLAS)** | 0.0206 | 0.4752 | 1.00x |
| W4A4 (TVM) | 1.0077 | 4.2232 | 0.02x |
| W4A8 (TVM) | 0.9910 | 4.2294 | 0.02x |
| W4A16 Hybrid (dequant+cuBLAS) | 0.6357 | 2.9389 | 0.03x |
| W4A16 Fused (TVM) | 1.3092 | 4.1953 | 0.02x |

### 11.3 推理速率估算

| 方案 | Per Layer | 24 Layers | Est. Hz |
|------|-----------|-----------|---------|
| FP8 cuBLAS | 0.52 ms | 12.39 ms | 80.7 Hz |
| W4A4 TVM | 6.24 ms | 149.73 ms | 6.7 Hz |
| W4A8 TVM | 6.21 ms | 149.07 ms | 6.7 Hz |
| W4A16 Fused TVM | 6.81 ms | 163.53 ms | 6.1 Hz |

### 11.4 分析

**为什么 TVM kernel 比 cuBLAS 慢 ~50x?**

当前 TVM kernel 是朴素实现:
```python
for each output element:
    C[i,j] = 0
    for k in T.serial(K):
        C[i,j] += A[i,k] * scale_A * B[j,k] * scale_B
```

缺少关键优化:
1. **无共享内存 tiling** - 每个线程独立从全局内存读取
2. **无 Tensor Cores** - 只使用标量 FP32 运算
3. **频繁全局内存写入** - 累加到全局内存而非寄存器

**cuBLAS 的优势**:
- 高度优化的 tiled GEMM
- 使用 Tensor Cores (SM 70+)
- 共享内存 + 寄存器缓存
- 向量化访存

### 11.5 关键发现

1. **FP8 cuBLAS 已经很快** - 80 Hz 远超 12 Hz 目标
2. **12 Hz 瓶颈可能在其他地方** - 非 GEMM 开销 (attention, normalization, etc.)
3. **朴素 TVM kernel 无法与 cuBLAS 竞争** - 需要高级优化

### 11.6 下一步建议

**短期方案** (推荐):
- 使用现有 FP8 cuBLAS，已经足够快
- 优化非 GEMM 开销

**中期方案**:
- 使用 TVM auto-scheduler 自动优化
- 或实现 tiled GEMM with shared memory

**长期方案**:
- 等待 NVIDIA 发布 SM110 的 mxf8f6f4 支持
- 或 CUTLASS 更新以支持 Thor

### 11.7 代码位置

TVM kernel 文件:
- `openpi/src/openpi/models_pytorch/tvm_kernels/nvfp4_gemm.py` - W4A4 GEMM
- `openpi/src/openpi/models_pytorch/tvm_kernels/w4a8_gemm.py` - W4A8 GEMM
- `openpi/src/openpi/models_pytorch/tvm_kernels/w4a16_dequant.py` - W4A16 dequant + fused GEMM
- `openpi/src/openpi/models_pytorch/tvm_kernels/benchmark_all.py` - 综合 benchmark

## 12. TVM → TensorRT Plugin 工作流 (2026-02-10)

### 12.1 突破：CUDA Source Export

发现了 TVM 导出 CUDA 源代码的方法：

```python
import tvm
from tvm.script import tir as T

# 定义 TensorIR kernel
@T.prim_func
def nvfp4_gemm(A, W, scale_A, scale_W, C):
    ...

# 编译
mod = tvm.build(nvfp4_gemm, target="cuda -arch=sm_110")

# 关键：从 imported modules 获取 CUDA 源代码
cuda_source = mod.imports_[0].inspect_source()
```

### 12.2 生成的 CUDA Kernel

TVM 生成的纯 CUDA kernel（无 CUTLASS/Tensor Core 特殊指令）：

```cuda
extern "C" __global__ void __launch_bounds__(256) nvfp4_gemm_kernel(
    float* __restrict__ A, float* __restrict__ C, float* __restrict__ W,
    float* __restrict__ scale_A, float* __restrict__ scale_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    C[idx] = 0.0f;
    for (int k = 0; k < K; ++k) {
        int block_idx = k / 32;
        float a_val = A[k] * scale_A[block_idx];
        float w_val = W[idx * K + k] * scale_W[idx * (K/32) + block_idx];
        C[idx] += a_val * w_val;
    }
}
```

### 12.3 TensorRT Plugin 生成

创建了完整的 TVM → TensorRT Plugin 工作流：

```
openpi/tvm_trt_plugin/
├── README.md                       # 使用说明
└── nvfp4_gemm/
    ├── CMakeLists.txt              # 构建配置
    ├── nvfp4_gemm_kernel.cu        # TVM 生成的 CUDA kernel
    ├── nvfp4_gemm_tvm_plugin.h     # TensorRT Plugin 头文件
    ├── nvfp4_tvm_plugin.cpp        # TensorRT IPluginV3 实现
    ├── nvfp4_gemm_launcher.cu      # Kernel 启动器
    └── test_nvfp4_tvm.cu           # 正确性/性能测试
```

**工具脚本**：
- `openpi/src/openpi/models_pytorch/tvm_kernels/tvm_to_trt_plugin.py`
  - 自动从 TVM TensorIR 生成 CUDA + TensorRT Plugin 代码

### 12.4 优势

| 对比 | CUTLASS | TVM → TRT Plugin |
|------|---------|-----------------|
| SM110 兼容性 | ❌ mxf8f6f4 问题 | ✅ 纯 CUDA |
| 构建复杂度 | 高 (模板元编程) | ✅ 简单 CMake |
| 调试 | 困难 | ✅ 清晰 CUDA 源码 |
| 自动调优 | 手动 | ✅ Auto-scheduler |
| TensorRT 集成 | 需要手动包装 | ✅ 自动生成 |

### 12.5 当前状态

- ✅ TVM CUDA source export 验证成功
- ✅ TensorRT Plugin 框架创建
- ✅ 朴素 kernel 可用
- ⏳ 待优化：
  - TVM auto-scheduler
  - Shared memory tiling
  - 向量化内存访问

### 12.6 使用方法

```bash
# 1. 生成 kernel
python tvm_to_trt_plugin.py --kernel nvfp4_gemm --N 3072 --K 3072

# 2. 构建 plugin
cd openpi/tvm_trt_plugin/nvfp4_gemm/build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=110 -DBUILD_TESTS=ON
make

# 3. 测试
./test_nvfp4_tvm 1 3072 3072

# 4. 在 TensorRT 中使用
import ctypes
ctypes.CDLL("libnvfp4_tvm_plugin.so")
# ... 创建 plugin 并添加到 network
```

## 13. 优化计划 (进行中)

### 13.1 TVM Auto-Scheduler

使用 TVM 的自动调优功能找到最优 schedule：

```python
from tvm import auto_scheduler

# 定义搜索空间
task = auto_scheduler.SearchTask(
    func=nvfp4_gemm,
    target="cuda -arch=sm_110",
)

# 自动搜索
tuner = auto_scheduler.TaskScheduler([task])
tuner.tune(auto_scheduler.TuningOptions(
    num_measure_trials=1000,
    measure_callbacks=[auto_scheduler.RecordToFile("nvfp4_gemm.json")],
))
```

### 13.2 Shared Memory Tiling

手动实现 tiled GEMM：

```python
@T.prim_func
def nvfp4_gemm_tiled(A, W, scale_A, scale_W, C):
    TILE_M, TILE_N, TILE_K = 16, 16, 32

    # Shared memory buffers
    A_shared = T.alloc_buffer((TILE_M, TILE_K), "float32", scope="shared")
    W_shared = T.alloc_buffer((TILE_N, TILE_K), "float32", scope="shared")

    for bx, by in T.thread_binding(...):
        # 协作加载 tiles 到 shared memory
        for k_tile in T.serial(K // TILE_K):
            # Load A tile
            # Load W tile
            T.tvm_storage_sync("shared")

            # 计算 partial sum
            for k in T.serial(TILE_K):
                ...

            T.tvm_storage_sync("shared")
```

### 13.3 向量化内存访问

使用 float4 向量化加载：

```python
# 向量化 load (每次加载 4 个 float)
for k in T.vectorized(4):
    A_local[k] = A[i, k_base + k]
```

### 13.4 预期优化效果

| 优化 | 预期加速比 |
|------|-----------|
| 朴素实现 | 1x (baseline) |
| Auto-scheduler | 5-10x |
| + Shared memory tiling | 10-20x |
| + 向量化访问 | 15-30x |
| 目标 | 接近 cuBLAS (50x) |

## 14. Benchmark 结果 (2026-02-10 最新)

### 14.1 Thor SM110 cuBLAS Baseline

| Kernel | Time (ms) | TFLOPS | vs FP32 |
|--------|-----------|--------|---------|
| **cuBLAS FP8** | 0.0142 | 1.32 | **11.5x** |
| cuBLAS BF16 | 0.0193 | 0.98 | 8.5x |
| cuBLAS FP32 | 0.1635 | 0.12 | 1.0x |
| cuBLAS FP32 + dequant | 0.6362 | 0.03 | 0.26x |
| TVM Naive (estimated) | 1.0000 | 0.02 | 0.16x |

**矩阵尺寸**: M=1, N=3072, K=3072 (Pi0 单 token 推理)

### 14.2 关键发现

1. **FP8 cuBLAS 极快**: 0.014 ms，比 FP32 快 11.5x
   - 对于 GEMM，FP8 已经是最优选择
   - 不需要 nvFP4 来加速 GEMM

2. **Dequant 开销巨大**: 0.47 ms (3x GEMM 时间!)
   - `A_dq = A * scale` 是瓶颈
   - 即使 GEMM 快，dequant 也会拖慢整体

3. **TVM 朴素 kernel 无法竞争**: ~1.0 ms
   - 比 cuBLAS FP32 还慢 6x
   - 需要高度优化才能接近 cuBLAS

### 14.3 nvFP4 的真正价值

nvFP4 不是为了加速 GEMM，而是为了：

1. **减少内存带宽**: 4-bit vs 16-bit = 4x 带宽节省
2. **减少显存占用**: 模型可以放入更小的 GPU
3. **预量化权重**: 避免运行时量化开销

**正确使用方式**:
```
权重: 离线量化为 nvFP4 (4-bit packed)
推理时:
  1. 加载 4-bit 权重 (带宽节省)
  2. Fused dequant + GEMM (避免分离开销)
  3. 输出 BF16/FP32
```

### 14.4 TVM Kernel 优化状态

已生成的优化 kernel:

| Kernel | 文件 | 优化 |
|--------|------|------|
| Naive | `nvfp4_gemm.py` | 基础实现 |
| Unroll 8x | `nvfp4_gemm_optimized.py` | 循环展开 |
| Vectorized | `nvfp4_gemm_optimized.py` | 4 元素批量处理 |

**代码位置**:
- `openpi/src/openpi/models_pytorch/tvm_kernels/nvfp4_gemm_optimized.py`
- `openpi/src/openpi/models_pytorch/tvm_kernels/benchmark_optimized.py`
- `openpi/tvm_trt_plugin/nvfp4_gemm/` (TensorRT Plugin)

### 14.5 结论与建议

**短期** (推荐):
- 继续使用 FP8 cuBLAS，已经非常快 (0.014 ms)
- 12 Hz 瓶颈不在 GEMM，在其他地方 (attention, data transfer, Python overhead)

**中期**:
- 如需 nvFP4，使用 TVM Fused kernel (dequant + GEMM)
- TensorRT Plugin 已准备好，待测试

**长期**:
- 等待 NVIDIA 官方 nvFP4 support for Thor SM110
- 或 CUTLASS 更新解决兼容性问题
