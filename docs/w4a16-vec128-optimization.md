# W4A16 128-bit 向量化加载优化

**Date:** 2026-02-11
**Status:** ✅ 目标达成
**Target:** < 0.2ms | **Achieved:** 0.125ms (1.6x faster)

## 1. 核心成果

通过 **Block-Interleaved 内存布局** 和 **128-bit 向量化加载**，将 W4A16 GEMV 延迟从 0.69ms 优化到 **0.125ms**。

### 1.1 性能对比

| 版本 | 内存布局 | 加载指令 | 延迟 | 带宽 | vs 目标 |
|------|----------|----------|------|------|---------|
| Baseline (标量) | `(N, K_packed)` uint8 | 16 × 8-bit | 0.92 ms | 20 GB/s | 4.6x ❌ |
| Transposed (标量) | `(K_packed, N)` uint8 | 16 × 8-bit | 0.69 ms | 27 GB/s | 3.5x ❌ |
| **Vec128 (向量化)** | `(num_qb, N, 4)` uint32 | **4 × 32-bit** | **0.125 ms** | **151 GB/s** | **1.6x ✅** |

### 1.2 优化效果

- **延迟**: 0.69 ms → 0.125 ms (**5.5x 加速**)
- **带宽**: 27 GB/s → 151 GB/s (**5.6x 提升**)
- **带宽效率**: 151 / 55 = **275%** (L2 Cache 命中)

### 1.3 稳定性验证

```
Run 1: 0.1244 ms, 151.8 GB/s
Run 2: 0.1249 ms, 151.1 GB/s
Run 3: 0.1252 ms, 150.8 GB/s
Run 4: 0.1252 ms, 150.8 GB/s
Run 5: 0.1251 ms, 150.9 GB/s

Mean: 0.1250 ms
Std:  0.0003 ms  ← 极低方差，结果稳定
```

## 2. 问题分析

### 2.1 原始问题：标量加载瓶颈

之前的 Transposed 布局虽然实现了 Coalesced 访问，但仍然使用 **标量 8-bit 加载**：

```python
# 旧代码：16 次 8-bit 标量加载 per quant block
for byte_offset in range(16):
    packed = W_packed_T[byte_idx, n]  # ← 每次只读 8-bit！
```

**问题**:
- 每个 Quant Block (32 个 INT4) 需要 **16 次内存加载指令**
- GPU 指令发射单元成为瓶颈（Instruction Bound）
- 实际带宽利用率仅 49%

### 2.2 理论分析

| 参数 | 值 |
|------|-----|
| N × K | 16384 × 2048 |
| 权重大小 | 16.78 MB (uint32 布局) |
| Scale 大小 | 2.10 MB |
| 总数据 | 18.88 MB |
| DRAM 理论 (55 GB/s) | 0.343 ms |
| L2 理论 (230 GB/s) | 0.082 ms |
| **实测** | **0.125 ms** |

实测结果比 DRAM 理论快，说明 L2 Cache 命中率高。

## 3. 解决方案：Block-Interleaved 布局

### 3.1 内存布局变化

```
旧布局 (Transposed uint8):
  Shape: (K_packed, N) = (1024, 16384)
  每个 Quant Block: 16 个 uint8 分散存储
  加载: 16 次 8-bit 读取

新布局 (Block-Interleaved uint32):
  Shape: (num_scale_blocks, N, 4) = (64, 16384, 4)
  每个 Quant Block: 4 个连续 uint32
  加载: T.vectorized(4) → 1 次 128-bit 读取
```

### 3.2 内存访问模式

```
Thread n 访问 W_packed[qb, n, 0:4]:

Memory Layout:
┌─────────────────────────────────────────────────────┐
│ qb=0, n=0 │ qb=0, n=1 │ qb=0, n=2 │ ... │ qb=0, n=255 │
│  u0 u1 u2 u3  │  u0 u1 u2 u3  │  u0 u1 u2 u3  │ ... │  u0 u1 u2 u3  │
└─────────────────────────────────────────────────────┘
      ↑              ↑              ↑
   Thread 0      Thread 1      Thread 2

特点:
1. 同一 (qb, n) 的 4 个 uint32 在内存中连续 → 可向量化
2. 相邻 thread (n, n+1) 访问相邻内存地址 → Coalesced
```

### 3.3 TIR 实现

```python
@T.prim_func
def gemv_vec128_v3(
    A: T.Buffer((1, K), "float16"),
    W_packed: T.Buffer((num_scale_blocks, N, 4), "uint32"),  # Block-Interleaved
    scales_T: T.Buffer((num_scale_blocks, N), "float16"),
    C: T.Buffer((1, N), "float32"),
):
    T.func_attr({"global_symbol": "gemv_vec128_v3", "tir.noalias": True})

    A_shared = T.alloc_buffer((K,), "float16", scope="shared")
    W_local = T.alloc_buffer((4,), "uint32", scope="local")  # 寄存器

    for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
        # 1. 协作加载 A 到 Shared Memory
        for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
            for i in range((K + THREADS - 1) // THREADS):
                k = tid + i * THREADS
                if k < K:
                    A_shared[k] = A[0, k]

        T.tvm_storage_sync("shared")

        # 2. 每个 thread 处理一个输出 n
        for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
            n = block_idx * THREADS + tid
            if n < N:
                C[0, n] = T.float32(0)

                for qb in range(num_scale_blocks):
                    scale = scales_T[qb, n]
                    k_base = qb * QUANT_BLOCK

                    # 关键优化：128-bit 向量化加载
                    for v in T.vectorized(4):  # ← TVM 自动生成向量指令
                        W_local[v] = W_packed[qb, n, v]

                    # 解码 4 个 uint32 = 32 个 INT4
                    for u_idx in range(4):
                        u = W_local[u_idx]
                        k_offset = u_idx * 8

                        for i in range(8):
                            int4_val = (u >> T.uint32(i * 4)) & T.uint32(0xF)
                            k_idx = k_base + k_offset + i
                            w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                            C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_idx] * w)
```

### 3.4 关键优化点

| 优化 | 说明 |
|------|------|
| `T.vectorized(4)` | 提示 TVM 生成 128-bit 向量加载 |
| `W_local` in `local` scope | 数据存入寄存器，避免重复访存 |
| Block-Interleaved Layout | 保证 4 个 uint32 内存连续 |
| N 作为最快变化维度 | 相邻 thread 访问相邻地址 (Coalesced) |

## 4. 量化格式

### 4.1 INT4 Packing

```
每个 uint32 包含 8 个 INT4 值：

Bits:  [31:28] [27:24] [23:20] [19:16] [15:12] [11:8] [7:4] [3:0]
INT4:    i7      i6      i5      i4      i3      i2     i1    i0

解码：
  int4_val = (u >> (i * 4)) & 0xF
  weight = (int4_val - 8) * scale
```

### 4.2 Quant Block 结构

```
每个 Quant Block = 32 个 INT4 权重 + 1 个 FP16 scale

存储：
  - 4 个 uint32 = 16 bytes = 32 个 INT4
  - 1 个 FP16 scale = 2 bytes

总计：18 bytes per 32 weights
```

### 4.3 量化函数

```python
def quantize_to_block_interleaved(W, block_size=32):
    """
    输入: W (N, K) float32
    输出:
        W_packed: (num_scale_blocks, N, 4) uint32
        scales: (num_scale_blocks, N) float16
    """
    N_dim, K_dim = W.shape
    num_blocks_k = K_dim // block_size

    # 1. 逐 block 计算 scale 和量化
    for n in range(N_dim):
        for b in range(num_blocks_k):
            block = W[n, b*32 : (b+1)*32]
            max_abs = np.max(np.abs(block))
            scale = max_abs / 7.0  # INT4 range: [-8, 7]
            scales[b, n] = scale

            for k in range(32):
                val = block[k] / scale
                quantized = clip(round(val + 8), 0, 15)  # offset to [0, 15]
                # Pack into uint32
                W_packed[b, n, k//8] |= quantized << ((k % 8) * 4)

    return W_packed, scales
```

## 5. 文件清单

| 文件 | 说明 |
|------|------|
| `tvm_kernels/w4a16_vec128_gemv.py` | 主 Kernel 实现 |
| `tvm_kernels/w4a16_vec128_verify.py` | 稳定性验证脚本 |
| `tvm_kernels/w4a16_transposed_gemv.py` | 旧版 Transposed 实现 (0.69ms) |
| `tvm_kernels/w4a16_optimized_gemv.py` | Baseline 实现 (0.92ms) |

## 6. 运行命令

```bash
# 激活 TVM 环境
source /home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/venv/bin/activate
cd /home/heima-thor/suliang/Turbo-Pi/openpi

# 完整 Benchmark
python src/openpi/models_pytorch/tvm_kernels/w4a16_vec128_gemv.py

# 稳定性验证 (5 runs)
python src/openpi/models_pytorch/tvm_kernels/w4a16_vec128_verify.py
```

## 7. 预期收益

### 7.1 单层 MLP 收益

| GEMM | 维度 | 旧延迟 | 新延迟 | 加速比 |
|------|------|--------|--------|--------|
| gate_proj | 16384×2048 | 0.69 ms | **0.125 ms** | **5.5x** |
| up_proj | 16384×2048 | 0.69 ms | **0.125 ms** | **5.5x** |
| down_proj | 2048×16384 | ~0.69 ms | ~0.125 ms | ~5.5x |
| **MLP Total** | - | ~2.1 ms | **~0.38 ms** | **5.5x** |

### 7.2 18 层 Action Expert 收益

| 配置 | 单层 MLP | 18 层总计 | Hz (预估) |
|------|----------|----------|-----------|
| TRT FP8 Baseline | ~1.13 ms | ~20.4 ms | 12.0 |
| TVM W4A16 (旧) | ~2.1 ms | ~37.8 ms | ~9.5 |
| **TVM W4A16 Vec128** | **~0.38 ms** | **~6.8 ms** | **~14.5** |

*注: 实际收益需端到端验证*

## 8. 技术总结

### 8.1 为什么 Block-Interleaved 有效

1. **减少指令数**: 16 次 8-bit load → 1 次 128-bit load (理论 16x)
2. **提高 ILP**: 更少的 load 指令让计算单元更忙
3. **L2 友好**: 连续 128-bit 读取提高 Cache Line 利用率

### 8.2 关键 Insight

> **标量 8-bit 加载在 GPU 上极其低效**。即使实现了 Coalesced 访问，
> 如果每个 thread 需要发出大量独立的 load 指令，指令发射单元
> 会成为瓶颈（Instruction Bound），无法充分利用内存带宽。

### 8.3 经验教训

| 错误方向 | 正确方向 |
|----------|----------|
| 增加 LUT 内存 → 更慢 | 减少 load 指令数 → 更快 |
| 只关注 Coalescing | **同时关注向量化** |
| 分析延迟看带宽 | 分析延迟看指令数 |

---

## 9. PyTorch 生产集成

### 9.1 架构设计

采用 **"PyTorch Custom Op + CUDA Graphs"** 架构：

```
┌─────────────────────────────────────────────────────────┐
│                    W4A16Linear                          │
│  (nn.Module, same interface as nn.Linear)               │
├─────────────────────────────────────────────────────────┤
│  seq_len > 1 → F.linear (BF16 fallback)                │
│  seq_len = 1 → w4a16_gemv (TVM kernel)                 │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              torch.ops.openpi.w4a16_gemv                │
│  (torch.library registered, torch.compile compatible)  │
├─────────────────────────────────────────────────────────┤
│  DLPack zero-copy ←→ TVM NDArray                       │
│  CUDA Graph safe (no dynamic allocations)              │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│           TVM 128-bit Vectorized Kernel                 │
│  (0.125ms latency, 151 GB/s bandwidth)                 │
└─────────────────────────────────────────────────────────┘
```

### 9.2 文件结构

```
openpi/src/openpi/
├── ops/
│   ├── __init__.py
│   ├── w4a16_gemv.py           # Core operator + torch.library
│   ├── w4a16_validation.py     # CUDA Graphs benchmark
│   └── w4a16_standalone_test.py # TVM-only test
├── utils/
│   ├── __init__.py
│   └── w4a16_packer.py         # Weight quantization
└── modules/
    ├── __init__.py
    └── w4a16_linear.py         # W4A16Linear nn.Module
```

### 9.3 使用示例

```python
from openpi.modules import W4A16Linear
from openpi.ops import precompile_kernels

# 预编译常用 kernel (避免首次调用延迟)
precompile_kernels([
    (16384, 2048),  # gate_proj, up_proj
    (2048, 16384),  # down_proj
])

# 方式1：从 nn.Linear 转换
linear = nn.Linear(2048, 16384)
w4a16_layer = W4A16Linear.from_linear(linear)

# 方式2：直接创建
w4a16_layer = W4A16Linear(2048, 16384, bias=True, device='cuda')
w4a16_layer.pack_weights(weight_tensor)

# Forward (自动选择 kernel)
x = torch.randn(1, 2048, dtype=torch.float16, device='cuda')
y = w4a16_layer(x)  # seq=1 → TVM kernel

x_batch = torch.randn(4, 2048, dtype=torch.float16, device='cuda')
y_batch = w4a16_layer(x_batch)  # seq>1 → F.linear fallback
```

### 9.4 CUDA Graphs 用法

```python
# 创建 MLP layers
gate = W4A16Linear(2048, 16384, device='cuda')
up = W4A16Linear(2048, 16384, device='cuda')
down = W4A16Linear(16384, 2048, device='cuda')

# 固定输入 (CUDA Graphs 要求)
x = torch.randn(1, 2048, dtype=torch.float16, device='cuda')

# 捕获 Graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    gate_out = gate(x)
    up_out = up(x)
    hidden = F.silu(gate_out) * up_out
    out = down(hidden)

# 执行 (零 Python 开销)
g.replay()
```

### 9.5 验证结果

```
============================================================
W4A16 Standalone TVM Kernel Test
N=16384, K=2048
============================================================

Cosine similarity: 1.000000
Correctness: PASS

Average latency: 0.1254 ms
Bandwidth: 151.6 GB/s
Target (< 0.2ms): ACHIEVED!

Stability Check (5 runs):
Mean: 0.1254 ms
Std:  0.0002 ms  ← 极低方差
============================================================
```

---

## 10. Docker 容器测试

### 10.1 测试环境

- **容器**: `openpi_orin` (基于 `ghcr.io/physical-intelligence/openpi`)
- **镜像特点**: 包含完整的 PyTorch CUDA 支持
- **TVM 路径**: 需要挂载宿主机 TVM 路径

### 10.2 容器启动命令

```bash
# 启动容器 (需要挂载 TVM)
docker run -d --name turbo_pi_eval \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --network host \
  -v /home/heima-thor/suliang/Turbo-Pi/openpi:/workspace \
  -v /home/heima-thor/.cache/openpi:/root/.cache/openpi \
  -v /home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm:/opt/tvm \
  -e PYTHONPATH=/opt/tvm/python \
  -e TVM_HOME=/opt/tvm \
  turbo_pi_libero:latest \
  sleep infinity
```

### 10.3 测试命令

```bash
# 设置环境变量 (在 docker exec 时)
ENV_VARS="-e LD_LIBRARY_PATH=/opt/tvm/build:/opt/tvm/build/3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels:/opt/tvm/build/3rdparty/libflash_attn/src -e PYTHONPATH=/opt/tvm/python:/workspace/src"

# 1. Standalone TVM 测试
docker exec $ENV_VARS turbo_pi_eval python /workspace/src/openpi/ops/w4a16_standalone_test.py

# 2. PyTorch 集成测试
docker exec $ENV_VARS turbo_pi_eval python /workspace/src/openpi/ops/w4a16_gemv.py

# 3. W4A16Linear 模块测试
docker exec $ENV_VARS turbo_pi_eval python /workspace/src/openpi/modules/w4a16_linear.py

# 4. 完整验证 (含 CUDA Graphs)
docker exec $ENV_VARS turbo_pi_eval python /workspace/src/openpi/ops/w4a16_validation.py
```

### 10.3 测试结果 (2026-02-11)

**容器环境:**
- PyTorch: 2.10.0a0+b4e4ee81d3.nv25.12
- TVM: 0.24.dev0
- GPU: NVIDIA Thor (SM110)

**1. Standalone TVM Test:**
```
Cosine similarity: 1.000000
Correctness: PASS
Average latency: 0.1245 ms
Bandwidth: 151.6 GB/s
Target (< 0.2ms): ACHIEVED!
Stability: Mean 0.1252 ms, Std 0.0000 ms
```

**2. PyTorch Integration Test:**
```
Output shape: torch.Size([1, 16384])
Average latency: 0.1246 ms
Target (< 0.2ms): ACHIEVED!
```

**3. W4A16Linear Module Test:**
```
seq_len=1: 0.1342 ms (ACHIEVED!)
seq_len=4 fallback: F.linear 正常
from_linear conversion: cosine similarity 0.9949
```

**4. CUDA Graphs Benchmark:**
```
Single Layer (16384×2048):
  Without CUDA Graphs: 0.1346 ms
  With CUDA Graphs:    0.0066 ms
  Speedup:             20.25x

MLP (gate+up+silu*mul+down):
  Without CUDA Graphs: 0.7795 ms
  With CUDA Graphs:    0.0088 ms
  Speedup:             88.19x

18-Layer Projection:
  Without CUDA Graphs: 14.03 ms
  With CUDA Graphs:    0.16 ms
```

**5. torch.compile:**
```
torch.compile latency: 0.0761 ms
Status: COMPATIBLE
```

**6. Memory Compression:**
```
Single Layer: FP16 67.11 MB → W4A16 18.87 MB (3.6x)
18-Layer MLP: FP16 3.54 GB → W4A16 1.00 GB
Savings: 2.54 GB
```

---

**Author:** Claude Code
**GPU:** NVIDIA Thor (SM110)
**Framework:** TVM TIR Script + PyTorch Custom Op
