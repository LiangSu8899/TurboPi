# Thor Triforce 优化工程计划 v2.0

## 基于带宽墙修正后的硬核加速方案

**核心修正**: KV Cache 瓶颈是 **MLP 权重的内存带宽读取**，不是 Attention 计算。FlashInfer 无法打破 22ms 的物理下限。

---

## 一、问题根因重新定位

### 1.1 带宽墙分析 (Memory Wall)

| 组件 | 权重大小 (BF16) | 占比 | 读取时间 (@200GB/s) |
|------|-----------------|------|---------------------|
| QKV + O Projection | 324 MB | 8.2% | 1.6 ms |
| **MLP (gate+up+down)** | **3.62 GB** | **91.4%** | **18.1 ms** |
| LayerNorm/RoPE | 18 MB | 0.4% | 0.1 ms |
| **Total** | **3.96 GB** | 100% | **~20 ms (理论下限)** |

### 1.2 为什么之前的方案过于乐观

| 方案 | 原假设 | 实际情况 |
|------|--------|----------|
| FlashInfer | Attention 是瓶颈 | ❌ MLP 带宽才是瓶颈 |
| Static Prompt Caching | 缓存省计算 | ❌ 省计算不省带宽 |
| TRT FP8 | FP8 减半权重 | ❌ Thor TRT scale 被忽略 |
| FP4 量化 | 1/4 权重 | ❌ Thor 不支持 (Segfault) |

### 1.3 唯一破局点

**必须把 3.62 GB 权重变小**：

| 量化方案 | 权重大小 | 读取时间 | 可行性 |
|----------|----------|----------|--------|
| BF16 | 3.62 GB | 18.1 ms | ✅ 当前 |
| FP8 | 1.81 GB | 9.0 ms | ⚠️ Thor TRT 有 bug |
| **INT4 (W4A16)** | **0.90 GB** | **4.5 ms** | 🎯 **唯一出路** |

---

## 二、修正后的目标设定

### 2.1 务实的延迟目标

| 组件 | 当前 | Plan A (保守) | Plan B (激进) |
|------|------|---------------|---------------|
| Vision TRT | 17.0 ms | 17.0 ms | 17.0 ms |
| KV Cache Prefill | 54.0 ms | **30 ms** | **12 ms** |
| Denoise (10 step) | 102.3 ms | 95 ms | 85 ms |
| Overhead | 3.2 ms | 1.0 ms | 0.5 ms |
| **Total** | **176.5 ms** | **143 ms** | **114.5 ms** |
| **Hz** | **5.7 Hz** | **7.0 Hz** | **8.7 Hz** |

### 2.2 两条路线对比

| 路线 | 核心手段 | 预期收益 | 风险 | 工作量 |
|------|----------|----------|------|--------|
| **Plan A** | FlashInfer + CUDA Graph | 5.7→7.0 Hz | 低 | 2 周 |
| **Plan B** | INT4 Triton Kernel | 5.7→8.7 Hz | 中-高 | 4 周 |

---

## 三、Phase 0: 环境验证 (Day 1-2)

**这是最关键的一步，决定后续路线**

### 3.1 验证脚本

```python
#!/usr/bin/env python3
"""
scripts/phase0_environment_check.py

验证 Thor 平台的软硬件支持情况，决定优化路线
"""

import torch
import time
import subprocess
import sys

def check_gpu_info():
    """检查 GPU 信息"""
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"Device: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"SM Count: {props.multi_processor_count}")

    # Thor 应该是 SM 10.0 或 11.0
    if props.major >= 10:
        print("✅ Blackwell architecture detected (Thor)")
    else:
        print(f"⚠️ Non-Blackwell GPU (SM {props.major}.{props.minor})")

    return True


def benchmark_memory_bandwidth():
    """
    测量实际内存带宽 - 这是最关键的数据
    模拟 MLP 的权重读取
    """
    print("\n" + "=" * 60)
    print("Memory Bandwidth Benchmark")
    print("=" * 60)

    device = "cuda"

    # 测试不同大小的 Linear 层
    configs = [
        (2048, 16384, "MLP Up/Gate (single)"),
        (16384, 2048, "MLP Down (single)"),
        (2048, 2048, "QKV/O Projection"),
    ]

    results = {}

    for in_dim, out_dim, name in configs:
        # 模拟 batch=1, seq=712
        x = torch.randn(1, 712, in_dim, device=device, dtype=torch.bfloat16)
        layer = torch.nn.Linear(in_dim, out_dim, bias=False,
                                device=device, dtype=torch.bfloat16)

        # 权重大小
        weight_bytes = in_dim * out_dim * 2  # BF16 = 2 bytes

        # Warmup
        for _ in range(10):
            _ = layer(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        num_iters = 100
        for _ in range(num_iters):
            _ = layer(x)
        torch.cuda.synchronize()

        avg_time_ms = (time.perf_counter() - start) / num_iters * 1000
        effective_bw = weight_bytes / (avg_time_ms / 1000) / 1e9  # GB/s

        results[name] = {
            "time_ms": avg_time_ms,
            "weight_mb": weight_bytes / 1e6,
            "bandwidth_gbps": effective_bw,
        }

        print(f"\n{name}:")
        print(f"  Shape: ({in_dim}, {out_dim})")
        print(f"  Weight: {weight_bytes/1e6:.2f} MB")
        print(f"  Time: {avg_time_ms:.3f} ms")
        print(f"  Effective Bandwidth: {effective_bw:.1f} GB/s")

    # 估算完整 KV Cache MLP 时间
    # 18 层 × (gate + up + down)
    mlp_time = (results["MLP Up/Gate (single)"]["time_ms"] * 2 +
                results["MLP Down (single)"]["time_ms"]) * 18

    print(f"\n" + "-" * 40)
    print(f"Estimated KV Cache MLP Time (18 layers): {mlp_time:.1f} ms")
    print(f"Theoretical Minimum (@200 GB/s): 18.1 ms")

    avg_bw = sum(r["bandwidth_gbps"] for r in results.values()) / len(results)
    print(f"\nAverage Effective Bandwidth: {avg_bw:.1f} GB/s")

    if avg_bw < 150:
        print("⚠️ Bandwidth significantly below theoretical (200 GB/s)")
        print("   Possible causes: CUDA driver, memory contention, thermal throttling")
    elif avg_bw > 180:
        print("✅ Bandwidth close to theoretical maximum")

    return results


def check_flashinfer():
    """检查 FlashInfer 是否可用"""
    print("\n" + "=" * 60)
    print("FlashInfer Check")
    print("=" * 60)

    try:
        import flashinfer
        print(f"✅ FlashInfer version: {flashinfer.__version__}")

        # 尝试简单操作
        q = torch.randn(1, 50, 8, 256, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 712, 1, 256, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 712, 1, 256, device="cuda", dtype=torch.float16)

        # 这里可能会失败，因为 Thor 可能不被支持
        try:
            # FlashInfer API 调用
            print("  Testing FlashInfer attention...")
            # out = flashinfer.single_prefill_with_kv_cache(q, k, v)
            print("  ⚠️ Need to test actual FlashInfer API on Thor")
        except Exception as e:
            print(f"  ❌ FlashInfer operation failed: {e}")

        return True
    except ImportError:
        print("❌ FlashInfer not installed")
        print("   Install: pip install flashinfer")
        return False


def check_triton():
    """检查 Triton 是否支持 Thor"""
    print("\n" + "=" * 60)
    print("Triton Check")
    print("=" * 60)

    try:
        import triton
        import triton.language as tl

        print(f"✅ Triton version: {triton.__version__}")

        # 简单的 Triton kernel 测试
        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        # 运行测试
        n = 1024
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        out = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)

        # 验证
        expected = x + y
        if torch.allclose(out, expected):
            print("✅ Triton kernel execution successful on Thor")
            return True
        else:
            print("❌ Triton kernel produced incorrect results")
            return False

    except ImportError:
        print("❌ Triton not installed")
        return False
    except Exception as e:
        print(f"❌ Triton error: {e}")
        return False


def check_int4_support():
    """检查 INT4 量化库支持"""
    print("\n" + "=" * 60)
    print("INT4 Quantization Check")
    print("=" * 60)

    libs = {
        "bitsandbytes": "INT8/FP4 quantization",
        "auto_gptq": "GPTQ INT4",
        "awq": "AWQ INT4",
    }

    available = []
    for lib, desc in libs.items():
        try:
            __import__(lib.replace("-", "_"))
            print(f"✅ {lib}: {desc}")
            available.append(lib)
        except ImportError:
            print(f"❌ {lib}: not installed")

    return available


def run_all_checks():
    """运行所有检查"""
    print("\n" + "=" * 60)
    print("THOR TRIFORCE ENVIRONMENT CHECK")
    print("=" * 60)

    results = {}

    # GPU Info
    results["gpu"] = check_gpu_info()

    # Bandwidth - 最关键的测试
    if results["gpu"]:
        results["bandwidth"] = benchmark_memory_bandwidth()

    # FlashInfer
    results["flashinfer"] = check_flashinfer()

    # Triton
    results["triton"] = check_triton()

    # INT4
    results["int4_libs"] = check_int4_support()

    # 决策建议
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if results.get("bandwidth"):
        avg_bw = sum(r["bandwidth_gbps"] for r in results["bandwidth"].values()) / len(results["bandwidth"])

        if avg_bw < 150:
            print("\n⚠️ 带宽受限严重，建议:")
            print("   1. 检查 CUDA driver 版本")
            print("   2. 检查是否有其他进程占用 GPU")
            print("   3. INT4 量化是唯一出路")

        if results.get("triton"):
            print("\n✅ Triton 可用，推荐 Plan B (INT4 Triton Kernel)")
        else:
            print("\n⚠️ Triton 不可用，只能走 Plan A (保守优化)")

    return results


if __name__ == "__main__":
    results = run_all_checks()
```

### 3.2 运行验证

```bash
# 在 Thor 上运行
docker exec turbo_pi_eval python /workspace/scripts/phase0_environment_check.py

# 保存结果
docker exec turbo_pi_eval python /workspace/scripts/phase0_environment_check.py > phase0_results.txt 2>&1
```

### 3.3 决策树

```
Phase 0 结果
    │
    ├── 带宽 > 180 GB/s?
    │   ├── Yes → FP16 还有救，尝试 FlashInfer
    │   └── No → 必须 INT4 量化
    │
    ├── Triton 可用?
    │   ├── Yes → Plan B (Triton INT4 Kernel)
    │   └── No → Plan A (保守优化) 或等待软件支持
    │
    └── FlashInfer 可用?
        ├── Yes → Attention 部分可用 FlashInfer
        └── No → 用 PyTorch + CUDA Graph
```

---

## 四、Plan A: 保守优化路线 (低风险)

**目标**: 5.7 Hz → 7.0 Hz
**预期 KV Cache**: 54 ms → 30 ms
**工作量**: 2 周

### 4.1 优化内容

| 组件 | 手段 | 预期节省 |
|------|------|----------|
| Attention | FlashInfer 或 Triton | 5-8 ms |
| Padding | 去除无效 token | 3-5 ms |
| CUDA Graph | 全图录制 | 2-3 ms |
| Kernel Fusion | torch.compile | 2-3 ms |
| **Total** | | **12-19 ms** |

### 4.2 实现步骤

#### Step 1: CUDA Graph 全图录制

```python
# src/openpi/inference/full_graph_policy.py

class FullGraphPolicy:
    """全图 CUDA Graph 录制"""

    def __init__(self, base_policy):
        self.base_policy = base_policy

        # 静态 buffer
        self.static_image = torch.zeros(1, 3, 224, 224, device="cuda", dtype=torch.float16)
        self.static_wrist = torch.zeros(1, 3, 224, 224, device="cuda", dtype=torch.float16)
        self.static_state = torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16)
        self.static_tokens = torch.zeros(1, 200, device="cuda", dtype=torch.long)

        # 捕获图
        self._capture_graph()

    def _capture_graph(self):
        """捕获完整计算图"""
        # Warmup
        for _ in range(5):
            self._forward()
        torch.cuda.synchronize()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._forward()

    def _forward(self):
        """被捕获的前向传播"""
        # Vision + KV + Denoise 全部在一个 graph 里
        self.static_output = self.base_policy._forward_impl(
            self.static_image,
            self.static_wrist,
            self.static_state,
            self.static_tokens,
        )

    def infer(self, image, wrist, state, tokens):
        # 复制输入
        self.static_image.copy_(image)
        self.static_wrist.copy_(wrist)
        self.static_state.copy_(state)
        self.static_tokens.copy_(tokens)

        # 执行
        self.graph.replay()

        return self.static_output.clone()
```

#### Step 2: Kernel Fusion (torch.compile)

```python
# 对碎片算子进行融合
import torch._dynamo as dynamo

@torch.compile(mode="reduce-overhead", fullgraph=True)
def fused_adaln(x, scale, shift):
    """融合 AdaLN: norm → scale → shift → silu"""
    normed = torch.nn.functional.layer_norm(x, x.shape[-1:])
    return (normed * (1 + scale) + shift) * torch.sigmoid(x) * x
```

### 4.3 Plan A 验证

```bash
# 验证脚本
python scripts/validate_plan_a.py \
    --checkpoint_dir /root/.cache/openpi/checkpoints/pi05_libero \
    --num_runs 100

# 预期输出:
# Baseline: 54.0 ms KV Cache
# Plan A: 35-40 ms KV Cache
# Speedup: 1.35-1.54x
```

### 4.4 Plan A 成功标准

- [ ] KV Cache: ≤ 35 ms
- [ ] Total: ≤ 150 ms
- [ ] Hz: ≥ 6.5 Hz
- [ ] LIBERO 精度: ≥ 95%

---

## 五、Plan B: 激进优化路线 (中-高风险)

**目标**: 5.7 Hz → 8.7 Hz
**预期 KV Cache**: 54 ms → 12 ms
**工作量**: 4 周
**核心**: Triton W4A16 (INT4) Kernel

### 5.1 为什么是 INT4

| 精度 | 权重大小 | 带宽时间 | Thor 支持 |
|------|----------|----------|-----------|
| BF16 | 3.62 GB | 18.1 ms | ✅ |
| FP8 | 1.81 GB | 9.0 ms | ⚠️ TRT bug |
| FP4 | 0.90 GB | 4.5 ms | ❌ Segfault |
| **INT4** | **0.90 GB** | **4.5 ms** | 🎯 **Triton 手写** |

### 5.2 INT4 量化策略

**只量化 KV Cache MLP，不动其他部分**：

| 组件 | 量化 | 原因 |
|------|------|------|
| Vision Encoder | ❌ 不量化 | CNN/ViT 对量化敏感 |
| LLM Attention | ❌ 不量化 | 计算量小，量化收益低 |
| **LLM MLP** | ✅ **INT4** | 瓶颈所在，必须量化 |
| Action Expert | ❌ 不量化 | 已经很快 |

### 5.3 Triton W4A16 Kernel

```python
# src/openpi/inference/triton_int4_linear.py

import triton
import triton.language as tl
import torch

@triton.jit
def int4_dequant_matmul_kernel(
    # Pointers
    A_ptr,           # Input: [M, K] FP16
    W_packed_ptr,    # Weight: [K, N//2] INT8 (每 byte 存 2 个 INT4)
    W_scale_ptr,     # Scale: [K//group_size, N] FP16
    W_zero_ptr,      # Zero point: [K//group_size, N] INT4
    C_ptr,           # Output: [M, N] FP16
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    W4A16 MatMul: INT4 weight, FP16 activation

    关键优化:
    1. 权重从 4.5ms 降到理论可以
    2. FP16 accumulation 保证精度
    3. Per-group scale (每 128 个元素一个 scale)
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator (FP32 for precision)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Main loop over K
    for k in range(0, K, BLOCK_K):
        # Load activation [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)

        # Load packed INT4 weights [BLOCK_K, BLOCK_N//2]
        # 每个 byte 存 2 个 INT4
        w_packed_ptrs = W_packed_ptr + (k + offs_k[:, None]) * stride_wk + (offs_n[None, :] // 2)
        w_packed = tl.load(w_packed_ptrs)

        # Dequantize INT4 → FP16
        # 低 4 位和高 4 位
        w_low = (w_packed & 0x0F).to(tl.float16)   # 0-15
        w_high = ((w_packed >> 4) & 0x0F).to(tl.float16)

        # 加载 scale 和 zero point
        group_idx = (k + offs_k[:, None]) // GROUP_SIZE
        scale_ptrs = W_scale_ptr + group_idx * N + offs_n[None, :]
        scale = tl.load(scale_ptrs)

        zero_ptrs = W_zero_ptr + group_idx * (N // 2) + (offs_n[None, :] // 2)
        zero_packed = tl.load(zero_ptrs)
        zero_low = (zero_packed & 0x0F).to(tl.float16)
        zero_high = ((zero_packed >> 4) & 0x0F).to(tl.float16)

        # Dequant: w_fp16 = (w_int4 - zero) * scale
        # 交替处理偶数列和奇数列
        w_dequant = tl.where(
            (offs_n[None, :] % 2) == 0,
            (w_low - zero_low) * scale,
            (w_high - zero_high) * scale,
        )

        # MatMul accumulate
        acc += tl.dot(a, w_dequant).to(tl.float32)

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class TritonINT4Linear(torch.nn.Module):
    """Triton INT4 Linear 层封装"""

    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Packed INT4 weights (每 byte 存 2 个 INT4)
        self.register_buffer(
            "w_packed",
            torch.zeros(in_features, out_features // 2, dtype=torch.uint8)
        )

        # Per-group scale
        num_groups = in_features // group_size
        self.register_buffer(
            "w_scale",
            torch.ones(num_groups, out_features, dtype=torch.float16)
        )

        # Per-group zero point (packed)
        self.register_buffer(
            "w_zero",
            torch.zeros(num_groups, out_features // 2, dtype=torch.uint8)
        )

    @classmethod
    def from_float(cls, linear: torch.nn.Linear, group_size: int = 128):
        """从 FP16 Linear 转换"""
        instance = cls(linear.in_features, linear.out_features, group_size)

        weight = linear.weight.data.float()  # [out, in]
        weight = weight.t()  # [in, out]

        # Per-group quantization
        K, N = weight.shape
        num_groups = K // group_size

        weight_grouped = weight.reshape(num_groups, group_size, N)

        # 计算 scale 和 zero point
        w_min = weight_grouped.min(dim=1).values  # [num_groups, N]
        w_max = weight_grouped.max(dim=1).values

        scale = (w_max - w_min) / 15.0  # INT4: 0-15
        zero = (-w_min / scale).round().clamp(0, 15)

        # 量化
        weight_int4 = ((weight_grouped - w_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
        weight_int4 = weight_int4.reshape(K, N).to(torch.uint8)

        # Pack: 2 个 INT4 → 1 个 byte
        w_packed = weight_int4[:, 0::2] | (weight_int4[:, 1::2] << 4)

        instance.w_packed.copy_(w_packed)
        instance.w_scale.copy_(scale.half())

        zero_packed = zero[:, 0::2].to(torch.uint8) | (zero[:, 1::2].to(torch.uint8) << 4)
        instance.w_zero.copy_(zero_packed)

        return instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        M = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        x_2d = x.reshape(M, self.in_features)

        out = torch.empty(M, self.out_features, device=x.device, dtype=x.dtype)

        # Grid
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 128
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(self.out_features, BLOCK_N))

        int4_dequant_matmul_kernel[grid](
            x_2d, self.w_packed, self.w_scale, self.w_zero, out,
            M, self.out_features, self.in_features,
            x_2d.stride(0), x_2d.stride(1),
            self.w_packed.stride(0), self.w_packed.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE=self.group_size,
        )

        return out.reshape(*x.shape[:-1], self.out_features)
```

### 5.4 INT4 量化流程

```python
# scripts/quantize_kv_cache_mlp.py

def quantize_kv_cache_mlp(model, group_size=128):
    """只量化 KV Cache 阶段的 MLP"""

    # 遍历 LLM backbone 的每一层
    for layer_idx in range(model.config.depth):  # 18 layers
        layer = model.paligemma_backbone.layers[layer_idx]

        # 量化 gate_proj, up_proj, down_proj
        for name in ["gate_proj", "up_proj", "down_proj"]:
            original = getattr(layer.mlp, name)
            quantized = TritonINT4Linear.from_float(original, group_size)
            setattr(layer.mlp, name, quantized)

            # 释放原始权重
            del original
            torch.cuda.empty_cache()

        print(f"Layer {layer_idx}: MLP quantized to INT4")

    return model
```

### 5.5 Plan B 验证

```bash
# 精度验证
python scripts/validate_int4_precision.py \
    --checkpoint_dir /root/.cache/openpi/checkpoints/pi05_libero \
    --group_size 128 \
    --num_samples 1000

# 延迟验证
python scripts/benchmark_int4_kv_cache.py \
    --num_runs 100

# LIBERO 评测
python scripts/libero_eval_int4.py \
    --quick --mode int4
```

### 5.6 Plan B 成功标准

- [ ] INT4 vs FP16 Cosine: ≥ 0.98
- [ ] KV Cache: ≤ 15 ms
- [ ] Total: ≤ 120 ms
- [ ] Hz: ≥ 8.3 Hz
- [ ] LIBERO 精度: ≥ 90%

---

## 六、Backup 方案

### 6.1 各阶段 Backup

| 阶段 | 主方案 | Backup 方案 | 触发条件 |
|------|--------|-------------|----------|
| Phase 0 | 环境验证 | 等待软件支持 | Triton/FlashInfer 不可用 |
| Plan A | FlashInfer | PyTorch Attention | FlashInfer 在 Thor 失败 |
| Plan B | Triton INT4 | CUTLASS INT4 | Triton kernel 性能差 |
| 精度 | Per-group INT4 | Per-channel INT4 | 精度损失 > 10% |

### 6.2 Backup: CUTLASS INT4

如果 Triton 在 Thor 上性能不佳，使用 CUTLASS 手写 INT4 kernel：

```cpp
// 使用 CUTLASS 的 INT4 GEMM
// 这是更底层但更可控的方案

#include <cutlass/gemm/device/gemm.h>

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::int4b_t,                    // Element A (INT4)
    cutlass::layout::RowMajor,           // Layout A
    cutlass::half_t,                     // Element B (FP16)
    cutlass::layout::ColumnMajor,        // Layout B
    cutlass::half_t,                     // Element C (FP16)
    cutlass::layout::RowMajor,           // Layout C
    int32_t,                             // Accumulator (INT32)
    cutlass::arch::OpClassTensorOp,      // Use Tensor Cores
    cutlass::arch::Sm100                 // Thor = SM 10.0
>;
```

### 6.3 Backup: 等待 NVIDIA 修复

如果所有方案都失败，等待：
- TensorRT 10.15+ 修复 FP8/FP4 scale bug
- TensorRT-LLM 支持 Thor
- NVIDIA 发布 Thor 专用优化库

---

## 七、最终决策树

```
                    Phase 0: 环境验证
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   Triton ✅          Triton ❌          全失败
        │                  │                  │
        ▼                  ▼                  ▼
   测试带宽            Plan A Only       等待软件支持
        │              (保守优化)         (7.0 Hz max)
        │
   ┌────┴────┐
   │         │
 >150GB/s  <150GB/s
   │         │
   ▼         ▼
Plan A    Plan B
(FP16)   (INT4 必须)
   │         │
   ▼         ▼
7.0 Hz    8.7 Hz
```

---

## 八、时间表

| 阶段 | 工作内容 | 时间 | 交付物 |
|------|----------|------|--------|
| **Week 1 Day 1-2** | Phase 0 环境验证 | 2 天 | 决策报告 |
| **Week 1 Day 3-5** | Plan A 实现 | 3 天 | CUDA Graph + Fusion |
| **Week 2** | Plan A 验证 + 调优 | 5 天 | 7.0 Hz 版本 |
| **Week 3** | Plan B INT4 Kernel | 5 天 | Triton INT4 实现 |
| **Week 4** | Plan B 验证 + 调优 | 5 天 | 8.7 Hz 版本 |
| **Week 5** | 集成测试 + 文档 | 5 天 | 最终发布 |

**总计: 5 周**

---

## 九、风险评估总结

| 风险 | 可能性 | 影响 | 缓解 |
|------|--------|------|------|
| Triton Thor 不支持 | 30% | 高 | 回退 CUTLASS |
| INT4 精度损失大 | 40% | 中 | 调整 group_size |
| 带宽测量偏低 | 20% | 中 | 检查驱动/散热 |
| FlashInfer 失败 | 50% | 低 | 用 PyTorch |

---

## 十、执行清单

### Day 1 立即执行

- [ ] 运行 `phase0_environment_check.py`
- [ ] 记录 Thor 实际带宽
- [ ] 确认 Triton/FlashInfer 可用性
- [ ] 基于结果决定 Plan A 或 Plan B

### Week 1 交付

- [ ] Plan A 基础实现
- [ ] CUDA Graph 全图录制
- [ ] 初步性能数据

### Week 2 交付

- [ ] Plan A 完整验证
- [ ] LIBERO 精度测试
- [ ] 决定是否启动 Plan B

### Week 3-4 交付 (如需 Plan B)

- [ ] INT4 量化实现
- [ ] Triton Kernel 调优
- [ ] 最终性能验证

---

## 十一、Phase 0 验证结果 (2026-02-08)

### 测试结果汇总

#### Triton 性能测试

| 实现 | 延迟 | vs cuBLAS |
|------|------|-----------|
| torch.matmul (cuBLAS) | 0.47 ms | 1.00x |
| **Triton FP16 MatMul** | **1.14 ms** | **0.41x** |

**结论**: ❌ Triton 在 Thor SM 11.0 上性能只有 cuBLAS 的 41%，不可用。

#### 量化库测试

| 方案 | 性能 vs BF16 | 状态 |
|------|-------------|------|
| Triton INT4 | 0.02x | ❌ Triton 本身慢 |
| torchao INT8 | 0.09x | ❌ 灾难性性能 |
| torchao INT4 | N/A | ❌ 缺少 fbgemm-gpu-genai |
| torch._int_mm (cuBLAS INT8) | 0.98x | ❌ 无加速 |
| CUDA Graph | 1.00x | ≈ 无提升 |

### 最终结论

**所有量化加速方案在 Thor 上都不可用**:

1. **Triton**: 基础 FP16 MatMul 就比 cuBLAS 慢 2.5x
2. **torchao**: 没有 Thor SM 11.0 优化的 kernel
3. **cuBLAS INT8**: 与 FP16 性能相同，无硬件加速
4. **CUDA Graph**: 几乎没有收益

### 更新后的优化路线

```
原计划:
  Plan A (保守): 5.7 Hz → 7.0 Hz
  Plan B (激进): 5.7 Hz → 8.7 Hz

验证后:
  ❌ Plan A/B 都不可行
  ✅ 当前最佳: 维持 5.7 Hz
```

### 后续建议

| 方向 | 优先级 | 预期收益 | 工作量 |
|------|--------|----------|--------|
| 等待 NVIDIA Thor 软件支持 | 高 | 未知 | 等待 |
| **减少 denoising steps** | **高** | **12 Hz** | **1 天** |
| 模型蒸馏 | 中 | 2-3x | 4-6 周 |
| 减少 Transformer 层数 | 中 | 1.5x | 2 周 |

### 执行清单更新

- [x] 运行 `phase0_environment_check.py`
- [x] 测试 Triton FP16 性能 (失败)
- [x] 测试 torchao INT8/INT4 (失败)
- [x] 测试 cuBLAS INT8 (无加速)
- [x] 测试 CUDA Graph (无收益)
- [ ] 验证 3-step denoising 精度 (下一步)

---

## Phase NVFP4: CUTLASS SM110a NVFP4 突破 (2025-02-08)

### 重大发现

经过深入调研和测试，成功在 Thor SM110 上运行 CUTLASS NVFP4 GEMM！

### 测试方法论

1. **TensorRT-LLM FP4 Ops**: 发现 TRT-LLM 的 NVFP4 内核只编译了 SM90a/SM100/SM120，**缺少 SM110**
2. **CUTLASS 源码编译**: 修改 CUTLASS 72a example 支持 SM110a 架构
3. **架构检查绕过**: 修改 `CUTLASS_ARCH_MMA_SM100_SUPPORTED` → `CUTLASS_ARCH_MMA_SM110_SUPPORTED`

### 性能对比 (NVFP4 vs cuBLAS BF16)

```
======================================================================
NVFP4 vs cuBLAS BF16 Benchmark on Thor SM110
======================================================================

Problem Size                   | BF16 (ms)    | NVFP4 (ms)   | Speedup
------------------------------------------------------------------------------------------
256x16384x2048                 | 0.356        | 0.082        | 4.34x
256x2048x16384                 | 0.449        | 0.057        | 7.82x
512x8192x2048                  | 0.231        | 0.082        | 2.82x
512x2048x8192                  | 0.162        | 0.061        | 2.63x
1024x4096x2048                 | 0.156        | 0.082        | 1.90x
```

### 关键限制

1. **尺寸限制**: 某些 M*N 组合会失败 (如 512x16384, 712x16384)
2. **对齐要求**: M 和 N 需要对齐到特定倍数
3. **Pi0.5 实际 batch 712 不支持**: 需要 padding 或拆分

### 架构兼容性发现

| 架构 | MMA 指令 | Thor 兼容性 |
|------|----------|-------------|
| SM100 (B100/B200) | tcgen05.mma.blockscaled | ✅ 部分兼容 |
| SM120 (RTX 50xx) | mma.sync.aligned.block_scale | ❌ 不兼容 |
| SM110 (Thor) | tcgen05.mma.blockscaled | ✅ 自编译成功 |

### 编译方法

```bash
# 在 Docker 容器中
cd /workspace/external/cutlass_sm110_build

# 复制并修改示例
cp /usr/local/lib/python3.12/dist-packages/cutlass_library/source/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu .

# 修改架构检查
sed -i 's/CUTLASS_ARCH_MMA_SM100_SUPPORTED/CUTLASS_ARCH_MMA_SM110_SUPPORTED/g' 72a_blackwell_nvfp4_bf16_gemm.cu

# 编译
CUTLASS_PATH=/usr/local/lib/python3.12/dist-packages/cutlass_library/source
nvcc -O3 -std=c++17 -arch=sm_110a \
    --expt-relaxed-constexpr \
    -I$CUTLASS_PATH/include \
    -I$CUTLASS_PATH/tools/util/include \
    -I$CUTLASS_PATH/examples/common \
    72a_blackwell_nvfp4_bf16_gemm.cu \
    -o nvfp4_gemm_sm110a
```

### 理论加速潜力

假设 NVFP4 可以用于 MLP:
- 当前 MLP 权重读取: 18.1 ms (BF16)
- NVFP4 理论: 4.5 ms (1/4 权重)
- 实测加速: 2.8x - 7.8x (取决于尺寸)

| 组件 | 当前 | NVFP4 优化 |
|------|------|-----------|
| MLP GEMM | 18.1 ms | ~4-6 ms |
| KV Cache 总计 | 54 ms | ~25-35 ms |

### 下一步计划

1. **封装 PyTorch Op**: 将 CUTLASS NVFP4 kernel 封装为可调用的 PyTorch 扩展
2. **量化权重**: 实现 Pi0.5 模型权重的 FP4 量化
3. **集成测试**: 验证端到端精度和性能
4. **解决尺寸限制**: 通过 padding 或拆分支持 batch=712

### 社区反馈建议

建议向 NVIDIA 反馈:
1. TensorRT-LLM 缺少 SM110 NVFP4 内核编译
2. CUTLASS 文档缺少 SM110 示例
3. 请求官方 Thor NVFP4 支持
