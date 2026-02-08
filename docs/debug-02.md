# Debug Session 02: Torch-TensorRT Backend Integration

**Date**: 2026-02-02 ~ 2026-02-03
**Platform**: NVIDIA Jetson Thor (Blackwell SM 11.0)
**Environment**: Docker nvcr.io/nvidia/pytorch:25.12-py3

---

## 1. Background & Problem Statement

### 1.1 Previous Attempts (Failed)

| Approach | Result | Issue |
|----------|--------|-------|
| ONNX → TensorRT (opt_level≥3) | ❌ Crash | Myelin kernel crash: `CUDA error 400 launching __myl_CastMulMeanAddSqrtDivMulMul` |
| ONNX → TensorRT (opt_level=2) | ⚠️ Unstable | ~58ms but occasional failures |
| ONNX → TensorRT (opt_level=0) | ✅ Works | 654ms latency (6x slower than PyTorch!) |
| TRT IAttention API | ⚠️ Limited | Only 1 attention layer per network allowed |
| W8A16 Quantization | ❌ Fail | Accuracy loss, cosine sim < 0.9 |

### 1.2 Root Cause Analysis

Thor (Blackwell) 是首款基于 Blackwell 架构的 Jetson 平台，TensorRT 的 Myelin 优化器在处理 RMSNorm 相关的融合算子时存在 bug：
- RMSNorm = Cast + Mul + Mean + Add + Sqrt + Div + Mul + Mul
- Myelin 尝试将这些算子融合成单一 kernel，但在 Thor 上执行时崩溃
- 相同的 ONNX 模型在 A100/H100/Orin 上正常工作

### 1.3 Solution: Torch-TensorRT

Torch-TensorRT 通过 PyTorch 的 `torch.compile` 机制编译模型，避开了 ONNX 导出路径和 Myelin 的某些优化：
- 直接从 PyTorch → TensorRT
- SDPA (Scaled Dot Product Attention) 原生支持
- 更好的数值稳定性控制

---

## 2. Implementation Details

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      UnifiedPolicy                               │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐  │
│  │ PyTorch     │ PyTorchPipe │ TensorRT    │ TorchTRT        │  │
│  │ Backend     │ Backend     │ Backend     │ Backend (NEW)   │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│                    TorchTRTKVCacheEngine                        │
│                    ┌───────────────────────┐                    │
│                    │  KVCacheModel         │                    │
│                    │  ├── 18x Transformer  │                    │
│                    │  │   └── RMSNorm      │                    │
│                    │  │   └── GQA Attn     │                    │
│                    │  │   └── MLP          │                    │
│                    │  └── Final Norm       │                    │
│                    └───────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Model Constants

```python
NUM_LAYERS = 18          # Transformer layers
NUM_HEADS = 8            # Query heads
NUM_KV_HEADS = 1         # KV heads (GQA ratio = 8:1)
HEAD_DIM = 256           # Head dimension
HIDDEN_SIZE = 2048       # Hidden dimension
MLP_DIM = 16384          # MLP intermediate dimension
SEQ_LEN = 970            # Sequence length (512 img + 458 text)
RMS_NORM_EPS = 1e-6      # RMSNorm epsilon
```

### 2.3 Files Created/Modified

| File | Type | Description |
|------|------|-------------|
| `openpi/src/openpi/inference/torch_trt_kv_cache.py` | NEW | TorchTRT KV Cache Engine |
| `openpi/src/openpi/inference/unified_policy.py` | MODIFIED | Added TorchTRTBackend |
| `openpi/src/openpi/inference/__init__.py` | MODIFIED | Added exports |
| `openpi/scripts/benchmark_torch_trt_backend.py` | NEW | Latency/Precision benchmark |
| `openpi/scripts/libero_eval_torch_trt.py` | NEW | LIBERO evaluation |

### 2.4 Key Design Decisions

#### 2.4.1 RMSNorm Numerical Stability
```python
class RMSNorm(nn.Module):
    def forward(self, x):
        input_dtype = x.dtype
        # Cast to float32 for numerical stability
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        # Cast back to original dtype
        return (x * self.weight.float()).to(input_dtype)
```

#### 2.4.2 GQA KV Expansion
```python
# Expand KV heads: 1 → 8 for GQA
k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
```

#### 2.4.3 KV Reuse Strategy
```python
class TorchTRTBackend:
    def __init__(self, config, kv_reuse_freq=2):
        # kv_reuse_freq: 1 = no reuse, 2 = every 2 frames, 3 = every 3 frames
        self.kv_reuse_freq = kv_reuse_freq

    def infer(self, obs, num_steps=3):
        if self.frame_count % self.kv_reuse_freq == 0:
            return self._full_frame_inference(obs, num_steps)
        else:
            return self._fast_frame_inference(obs, num_steps)
```

---

## 3. Backend Registry

```python
BACKENDS = {
    # PyTorch backends
    "pytorch": PyTorchBackend,
    "pytorch_pipelined": PyTorchPipelinedBackend,

    # TensorRT backends
    "tensorrt": TurboTitanBackend,
    "tensorrt_pipelined": TripleStreamBackend,

    # Torch-TensorRT backends (NEW)
    "torch_trt": TorchTRTBackend,
    "torch_trt_freq1": lambda cfg: TorchTRTBackend(cfg, kv_reuse_freq=1),
    "torch_trt_freq2": lambda cfg: TorchTRTBackend(cfg, kv_reuse_freq=2),
    "torch_trt_freq3": lambda cfg: TorchTRTBackend(cfg, kv_reuse_freq=3),
}
```

---

## 4. Test Results (4-Layer Validation)

| Metric | Value |
|--------|-------|
| PyTorch Latency | 19.0 ms |
| TorchTRT Latency | 13.5 ms |
| Speedup | **1.45x** |
| Cosine Similarity | 0.999999 |
| Compilation Time | ~10-27s |

---

## 5. Usage Instructions

### 5.1 Compile Model
```bash
cd /workspace/openpi
python scripts/benchmark_torch_trt_backend.py \
    --checkpoint ~/.cache/openpi/checkpoints/pi05_libero \
    --compile-and-save
```

### 5.2 Run Latency Benchmark
```bash
python scripts/benchmark_torch_trt_backend.py \
    --checkpoint ~/.cache/openpi/checkpoints/pi05_libero \
    --num-warmup 10 \
    --num-iters 50
```

### 5.3 Run LIBERO Evaluation
```bash
python scripts/libero_eval_torch_trt.py \
    --checkpoint ~/.cache/openpi/checkpoints/pi05_libero \
    --backend torch_trt \
    --compare-pytorch \
    --num-episodes 5
```

---

## 6. Engine Verification (Critical!)

**⚠️ 重要**: 测试前必须验证 engine 对应正确！

```python
def verify_backend_engine(policy, obs, reference_policy=None):
    """Verify engine produces valid output."""
    result = policy.infer(obs)

    # Check for NaN/Inf
    if np.isnan(result['actions']).any() or np.isinf(result['actions']).any():
        return False, "Output contains NaN/Inf"

    # Compare with reference if available
    if reference_policy:
        ref_result = reference_policy.infer(obs)
        cos_sim = cosine_similarity(result['actions'], ref_result['actions'])
        if cos_sim < 0.99:
            return False, f"Low similarity: {cos_sim}"

    return True, "Verified"
```

---

## 7. Known Issues

### 7.1 IAttention Single-Layer Limit
```
Error: mScopedOps.size() == mGraph.scopedOps.size()
```
TensorRT 10.14.1 的 `network.add_attention()` API 只允许每个 network 有一个 attention layer。

### 7.2 Myelin Crash on Thor
当 `builder_optimization_level >= 3` 时，Myelin 创建的融合 kernel 在 Thor 上崩溃。

### 7.3 Import in Docker
Torch-TensorRT 相关代码只能在 Docker 容器内运行：
```bash
docker exec -it openpi-thor bash
cd /workspace/openpi
python scripts/benchmark_torch_trt_backend.py
```

---

## 8. FP8 Mixed Precision Testing (2026-02-03)

### 8.1 FP8 Availability on Thor

| Component | Status |
|-----------|--------|
| PyTorch FP8 dtype | ✅ `torch.float8_e4m3fn` available |
| TensorRT FP8 DataType | ✅ `trt.DataType.FP8` available |
| TensorRT FP8 BuilderFlag | ✅ `trt.BuilderFlag.FP8` can be set |
| torch._scaled_mm | ✅ Works for FP8 GEMM |

### 8.2 FP8 MatMul Benchmark (Single Op)

测试 `(970, 2048) @ (2048, 16384)` 矩阵乘法：

| Precision | Latency | Speedup | Cosine Sim |
|-----------|---------|---------|------------|
| FP16 | 0.69 ms | 1.0x | baseline |
| FP8 (_scaled_mm) | 0.38 ms | **1.82x** | 0.9993 |

**结论**: FP8 Tensor Core 在 Thor 上提供接近 2x 的 GEMM 加速。

### 8.3 FP8 MLP Benchmark (Full Layer)

测试完整 MLP 层 (gate + up + silu + down)：

| Mode | Latency | Speedup | Cosine Sim |
|------|---------|---------|------------|
| FP16 PyTorch | 3.52 ms | 1.0x | baseline |
| FP8 (online quant) | 3.39 ms | 1.04x | 0.9979 |
| FP8 (pre-quant scale) | 2.87 ms | **1.23x** | 0.9979 |

**分析**:
- 单个 GEMM 获得 1.82x 加速
- 完整 MLP (3个 GEMM + activations) 只有 1.23x 加速
- 原因：量化/反量化开销 + SiLU/elementwise 操作仍在 FP16

### 8.4 FP8 量化策略

```python
# Per-tensor absmax scaling for FP8
FP8_MAX = 448.0  # FP8 e4m3fn 最大值
scale = tensor.abs().max() / FP8_MAX
tensor_fp8 = (tensor / scale).to(torch.float8_e4m3fn)

# FP8 MatMul with scale
output = torch._scaled_mm(
    a_fp8, b_fp8.t(),
    scale_a=scale_a, scale_b=scale_b,
    out_dtype=torch.float16
)
```

### 8.5 Torch-TensorRT FP8 现状

| 测试 | 结果 |
|------|------|
| 声明 FP8 支持 | ✅ `enabled_precisions={torch.float16, torch.float8_e4m3fn}` |
| 实际使用 FP8 | ❌ 无加速，可能 fallback 到 FP16 |
| 原因分析 | Torch-TRT 可能不会自动插入 Q/DQ 节点 |

**结论**: Torch-TensorRT 的 FP8 支持目前是"声明式"的，不会自动进行 FP8 量化。需要手动使用 `torch._scaled_mm` 或 TensorRT-LLM 的量化工具。

### 8.6 18-Layer Model Benchmark

| Backend | KV Cache Latency | Speedup | Cosine Sim |
|---------|------------------|---------|------------|
| PyTorch FP16 | 87.65 ms | 1.0x | baseline |
| Torch-TRT FP16 | 59.84 ms | **1.46x** | ~0.9999 |
| Torch-TRT FP8 | 60.03 ms | 1.46x | ~0.9999 |

**注**: Torch-TRT 的 FP8 和 FP16 延迟相同，确认 FP8 未被实际使用。

---

## 9. Optimization Analysis (2026-02-03)

### 9.1 带宽瓶颈分析

用户指出：Thor 是 16/32 字节对齐的，FP8 ↔ FP16 混合精度会触发自动 reformat，导致带宽瓶颈。

**Per-Layer Compute Analysis:**
```
Op              Shape                    GFLOPS   AI (FLOPS/Byte)
----------------------------------------------------------------
Q_proj          (970x2048)@(2048x2048)    8.14    498.1
K_proj          (970x2048)@(2048x256)     1.02    184.3
V_proj          (970x2048)@(2048x256)     1.02    184.3
O_proj          (970x2048)@(2048x2048)    8.14    498.1
gate_proj       (970x2048)@(2048x16384)  65.10    632.8
up_proj         (970x2048)@(2048x16384)  65.10    632.8
down_proj       (970x16384)@(16384x2048) 65.10    632.8
----------------------------------------------------------------
TOTAL                                   213.59
```

**结论**: MLP 的 Arithmetic Intensity ≈ 715 FLOPS/Byte，低于 Thor FP16 峰值 (~1000 FLOPS/Byte)，说明**内存带宽受限**。

### 9.2 垂直融合测试结果

| 融合策略 | Separate | Fused | Speedup |
|---------|----------|-------|---------|
| QKV Fusion | 0.103 ms | 0.094 ms | **1.10x** |
| Gate-Up Fusion | 1.359 ms | 1.256 ms | **1.08x** |
| SwiGLU (torch.compile) | 0.693 ms | 1.036 ms | **0.67x** ⚠️ |
| Full MLP (torch.compile) | 3.490 ms | 3.219 ms | **1.08x** |

**分析**:
- 垂直融合收益有限 (~1.1x)，因为单个 GEMM 已经足够大
- `torch.compile` 在小 batch size 下有 JIT 开销
- 真正瓶颈在于 FP8 ↔ FP16 精度转换

### 9.3 FP4/INT4 支持检查

| Feature | Status |
|---------|--------|
| `trt.DataType.FP4` | ✅ Available |
| `trt.DataType.INT4` | ✅ Available |
| `trt.BuilderFlag.INT4` | ✅ Available |
| `torch.float4_e2m1fn_x2` | ✅ Available (packed) |
| modelopt 0.39.0 | ✅ Installed |

**问题**: TensorRT 原生 API 在 Thor 上仍然触发 Myelin crash。

### 9.4 FP8 `torch._scaled_mm` 性能

| Mode | Latency | Speedup | Cosine Sim |
|------|---------|---------|------------|
| FP16 matmul | 0.669 ms | 1.0x | baseline |
| FP8 (static scale) | 0.381 ms | **1.75x** | 0.999299 |
| FP8 (online quant) | 0.540 ms | **1.24x** | 0.999299 |

**关键发现**:
- `torch._scaled_mm` 提供真正的 FP8 Tensor Core 加速
- 静态 scale 比动态 scale 快 0.159 ms
- 精度损失极小 (cosine sim > 0.999)

### 9.5 推荐优化路径

```
┌─────────────────────────────────────────────────────────────────┐
│                 Current: Torch-TRT FP16                          │
│                 KV Cache: 59.84 ms (1.46x vs PyTorch)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          Option 1: FP8 MLP with torch._scaled_mm                 │
│          Expected: ~45 ms (1.33x additional speedup)             │
│          Pros: Uses FP8 Tensor Cores, high accuracy              │
│          Cons: Requires custom CUDA graph for best perf          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          Option 2: QKV + Gate-Up Fusion + FP8                    │
│          Expected: ~40 ms (additional 1.1x from fusion)          │
│          Pros: Reduces kernel launches and memory traffic        │
│          Cons: More complex implementation                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          Option 3: INT4/FP4 Weight-Only Quantization             │
│          Expected: ~30 ms (if TRT Myelin fixed)                  │
│          Pros: 4x weight bandwidth savings                       │
│          Cons: TRT Myelin crash on Thor, needs NVIDIA fix        │
└─────────────────────────────────────────────────────────────────┘
```

### 9.6 Recommended Implementation

**短期 (立即可用)**:
```python
# FP8 MLP with torch._scaled_mm
class FP8MLP(nn.Module):
    def __init__(self, gate_w, up_w, down_w):
        # Pre-quantize weights to FP8
        self.gate_fp8 = quantize_to_fp8(gate_w)
        self.up_fp8 = quantize_to_fp8(up_w)
        self.down_fp8 = quantize_to_fp8(down_w)
        self.scale_gate = compute_scale(gate_w)
        # ...

    def forward(self, x):
        # FP8 GEMM with static scales
        gate = scaled_mm(x_fp8, self.gate_fp8, scale_x, self.scale_gate)
        up = scaled_mm(x_fp8, self.up_fp8, scale_x, self.scale_up)
        hidden = F.silu(gate) * up  # Still FP16
        return scaled_mm(hidden_fp8, self.down_fp8, scale_h, self.scale_down)
```

**中期 (需要 NVIDIA 修复)**:
- 等待 TRT 10.15+ 修复 Thor Myelin crash
- 使用 TRT INT4/FP4 weight-only quantization

---

## 10. Next Steps

1. ✅ Complete Torch-TensorRT backend integration
2. ✅ Test FP8 availability on Thor
3. ✅ Benchmark FP8 MLP vs FP16 MLP
4. ✅ Analyze bandwidth bottleneck and vertical fusion
5. ✅ Verify FP4/INT4 support on Thor (available but TRT crashes)
6. ✅ Implement FP8 MLP layer with `torch._scaled_mm`
7. ✅ Create `FP8MLPBackend` for UnifiedPolicy
8. ⏳ Run LIBERO evaluation with FP8 backend
9. ⏳ Report Myelin crash to NVIDIA for FP4/INT4 support

---

## 10. FP8 Hybrid MLP Implementation (2026-02-03)

### 10.1 Key Discovery: Hidden Quantization Bottleneck

Full FP8 MLP is **SLOWER** than FP16 due to hidden tensor quantization overhead:

| Component | Tensor Size | Time |
|-----------|-------------|------|
| Input quantization | 970 x 2048 = 2M | 0.12 ms |
| Hidden quantization | 970 x 16384 = 15.9M | **1.99 ms** |
| FP8 matmul (gate) | - | 0.40 ms |
| FP8 matmul (up) | - | 0.37 ms |
| FP8 matmul (down) | - | 0.49 ms |

**Root cause**: The hidden tensor (seq_len × mlp_dim) is 8x larger than the input tensor. Quantizing it requires:
1. `hidden.float()` - type conversion (0.51 ms)
2. `/ scale` - division (0.57 ms)
3. `.clamp()` - range clipping (0.55 ms)
4. `.to(fp8)` - dtype conversion (0.35 ms)

This ~2ms overhead completely negates the 0.3ms FP8 matmul speedup.

### 10.2 Solution: FP8 Hybrid MLP

Use FP8 only where it's beneficial:

```
Input (FP16) → [Quant 0.12ms] → FP8
    ↓
Gate FP8 GEMM (0.40ms) ─┐
Up FP8 GEMM (0.37ms) ───┼→ SiLU * Up (FP16)
    ↓
Hidden (FP16) → Down FP16 GEMM (0.68ms) → Output (FP16)
```

**Key insight**: FP8 on gate/up (hidden→mlp_dim) saves time. FP8 on down (mlp_dim→hidden) loses time.

### 10.3 Benchmark Results

| Mode | Latency | Speedup | Cosine Sim |
|------|---------|---------|------------|
| FP16 full | 3.47 ms | 1.00x | baseline |
| FP8 full | 4.11 ms | **0.84x** ❌ | 0.997 |
| **FP8 hybrid** | **3.09 ms** | **1.12x** ✓ | 0.996 |

### 10.4 Single Operation Speedups

| Operation | FP16 | FP8 | Speedup |
|-----------|------|-----|---------|
| Gate matmul (970,2048)@(2048,16384) | 0.68 ms | 0.40 ms | **1.70x** |
| Up matmul | 0.68 ms | 0.37 ms | **1.84x** |
| Down matmul (970,16384)@(16384,2048) | 0.68 ms | 0.49 ms | **1.39x** |
| Gate+Up combined | 1.36 ms | 0.77 ms | **1.77x** |

### 10.5 Implementation

Created `FP8HybridMLP` class in `src/openpi/inference/fp8_mlp.py`:

```python
class FP8HybridMLP(nn.Module):
    """FP8 for gate/up, FP16 for down projection."""

    def forward(self, x):
        # Quantize input (small, fast)
        x_fp8 = quantize_to_fp8(x, self.scale_input)

        # Gate + Up in FP8 (1.77x speedup)
        gate = torch._scaled_mm(x_fp8, self.gate_w_fp8.t(), ...)
        up = torch._scaled_mm(x_fp8, self.up_w_fp8.t(), ...)

        # SiLU + mul (stays FP16)
        hidden = F.silu(gate) * up

        # Down in FP16 (avoids 2ms hidden quantization)
        return hidden @ self.down_w.t()
```

### 10.6 torch._scaled_mm Layout Requirements

Critical: `torch._scaled_mm` has specific memory layout requirements:

```python
# CORRECT: b must be column-major (transposed view)
result = torch._scaled_mm(a, W.t(), scale_a, scale_b, out_dtype=torch.float16)

# WRONG: .contiguous() converts back to row-major
result = torch._scaled_mm(a, W.t().contiguous(), ...)  # Will fail!
```

Weight storage:
- Store weights as (out_features, in_features) in row-major
- At runtime: call `.t()` to get column-major view (don't call .contiguous()!)

### 10.7 18-Layer KV Cache Impact

| Component | FP16 | FP8 Hybrid | Savings |
|-----------|------|------------|---------|
| MLP (per layer) | 3.47 ms | 3.09 ms | 0.38 ms |
| MLP (18 layers) | 62.5 ms | 55.7 ms | **6.8 ms** |

Combined with Torch-TensorRT attention (1.45x speedup), expected total:
- FP16 baseline KV Cache: 87.6 ms
- FP8 Hybrid + Torch-TRT: ~75 ms → **~1.17x** total speedup

### 10.8 Registered Backends

New backends added to `UnifiedPolicy`:

```python
BACKENDS = {
    # FP8 MLP backends (FP16 Attention + FP8 MLP using torch._scaled_mm)
    "fp8_mlp": FP8MLPBackend,              # Freq=2 default
    "fp8_mlp_freq1": lambda cfg: FP8MLPBackend(cfg, kv_reuse_freq=1),
    "fp8_mlp_freq2": lambda cfg: FP8MLPBackend(cfg, kv_reuse_freq=2),
    "fp8_mlp_freq3": lambda cfg: FP8MLPBackend(cfg, kv_reuse_freq=3),
    "fp8_mlp_dynamic": lambda cfg: FP8MLPBackend(cfg, use_static_scale=False),
}
```

### 10.9 Files Created

| File | Description |
|------|-------------|
| `src/openpi/inference/fp8_mlp.py` | FP8 MLP implementations (FP8MLP, FP8HybridMLP, FP8MLPWithStaticScale) |
| `src/openpi/inference/fp8_kv_cache.py` | FP8 KV Cache model (FP16 Attention + FP8 MLP) |
| `scripts/benchmark_fp8_mixed_precision.py` | Comprehensive FP8 benchmark suite |

---

## 11. References

- [TensorRT GitHub Issues](https://github.com/NVIDIA/TensorRT/issues)
- [Torch-TensorRT Documentation](https://pytorch.org/TensorRT/)
- [NVIDIA Thor TRT Issues Report](./NVIDIA_THOR_TRT_ISSUES.md)
- [PyTorch FP8 Documentation](https://pytorch.org/docs/stable/generated/torch._scaled_mm.html)

---

## 12. Appendix: Test Scripts

| Script | Description |
|--------|-------------|
| `scripts/benchmark_fp8_mlp_fp32_attn.py` | FP8 MLP + FP32 Attention mixed precision benchmark |
| `scripts/test_native_trt_fp8_mlp.py` | Native TensorRT FP8 MLP test |
| `scripts/benchmark_torch_trt_backend.py` | Torch-TensorRT backend latency/precision benchmark |
| `scripts/libero_eval_torch_trt.py` | LIBERO evaluation with engine verification |
| `scripts/benchmark_fp8_mixed_precision.py` | FP8 Hybrid MLP benchmark suite |

---

*Last Updated: 2026-02-03 06:30 UTC*
