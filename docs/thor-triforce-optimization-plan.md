# Thor Triforce 优化工程计划

## Triton + TRT + TVM 三位一体硬核加速方案

**目标**: 在 Jetson Thor 上，保持 10-Step Denoising 的前提下，将 Pi0.5 推理频率从 **5.7 Hz** 提升至 **12-14 Hz**。

---

## 一、当前状态基线 (Baseline)

### 1.1 延迟分解 (10-Step Denoising)

| 组件 | 当前延迟 | 占比 | 瓶颈类型 |
|------|----------|------|----------|
| Vision TRT (FP16) | 17.0 ms | 9.6% | 计算受限 |
| KV Cache Prefill | 54.0 ms | 30.6% | **内存带宽受限** |
| Denoise (10 step) | 102.3 ms | 58.0% | 计算+内存混合 |
| Overhead | ~3.2 ms | 1.8% | CPU/CUDA 同步 |
| **Total** | **176.5 ms** | 100% | **5.7 Hz** |

### 1.2 模型架构关键参数

```
PaliGemma LLM (gemma_2b):
├── Layers: 18
├── Hidden: 2048
├── MLP Dim: 16,384 (瓶颈!)
├── Heads: 8 (Query) / 1 (KV, GQA)
└── Head Dim: 256

Prefix Sequence:
├── Base Camera: 256 tokens (16×16)
├── Wrist Camera: 256 tokens (16×16)
├── Language: ~200 tokens
└── Total: ~712 tokens

Action Expert (gemma_300m):
├── Layers: 18
├── Hidden: 1024
├── MLP Dim: 4,096
└── Action Horizon: 50
```

### 1.3 权重量分析 (KV Cache 瓶颈根因)

| 组件 | 每层权重 | 18层总计 | 读取时间 (@180GB/s) |
|------|----------|----------|---------------------|
| QKV Projection | 10 MB | 180 MB | 1.0 ms |
| O Projection | 8 MB | 144 MB | 0.8 ms |
| MLP (gate+up+down) | 201 MB | **3.62 GB** | **20.1 ms** |
| LayerNorm等 | ~1 MB | 18 MB | 0.1 ms |
| **Total** | 220 MB | **3.96 GB** | **22.0 ms** |

**关键发现**: MLP 权重占 91.4%，是内存带宽瓶颈的根本原因。

---

## 二、优化目标设定

### 2.1 分阶段目标

| 阶段 | 目标延迟 | 目标 Hz | 精度要求 | 风险等级 |
|------|----------|---------|----------|----------|
| Phase 0 (Baseline) | 176.5 ms | 5.7 Hz | 100% | - |
| **Phase 1**: FlashInfer KV | 140 ms | 7.1 Hz | ≥95% | 中 |
| **Phase 2**: Triton Fused Attn | 120 ms | 8.3 Hz | ≥95% | 中 |
| **Phase 3**: CUDA Graph 全图 | 105 ms | 9.5 Hz | ≥95% | 低 |
| **Phase 4**: TVM Kernel Fusion | 95 ms | 10.5 Hz | ≥95% | 中 |
| **Phase 5**: FP8 Accumulation | 85 ms | 11.8 Hz | ≥90% | 高 |
| **Phase 6** (激进): NVFP4 MLP | 75 ms | 13.3 Hz | ≥85% | 极高 |

### 2.2 精度验证标准

| 指标 | 验证方法 | 通过阈值 |
|------|----------|----------|
| Cosine Similarity | 对比 baseline 输出 | ≥ 0.99 |
| LIBERO Success Rate | 3 tasks × 3 trials | ≥ 85% (baseline 100%) |
| Action MSE | 与 FP32 参考比较 | ≤ 0.01 |
| 数值稳定性 | 100次推理无 NaN/Inf | 100% |

---

## 三、Phase 1: FlashInfer KV Cache 优化

### 3.1 技术方案

**问题诊断**:
- 当前 TRT KV Cache 对 Ragged Input (Vision 512 + Text 200) 效率低
- Padding 到固定长度导致 30-40% 无效计算

**解决方案**:
```python
# 当前方案 (TRT Padded Attention)
# 输入: [B, 968, 2048] (固定长度，含 padding)
# 问题: 对 968 长度做完整 Attention，浪费带宽

# 优化方案 (FlashInfer Ragged Attention)
# 输入:
#   - Vision Tokens: [B, 512, 2048] (无 padding)
#   - Text Tokens: [B, ~200, 2048] (变长)
# 使用 FlashInfer 的 Ragged Batch API 处理
```

### 3.2 实现步骤

```python
# scripts/phase1_flashinfer_kv.py

import torch
import flashinfer

class FlashInferKVCache:
    """使用 FlashInfer 替代 TRT Attention 处理 KV Cache Prefill"""

    def __init__(self, num_layers: int = 18, num_heads: int = 8,
                 num_kv_heads: int = 1, head_dim: int = 256):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Pre-allocate KV cache
        self.max_seq_len = 1024
        self.k_cache = torch.zeros(
            num_layers, 1, self.max_seq_len, num_kv_heads, head_dim,
            dtype=torch.float16, device="cuda"
        )
        self.v_cache = torch.zeros_like(self.k_cache)

    def prefill(self, hidden_states: torch.Tensor,
                seq_lens: torch.Tensor) -> tuple:
        """
        Ragged Prefill: 只计算实际 token 的 Attention

        Args:
            hidden_states: [B, actual_len, hidden_dim] (无 padding)
            seq_lens: [B] 每个样本的实际长度
        """
        # 使用 FlashInfer 的 ragged prefill
        # 关键: 不需要 padding 到固定长度
        for layer_idx in range(self.num_layers):
            # QKV Projection (保持 TRT FP8)
            q, k, v = self.qkv_proj[layer_idx](hidden_states)

            # FlashInfer Ragged Attention
            # 这里是核心优化点
            attn_out = flashinfer.single_prefill_with_kv_cache(
                q=q,  # [total_tokens, num_heads, head_dim]
                k=k,
                v=v,
                kv_cache=(self.k_cache[layer_idx], self.v_cache[layer_idx]),
                qo_indptr=self._compute_indptr(seq_lens),
                kv_indptr=self._compute_indptr(seq_lens),
                causal=True,
            )

            # Update hidden states
            hidden_states = self.post_attn[layer_idx](attn_out, hidden_states)

        return hidden_states, (self.k_cache, self.v_cache)
```

### 3.3 验证脚本

```bash
# 运行 Phase 1 验证
python scripts/validate_phase1_flashinfer.py \
    --checkpoint_dir /root/.cache/openpi/checkpoints/pi05_libero \
    --num_warmup 10 \
    --num_runs 100

# 预期输出:
# Baseline KV Cache: 54.0 ms
# FlashInfer KV Cache: 35-40 ms (预期)
# Speedup: 1.35-1.54x
# Cosine Similarity: 0.998+
```

### 3.4 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| FlashInfer 不支持 Thor SM 11.0 | 中 | 高 | 使用 CUTLASS 重写 |
| GQA (1 KV head) 兼容性问题 | 低 | 中 | 手写 Triton Kernel |
| 精度损失 | 低 | 高 | FP16 Accumulation |

### 3.5 Phase 1 成功标准

- [ ] 延迟: KV Cache < 40 ms
- [ ] 精度: Cosine Similarity ≥ 0.995
- [ ] 稳定性: 1000 次推理无异常

---

## 四、Phase 2: Triton Fused Attention

### 4.1 技术方案

**问题**: Denoise 阶段每步需要 Cross-Attention (Action × Prefix KV)，TRT 对此优化有限。

**解决方案**: 手写 Triton FlashAttention，针对 Pi0.5 的特殊 Shape 优化。

### 4.2 Triton Kernel 实现

```python
# src/openpi/inference/triton_flash_attn.py

import triton
import triton.language as tl

@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N_CTX: tl.constexpr,  # Prefix length (968)
    M_CTX: tl.constexpr,  # Action length (50)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Pi0.5 专用 FlashAttention:
    - Q: Action tokens [B, 50, 8, 256]
    - K/V: Prefix KV Cache [B, 968, 1, 256] (GQA)
    - 特点: M << N (50 << 968), GQA (8 query heads, 1 KV head)
    """
    # Block indices
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load Q block (FP16)
    q_ptrs = Q + pid_batch * stride_qb + pid_head * stride_qh + \
             offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M_CTX)

    # Initialize accumulator (FP32 for precision!)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # GQA: KV head index (all query heads share same KV)
    kv_head_idx = 0  # 因为 num_kv_heads = 1

    # Iterate over K/V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K block
        k_ptrs = K + pid_batch * stride_kb + kv_head_idx * stride_kh + \
                 (start_n + offs_n[:, None]) * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX)

        # Compute attention scores (QK^T)
        qk = tl.dot(q, tl.trans(k))
        qk *= 0.0625  # 1/sqrt(256) = 1/16

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)

        # Load V and accumulate
        v_ptrs = V + pid_batch * stride_vb + kv_head_idx * stride_vh + \
                 (start_n + offs_n[:, None]) * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX)

        p = tl.exp(qk - m_new[:, None])
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.float16), v).to(tl.float32)

        # Update running stats
        l_i = l_new
        m_i = m_new

    # Final normalization and store (back to FP16)
    acc = acc / l_i[:, None]
    out_ptrs = Out + pid_batch * stride_ob + pid_head * stride_oh + \
               offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < M_CTX)


class TritonFlashAttention:
    """Pi0.5 专用 Triton FlashAttention 封装"""

    @staticmethod
    def forward(q, k, v, causal=False):
        """
        Args:
            q: [B, M, H, D] - Action query (B=1, M=50, H=8, D=256)
            k: [B, N, Hkv, D] - Prefix key (B=1, N=968, Hkv=1, D=256)
            v: [B, N, Hkv, D] - Prefix value
        """
        B, M, H, D = q.shape
        _, N, _, _ = k.shape

        # Output tensor
        out = torch.empty_like(q)

        # Grid configuration
        BLOCK_M = 32  # Action chunk size
        BLOCK_N = 64  # Prefix chunk size
        BLOCK_K = 256  # Head dim (full)

        grid = (triton.cdiv(M, BLOCK_M), B, H)

        flash_attn_fwd_kernel[grid](
            q, k, v, out,
            *q.stride(), *k.stride(), *v.stride(), *out.stride(),
            N_CTX=N, M_CTX=M,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        return out
```

### 4.3 Phase 2 验证

```bash
# 精度验证
python scripts/validate_triton_flash_attn.py \
    --compare_with_pytorch \
    --num_runs 1000

# 性能对比
python scripts/benchmark_attn_implementations.py \
    --implementations pytorch,trt,triton \
    --shapes "1,50,8,256;1,968,1,256"

# 预期结果:
# PyTorch Attention: 2.5 ms/step
# TRT Attention: 1.8 ms/step
# Triton Flash Attention: 0.8 ms/step
# 10 steps saving: ~17 ms
```

### 4.4 Phase 2 成功标准

- [ ] 单步 Attention: < 1.0 ms
- [ ] 10步总节省: ≥ 15 ms
- [ ] 精度: MSE < 1e-5 vs PyTorch

---

## 五、Phase 3: CUDA Graph 全图录制

### 5.1 当前问题分析

```python
# 当前: 部分 CUDA Graph (只录制 Denoise 循环)
# 问题: Vision → KV → Denoise 之间有 Python 胶水代码
# 导致: ~3.2 ms CPU overhead
```

### 5.2 全图录制方案

```python
# src/openpi/inference/cuda_graph_full.py

class FullGraphPolicy:
    """完整 CUDA Graph 录制，包含 Vision + KV + Denoise"""

    def __init__(self, ...):
        # 静态输入 buffer
        self.static_image = torch.zeros(1, 3, 224, 224, device="cuda", dtype=torch.float16)
        self.static_wrist = torch.zeros(1, 3, 224, 224, device="cuda", dtype=torch.float16)
        self.static_tokens = torch.zeros(1, 200, device="cuda", dtype=torch.long)
        self.static_state = torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16)

        # 静态输出 buffer
        self.static_actions = torch.zeros(1, 50, 32, device="cuda", dtype=torch.bfloat16)

        # 捕获完整图
        self._capture_full_graph()

    def _capture_full_graph(self):
        """捕获从输入到输出的完整计算图"""
        # Warmup
        for _ in range(5):
            self._forward_impl()
        torch.cuda.synchronize()

        # Capture
        self.full_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.full_graph):
            self._forward_impl()

        logger.info("Full CUDA Graph captured: Vision + KV + Denoise")

    def _forward_impl(self):
        """完整前向传播 (被 Graph 录制)"""
        # Stage 1: Vision TRT
        vision_out = self.vision_trt(self.static_image, self.static_wrist)

        # Stage 2: KV Cache (FlashInfer/Triton)
        prefix_kv = self.kv_cache.prefill(vision_out, self.static_tokens)

        # Stage 3: Denoise (10 steps)
        x_t = self.static_noise.clone()
        for step in range(self.num_steps):
            v_t = self.denoise_step(prefix_kv, x_t, self.timesteps[step])
            x_t = x_t + self.dts[step] * v_t

        self.static_actions.copy_(x_t)

    def infer(self, image, wrist, tokens, state):
        """推理: 复制输入 → 执行图 → 返回输出"""
        # Copy inputs (async)
        self.static_image.copy_(image)
        self.static_wrist.copy_(wrist)
        self.static_tokens.copy_(tokens)
        self.static_state.copy_(state)

        # Replay graph (所有计算一次性执行)
        self.full_graph.replay()

        return self.static_actions.clone()
```

### 5.3 Phase 3 验证

```bash
# CUDA Graph 覆盖率验证
python scripts/validate_cuda_graph_coverage.py \
    --profile_mode \
    --num_runs 100

# 预期:
# Before: 3.2 ms overhead (CPU dispatch + sync)
# After: 0.5 ms overhead (graph replay only)
# Saving: 2.7 ms
```

### 5.4 Phase 3 成功标准

- [ ] CPU Overhead: < 1.0 ms
- [ ] Graph Replay Time: < 0.5 ms
- [ ] 无内存泄漏: 1000 次推理内存稳定

---

## 六、Phase 4: TVM/Torch.compile Kernel Fusion

### 6.1 目标算子

```python
# Denoise 网络中的碎片算子:
# 1. AdaLN (Adaptive Layer Norm): scale, shift, silu, mul, add
# 2. Action Head: linear → silu → linear
# 3. Timestep Embedding: sin/cos → linear → silu → linear

# 问题: 每个操作一个 CUDA Kernel = 频繁的内存读写
# 目标: 融合成单个 Kernel
```

### 6.2 Torch.compile 方案

```python
# src/openpi/inference/fused_ops.py

import torch
from torch import nn

class FusedAdaLN(nn.Module):
    """融合的 Adaptive LayerNorm"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x, scale, shift):
        # 原始: 5 个 Kernel
        # norm → scale → shift → silu → mul

        # 融合后: 1 个 Kernel
        return (self.norm(x) * (1 + scale) + shift) * torch.sigmoid(x) * x

# 使用 torch.compile 自动融合
fused_adaln = torch.compile(
    FusedAdaLN(2048),
    backend="inductor",  # Triton codegen
    mode="max-autotune",  # 自动调优
    fullgraph=True,  # 完整图编译
)


class FusedDenoiseMLP(nn.Module):
    """融合 Denoise MLP: Linear + SiLU + Linear"""

    def __init__(self, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU: down(gate(x) * silu(up(x)))
        return self.down_proj(
            self.gate_proj(x) * torch.nn.functional.silu(self.up_proj(x))
        )

# Torch.compile 自动融合 SwiGLU
fused_mlp = torch.compile(
    FusedDenoiseMLP(1024, 4096),  # Action Expert
    backend="inductor",
    mode="reduce-overhead",
)
```

### 6.3 TVM 手动融合 (备选)

```python
# 如果 torch.compile 效果不佳，使用 TVM 手动融合

import tvm
from tvm import relay

def create_fused_adaln_tvm():
    """TVM 版本的融合 AdaLN"""
    # 定义计算图
    x = relay.var("x", shape=(1, 50, 2048), dtype="float16")
    scale = relay.var("scale", shape=(1, 50, 2048), dtype="float16")
    shift = relay.var("shift", shape=(1, 50, 2048), dtype="float16")

    # LayerNorm
    mean = relay.mean(x, axis=-1, keepdims=True)
    var = relay.variance(x, axis=-1, keepdims=True)
    norm = (x - mean) / relay.sqrt(var + 1e-6)

    # Scale + Shift + SiLU
    out = norm * (1 + scale) + shift
    silu = out * relay.sigmoid(out)

    # 编译
    func = relay.Function([x, scale, shift], silu)
    mod = tvm.IRModule.from_expr(func)

    # 针对 Thor 优化
    target = tvm.target.cuda(arch="sm_100")  # Thor = SM 10.0 / 11.0
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    return lib
```

### 6.4 Phase 4 验证

```bash
# Kernel 数量对比
python scripts/count_cuda_kernels.py \
    --mode before_fusion \
    --mode after_fusion

# 预期:
# Before: 47 kernels per denoise step
# After: 12 kernels per denoise step
# Kernel 减少: 74%
# 预期节省: 5-10 ms (10 steps)
```

### 6.5 Phase 4 成功标准

- [ ] Kernel 数量: 减少 50%+
- [ ] 单步延迟: 减少 0.5 ms
- [ ] 精度: bit-exact (无精度损失)

---

## 七、Phase 5: FP8 Accumulation 优化

### 7.1 问题诊断

当前 Thor TRT FP8 存在 Scale 被忽略的问题:
```
[DEQUANTIZE] [SCALE] has invalid precision FP8, ignored.
```

这导致 FP8 实际以 FP16 运行，无加速。

### 7.2 Triton FP8 方案

```python
# src/openpi/inference/triton_fp8_matmul.py

import triton
import triton.language as tl

@triton.jit
def fp8_matmul_kernel(
    A, B, C,
    A_scale, B_scale,  # 手动处理 scale
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    手写 FP8 MatMul，绕过 TRT scale bug

    关键: 使用 FP16 accumulation 保证精度
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in FP32 (关键!)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Load scales
    a_scale = tl.load(A_scale)  # per-tensor scale
    b_scale = tl.load(B_scale)
    combined_scale = a_scale * b_scale

    for k in range(0, K, BLOCK_K):
        # Load FP8 tiles
        a_ptrs = A + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = B + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=offs_m[:, None] < M)  # e4m3
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N)  # e4m3

        # Convert to FP16 for computation
        a_fp16 = a.to(tl.float16)
        b_fp16 = b.to(tl.float16)

        # Accumulate in FP32
        acc += tl.dot(a_fp16, b_fp16).to(tl.float32)

    # Apply scale and store as FP16
    acc = acc * combined_scale
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 7.3 Phase 5 验证

```bash
# FP8 精度验证
python scripts/validate_triton_fp8.py \
    --compare_with_fp16 \
    --compare_with_fp32 \
    --num_samples 1000

# 预期:
# FP8 vs FP16 MSE: < 0.001
# FP8 vs FP32 MSE: < 0.005
# 速度提升: 1.5-2x (内存带宽减半)
```

### 7.4 Phase 5 成功标准

- [ ] MSE vs FP16: < 0.001
- [ ] LIBERO 精度: ≥ 90%
- [ ] 延迟减少: ≥ 10 ms

---

## 八、Phase 6 (激进): NVFP4 MLP 量化

### 8.1 风险评估

**这是最高风险的优化**:
- NVIDIA TRT 在 Thor 上对 FP4 支持有 bug
- 需要完全手写 Triton Kernel
- 精度损失可能显著

### 8.2 Triton FP4 Dequantize Kernel

```python
# src/openpi/inference/triton_fp4_mlp.py

@triton.jit
def fp4_dequant_matmul_kernel(
    A,           # Input activation (FP16)
    W_packed,    # FP4 packed weights (每个 byte 存 2 个 FP4)
    W_scale,     # Scale per block
    C,           # Output
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,  # 量化块大小 (e.g., 128)
):
    """
    FP4 量化 MatMul:
    1. 读取 packed FP4 weights
    2. Dequantize to FP16
    3. 执行 MatMul
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load input (FP16)
        a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M)

        # Load packed FP4 weights (2 values per byte)
        w_packed_ptrs = W_packed + (offs_k[:, None] * N + offs_n[None, :]) // 2
        w_packed = tl.load(w_packed_ptrs)

        # Dequantize FP4 → FP16
        # FP4 E2M1: 4 exponent levels, 2 mantissa levels
        w_low = (w_packed & 0x0F).to(tl.float16)   # Low nibble
        w_high = ((w_packed >> 4) & 0x0F).to(tl.float16)  # High nibble

        # Apply scale
        scale_idx = offs_k[:, None] // QUANT_BLOCK
        scale = tl.load(W_scale + scale_idx * (N // QUANT_BLOCK) + offs_n[None, :] // QUANT_BLOCK)

        w_dequant_low = (w_low - 8) * scale   # Center around 0
        w_dequant_high = (w_high - 8) * scale

        # Interleave and matmul
        # ... (complex indexing)

        acc += tl.dot(a, w_dequant).to(tl.float32)

    # Store
    c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 8.3 Phase 6 验证

```bash
# FP4 精度验证 (严格)
python scripts/validate_fp4_mlp.py \
    --compare_with_fp16 \
    --num_samples 10000 \
    --layers all

# 预期:
# FP4 vs FP16 Cosine: ≥ 0.95
# LIBERO 精度: ≥ 85% (可接受的损失)
# 内存减少: 4x (BF16 → FP4)
# 速度提升: 2-3x
```

### 8.4 Phase 6 成功标准

- [ ] Cosine Similarity: ≥ 0.95
- [ ] LIBERO 精度: ≥ 85%
- [ ] 延迟减少: ≥ 20 ms

---

## 九、集成测试与最终验证

### 9.1 全链路测试脚本

```python
# scripts/triforce_full_benchmark.py

import torch
import time
from openpi.inference.triforce_engine import TriforceEngine

def benchmark_full_pipeline(engine, num_runs=100):
    """完整 Pipeline Benchmark"""

    # Warmup
    for _ in range(10):
        engine.infer(dummy_input)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = engine.infer(dummy_input)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p99_ms": np.percentile(latencies, 99),
        "hz": 1000 / np.mean(latencies),
    }


def run_libero_accuracy_test(engine, num_tasks=10, num_trials=10):
    """LIBERO 精度测试"""
    # ... 使用标准 LIBERO 评测流程
    pass


if __name__ == "__main__":
    # 测试所有配置
    configs = [
        ("baseline", BaselineEngine()),
        ("phase1_flashinfer", Phase1Engine()),
        ("phase2_triton_attn", Phase2Engine()),
        ("phase3_cuda_graph", Phase3Engine()),
        ("phase4_kernel_fusion", Phase4Engine()),
        ("phase5_fp8", Phase5Engine()),
        ("phase6_fp4", Phase6Engine()),
        ("triforce_full", TriforceEngine()),
    ]

    for name, engine in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        # Latency
        latency = benchmark_full_pipeline(engine)
        print(f"Latency: {latency['mean_ms']:.1f} ms ({latency['hz']:.1f} Hz)")

        # Accuracy
        accuracy = run_libero_accuracy_test(engine)
        print(f"Accuracy: {accuracy['success_rate']:.1f}%")
```

### 9.2 最终目标验证

| 指标 | Baseline | Triforce 目标 | 验收标准 |
|------|----------|---------------|----------|
| 总延迟 | 176.5 ms | ≤ 85 ms | 必须达成 |
| 推理频率 | 5.7 Hz | ≥ 11.8 Hz | 必须达成 |
| LIBERO 精度 | 100% | ≥ 90% | 必须达成 |
| 内存占用 | 12 GB | ≤ 10 GB | 建议达成 |
| 启动时间 | 120s | ≤ 60s | 建议达成 |

---

## 十、风险与回退策略

### 10.1 风险矩阵

| 风险 | 可能性 | 影响 | 缓解策略 |
|------|--------|------|----------|
| FlashInfer Thor 不兼容 | 40% | 高 | 使用 CUTLASS 重写 |
| CUDA Graph 捕获失败 | 20% | 中 | 保持部分图录制 |
| FP8 精度损失超预期 | 30% | 中 | 回退到 FP16 |
| FP4 完全不可用 | 60% | 低 | 跳过 Phase 6 |
| Triton Kernel 性能不如 TRT | 25% | 中 | 混合使用 |

### 10.2 回退方案

```
如果 Triforce 全方案失败，回退到:

1. 保守方案 (8-9 Hz):
   - Vision TRT (保持)
   - KV Cache TRT FP8 (保持)
   - Denoise CUDA Graph (保持)
   - 只优化: Kernel Fusion (Phase 4)

2. 最小改动方案 (6-7 Hz):
   - 只做 CUDA Graph 全图录制
   - 消除 Python overhead

3. 硬件升级方案:
   - 等待 NVIDIA TRT 10.15+ 修复 FP8/FP4 bug
   - 使用 TensorRT-LLM (如果支持 Thor)
```

---

## 十一、时间表

| 阶段 | 工作内容 | 预计时间 | 里程碑 |
|------|----------|----------|--------|
| **Week 1** | Phase 1: FlashInfer 集成 | 5 天 | KV Cache < 40 ms |
| **Week 2** | Phase 2: Triton Flash Attention | 5 天 | Denoise -15 ms |
| **Week 3** | Phase 3: CUDA Graph 全图 | 3 天 | Overhead < 1 ms |
| **Week 3** | Phase 4: Kernel Fusion | 2 天 | -10 ms |
| **Week 4** | Phase 5: FP8 优化 | 3 天 | 总体 < 100 ms |
| **Week 4** | Phase 6: FP4 探索 (可选) | 2 天 | 如可行，< 85 ms |
| **Week 5** | 集成测试 + 文档 | 5 天 | 发布 Triforce v1.0 |

**总计: 4-5 周**

---

## 十二、参考资料

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): Ragged Attention 库
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/): Triton 入门
- [TVM Documentation](https://tvm.apache.org/docs/): TVM 编译器
- [CUTLASS](https://github.com/NVIDIA/cutlass): CUDA 模板库
- [debug-08.md](./debug-08.md): TRT 优化记录
- [kv-cache-profill-report.md](./kv-cache-profill-report.md): KV Cache 分析
- [cliff_report.md](./cliff_report.md): 精度悬崖分析

---

## 附录 A: 目录结构

```
openpi/
├── src/openpi/inference/
│   ├── triforce_engine.py         # 主引擎
│   ├── flashinfer_kv_cache.py     # Phase 1
│   ├── triton_flash_attn.py       # Phase 2
│   ├── cuda_graph_full.py         # Phase 3
│   ├── fused_ops.py               # Phase 4
│   ├── triton_fp8_matmul.py       # Phase 5
│   └── triton_fp4_mlp.py          # Phase 6
├── scripts/
│   ├── validate_phase1_flashinfer.py
│   ├── validate_phase2_triton_attn.py
│   ├── validate_phase3_cuda_graph.py
│   ├── validate_phase4_fusion.py
│   ├── validate_phase5_fp8.py
│   ├── validate_phase6_fp4.py
│   └── triforce_full_benchmark.py
└── docs/
    └── thor-triforce-optimization-plan.md  # 本文档
```
