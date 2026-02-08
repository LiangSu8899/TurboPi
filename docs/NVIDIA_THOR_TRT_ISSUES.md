# NVIDIA Jetson Thor TensorRT Issues Report

## Environment

| Component | Version |
|-----------|---------|
| Platform | NVIDIA Jetson Thor |
| GPU Architecture | Blackwell (SM 11.0) |
| CUDA Compute Cap | 11.0 |
| Driver Version | 580.00 |
| JetPack | R38.2.1 |
| TensorRT (system) | 10.13.3.9 |
| TensorRT (Docker) | 10.14.1.48 |
| torch_tensorrt | 2.10.0a0 |
| CUDA | 13.0 (inferred) |

## Issue 1: Myelin Kernel Crash on Thor

### Symptom
```
CUDA error 400 launching __myl_CastMulMeanAddSqrtDivMulMul kernel
```

### Description
When building TensorRT engines with optimization level >= 3, the Myelin optimizer creates fused kernels (Cast+Mul+Mean+Add+Sqrt+Div+Mul+Mul) that crash at runtime on Thor GPU. This pattern matches RMSNorm operations in transformer models.

### Reproduction
1. Build a TRT engine with 18-layer transformer (GQA attention + MLP)
2. Use `builder_optimization_level >= 3`
3. Run inference - crashes during first layer's RMSNorm

### Attempted Workarounds (All Failed)

| Workaround | Result |
|------------|--------|
| `opt_level=0` | Works but 654ms latency (6x slower than PyTorch) |
| `opt_level=1` | Works but still slow (~400ms) |
| `opt_level=2` | Works, ~58ms, but Myelin still causes issues |
| `opt_level>=3` | Crashes with CUDA error 400 |
| `TRT_DISABLE_MYELIN=1` env var | No effect (env var not recognized) |
| Exclude Myelin from tactic_sources | API doesn't expose Myelin control |
| Pure FP16 pipeline (no Cast ops) | Myelin still creates fused kernels internally |
| Custom TurboAttention plugin | Works for attention but RMSNorm still fused by Myelin |

### Impact
- Cannot achieve optimal TRT performance on Thor
- PyTorch baseline (109ms) faster than TRT with safe settings (654ms @ opt_level=0)
- Current best TRT: ~58ms @ opt_level=2 (with occasional instability)

### Suspected Root Cause
Thor/Blackwell-specific TensorRT bug. The same ONNX model + TRT settings work on:
- A100 (Ampere)
- H100 (Hopper)
- Orin (Ampere-based Jetson)

But fails on Thor (first Blackwell-based Jetson).

### Related NVIDIA GitHub Issues
- [#4590](https://github.com/NVIDIA/TensorRT/issues/4590): FP8/FP4 silently falls back to FP32 on Blackwell
- [#4599](https://github.com/NVIDIA/TensorRT/issues/4599): ViT FP8 low performance on Blackwell

---

## Issue 2: TRT IAttention API Status

### Update (2026-02-02)
Docker container (nvcr.io/nvidia/pytorch:25.12-py3) has TensorRT **10.14.1.48**, which includes the IAttention API.

### Available API (TRT 10.14.1+)
```python
# This API should bypass Myelin pattern matching
attention_layer = network.add_attention(
    query=query, key=key, value=value,
    norm_op=trt.AttentionNormalizationOp.SOFTMAX,
    causal=False
)
```

### Testing Status (2026-02-03)
- [x] Test if IAttention API works on Thor - **WORKS (single layer only)**
- [x] Test if it bypasses Myelin crash - **YES**
- [x] Benchmark performance vs PyTorch - **0.34ms per attention layer**

### Critical Limitation
**Only ONE IAttention layer allowed per network!**
```
Error: mScopedOps.size() == mGraph.scopedOps.size()
```
This makes IAttention unsuitable for multi-layer transformers.

### Key Findings
1. Requires `STRONGLY_TYPED` network flag
2. Uses scale=1 by default (must manually scale Q by 1/sqrt(head_dim))
3. Works perfectly for single-layer attention

---

## Issue 3: Torch-TensorRT Status

### Testing Results (2026-02-03)
**STATUS: WORKS (Recommended approach)**

| Metric | Result |
|--------|--------|
| Availability | torch_tensorrt 2.10.0a0 in Docker |
| SDPA conversion | SUCCESS |
| Cosine similarity | 0.999999 |
| Speedup | 1.45x |

### Test Results (4-layer transformer)
- PyTorch latency: 19.0 ms
- TensorRT latency: 13.5 ms
- Compilation time: ~10-27s

---

## Issue 4: TensorRT-LLM Jetson Support

### Status
TensorRT-LLM has `v0.12.0-jetson` branch with Jetson-specific optimizations.

### Questions
1. Does v0.12.0-jetson support Thor (Blackwell)?
2. Can we extract and use GPT attention kernels standalone?
3. What's the integration complexity?

---

## Recommended Actions

### Immediate (Validated)
1. **Use Torch-TensorRT for KV Cache optimization**
   - Already validated on Thor
   - 1.45x speedup on 4-layer test
   - Expected 1.5-2x speedup on full 18-layer model
   ```bash
   python scripts/build_torch_trt_kv_cache.py
   ```

### Short Term
1. Integrate Torch-TensorRT compiled model into `unified_policy.py`
2. Create `TorchTRTBackend` for seamless switching

### Medium Term
1. Wait for TRT update allowing multiple IAttention layers
2. Report IAttention limitation to NVIDIA

### Long Term
1. Integrate TensorRT-LLM when Thor support improves
2. Consider custom CUDA kernels for maximum performance

---

## Benchmark Summary

| Backend | Component | Latency | Notes |
|---------|-----------|---------|-------|
| PyTorch | Full Inference | 183.7 ms | Baseline |
| PyTorch | KV Cache (18 layers) | 87.2 ms | - |
| Torch-TRT | KV Cache (4 layers) | 13.5 ms | 1.45x faster |
| TRT IAttention | Single Attention | 0.34 ms | Best but single-layer only |

---

## Contact Information

**For NVIDIA Bug Report:**
- Platform: Jetson Thor
- JetPack: R38.2.1
- TensorRT (Docker): 10.14.1.48
- torch_tensorrt: 2.10.0a0
- Model: PaliGemma-based VLA (pi0.5)
- Workload: 18-layer transformer, GQA (8 heads, 1 KV head), seq_len=970

**Issues to Report:**
1. IAttention single-layer limit (Error: mScopedOps.size() check)
2. IAttention undocumented scale=1 default
3. Myelin crash on Thor with RMSNorm fusion

---

*Last Updated: 2026-02-03*
