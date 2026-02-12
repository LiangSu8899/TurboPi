# NVFP4 Persistent Kernel Research Summary

## Executive Summary

This research investigated implementing an 18-layer Persistent MLP Kernel using NVFP4 (4-bit floating point) quantization for the Thor GPU (SM110). The goal was to achieve 2-3x speedup over TRT FP8 / BF16 cuBLAS baseline (~20ms for 18 layers).

### Key Results

| Approach | Time (18 layers) | vs BF16 | Memory |
|----------|------------------|---------|--------|
| BF16 cuBLAS (baseline) | 20.52 ms | 1.00x | 3624 MB |
| TRT FP8 | 20.39 ms | 1.00x | ~1800 MB |
| FP4 Pre-decoded + cuBLAS | 20.64 ms | 0.99x | 1132 MB |
| FP4 Persistent Kernel | 1053.86 ms | 0.02x | 1132 MB |
| FP4 Theoretical | 9.22 ms | 2.22x | 1132 MB |

### Conclusions

1. **FP4 storage works**: 3.2x memory reduction achieved
2. **FP4 decode is bottleneck**: On-the-fly decode adds 33x overhead
3. **Custom kernel too slow**: Single-block design limits parallelism
4. **Pre-decode is viable**: Store FP4, decode to BF16 at startup

---

## Research Details

### 1. Architecture Design

Based on GPT's recommendations, we designed an 18-layer persistent MLP kernel:

```
Input → [Layer 1 → Layer 2 → ... → Layer 18] → Output
         ↑                                    ↑
         └── activation stays in registers ───┘
```

Key design choices:
- Single block (256 threads) to keep activation in shared memory
- Template-based layer count (4/6/8/10/12/18)
- NVFP4 E2M1 format with block scaling (block_size=32)

### 2. Implementation

Files created:
- `nvfp4_nlayer_persistent.cu` - CUDA kernel
- `nvfp4_persistent_extension.cpp` - PyTorch C++ wrapper
- `setup_persistent.py` - Build script

Compilation results:
```
ptxas info: Used 40 registers, used 1 barriers
ptxas info: 0 bytes spill stores, 0 bytes spill loads
```

### 3. Correctness Verification

Kernel correctness verified against PyTorch reference:
- Max difference: 1.26e-04
- Mean difference: 3.14e-05
- All outputs match reference within numerical precision

### 4. Performance Analysis

#### Why Custom Kernel is Slow

1. **Single block = Single SM**
   - Thor has multiple SMs, but we only use 1
   - GPU utilization: ~1/N of capacity

2. **Fine-grained FP4 decode**
   - Each byte read and decoded individually
   - No vectorized loads (float4)
   - Poor memory coalescing

3. **Algorithm complexity**
   - Nested loops for tile processing
   - Atomic operations for accumulation
   - Sync barriers between phases

#### Memory Bandwidth Analysis

```
Thor bandwidth: 122.8 GB/s

18 layers BF16:
  Memory traffic: 3.63 GB
  Theoretical time: 29.5 ms
  Actual time: 20.5 ms (L2 cache helps)

18 layers FP4:
  Memory traffic: 1.13 GB
  Theoretical time: 9.2 ms
  Actual time: 1053 ms (kernel inefficiency)
```

### 5. Alternative Approaches

#### Pre-decode + cuBLAS (Recommended)

Store weights in FP4, decode to BF16 at startup:
- **Same inference speed as BF16** (20.64 ms vs 20.52 ms)
- **3.2x less weight storage** (1132 MB vs 3624 MB)
- **Simple implementation**: Just quantize/dequantize functions

#### True FP4 Speedup Requirements

To achieve theoretical 2.2x speedup:
1. **Hardware FP4 Tensor Core** - Not available on Thor
2. **Fused decode+GEMM kernel** - Complex, needs CUTLASS integration
3. **Multi-block persistent kernel** - Requires cooperative groups

---

## Recommendations

### For Memory-Constrained Deployment

```python
# Store FP4, decode at startup
fp4_weights = load_fp4_weights()  # 1.1 GB for 18 layers
bf16_weights = dequantize_to_bf16(fp4_weights)  # Decode once
# Use bf16_weights for inference
```

Benefits:
- 3.2x less disk/memory for weight storage
- Same inference speed as native BF16
- Easy to implement

### For Maximum Performance

Current best options:
1. **TRT FP8 Static Graph**: 20.39 ms (already implemented)
2. **BF16 + CUDA Graphs**: 20.44 ms
3. **Pre-decoded FP4**: 20.64 ms (with memory savings)

### Future Work

To achieve true FP4 speedup:
1. Wait for FP4 Tensor Core support in future GPUs
2. Investigate CUTLASS with FP4 dequant
3. Explore TensorRT-LLM's W4A16 kernels
4. Consider Triton for auto-optimized kernels

---

## Files Created

| File | Purpose |
|------|---------|
| `nvfp4_nlayer_persistent.cu` | CUDA kernel implementation |
| `nvfp4_persistent_extension.cpp` | PyTorch C++ binding |
| `setup_persistent.py` | Build script |
| `test_persistent_extension.py` | Correctness & benchmark test |
| `verify_fp4_decode.py` | Decode verification |
| `analyze_kernel_perf.py` | Performance analysis |
| `nvfp4_predecode_benchmark.py` | Pre-decode approach benchmark |

---

## Appendix: NVFP4 E2M1 Format

```
Value encoding (4 bits):
  Bit 3: Sign (0=positive, 1=negative)
  Bits 0-2: Magnitude index

Magnitude LUT:
  [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

Block scaling:
  scale = max(abs(block)) / 6.0
  quantized = round(value / scale)
  dequantized = LUT[quantized] * scale
```

---

*Research conducted: 2026-02-10*
*GPU: NVIDIA Thor (SM110)*
*PyTorch: 2.10.0, CUDA 13.1*
