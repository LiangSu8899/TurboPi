# W4A16 Packed FP4 Kernel 开发进展

**Date:** 2026-02-10

## 1. 核心成果

成功实现了 TVM 生成的 W4A16 Packed FP4 GEMV kernel，在 Thor SM110 上实现了显著加速。

### 1.1 性能数据

| GEMM | Dimensions (N×K) | W4A16 Fast | TRT FP8 | Speedup | 正确性 |
|------|------------------|------------|---------|---------|--------|
| gate_proj | 16384×2048 | **0.224ms** | 0.53ms | **2.37x** | ✅ cos=1.0 |
| up_proj | 16384×2048 | **0.224ms** | 0.53ms | **2.37x** | ✅ cos=1.0 |
| down_proj | 2048×16384 | **0.202ms** | 0.53ms | **2.62x** | ✅ cos=1.0 |
| **MLP Total** | - | **0.65ms** | 1.59ms | **2.45x** | ✅ |

### 1.2 18层 MLP 预期收益

| 配置 | 单层 MLP | 18层总计 | 相对加速 |
|------|----------|----------|----------|
| TRT FP8/BF16 (实测) | ~1.13ms | ~20.4ms | 1.00x |
| **W4A16 Packed (预期)** | ~0.65ms | **~11.7ms** | **1.74x** |

### 1.3 内存节省

| 指标 | BF16 | W4A16 Packed | 压缩比 |
|------|------|--------------|--------|
| 单层权重 | 134 MB | **17 MB** | **8x** |
| 18层总计 | 2.4 GB | **0.3 GB** | **8x** |

## 2. 技术实现

### 2.1 Kernel 设计

```
W4A16 Packed FP4 GEMV:
- 权重: uint8 packed (2 FP4 values per byte)
- 激活: float32
- 计算: In-register dequant + CUDA Core accumulation
- Reduction: Shared memory parallel reduction

Thread Block Organization:
- 256 threads per block
- 4 outputs per block (64 threads per output)
- K-dimension tiling for large K (TILE_K = 2048)
```

### 2.2 nvFP4 E2M1 格式

```
4-bit encoding: [sign][exp1][exp0][mantissa]

LUT values:
  0x0-0x7: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
  0x8-0xF: [0, -0.5, -1, -1.5, -2, -3, -4, -6]

Block scaling: scale_per_32_elements
```

### 2.3 关键优化

1. **Packed 存储**: 8x 内存带宽节省
2. **Shared Memory LUT**: 16 entries for fast dequant lookup
3. **K-dimension Tiling**: 处理大 K 值 (16384)
4. **Parallel Reduction**: 6-step log2 reduction (64→1)
5. **Activation Caching**: A tile in shared memory

## 3. 文件位置

| 文件 | 用途 |
|------|------|
| `src/openpi/models_pytorch/tvm_kernels/w4a16_packed_gemm.py` | TVM kernel 实现 |
| `tvm_trt_plugin/w4a16_mlp/w4a16_packed_gemv.cu` | gate/up_proj CUDA 源码 |
| `tvm_trt_plugin/w4a16_mlp/w4a16_down_proj.cu` | down_proj CUDA 源码 |

## 4. 下一步计划

### 4.1 TRT Plugin 集成 (进行中)
- [ ] 封装为 IPluginV3
- [ ] 支持静态 shape
- [ ] 集成到推理 pipeline

### 4.2 Fusion 优化
- [ ] gate_proj + up_proj fusion (减少 A 加载)
- [ ] SiLU * mul fusion
- [ ] Multi-layer persistent kernel (可选)

### 4.3 端到端验证
- [ ] 全模型精度验证 (cos > 0.99)
- [ ] LIBERO 任务成功率验证
- [ ] 端到端延迟测试

## 5. 预期最终收益

| 阶段 | KV Cache MLP | 总 Pipeline | Hz |
|------|--------------|-------------|-----|
| 当前 (TRT FP8) | 20.4ms | 83.5ms | 12.0 |
| **W4A16 (预期)** | **11.7ms** | **~75ms** | **~13.3** |

*注: 实际收益需要端到端验证确认*

---

## 验证命令

```bash
# 运行 benchmark
source /home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/venv/bin/activate
cd /home/heima-thor/suliang/Turbo-Pi/openpi

# gate/up_proj 测试
python src/openpi/models_pytorch/tvm_kernels/w4a16_packed_gemm.py --M 1 --N 16384 --K 2048

# down_proj 测试
python src/openpi/models_pytorch/tvm_kernels/w4a16_packed_gemm.py --M 1 --N 2048 --K 16384
```
