# NVFP4 Packed GEMV Plugin

高性能 4-bit 量化 GEMV 实现，针对 Thor (SM110) 优化。

## 性能

| Kernel | 延迟 | vs TRT FP8 |
|--------|------|------------|
| TRT FP8 Baseline | 0.53 ms | 1.0x |
| **NVFP4 Packed Warp Reduce** | **0.36 ms** | **1.46x** |
| NVFP4 Packed + Bias + GELU | 0.37 ms | 1.43x |

## 关键优化

1. **真正的 4-bit Packed 格式**
   - 使用 `uint8` 存储 2 个 FP4 值
   - 内存读取量从 36MB 降到 4.5MB (8x 节省)

2. **Warp-level Reduction**
   - 32 个线程协作计算一个输出元素
   - K 维度在线程间并行，最后用 `__shfl_down_sync` 规约

3. **算子融合**
   - GEMV + Bias + GELU 融合在一个 kernel
   - 避免中间结果写回显存

## 目录结构

```
nvfp4_packed_plugin/
├── src/
│   ├── nvfp4_packed_kernel.cu    # CUDA kernel 实现
│   ├── nvfp4_packed_plugin.h     # TRT Plugin 头文件
│   └── nvfp4_packed_plugin.cpp   # TRT Plugin 实现
├── tests/
│   ├── benchmark_kernel.cu       # 独立 kernel benchmark
│   └── test_nvfp4_packed.cu      # TRT 集成测试
├── scripts/
│   └── prepare_weights.py        # 权重预处理脚本
├── python/
│   └── nvfp4_packed.py           # PyTorch 集成
└── CMakeLists.txt
```

## 快速开始

### 1. 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89;90;100;110"
make -j$(nproc)
```

### 2. 运行 Benchmark

```bash
# 独立 kernel benchmark (无 TRT 依赖)
./benchmark_nvfp4_kernel 3072 3072 50 200

# TRT 集成测试
./test_nvfp4_packed
```

### 3. 预处理权重

```bash
python scripts/prepare_weights.py \
    --input model.safetensors \
    --output weights_packed.npz \
    --block-size 32
```

### 4. Python 集成

```python
from nvfp4_packed_plugin.python.nvfp4_packed import (
    NVFP4PackedLinear,
    replace_mlp_with_nvfp4_packed
)

# 替换模型 MLP 层
replace_mlp_with_nvfp4_packed(model)

# 推理
output = model(input)
```

## TRT Plugin 使用

### 输入张量

| Index | Name | Shape | Type | Description |
|-------|------|-------|------|-------------|
| 0 | activation | [M, K] | FP32/BF16 | 输入激活 |
| 1 | weight_packed | [N, K/2] | INT8 | Packed FP4 权重 |
| 2 | scale_A | [M, K/32] | FP32/BF16 | 激活 scale |
| 3 | scale_W | [N, K/32] | FP32/BF16 | 权重 scale |
| 4 | bias (可选) | [N] | FP32/BF16 | Bias |

### 输出张量

| Index | Name | Shape | Type |
|-------|------|-------|------|
| 0 | output | [M, N] | FP32/BF16 |

### Plugin 参数

| Name | Type | Description |
|------|------|-------------|
| in_features | int32 | K 维度 |
| out_features | int32 | N 维度 |
| activation_type | int32 | 0=None, 1=GELU, 2=SiLU |
| has_bias | int32 | 0=无, 1=有 |

## NVFP4 E2M1 编码

```
4-bit encoding: [sign(1)] [magnitude_index(3)]

magnitude_index -> value:
  0 -> 0.0
  1 -> 0.5
  2 -> 1.0
  3 -> 1.5
  4 -> 2.0
  5 -> 3.0
  6 -> 4.0
  7 -> 6.0

Packed format (uint8):
  byte = low_nibble | (high_nibble << 4)
  - low_nibble:  even index element
  - high_nibble: odd index element
```

## Block Scaling

- Block size: 32 elements
- Scale factor: FP32 (可选 FP8)
- Scale 计算: `scale = max(abs(block)) / 6.0`

## 下一步

1. **集成到 Pi0 推理流程**
   - 替换 PaliGemma MLP 层
   - 测量整体 Hz 提升

2. **进一步优化**
   - 更激进的向量化 (uint128)
   - Persistent kernel
   - Batched GEMV

3. **扩展量化方案**
   - W4A8: 权重 FP4, 激活 FP8
   - W4A16: 权重 FP4, 激活 FP16

## 参考

- [NVFP4 Optimization Results](../../docs/nvfp4-optimization-results.md)
- [TVM-TRT FP4 Optimization Plan](../../docs/tvm-trt-fp4-opt-plan.md)
