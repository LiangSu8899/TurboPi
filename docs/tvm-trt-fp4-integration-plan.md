# TVM + TRT FP4 集成方案

## 目标

通过TVM编译绕过Thor SM110官方支持限制，实现FP4加速。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    TensorRT 推理图                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Attention (TRT FP8)  →  MLP (TVM FP4 Plugin)  →  ...       ││
│  │                              ↑                              ││
│  │                    TVM编译的W4A16 kernel                    ││
│  │                    (绕过SM110限制)                          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 实现步骤

### Phase 1: 验证TVM Kernel性能

**目标**: 确认TVM生成的kernel能达到理论性能的50%以上

```bash
# 运行现有优化kernel的benchmark
cd openpi/src/openpi/models_pytorch/tvm_kernels
python nvfp4_gemm_thor_optimized.py --M 1 --N 16384 --K 2048
```

**预期结果**:
- BF16 cuBLAS baseline: ~1.13ms (单层gate+up)
- FP4 理论: ~0.35ms (3.2x内存减少)
- TVM kernel目标: <0.7ms (50%效率)

### Phase 2: 导出TRT Plugin

```bash
# 生成MLP维度的kernel
python tvm_to_trt_plugin.py \
    --kernel w4a16_gemm \
    --M 1 \
    --N 16384 \
    --K 2048 \
    --output openpi/tvm_trt_plugin/mlp_gate

python tvm_to_trt_plugin.py \
    --kernel w4a16_gemm \
    --M 1 \
    --N 2048 \
    --K 16384 \
    --output openpi/tvm_trt_plugin/mlp_down
```

### Phase 3: 构建TRT Plugin

```bash
cd openpi/tvm_trt_plugin/mlp_gate
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=110
make -j$(nproc)
```

### Phase 4: 集成到推理Pipeline

```python
# 在 openpi/src/openpi/inference/trt_pipeline.py 中
from openpi.tvm_trt_plugin import load_w4a16_plugin

# 注册plugin
load_w4a16_plugin("libnvfp4_mlp_gate.so")
load_w4a16_plugin("libnvfp4_mlp_down.so")

# 构建TRT engine时使用plugin
def build_engine_with_fp4_mlp(model, fp4_weights):
    # 替换Linear层为FP4 Plugin
    ...
```

## 性能预期

| 配置 | 18层MLP时间 | vs BF16 |
|------|-------------|---------|
| BF16 cuBLAS | 20.5ms | 1.00x |
| TRT FP8 | 20.4ms | 1.00x |
| TVM FP4 (50%效率) | ~13.7ms | 1.50x |
| TVM FP4 (理论) | ~6.4ms | 3.20x |

## 优化路径

### Level 1: 基础实现 (当前)
- ✅ Shared memory tiling
- ✅ Register accumulation
- ✅ Loop unrolling

### Level 2: 进阶优化
- [ ] 向量化load (float4)
- [ ] Double buffering
- [ ] Warp specialization

### Level 3: Tensor Core
- [ ] 检查Thor SM110是否支持FP4 MMA指令
- [ ] 如果支持，使用TVM的tensorcore primitives

## 验证清单

- [ ] TVM kernel正确性验证
- [ ] TVM kernel性能benchmark
- [ ] TRT Plugin编译通过
- [ ] TRT Plugin集成测试
- [ ] 端到端推理性能测试
- [ ] 模型精度验证

## 参考

- TensorRT-LLM W4A16: https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv
- TVM Tensor Core: https://tvm.apache.org/docs/how_to/work_with_tensor_core.html
- CUTLASS W4A16: CUTLASS Example 79

## 风险

1. **Thor SM110可能不支持某些指令**
   - 缓解: TVM生成纯CUDA代码，避免特殊PTX

2. **TVM kernel效率可能不高**
   - 缓解: 使用auto_scheduler自动调优

3. **TRT Plugin集成复杂度**
   - 缓解: 已有tvm_to_trt_plugin.py自动生成wrapper
