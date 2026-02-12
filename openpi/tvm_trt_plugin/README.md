# TVM → TensorRT Plugin Integration

This directory contains TVM-generated CUDA kernels packaged as TensorRT plugins for Jetson Thor (SM110).

## Why TVM?

1. **CUTLASS Compatibility Issues**: CUTLASS examples for nvFP4 (e.g., Example 79) use `mxf8f6f4` instructions that may not work on SM110
2. **Absolute Control**: TVM TensorIR gives you precise control over the generated code
3. **No External Dependencies**: Pure CUDA kernels without CUTLASS or special PTX instructions
4. **TensorRT Integration**: Kernels can be wrapped as TensorRT plugins for graph fusion

## Workflow

```
┌─────────────────────┐
│  TVM TensorIR       │  Python: Define kernel using @T.prim_func
│  nvfp4_gemm.py      │
└─────────┬───────────┘
          │
          ▼ tvm.build() + inspect_source()
┌─────────────────────┐
│  CUDA Source Code   │  Pure CUDA, no special instructions
│  nvfp4_gemm.cu      │
└─────────┬───────────┘
          │
          ▼ CMake + nvcc
┌─────────────────────┐
│  TensorRT Plugin    │  IPluginV3 interface
│  libnvfp4_tvm.so    │
└─────────┬───────────┘
          │
          ▼ TensorRT engine
┌─────────────────────┐
│  Inference Graph    │  Fused execution
│  Pi0.5 VLA Model    │
└─────────────────────┘
```

## Directory Structure

```
tvm_trt_plugin/
├── README.md
├── nvfp4_gemm/                    # nvFP4 GEMM kernel
│   ├── CMakeLists.txt             # Build configuration
│   ├── nvfp4_gemm_kernel.cu       # TVM-generated CUDA kernel
│   ├── nvfp4_gemm_launcher.cu     # Kernel launcher
│   ├── nvfp4_gemm_tvm_plugin.h    # Plugin header
│   ├── nvfp4_tvm_plugin.cpp       # Plugin implementation
│   └── test_nvfp4_tvm.cu          # Test/benchmark
```

## Quick Start

### 1. Generate CUDA Kernel from TVM

```bash
# Set up TVM environment
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export LD_LIBRARY_PATH=$TVM_HOME/build:$LD_LIBRARY_PATH

# Generate kernel for your dimensions
python openpi/src/openpi/models_pytorch/tvm_kernels/tvm_to_trt_plugin.py \
    --kernel nvfp4_gemm \
    --M 1 \
    --N 3072 \
    --K 3072 \
    --output openpi/tvm_trt_plugin/nvfp4_gemm
```

### 2. Build TensorRT Plugin

```bash
cd openpi/tvm_trt_plugin/nvfp4_gemm
mkdir build && cd build

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=110 \
    -DTENSORRT_ROOT=/usr/local/tensorrt \
    -DBUILD_TESTS=ON

make -j$(nproc)
```

### 3. Test Kernel

```bash
./test_nvfp4_tvm 1 3072 3072
```

### 4. Use in TensorRT

```python
import tensorrt as trt
import ctypes

# Load plugin library
ctypes.CDLL("libnvfp4_tvm_plugin.so")

# Create plugin
registry = trt.get_plugin_registry()
creator = registry.get_creator("NVFP4GemmTVMPlugin", "1", "")

plugin_fields = trt.PluginFieldCollection([
    trt.PluginField("in_features", np.array([3072], dtype=np.int32), trt.PluginFieldType.INT32),
    trt.PluginField("out_features", np.array([3072], dtype=np.int32), trt.PluginFieldType.INT32),
    trt.PluginField("block_size", np.array([32], dtype=np.int32), trt.PluginFieldType.INT32),
])

plugin = creator.create_plugin("nvfp4_gemm", plugin_fields, trt.TensorRTPhase.kRUNTIME)

# Add to network
network.add_plugin_v3(
    [activation, weight, scale_A, scale_W],
    [],
    plugin
)
```

## Performance Notes

### Current Implementation

The current TVM-generated kernel is a **naive implementation**:
- One thread per output element
- No shared memory tiling
- No Tensor Core usage
- Sequential K-dimension accumulation

This is intentionally simple to ensure correctness and SM110 compatibility.

### Optimization Path

For higher performance, consider:

1. **TVM Auto-Scheduler**: Use `tvm.auto_scheduler` to find optimized schedules
2. **Manual Tiling**: Add shared memory tiling in TensorIR
3. **Vectorized Loads**: Use float4 loads for better memory bandwidth
4. **Loop Unrolling**: Unroll the K-loop for instruction-level parallelism

Example optimized kernel (future work):
```python
@T.prim_func
def nvfp4_gemm_tiled(A, W, scale_A, scale_W, C):
    # Tiled implementation with shared memory
    A_shared = T.alloc_buffer((TILE_SIZE, TILE_SIZE), "float32", scope="shared")
    W_shared = T.alloc_buffer((TILE_SIZE, TILE_SIZE), "float32", scope="shared")

    for bx in T.thread_binding(...):
        for k_tile in T.serial(K // TILE_SIZE):
            # Load tiles to shared memory
            ...
            T.tvm_storage_sync("shared")
            # Compute partial sums
            ...
```

## Advantages Over CUTLASS

| Aspect | CUTLASS | TVM |
|--------|---------|-----|
| SM110 Compatibility | ❌ mxf8f6f4 issues | ✅ Pure CUDA |
| Control | Limited templates | ✅ Full TensorIR |
| Build Complexity | High (template metaprogramming) | ✅ Simple CMake |
| Debugging | Difficult | ✅ Clear CUDA source |
| Auto-tuning | Manual | ✅ Auto-scheduler |

## Integration with Pi0.5 Pipeline

To integrate with the existing inference pipeline:

1. **Replace MLP Layers**: Swap torch.nn.Linear with TVM plugin calls
2. **Weight Conversion**: Pre-quantize weights to nvFP4 format
3. **TensorRT Graph**: Build engine with custom plugins

```python
# In openpi/src/openpi/inference/trt_pipeline.py
from openpi.tvm_trt_plugin import load_nvfp4_plugin

# Register plugin
load_nvfp4_plugin()

# Build engine with quantized weights
engine = build_engine_with_nvfp4_gemm(model, quantized_weights)
```

## Troubleshooting

### "Module has no function 'get_source'"
TVM 0.24+ changed API. Use `mod.imports_[0].inspect_source()` instead.

### Kernel produces zeros
Check that you're accumulating to output buffer, not local variable:
```python
# Wrong (TVM 0.24 bug)
acc = T.float32(0)
for k in T.serial(K):
    acc = acc + ...
C[i, j] = acc

# Correct
C[i, j] = T.float32(0)
for k in T.serial(K):
    C[i, j] = C[i, j] + ...
```

### Build errors with TensorRT
Ensure TensorRT 10.x is installed and TENSORRT_ROOT is set correctly.

## References

- [TVM TensorIR Documentation](https://tvm.apache.org/docs/tutorial/tensor_ir_blitz_course.html)
- [TensorRT IPluginV3 Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html)
- [TPAT: TensorRT Plugin Autogen Tool](https://github.com/Tencent/TPAT)
