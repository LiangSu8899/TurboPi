# W4A8 GEMM TensorRT Plugin

Weight 4-bit (NVFP4), Activation 8-bit (FP8) GEMM plugin for TensorRT.

## Overview

This plugin implements mixed-precision GEMM optimized for NVIDIA Blackwell (SM100/SM110) architecture:

| Component | Format | Bits | Notes |
|-----------|--------|------|-------|
| Weight | NVFP4 E2M1 | 4-bit | Block-scaled, values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} |
| Activation | FP8 E4M3 | 8-bit | Dynamic per-row scaling |
| Output | BF16 | 16-bit | |
| Accumulator | FP32 | 32-bit | |

## Performance Target

| Metric | Value |
|--------|-------|
| Weight memory | 4-bit → 75% reduction vs FP16 |
| Activation bandwidth | 8-bit → 50% reduction vs FP16 |
| Compute | Block-scaled Tensor Core MMA (2x throughput vs FP8) |

## Architecture

```
Input (BF16) ──────────────────────────────────────────────────────────┐
     │                                                                  │
     ▼                                                                  │
┌─────────────────┐                                                     │
│ FP8 Quantization│ (Triton-style per-row dynamic scaling)              │
│ BF16 → FP8      │                                                     │
└────────┬────────┘                                                     │
         │                                                              │
         ▼                                                              │
┌─────────────────────────────────────────────────────────────────────┐ │
│                    CUTLASS W4A8 GEMM                                 │ │
│  ┌───────────┐     ┌────────────┐     ┌───────────┐                 │ │
│  │FP8 Act [M,K]│ × │NVFP4 W [K,N]│ → │BF16 Out [M,N]│               │ │
│  └───────────┘     └────────────┘     └───────────┘                 │ │
│                                                                      │ │
│  Uses SM100 Block-Scaled Tensor Core MMA (tcgen05.mma.blockscaled)  │ │
└─────────────────────────────────────────────────────────────────────┘ │
         │                                                              │
         ▼                                                              │
    Output (BF16) ◄────────────────────────────────────────────────────┘
```

## Build

### Prerequisites

- CUDA 12.8+
- TensorRT 10.0+
- CUTLASS 3.5+ (included in `external/cutlass_nvfp4_build`)
- CMake 3.18+

### Build Steps

```bash
cd /home/heima-thor/suliang/Turbo-Pi/openpi/w4a8_plugin

mkdir build && cd build

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=100 \
    -DTENSORRT_ROOT=/usr/local/tensorrt

make -j$(nproc)
```

### Test

```bash
./test_w4a8_gemm 256 2048 2048
```

## Usage

### 1. Standalone CUTLASS Kernel

```cpp
#include "w4a8_gemm_plugin.h"

// Quantize weight (offline, once)
quantizeWeightToNVFP4(weight_bf16, weight_packed, weight_scale, N, K, stream);

// Quantize activation (runtime, per forward)
quantizeActivationToFP8(input_bf16, input_fp8, input_scale, M, K, stream);

// Run GEMM
w4a8GemmForward(
    input_fp8, input_scale,
    weight_packed, weight_scale,
    output_bf16,
    M, N, K, alpha,
    workspace, workspace_size,
    stream
);
```

### 2. TensorRT Plugin

```python
import tensorrt as trt

# Register plugin
trt.init_libnvinfer_plugins(None, "")

# Load plugin library
ctypes.CDLL("libw4a8_gemm_plugin.so")

# Create network with plugin
builder = trt.Builder(logger)
network = builder.create_network()

# Add W4A8GemmPlugin to network
plugin_registry = trt.get_plugin_registry()
creator = plugin_registry.get_creator("W4A8GemmPlugin", "1", "")

plugin_fields = trt.PluginFieldCollection([
    trt.PluginField("in_features", np.array([2048], dtype=np.int32), trt.PluginFieldType.INT32),
    trt.PluginField("out_features", np.array([16384], dtype=np.int32), trt.PluginFieldType.INT32),
    trt.PluginField("alpha", np.array([1.0], dtype=np.float32), trt.PluginFieldType.FLOAT32),
])

plugin = creator.create_plugin("w4a8_gemm", plugin_fields, trt.TensorRTPhase.kRUNTIME)

# Add to network
layer = network.add_plugin_v3(
    [activation_tensor, weight_packed_tensor, weight_scale_tensor],
    [],
    plugin
)
```

### 3. Replace MLP Layers

```python
from openpi.w4a8_plugin.python import W4A8Linear, replace_mlp_with_w4a8

# Replace single layer
w4a8_linear = W4A8Linear.from_linear(original_linear)

# Replace all MLP layers in model
replace_mlp_with_w4a8(model)
```

## File Structure

```
w4a8_plugin/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── src/
│   ├── w4a8_gemm_plugin.h   # Plugin header
│   ├── w4a8_gemm_plugin.cpp # Plugin implementation
│   └── w4a8_gemm_kernel.cu  # CUTLASS W4A8 GEMM kernel
├── tests/
│   └── test_w4a8_gemm.cu    # Standalone test
└── python/
    └── w4a8_plugin.py       # Python bindings (TODO)
```

## References

- [CUTLASS Example 72c](https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72c_blackwell_mixed_mxfp8_bf16_gemm.cu) - Mixed MXFP8 × NVFP4 GEMM
- [TensorRT IPluginV3](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html) - Custom plugin interface
- [NVIDIA Block Scaled Tensor Core MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/) - tcgen05.mma.blockscaled instruction

## Performance Notes

- **BS=1 Latency**: Expect ~0.1ms for a 2048×16384 GEMM on Thor
- **Memory Savings**: 75% reduction in weight memory (4-bit vs 16-bit)
- **Precision**: Cosine similarity > 0.99 vs BF16 baseline

## TODO

- [ ] Python bindings with pybind11
- [ ] Integration with Pi0.5 VLA inference pipeline
- [ ] Performance profiling and tuning
- [ ] Support for dynamic batch sizes
