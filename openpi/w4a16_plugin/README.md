# W4A16 GEMM TensorRT Plugin

4-bit Weight, 16-bit Activation GEMM for Jetson Thor (SM110).

## Overview

This plugin implements mixed-precision GEMM using CUTLASS:
- **Weight**: INT4 (4-bit integer with per-group scaling)
- **Activation**: BF16 (16-bit bfloat16)
- **Output**: BF16

Based on CUTLASS Example 86: `blackwell_mixed_dtype_gemm`

## Key Advantage

Unlike W4A8 (FP8×NVFP4) or high-performance W4A4 kernels, W4A16 does **NOT** require:
- `tcgen05.mma.block_scale` instruction
- `.cta_group::2` feature
- `.kind::mxf4nvf4` feature

This makes it **fully compatible with SM110 (Jetson Thor)**, while still achieving excellent performance.

## Performance

Tested on Jetson Thor (SM110):

| Size (M×N×K) | Latency | TFLOPS |
|--------------|---------|--------|
| 256×256×256 | 0.013ms | 2.5 |
| 2048×2048×2048 | 0.19ms | 87.5 |
| 4096×5120×8192 | 3.5ms | **99.4** |

Peak performance: **99.4 TFLOPS** (4096×5120×8192)

## Build

```bash
mkdir build && cd build
cmake .. \
    -DCUTLASS_DIR=$HOME/cutlass_build/cutlass/include \
    -DCUTLASS_UTIL_DIR=$HOME/cutlass_build/cutlass/tools/util/include
make -j$(nproc)
```

## Test

```bash
./test_w4a16_gemm
```

The test includes:
1. **Precision verification** against cuBLAS BF16 baseline
2. **Latency benchmarks** for various GEMM sizes
3. **TFLOPS calculation**

## Usage

### TensorRT Plugin

```cpp
#include "w4a16_gemm_plugin.h"

// Create plugin with dimensions
auto plugin = new turbo_pi::W4A16GemmPlugin(
    inFeatures,   // K dimension
    outFeatures,  // N dimension
    128,          // group size (default)
    true,         // use zero point
    1.0f,         // alpha
    0.0f          // beta
);
```

### Direct CUDA Kernel

```cpp
#include "w4a16_gemm_plugin.h"

// Quantize weights (offline)
quantizeWeightToINT4(
    bf16_weights,      // [N, K] BF16
    int4_packed,       // [N, K/2] INT8 (packed INT4)
    scales,            // [N, K/128] BF16
    zeros,             // [N, K/128] BF16
    N, K, 128, stream
);

// Run GEMM
w4a16GemmForward(
    activation,        // [M, K] BF16
    int4_packed,       // [N, K/2] INT8
    scales,            // [N, K/128] BF16
    zeros,             // [N, K/128] BF16 or nullptr
    output,            // [M, N] BF16
    M, N, K,
    1.0f, 0.0f,
    workspace, workspace_size, stream
);
```

## File Structure

```
w4a16_plugin/
├── src/
│   ├── w4a16_gemm_plugin.h      # TensorRT plugin header
│   ├── w4a16_gemm_plugin.cpp    # TensorRT plugin implementation
│   └── w4a16_gemm_kernel.cu     # CUTLASS kernel implementation
├── tests/
│   └── test_w4a16_gemm.cu       # Precision & latency tests
├── CMakeLists.txt
└── README.md
```

## Quantization Details

- **Group Size**: 128 (configurable, along K dimension)
- **Quantization**: Symmetric INT4 [-8, 7]
- **Scale Format**: BF16 per-group scales
- **Zero Point**: BF16 (optional, for asymmetric quantization)

## Comparison with Other Approaches

| Approach | SM110 Support | Peak TFLOPS | Notes |
|----------|---------------|-------------|-------|
| W4A4 (NVFP4×NVFP4) | ⚠️ Limited | 141 | Basic kernel only |
| W4A8 (FP8×NVFP4) | ❌ No | N/A | block_scale not supported |
| **W4A16 (INT4×BF16)** | ✅ Yes | **99.4** | Fully optimized |
| TRT FP8 (W8A8) | ✅ Yes | ~50 | May fallback to FP32 |

## References

- [CUTLASS Example 86](https://github.com/NVIDIA/cutlass/tree/main/examples/86_blackwell_mixed_dtype_gemm)
- [CUTLASS Mixed-Dtype Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
