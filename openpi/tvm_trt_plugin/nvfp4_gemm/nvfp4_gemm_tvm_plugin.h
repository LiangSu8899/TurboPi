/*
 * NVFP4_GEMM TensorRT Plugin - TVM Generated
 *
 * Auto-generated from TVM TensorIR
 * Target: CUDA SM110 (Jetson Thor)
 *
 * Matrix dimensions: M=1, N=3072, K=3072
 */

#ifndef NVFP4_GEMM_TVM_PLUGIN_H
#define NVFP4_GEMM_TVM_PLUGIN_H

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

namespace turbo_pi {

// Kernel launcher (W4A4 or W4A8)
void launch_nvfp4_gemm(
    const float* A,
    const float* W,
    const float* scale_A,
    const float* scale_W,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
);

}  // namespace turbo_pi

#endif  // NVFP4_GEMM_TVM_PLUGIN_H
