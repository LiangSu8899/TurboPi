/*
 * W4A16_GEMM TensorRT Plugin - TVM Generated
 *
 * Auto-generated from TVM TensorIR
 * Target: CUDA SM110 (Jetson Thor)
 *
 * Matrix dimensions: M=1, N=3072, K=3072
 * Note: W4A16 - only weight has scale, activation is full precision
 */

#ifndef W4A16_GEMM_TVM_PLUGIN_H
#define W4A16_GEMM_TVM_PLUGIN_H

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

namespace turbo_pi {

// Kernel launcher (W4A16 - no activation scale)
void launch_w4a16_gemm(
    const float* A,      // [M, K] full precision activation
    const float* W,      // [N, K] nvFP4 weight
    const float* scale_W, // [N, num_blocks_k] weight scales only
    float* C,            // [M, N] output
    int M, int N, int K,
    cudaStream_t stream
);

}  // namespace turbo_pi

#endif  // W4A16_GEMM_TVM_PLUGIN_H
