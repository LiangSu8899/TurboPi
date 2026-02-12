/*
 * W4A8_GEMM Kernel Launcher - TVM Generated
 */

#include "w4a8_gemm_tvm_plugin.h"
#include <cuda.h>

namespace turbo_pi {

// Include TVM-generated kernel
#include <cuda.h>

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef __CUDACC_RTC__
using int64_t = long long;
using uint64_t = unsigned long long;
#else
#include <cstdint>
#endif
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ void __launch_bounds__(256) w4a8_gemm_kernel(float* __restrict__ A, float* __restrict__ C, float* __restrict__ W, float* __restrict__ scale_A, float* __restrict__ scale_W);
extern "C" __global__ void __launch_bounds__(256) w4a8_gemm_kernel(float* __restrict__ A, float* __restrict__ C, float* __restrict__ W, float* __restrict__ scale_A, float* __restrict__ scale_W) {
  C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = 0x0p+0f/*0.000000e+00*/;
  for (int k = 0; k < 3072; ++k) {
    float a_val = (A[k] * scale_A[(k >> 5)]);
    float w_val = (W[(((((int)blockIdx.x) * 786432) + (((int)threadIdx.x) * 3072)) + k)] * scale_W[(((((int)blockIdx.x) * 24576) + (((int)threadIdx.x) * 96)) + (k >> 5))]);
    C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = (C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] + (a_val * w_val));
  }
}



void launch_w4a8_gemm(
    const float* A,
    const float* W,
    const float* scale_A,
    const float* scale_W,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    // Calculate grid dimensions
    const int THREADS_PER_BLOCK = 256;
    const int total_elements = M * N;
    const int num_blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch TVM-generated kernel
    w4a8_gemm_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        const_cast<float*>(A),
        C,
        const_cast<float*>(W),
        const_cast<float*>(scale_A),
        const_cast<float*>(scale_W)
    );
}

}  // namespace turbo_pi
