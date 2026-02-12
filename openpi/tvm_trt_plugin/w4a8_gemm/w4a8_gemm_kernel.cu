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

