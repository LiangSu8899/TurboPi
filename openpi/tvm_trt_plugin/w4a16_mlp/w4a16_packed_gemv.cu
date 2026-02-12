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

extern "C" __global__ void __launch_bounds__(256) w4a16_packed_gemv_kernel(float* __restrict__ A, float* __restrict__ C, uchar* __restrict__ W_packed, float* __restrict__ scales);
extern "C" __global__ void __launch_bounds__(256) w4a16_packed_gemv_kernel(float* __restrict__ A, float* __restrict__ C, uchar* __restrict__ W_packed, float* __restrict__ scales) {
  __shared__ float lut[16];
  if (((int)threadIdx.x) < 16) {
    if (((int)threadIdx.x) == 0) {
      lut[0] = 0x0p+0f/*0.000000e+00*/;
    } else {
      if (((int)threadIdx.x) == 1) {
        lut[1] = 0x1p-1f/*5.000000e-01*/;
      } else {
        if (((int)threadIdx.x) == 2) {
          lut[2] = 0x1p+0f/*1.000000e+00*/;
        } else {
          if (((int)threadIdx.x) == 3) {
            lut[3] = 0x1.8p+0f/*1.500000e+00*/;
          } else {
            if (((int)threadIdx.x) == 4) {
              lut[4] = 0x1p+1f/*2.000000e+00*/;
            } else {
              if (((int)threadIdx.x) == 5) {
                lut[5] = 0x1.8p+1f/*3.000000e+00*/;
              } else {
                if (((int)threadIdx.x) == 6) {
                  lut[6] = 0x1p+2f/*4.000000e+00*/;
                } else {
                  if (((int)threadIdx.x) == 7) {
                    lut[7] = 0x1.8p+2f/*6.000000e+00*/;
                  } else {
                    if (((int)threadIdx.x) == 8) {
                      lut[8] = 0x0p+0f/*0.000000e+00*/;
                    } else {
                      if (((int)threadIdx.x) == 9) {
                        lut[9] = -0x1p-1f/*-5.000000e-01*/;
                      } else {
                        if (((int)threadIdx.x) == 10) {
                          lut[10] = -0x1p+0f/*-1.000000e+00*/;
                        } else {
                          if (((int)threadIdx.x) == 11) {
                            lut[11] = -0x1.8p+0f/*-1.500000e+00*/;
                          } else {
                            if (((int)threadIdx.x) == 12) {
                              lut[12] = -0x1p+1f/*-2.000000e+00*/;
                            } else {
                              if (((int)threadIdx.x) == 13) {
                                lut[13] = -0x1.8p+1f/*-3.000000e+00*/;
                              } else {
                                if (((int)threadIdx.x) == 14) {
                                  lut[14] = -0x1p+2f/*-4.000000e+00*/;
                                } else {
                                  lut[15] = -0x1.8p+2f/*-6.000000e+00*/;
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();
  C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = 0x0p+0f/*0.000000e+00*/;
  for (int k = 0; k < 2048; ++k) {
    uchar packed_byte = W_packed[(((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 1024)) + (k >> 1))];
    uchar condval;
    if (((k % 2) == 0)) {
      condval = (packed_byte & (uchar)15);
    } else {
      condval = ((packed_byte >> (uchar)4) & (uchar)15);
    }
    uchar fp4_idx = condval;
    float w_val = lut[((int)fp4_idx)];
    float scale = scales[(((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 64)) + (k >> 5))];
    float w_dequant = (w_val * scale);
    float a_val = A[k];
    C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = (C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] + (a_val * w_dequant));
  }
}

