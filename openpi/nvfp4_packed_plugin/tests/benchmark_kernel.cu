/**
 * Standalone NVFP4 Packed GEMV Kernel Benchmark
 *
 * No TensorRT dependency - pure CUDA benchmark
 *
 * Usage:
 *   ./benchmark_nvfp4_kernel [N] [K] [warmup] [runs]
 *
 * Example:
 *   ./benchmark_nvfp4_kernel 3072 3072 50 200
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define BLOCK_SIZE_SCALE 32
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256

// NVFP4 decode table
__constant__ float NVFP4_DECODE_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float2 unpack_nvfp4_pair(uint8_t packed) {
    float2 result;
    result.x = NVFP4_DECODE_TABLE[packed & 0xF];
    result.y = NVFP4_DECODE_TABLE[(packed >> 4) & 0xF];
    return result;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coef * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ============================================================================
// NVFP4 GEMV Kernels
// ============================================================================

__global__ void nvfp4_gemv_packed_warp_reduce(
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    float* __restrict__ C,
    int M, int N, int K
) {
    int num_blocks_k = K / BLOCK_SIZE_SCALE;
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int total_outputs = M * N;
    if (global_warp_id >= total_outputs) return;

    int m = global_warp_id / N;
    int n = global_warp_id % N;

    float local_sum = 0.0f;
    int k_per_lane = K / WARP_SIZE;
    int k_start = (lane_id * k_per_lane / 2) * 2;
    int k_end = k_start + k_per_lane;

    for (int k = k_start; k < k_end; k += 2) {
        int block_idx = k / BLOCK_SIZE_SCALE;
        float a_scale = scale_A[m * num_blocks_k + block_idx];
        float w_scale = scale_W[n * num_blocks_k + block_idx];

        uint8_t w_packed = W_packed[n * (K / 2) + k / 2];
        float2 w_vals = unpack_nvfp4_pair(w_packed);

        float a0 = A[m * K + k];
        float a1 = A[m * K + k + 1];

        local_sum += a0 * a_scale * w_vals.x * w_scale;
        local_sum += a1 * a_scale * w_vals.y * w_scale;
    }

    float total = warpReduceSum(local_sum);

    if (lane_id == 0) {
        C[m * N + n] = total;
    }
}

__global__ void nvfp4_gemv_packed_bias_gelu(
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K
) {
    int num_blocks_k = K / BLOCK_SIZE_SCALE;
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int total_outputs = M * N;
    if (global_warp_id >= total_outputs) return;

    int m = global_warp_id / N;
    int n = global_warp_id % N;

    float local_sum = 0.0f;
    int k_per_lane = K / WARP_SIZE;
    int k_start = (lane_id * k_per_lane / 2) * 2;
    int k_end = k_start + k_per_lane;

    for (int k = k_start; k < k_end; k += 2) {
        int block_idx = k / BLOCK_SIZE_SCALE;
        float a_scale = scale_A[m * num_blocks_k + block_idx];
        float w_scale = scale_W[n * num_blocks_k + block_idx];

        uint8_t w_packed = W_packed[n * (K / 2) + k / 2];
        float2 w_vals = unpack_nvfp4_pair(w_packed);

        float a0 = A[m * K + k];
        float a1 = A[m * K + k + 1];

        local_sum += a0 * a_scale * w_vals.x * w_scale;
        local_sum += a1 * a_scale * w_vals.y * w_scale;
    }

    float total = warpReduceSum(local_sum);

    if (lane_id == 0) {
        float val = total + bias[n];
        C[m * N + n] = gelu_tanh(val);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

void pack_nvfp4(const float* input, uint8_t* packed, int N, int K) {
    auto encode_nvfp4 = [](float val) -> uint8_t {
        static const float values[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
        int sign = (val < 0) ? 8 : 0;
        float abs_val = fabsf(val);
        int idx = 0;
        float min_dist = fabsf(abs_val - values[0]);
        for (int i = 1; i < 8; ++i) {
            float dist = fabsf(abs_val - values[i]);
            if (dist < min_dist) {
                min_dist = dist;
                idx = i;
            }
        }
        return (uint8_t)(sign | idx);
    };

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; k += 2) {
            uint8_t low = encode_nvfp4(input[i * K + k]);
            uint8_t high = encode_nvfp4(input[i * K + k + 1]);
            packed[i * (K / 2) + k / 2] = low | (high << 4);
        }
    }
}

void benchmark_kernel(
    const char* name,
    void (*kernel)(const float*, const uint8_t*, const float*, const float*, float*, int, int, int),
    const float* d_A, const uint8_t* d_W_packed,
    const float* d_scale_A, const float* d_scale_W, float* d_C,
    int M, int N, int K, int warmup, int runs
) {
    int num_warps = M * N;
    int warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;
    int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_A, d_W_packed, d_scale_A, d_scale_W, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_A, d_W_packed, d_scale_A, d_scale_W, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / runs;

    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms / 1000.0) / 1e12;
    double read_bytes = M * K * 4.0 + N * K / 2.0 + M * (K / 32) * 4.0 + N * (K / 32) * 4.0;
    double write_bytes = M * N * 4.0;
    double bandwidth_gb = (read_bytes + write_bytes) / (avg_ms / 1000.0) / 1e9;

    printf("%-35s: %.4f ms | %.4f TFLOPS | %.2f GB/s\n", name, avg_ms, tflops, bandwidth_gb);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void benchmark_fused_kernel(
    const char* name,
    const float* d_A, const uint8_t* d_W_packed,
    const float* d_scale_A, const float* d_scale_W,
    const float* d_bias, float* d_C,
    int M, int N, int K, int warmup, int runs
) {
    int num_warps = M * N;
    int warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;
    int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        nvfp4_gemv_packed_bias_gelu<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_A, d_W_packed, d_scale_A, d_scale_W, d_bias, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        nvfp4_gemv_packed_bias_gelu<<<num_blocks, THREADS_PER_BLOCK>>>(
            d_A, d_W_packed, d_scale_A, d_scale_W, d_bias, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / runs;

    double flops = 2.0 * M * N * K + M * N * 20;  // GEMV + GELU approx
    double tflops = flops / (avg_ms / 1000.0) / 1e12;

    printf("%-35s: %.4f ms | %.4f TFLOPS\n", name, avg_ms, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    int M = 1;
    int N = 3072;
    int K = 3072;
    int warmup = 50;
    int runs = 200;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    if (argc > 3) warmup = atoi(argv[3]);
    if (argc > 4) runs = atoi(argv[4]);

    int num_blocks_k = K / BLOCK_SIZE_SCALE;

    printf("======================================================================\n");
    printf("NVFP4 Packed GEMV Kernel Benchmark\n");
    printf("======================================================================\n");
    printf("M=%d, N=%d, K=%d, warmup=%d, runs=%d\n\n", M, N, K, warmup, runs);

    printf("Memory comparison:\n");
    printf("  float32 weights: %.2f MB\n", N * K * 4.0 / 1e6);
    printf("  packed uint8:    %.2f MB (8x smaller!)\n", N * K / 2.0 / 1e6);
    printf("\n");

    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_W = (float*)malloc(N * K * sizeof(float));
    uint8_t *h_W_packed = (uint8_t*)malloc(N * K / 2);
    float *h_scale_A = (float*)malloc(M * num_blocks_k * sizeof(float));
    float *h_scale_W = (float*)malloc(N * num_blocks_k * sizeof(float));
    float *h_bias = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize
    srand(42);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(rand() % 100) / 100.0f;
    static const float fp4_vals[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6};
    for (int i = 0; i < N * K; ++i) h_W[i] = fp4_vals[rand() % 15];
    for (int i = 0; i < M * num_blocks_k; ++i) h_scale_A[i] = 0.1f;
    for (int i = 0; i < N * num_blocks_k; ++i) h_scale_W[i] = 0.1f;
    for (int i = 0; i < N; ++i) h_bias[i] = 0.01f;

    pack_nvfp4(h_W, h_W_packed, N, K);

    // Allocate device memory
    float *d_A, *d_scale_A, *d_scale_W, *d_bias, *d_C;
    uint8_t *d_W_packed;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_W_packed, N * K / 2);
    cudaMalloc(&d_scale_A, M * num_blocks_k * sizeof(float));
    cudaMalloc(&d_scale_W, N * num_blocks_k * sizeof(float));
    cudaMalloc(&d_bias, N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_packed, h_W_packed, N * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_A, h_scale_A, M * num_blocks_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_W, h_scale_W, N * num_blocks_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, N * sizeof(float), cudaMemcpyHostToDevice);

    printf("Benchmarking kernels...\n\n");

    benchmark_kernel("GEMV Warp Reduce",
        nvfp4_gemv_packed_warp_reduce,
        d_A, d_W_packed, d_scale_A, d_scale_W, d_C,
        M, N, K, warmup, runs);

    benchmark_fused_kernel("GEMV + Bias + GELU (Fused)",
        d_A, d_W_packed, d_scale_A, d_scale_W, d_bias, d_C,
        M, N, K, warmup, runs);

    printf("\n======================================================================\n");
    printf("Baseline Comparison:\n");
    printf("  TRT FP8 GEMV:     ~0.53 ms\n");
    printf("  Target:           < 0.53 ms\n");
    printf("======================================================================\n");

    // Verify correctness
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nSample output[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
           h_C[0], h_C[1], h_C[2], h_C[3], h_C[4]);

    // Cleanup
    free(h_A); free(h_W); free(h_W_packed);
    free(h_scale_A); free(h_scale_W); free(h_bias); free(h_C);
    cudaFree(d_A); cudaFree(d_W_packed);
    cudaFree(d_scale_A); cudaFree(d_scale_W); cudaFree(d_bias); cudaFree(d_C);

    return 0;
}
