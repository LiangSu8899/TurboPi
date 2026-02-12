/**
 * Test Packed FP4 Plugin
 *
 * Validates correctness and benchmarks performance of the packed FP4 GEMV kernel.
 *
 * Build:
 *   nvcc -O3 -arch=sm_110 test_packed_plugin.cu nvfp4_gemm_packed_launcher.cu -o test_packed_plugin
 *
 * Run:
 *   ./test_packed_plugin [N] [K] [warmup] [runs]
 *   ./test_packed_plugin 3072 3072 50 200
 *
 * Author: Claude Code
 * Date: 2026-02-10
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

namespace turbo_pi {
    void launch_nvfp4_gemm_packed(
        const float* A, const uint8_t* W_packed,
        const float* scale_A, const float* scale_W,
        float* C, int M, int N, int K, cudaStream_t stream);
}

#define BLOCK_SIZE 32
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// NVFP4 E2M1 values
static const float NVFP4_VALUES[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// CPU reference implementation
void cpu_gemv_reference(
    const float* A, const uint8_t* W_packed,
    const float* scale_A, const float* scale_W,
    float* C, int N, int K
) {
    int num_blocks_k = K / BLOCK_SIZE;

    for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < K; k += 2) {
            int block_idx = k / BLOCK_SIZE;
            float a_scale = scale_A[block_idx];
            float w_scale = scale_W[j * num_blocks_k + block_idx];

            uint8_t packed = W_packed[j * (K / 2) + k / 2];
            float w0 = NVFP4_VALUES[packed & 0xF] * w_scale;
            float w1 = NVFP4_VALUES[(packed >> 4) & 0xF] * w_scale;

            acc += A[k] * a_scale * w0;
            acc += A[k + 1] * a_scale * w1;
        }
        C[j] = acc;
    }
}

// Pack weights to uint8 format
void pack_weights(const float* input, uint8_t* packed, int N, int K) {
    auto encode_nvfp4 = [](float val) -> uint8_t {
        int sign = (val < 0) ? 8 : 0;
        float abs_val = fabsf(val);

        // Find closest representable value
        int idx = 0;
        float min_dist = fabsf(abs_val);
        for (int i = 1; i < 8; ++i) {
            float dist = fabsf(abs_val - NVFP4_VALUES[i]);
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

int main(int argc, char** argv) {
    int N = 3072;
    int K = 3072;
    int warmup = 50;
    int runs = 200;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    if (argc > 3) warmup = atoi(argv[3]);
    if (argc > 4) runs = atoi(argv[4]);

    int num_blocks_k = K / BLOCK_SIZE;
    int M = 1;  // Single token inference

    printf("======================================================================\n");
    printf("Packed FP4 Plugin Test\n");
    printf("======================================================================\n");
    printf("Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Warmup: %d, Runs: %d\n\n", warmup, runs);

    // Memory sizes
    size_t A_size = K * sizeof(float);
    size_t W_packed_size = N * K / 2;
    size_t scale_A_size = num_blocks_k * sizeof(float);
    size_t scale_W_size = N * num_blocks_k * sizeof(float);
    size_t C_size = N * sizeof(float);

    printf("Memory:\n");
    printf("  A:        %.2f KB\n", A_size / 1024.0);
    printf("  W_packed: %.2f KB (%.2f MB for full model)\n",
           W_packed_size / 1024.0, W_packed_size * 18 * 3 / 1e6);  // 18 layers, 3 GEMMs
    printf("  scale_W:  %.2f KB\n", scale_W_size / 1024.0);
    printf("  C:        %.2f KB\n\n", C_size / 1024.0);

    // Allocate host memory
    float* h_A = (float*)malloc(A_size);
    float* h_W = (float*)malloc(N * K * sizeof(float));
    uint8_t* h_W_packed = (uint8_t*)malloc(W_packed_size);
    float* h_scale_A = (float*)malloc(scale_A_size);
    float* h_scale_W = (float*)malloc(scale_W_size);
    float* h_C_gpu = (float*)malloc(C_size);
    float* h_C_cpu = (float*)malloc(C_size);

    // Initialize data
    srand(42);
    for (int i = 0; i < K; ++i) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
    }

    // Use NVFP4 representable values
    for (int i = 0; i < N * K; ++i) {
        h_W[i] = NVFP4_VALUES[rand() % 15];
    }

    for (int i = 0; i < num_blocks_k; ++i) {
        h_scale_A[i] = 0.1f + (float)(rand() % 100) / 1000.0f;
    }

    for (int i = 0; i < N * num_blocks_k; ++i) {
        h_scale_W[i] = 0.1f + (float)(rand() % 100) / 1000.0f;
    }

    // Pack weights
    pack_weights(h_W, h_W_packed, N, K);

    // Allocate device memory
    float *d_A, *d_scale_A, *d_scale_W, *d_C;
    uint8_t *d_W_packed;
    CHECK_CUDA(cudaMalloc(&d_A, A_size));
    CHECK_CUDA(cudaMalloc(&d_W_packed, W_packed_size));
    CHECK_CUDA(cudaMalloc(&d_scale_A, scale_A_size));
    CHECK_CUDA(cudaMalloc(&d_scale_W, scale_W_size));
    CHECK_CUDA(cudaMalloc(&d_C, C_size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_packed, h_W_packed, W_packed_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scale_A, h_scale_A, scale_A_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scale_W, h_scale_W, scale_W_size, cudaMemcpyHostToDevice));

    // ========================================================================
    // Correctness Test
    // ========================================================================
    printf("Testing correctness...\n");

    // CPU reference
    cpu_gemv_reference(h_A, h_W_packed, h_scale_A, h_scale_W, h_C_cpu, N, K);

    // GPU computation
    turbo_pi::launch_nvfp4_gemm_packed(d_A, d_W_packed, d_scale_A, d_scale_W, d_C, M, N, K, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, C_size, cudaMemcpyDeviceToHost));

    // Compare
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    int num_mismatch = 0;

    for (int i = 0; i < N; ++i) {
        float diff = fabsf(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > max_diff) max_diff = diff;
        mean_diff += diff;

        if (diff > 1e-3) {
            if (num_mismatch < 5) {
                printf("  Mismatch at %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                       i, h_C_gpu[i], h_C_cpu[i], diff);
            }
            num_mismatch++;
        }
    }
    mean_diff /= N;

    printf("  Max diff:  %.6f\n", max_diff);
    printf("  Mean diff: %.6f\n", mean_diff);
    printf("  Mismatches (>1e-3): %d / %d\n", num_mismatch, N);

    if (max_diff < 1e-2) {
        printf("  [PASS] Correctness verified\n\n");
    } else {
        printf("  [FAIL] Correctness check failed!\n\n");
    }

    // ========================================================================
    // Performance Benchmark
    // ========================================================================
    printf("Benchmarking...\n");

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        turbo_pi::launch_nvfp4_gemm_packed(d_A, d_W_packed, d_scale_A, d_scale_W, d_C, M, N, K, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < runs; ++i) {
        turbo_pi::launch_nvfp4_gemm_packed(d_A, d_W_packed, d_scale_A, d_scale_W, d_C, M, N, K, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / runs;

    // Calculate metrics
    double flops = 2.0 * N * K;
    double tflops = flops / (avg_ms / 1000.0) / 1e12;

    // Memory bandwidth (packed format)
    double read_bytes = K * 4.0 + N * K / 2.0 + num_blocks_k * 4.0 + N * num_blocks_k * 4.0;
    double write_bytes = N * 4.0;
    double bandwidth_gb = (read_bytes + write_bytes) / (avg_ms / 1000.0) / 1e9;

    printf("  Avg time:    %.4f ms\n", avg_ms);
    printf("  Throughput:  %.4f TFLOPS\n", tflops);
    printf("  Bandwidth:   %.2f GB/s\n\n", bandwidth_gb);

    // ========================================================================
    // Comparison with TRT FP8 Baseline
    // ========================================================================
    float trt_fp8_baseline = 0.53f;  // From docs/tvm-trt-fp4-opt-plan.md
    float speedup = trt_fp8_baseline / avg_ms;

    printf("======================================================================\n");
    printf("COMPARISON vs TRT FP8\n");
    printf("======================================================================\n");
    printf("  TRT FP8 baseline:  %.2f ms\n", trt_fp8_baseline);
    printf("  Packed FP4:        %.4f ms\n", avg_ms);
    printf("  Speedup:           %.2fx\n\n", speedup);

    if (speedup > 1.0f) {
        printf("  [SUCCESS] Packed FP4 is %.2fx FASTER than TRT FP8!\n", speedup);
    } else {
        printf("  [NEEDS WORK] Packed FP4 is %.2fx slower than TRT FP8\n", 1.0f / speedup);
    }
    printf("======================================================================\n");

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_A); free(h_W); free(h_W_packed);
    free(h_scale_A); free(h_scale_W);
    free(h_C_gpu); free(h_C_cpu);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_W_packed));
    CHECK_CUDA(cudaFree(d_scale_A));
    CHECK_CUDA(cudaFree(d_scale_W));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
