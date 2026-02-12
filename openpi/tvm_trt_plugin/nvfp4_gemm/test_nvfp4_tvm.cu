/*
 * Test for nvFP4 GEMM TVM Plugin
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DBUILD_TESTS=ON
 *   make
 *   ./test_nvfp4_tvm
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

namespace turbo_pi {
void launch_nvfp4_gemm(
    const float* A,
    const float* W,
    const float* scale_A,
    const float* scale_W,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
);
}

// Reference CPU implementation
void reference_gemm(
    const float* A,
    const float* W,
    const float* scale_A,
    const float* scale_W,
    float* C,
    int M, int N, int K,
    int block_size
) {
    int num_blocks_k = (K + block_size - 1) / block_size;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                int block_idx = k / block_size;
                float a_val = A[i * K + k] * scale_A[i * num_blocks_k + block_idx];
                float w_val = W[j * K + k] * scale_W[j * num_blocks_k + block_idx];
                acc += a_val * w_val;
            }
            C[i * N + j] = acc;
        }
    }
}

int main(int argc, char** argv) {
    // Default dimensions (Pi0.5 typical)
    int M = 1;
    int N = 3072;
    int K = 3072;
    int block_size = 32;
    int warmup_iters = 10;
    int bench_iters = 100;

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    printf("nvFP4 GEMM TVM Test\n");
    printf("  M=%d, N=%d, K=%d\n", M, N, K);
    printf("  Block size=%d\n", block_size);

    int num_blocks_k = (K + block_size - 1) / block_size;

    // Allocate host memory
    size_t A_size = M * K * sizeof(float);
    size_t W_size = N * K * sizeof(float);
    size_t scale_A_size = M * num_blocks_k * sizeof(float);
    size_t scale_W_size = N * num_blocks_k * sizeof(float);
    size_t C_size = M * N * sizeof(float);

    float* h_A = (float*)malloc(A_size);
    float* h_W = (float*)malloc(W_size);
    float* h_scale_A = (float*)malloc(scale_A_size);
    float* h_scale_W = (float*)malloc(scale_W_size);
    float* h_C = (float*)malloc(C_size);
    float* h_C_ref = (float*)malloc(C_size);

    // Initialize with random data
    srand(42);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < N * K; ++i) h_W[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < M * num_blocks_k; ++i) h_scale_A[i] = 1.0f;
    for (int i = 0; i < N * num_blocks_k; ++i) h_scale_W[i] = 1.0f;

    // Allocate device memory
    float *d_A, *d_W, *d_scale_A, *d_scale_W, *d_C;
    cudaMalloc(&d_A, A_size);
    cudaMalloc(&d_W, W_size);
    cudaMalloc(&d_scale_A, scale_A_size);
    cudaMalloc(&d_scale_W, scale_W_size);
    cudaMalloc(&d_C, C_size);

    // Copy to device
    cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, W_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_A, h_scale_A, scale_A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_W, h_scale_W, scale_W_size, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    printf("\nWarming up...\n");
    for (int i = 0; i < warmup_iters; ++i) {
        turbo_pi::launch_nvfp4_gemm(d_A, d_W, d_scale_A, d_scale_W, d_C, M, N, K, stream);
    }
    cudaStreamSynchronize(stream);

    // Benchmark
    printf("Benchmarking...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        turbo_pi::launch_nvfp4_gemm(d_A, d_W, d_scale_A, d_scale_W, d_C, M, N, K, stream);
    }
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / bench_iters;
    double flops = 2.0 * M * N * K;  // GEMM has 2*M*N*K FLOPs
    double tflops = flops / (avg_ms / 1000.0) / 1e12;

    printf("\nPerformance:\n");
    printf("  Avg time: %.4f ms\n", avg_ms);
    printf("  Throughput: %.2f TFLOPS\n", tflops);

    // Correctness check
    printf("\nChecking correctness...\n");
    cudaMemcpy(h_C, d_C, C_size, cudaMemcpyDeviceToHost);

    reference_gemm(h_A, h_W, h_scale_A, h_scale_W, h_C_ref, M, N, K, block_size);

    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(h_C[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("  Max diff: %.6f\n", max_diff);
    if (max_diff < 1e-3) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED\n");
        // Print first few elements for debugging
        printf("  First 5 elements:\n");
        for (int i = 0; i < 5 && i < M * N; ++i) {
            printf("    [%d] GPU=%.6f, CPU=%.6f\n", i, h_C[i], h_C_ref[i]);
        }
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_scale_A);
    cudaFree(d_scale_W);
    cudaFree(d_C);
    free(h_A);
    free(h_W);
    free(h_scale_A);
    free(h_scale_W);
    free(h_C);
    free(h_C_ref);

    return 0;
}
