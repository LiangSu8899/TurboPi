/*
 * W4A16 GEMM Test Suite
 *
 * Features:
 * 1. Precision verification against cuBLAS BF16 baseline
 * 2. Latency benchmarking
 * 3. TFLOPS calculation
 *
 * Author: Claude Code
 * Date: 2026-02-09
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

// Forward declarations from kernel
extern "C" void w4a16GemmForward(
    const void* activation,
    const void* weight_packed,
    const void* weight_scale,
    const void* weight_zero,
    void* output,
    int M, int N, int K,
    float alpha, float beta,
    void* workspace,
    size_t workspaceSize,
    cudaStream_t stream
);

extern "C" size_t getW4A16GemmWorkspaceSize(int M, int N, int K);

extern "C" void quantizeWeightToINT4(
    const __nv_bfloat16* input,
    int8_t* output_packed,
    __nv_bfloat16* scales,
    __nv_bfloat16* zeros,
    int N, int K,
    int group_size,
    cudaStream_t stream
);

extern "C" void dequantizeINT4ToBF16(
    const int8_t* input_packed,
    const __nv_bfloat16* scales,
    const __nv_bfloat16* zeros,
    __nv_bfloat16* output,
    int N, int K,
    int group_size,
    cudaStream_t stream
);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while (0)

// ============================================================================
// Utility Functions
// ============================================================================

void initializeRandomBF16(float* host_data, int size, float range = 1.0f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-range, range);
    for (int i = 0; i < size; ++i) {
        host_data[i] = dist(gen);
    }
}

void convertFloatToBF16(__nv_bfloat16* dst, const float* src, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

void convertBF16ToFloat(float* dst, const __nv_bfloat16* src, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = __bfloat162float(src[i]);
    }
}

// Compute relative error statistics
struct ErrorStats {
    float max_abs_error;
    float mean_abs_error;
    float max_rel_error;
    float mean_rel_error;
    float rmse;
    int num_large_errors;  // errors > 1%
};

ErrorStats computeErrorStats(const float* ref, const float* test, int size) {
    ErrorStats stats = {0, 0, 0, 0, 0, 0};
    double sum_abs_error = 0;
    double sum_rel_error = 0;
    double sum_sq_error = 0;

    for (int i = 0; i < size; ++i) {
        float abs_error = std::abs(ref[i] - test[i]);
        float rel_error = abs_error / (std::abs(ref[i]) + 1e-6f);

        stats.max_abs_error = std::max(stats.max_abs_error, abs_error);
        stats.max_rel_error = std::max(stats.max_rel_error, rel_error);
        sum_abs_error += abs_error;
        sum_rel_error += rel_error;
        sum_sq_error += abs_error * abs_error;

        if (rel_error > 0.01f) {  // > 1%
            stats.num_large_errors++;
        }
    }

    stats.mean_abs_error = sum_abs_error / size;
    stats.mean_rel_error = sum_rel_error / size;
    stats.rmse = std::sqrt(sum_sq_error / size);

    return stats;
}

// ============================================================================
// cuBLAS Reference Implementation
// ============================================================================

void runCublasReference(
    cublasHandle_t handle,
    const __nv_bfloat16* A,  // [M, K]
    const __nv_bfloat16* B,  // [K, N] (transposed from [N, K])
    __nv_bfloat16* C,        // [M, N]
    int M, int N, int K,
    float alpha, float beta
) {
    // cuBLAS uses column-major, so we compute C = alpha * A * B + beta * C
    // with appropriate transposes

    // Convert to float for cuBLAS (using CUBLAS_COMPUTE_32F)
    float alpha_f = alpha;
    float beta_f = beta;

    // cuBLAS gemm: C = alpha * op(A) * op(B) + beta * C
    // We have A [M, K] row-major = A^T [K, M] col-major
    // We have B [K, N] row-major = B^T [N, K] col-major
    // We want C [M, N] row-major = C^T [N, M] col-major

    // So we compute: C^T = alpha * B^T * A^T + beta * C^T
    // Which gives us C = alpha * A * B + beta * C

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,  // B is [N, K] col-major (transposed)
        CUBLAS_OP_N,  // A is [K, M] col-major (transposed)
        N, M, K,      // dimensions for C^T = B^T * A^T
        &alpha_f,
        B, CUDA_R_16BF, N,  // B^T [N, K] col-major
        A, CUDA_R_16BF, K,  // A^T [K, M] col-major
        &beta_f,
        C, CUDA_R_16BF, N,  // C^T [N, M] col-major
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));
}

// ============================================================================
// Test Functions
// ============================================================================

bool testPrecision(int M, int N, int K, int group_size = 128, bool verbose = true) {
    if (verbose) {
        printf("\n=== Precision Test: M=%d, N=%d, K=%d, group_size=%d ===\n", M, N, K, group_size);
    }

    // Host allocations
    std::vector<float> h_A_float(M * K);
    std::vector<float> h_B_float(N * K);
    std::vector<__nv_bfloat16> h_A(M * K);
    std::vector<__nv_bfloat16> h_B(N * K);
    std::vector<__nv_bfloat16> h_B_transposed(K * N);

    // Initialize with random data
    initializeRandomBF16(h_A_float.data(), M * K, 1.0f);
    initializeRandomBF16(h_B_float.data(), N * K, 1.0f);
    convertFloatToBF16(h_A.data(), h_A_float.data(), M * K);
    convertFloatToBF16(h_B.data(), h_B_float.data(), N * K);

    // Transpose B from [N, K] to [K, N] for cuBLAS
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            h_B_transposed[k * N + n] = h_B[n * K + k];
        }
    }

    // Device allocations
    __nv_bfloat16 *d_A, *d_B, *d_B_transposed, *d_C_ref, *d_C_test;
    int8_t *d_B_packed;
    __nv_bfloat16 *d_B_scales, *d_B_zeros;

    int num_groups = (K + group_size - 1) / group_size;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B_transposed, K * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B_packed, N * K / 2));  // INT4 packed
    CUDA_CHECK(cudaMalloc(&d_B_scales, N * num_groups * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B_zeros, N * num_groups * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C_test, M * N * sizeof(__nv_bfloat16)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_transposed, h_B_transposed.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C_ref, 0, M * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_C_test, 0, M * N * sizeof(__nv_bfloat16)));

    // Quantize weights to INT4
    quantizeWeightToINT4(d_B, d_B_packed, d_B_scales, d_B_zeros, N, K, group_size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run cuBLAS reference
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    runCublasReference(cublas_handle, d_A, d_B_transposed, d_C_ref, M, N, K, 1.0f, 0.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run W4A16 GEMM
    size_t workspace_size = getW4A16GemmWorkspaceSize(M, N, K);
    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    w4a16GemmForward(
        d_A, d_B_packed, d_B_scales, d_B_zeros,
        d_C_test,
        M, N, K,
        1.0f, 0.0f,
        d_workspace, workspace_size,
        0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    std::vector<__nv_bfloat16> h_C_ref(M * N), h_C_test(M * N);
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_test.data(), d_C_test, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Convert to float for comparison
    std::vector<float> h_C_ref_float(M * N), h_C_test_float(M * N);
    convertBF16ToFloat(h_C_ref_float.data(), h_C_ref.data(), M * N);
    convertBF16ToFloat(h_C_test_float.data(), h_C_test.data(), M * N);

    // Compute error statistics
    ErrorStats stats = computeErrorStats(h_C_ref_float.data(), h_C_test_float.data(), M * N);

    if (verbose) {
        printf("Precision Results:\n");
        printf("  Max Abs Error:  %.6e\n", stats.max_abs_error);
        printf("  Mean Abs Error: %.6e\n", stats.mean_abs_error);
        printf("  Max Rel Error:  %.4f%%\n", stats.max_rel_error * 100);
        printf("  Mean Rel Error: %.4f%%\n", stats.mean_rel_error * 100);
        printf("  RMSE:           %.6e\n", stats.rmse);
        printf("  Large Errors (>1%%): %d / %d (%.2f%%)\n",
               stats.num_large_errors, M * N,
               100.0f * stats.num_large_errors / (M * N));
    }

    // Cleanup
    cublasDestroy(cublas_handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B_transposed);
    cudaFree(d_B_packed);
    cudaFree(d_B_scales);
    cudaFree(d_B_zeros);
    cudaFree(d_C_ref);
    cudaFree(d_C_test);
    if (d_workspace) cudaFree(d_workspace);

    // Pass if mean relative error < 5% (INT4 quantization has inherent error)
    bool passed = stats.mean_rel_error < 0.05f;
    if (verbose) {
        printf("  Result: %s\n", passed ? "PASSED" : "FAILED");
    }

    return passed;
}

void benchmarkLatency(int M, int N, int K, int group_size = 128, int warmup = 10, int iterations = 100) {
    printf("\n=== Latency Benchmark: M=%d, N=%d, K=%d ===\n", M, N, K);

    // Host allocations
    std::vector<float> h_A_float(M * K);
    std::vector<float> h_B_float(N * K);
    std::vector<__nv_bfloat16> h_A(M * K);
    std::vector<__nv_bfloat16> h_B(N * K);

    initializeRandomBF16(h_A_float.data(), M * K, 1.0f);
    initializeRandomBF16(h_B_float.data(), N * K, 1.0f);
    convertFloatToBF16(h_A.data(), h_A_float.data(), M * K);
    convertFloatToBF16(h_B.data(), h_B_float.data(), N * K);

    // Device allocations
    __nv_bfloat16 *d_A, *d_B, *d_C;
    int8_t *d_B_packed;
    __nv_bfloat16 *d_B_scales, *d_B_zeros;

    int num_groups = (K + group_size - 1) / group_size;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B_packed, N * K / 2));
    CUDA_CHECK(cudaMalloc(&d_B_scales, N * num_groups * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_B_zeros, N * num_groups * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Quantize weights
    quantizeWeightToINT4(d_B, d_B_packed, d_B_scales, d_B_zeros, N, K, group_size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate workspace
    size_t workspace_size = getW4A16GemmWorkspaceSize(M, N, K);
    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    // Create events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        w4a16GemmForward(
            d_A, d_B_packed, d_B_scales, d_B_zeros,
            d_C, M, N, K, 1.0f, 0.0f,
            d_workspace, workspace_size, 0
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        w4a16GemmForward(
            d_A, d_B_packed, d_B_scales, d_B_zeros,
            d_C, M, N, K, 1.0f, 0.0f,
            d_workspace, workspace_size, 0
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iterations;

    // Compute TFLOPS
    double flops = 2.0 * M * N * K;  // 2 ops per MAC
    double tflops = flops / (avg_ms * 1e9);  // TFLOPS

    printf("Results:\n");
    printf("  Avg Latency:  %.3f ms\n", avg_ms);
    printf("  Throughput:   %.2f TFLOPS\n", tflops);
    printf("  Memory BW:    %.2f GB/s (weight load)\n",
           (N * K / 2.0) / (avg_ms * 1e6));  // INT4 weight only

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B_packed);
    cudaFree(d_B_scales);
    cudaFree(d_B_zeros);
    cudaFree(d_C);
    if (d_workspace) cudaFree(d_workspace);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("W4A16 GEMM Test Suite\n");
    printf("=====================\n");

    // Check GPU
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("Device: %s (SM %d.%d)\n", props.name, props.major, props.minor);

    if (props.major < 10) {
        printf("Error: This test requires SM100 or higher (Blackwell architecture)\n");
        return 1;
    }

    printf("\n");

    // ========================================
    // Precision Tests
    // ========================================
    printf("========================================\n");
    printf("PRECISION VERIFICATION\n");
    printf("========================================\n");

    bool all_passed = true;

    // Small test
    all_passed &= testPrecision(256, 256, 256);

    // Medium tests
    all_passed &= testPrecision(512, 512, 512);
    all_passed &= testPrecision(1024, 1024, 1024);

    // Large test (Pi0.5 typical sizes)
    all_passed &= testPrecision(256, 2048, 2048);
    all_passed &= testPrecision(256, 4096, 2048);

    // Different group sizes
    all_passed &= testPrecision(512, 512, 512, 64);
    all_passed &= testPrecision(512, 512, 512, 256);

    printf("\n========================================\n");
    printf("PRECISION SUMMARY: %s\n", all_passed ? "ALL PASSED" : "SOME FAILED");
    printf("========================================\n");

    // ========================================
    // Latency Benchmarks
    // ========================================
    printf("\n========================================\n");
    printf("LATENCY BENCHMARKS\n");
    printf("========================================\n");

    // Standard GEMM sizes
    benchmarkLatency(256, 256, 256);
    benchmarkLatency(512, 512, 512);
    benchmarkLatency(1024, 1024, 1024);
    benchmarkLatency(2048, 2048, 2048);
    benchmarkLatency(4096, 4096, 4096);

    // Pi0.5 typical sizes (batch=256, varying hidden dims)
    printf("\n--- Pi0.5 Typical Sizes (batch=256) ---\n");
    benchmarkLatency(256, 2048, 2048);   // hidden layer
    benchmarkLatency(256, 4096, 2048);   // FFN up
    benchmarkLatency(256, 2048, 4096);   // FFN down

    // Larger batches
    printf("\n--- Larger Batches ---\n");
    benchmarkLatency(512, 2048, 2048);
    benchmarkLatency(1024, 2048, 2048);

    // Peak performance test
    printf("\n--- Peak Performance Test ---\n");
    benchmarkLatency(4096, 5120, 8192, 128, 50, 200);

    printf("\n========================================\n");
    printf("TEST COMPLETE\n");
    printf("========================================\n");

    return all_passed ? 0 : 1;
}
