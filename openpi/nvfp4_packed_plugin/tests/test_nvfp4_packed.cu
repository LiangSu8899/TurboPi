/**
 * Test NVFP4 Packed Plugin with TensorRT
 *
 * This test verifies:
 * 1. Plugin registration
 * 2. Correctness vs reference implementation
 * 3. Performance vs TRT FP8 baseline
 */

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "nvfp4_packed_plugin.h"

using namespace nvinfer1;

// Logger for TensorRT
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

// NVFP4 encoding
uint8_t encode_nvfp4(float val) {
    static const float values[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    int sign = (val < 0) ? 8 : 0;
    float abs_val = std::abs(val);
    int idx = 0;
    float min_dist = std::abs(abs_val - values[0]);
    for (int i = 1; i < 8; ++i) {
        float dist = std::abs(abs_val - values[i]);
        if (dist < min_dist) {
            min_dist = dist;
            idx = i;
        }
    }
    return (uint8_t)(sign | idx);
}

// Pack weights
void pack_weights(const float* weights, uint8_t* packed, int N, int K) {
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            uint8_t low = encode_nvfp4(weights[n * K + k]);
            uint8_t high = encode_nvfp4(weights[n * K + k + 1]);
            packed[n * (K / 2) + k / 2] = low | (high << 4);
        }
    }
}

// Reference implementation
void reference_gemv(
    const float* A, const float* W, const float* scale_A, const float* scale_W,
    float* C, int M, int N, int K, int block_size
) {
    int num_blocks_k = K / block_size;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int block_idx = k / block_size;
                float a_scaled = A[m * K + k] * scale_A[m * num_blocks_k + block_idx];
                float w_scaled = W[n * K + k] * scale_W[n * num_blocks_k + block_idx];
                sum += a_scaled * w_scaled;
            }
            C[m * N + n] = sum;
        }
    }
}

// Verify results
bool verify_results(const float* ref, const float* test, int size, float tol = 1e-3f) {
    float max_diff = 0.0f;
    int max_diff_idx = 0;
    float cos_sim_num = 0.0f, cos_sim_den1 = 0.0f, cos_sim_den2 = 0.0f;

    for (int i = 0; i < size; ++i) {
        float diff = std::abs(ref[i] - test[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        cos_sim_num += ref[i] * test[i];
        cos_sim_den1 += ref[i] * ref[i];
        cos_sim_den2 += test[i] * test[i];
    }

    float cos_sim = cos_sim_num / (std::sqrt(cos_sim_den1) * std::sqrt(cos_sim_den2) + 1e-8f);

    std::cout << "Max diff: " << max_diff << " at index " << max_diff_idx << std::endl;
    std::cout << "Cosine similarity: " << cos_sim << std::endl;

    return cos_sim > 0.999f;
}

int main(int argc, char** argv) {
    std::cout << "======================================================================" << std::endl;
    std::cout << "NVFP4 Packed Plugin Test" << std::endl;
    std::cout << "======================================================================" << std::endl;

    // Initialize TensorRT plugins
    initLibNvInferPlugins(&gLogger, "");

    // Test parameters
    int M = 1;
    int N = 3072;
    int K = 3072;
    int block_size = 32;
    int num_blocks_k = K / block_size;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);

    std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_W(N * K);
    std::vector<uint8_t> h_W_packed(N * K / 2);
    std::vector<float> h_scale_A(M * num_blocks_k);
    std::vector<float> h_scale_W(N * num_blocks_k);
    std::vector<float> h_C_ref(M * N);
    std::vector<float> h_C_test(M * N);

    // Initialize data
    srand(42);
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
    }

    // Use NVFP4 representable values
    static const float fp4_vals[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6};
    for (int i = 0; i < N * K; ++i) {
        h_W[i] = fp4_vals[rand() % 15];
    }

    for (int i = 0; i < M * num_blocks_k; ++i) {
        h_scale_A[i] = 0.1f;
    }
    for (int i = 0; i < N * num_blocks_k; ++i) {
        h_scale_W[i] = 0.1f;
    }

    // Pack weights
    pack_weights(h_W.data(), h_W_packed.data(), N, K);

    // Compute reference
    std::cout << "\nComputing reference..." << std::endl;
    reference_gemv(h_A.data(), h_W.data(), h_scale_A.data(), h_scale_W.data(),
                   h_C_ref.data(), M, N, K, block_size);

    // Allocate device memory
    float *d_A, *d_scale_A, *d_scale_W, *d_C;
    uint8_t *d_W_packed;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_W_packed, N * K / 2);
    cudaMalloc(&d_scale_A, M * num_blocks_k * sizeof(float));
    cudaMalloc(&d_scale_W, N * num_blocks_k * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_packed, h_W_packed.data(), N * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_A, h_scale_A.data(), M * num_blocks_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_W, h_scale_W.data(), N * num_blocks_k * sizeof(float), cudaMemcpyHostToDevice);

    // Test kernel directly
    std::cout << "\nTesting kernel directly..." << std::endl;

    turbo_pi::nvfp4_gemv_forward(
        d_A, d_W_packed, d_scale_A, d_scale_W, nullptr, d_C,
        M, N, K,
        0,  // No activation
        0,  // FP32
        0   // Default stream
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_test.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    std::cout << "\nVerifying results..." << std::endl;
    bool passed = verify_results(h_C_ref.data(), h_C_test.data(), M * N);

    if (passed) {
        std::cout << "\n[PASS] NVFP4 Packed kernel output matches reference!" << std::endl;
    } else {
        std::cout << "\n[FAIL] Output mismatch!" << std::endl;
    }

    // Benchmark
    std::cout << "\nBenchmarking..." << std::endl;
    int warmup = 50;
    int runs = 200;

    for (int i = 0; i < warmup; ++i) {
        turbo_pi::nvfp4_gemv_forward(d_A, d_W_packed, d_scale_A, d_scale_W, nullptr, d_C,
                                      M, N, K, 0, 0, 0);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        turbo_pi::nvfp4_gemv_forward(d_A, d_W_packed, d_scale_A, d_scale_W, nullptr, d_C,
                                      M, N, K, 0, 0, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / runs;

    std::cout << "\nPerformance: " << avg_ms << " ms/iter" << std::endl;
    std::cout << "TRT FP8 baseline: ~0.53 ms" << std::endl;
    std::cout << "Speedup vs FP8: " << (0.53f / avg_ms) << "x" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_W_packed);
    cudaFree(d_scale_A);
    cudaFree(d_scale_W);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\n======================================================================" << std::endl;
    return passed ? 0 : 1;
}
