/*
 * W4A8 GEMM Test
 *
 * Tests the CUTLASS W4A8 GEMM kernel standalone (without TensorRT).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// Forward declarations from kernel
extern "C" void quantizeActivationToFP8(
    const __nv_bfloat16* input,
    __nv_fp8_e4m3* output,
    __nv_fp8_e4m3* scales,
    int M, int K,
    cudaStream_t stream
);

extern "C" void quantizeWeightToNVFP4(
    const __nv_bfloat16* input,
    uint8_t* output_packed,
    __nv_fp8_e4m3* scales,
    int N, int K,
    cudaStream_t stream
);

extern "C" void w4a8GemmForward(
    const void* activation,
    const void* activation_scale,
    const void* weight_packed,
    const void* weight_scale,
    void* output,
    int M, int N, int K,
    float alpha,
    void* workspace,
    size_t workspaceSize,
    cudaStream_t stream
);

extern "C" size_t getW4A8GemmWorkspaceSize(int M, int N, int K);

// Helper to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// BF16 reference GEMM on host
void reference_gemm_bf16(
    const std::vector<float>& A,  // [M, K]
    const std::vector<float>& B,  // [N, K] transposed
    std::vector<float>& C,        // [M, N]
    int M, int N, int K, float alpha
) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = alpha * sum;
        }
    }
}

// Compute cosine similarity
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-8f);
}

int main(int argc, char** argv) {
    // Problem size
    int M = 256;   // Batch size (tokens)
    int N = 2048;  // Output features
    int K = 2048;  // Input features

    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (argc > 3) K = std::atoi(argv[3]);

    std::cout << "======================================" << std::endl;
    std::cout << "W4A8 GEMM Test" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    float alpha = 1.0f;
    int block_size = 16;
    int num_k_blocks = (K + block_size - 1) / block_size;

    // Allocate host memory
    std::vector<float> h_activation(M * K);
    std::vector<float> h_weight(N * K);
    std::vector<float> h_output_ref(M * N);
    std::vector<float> h_output(M * N);

    // Initialize with random values
    srand(42);
    for (int i = 0; i < M * K; ++i) {
        h_activation[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }
    for (int i = 0; i < N * K; ++i) {
        h_weight[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }

    // Reference computation
    std::cout << "\nComputing reference (BF16)..." << std::endl;
    reference_gemm_bf16(h_activation, h_weight, h_output_ref, M, N, K, alpha);

    // Allocate device memory
    __nv_bfloat16 *d_activation_bf16, *d_weight_bf16;
    __nv_fp8_e4m3 *d_activation_fp8, *d_activation_scale;
    uint8_t *d_weight_packed;
    __nv_fp8_e4m3 *d_weight_scale;
    __nv_bfloat16 *d_output;
    void *d_workspace;

    CUDA_CHECK(cudaMalloc(&d_activation_bf16, M * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_weight_bf16, N * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_activation_fp8, M * K * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_activation_scale, M * num_k_blocks * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_weight_packed, (K / 2) * N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_weight_scale, num_k_blocks * N * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(__nv_bfloat16)));

    size_t workspace_size = getW4A8GemmWorkspaceSize(M, N, K);
    std::cout << "Workspace size: " << workspace_size << " bytes" << std::endl;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

    // Convert to BF16 and copy to device
    std::vector<__nv_bfloat16> h_activation_bf16(M * K);
    std::vector<__nv_bfloat16> h_weight_bf16(N * K);
    for (int i = 0; i < M * K; ++i) {
        h_activation_bf16[i] = __float2bfloat16(h_activation[i]);
    }
    for (int i = 0; i < N * K; ++i) {
        h_weight_bf16[i] = __float2bfloat16(h_weight[i]);
    }

    CUDA_CHECK(cudaMemcpy(d_activation_bf16, h_activation_bf16.data(),
                          M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight_bf16, h_weight_bf16.data(),
                          N * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Quantize weight (offline)
    std::cout << "Quantizing weight to NVFP4..." << std::endl;
    quantizeWeightToNVFP4(d_weight_bf16, d_weight_packed, d_weight_scale, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Quantize activation (runtime)
    std::cout << "Quantizing activation to FP8..." << std::endl;
    quantizeActivationToFP8(d_activation_bf16, d_activation_fp8, d_activation_scale, M, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Run W4A8 GEMM
    std::cout << "Running W4A8 GEMM..." << std::endl;
    w4a8GemmForward(
        d_activation_fp8, d_activation_scale,
        d_weight_packed, d_weight_scale,
        d_output,
        M, N, K, alpha,
        d_workspace, workspace_size,
        stream
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy output back
    std::vector<__nv_bfloat16> h_output_bf16(M * N);
    CUDA_CHECK(cudaMemcpy(h_output_bf16.data(), d_output,
                          M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        h_output[i] = __bfloat162float(h_output_bf16[i]);
    }

    // Compare results
    float cos_sim = cosine_similarity(h_output_ref, h_output);
    float max_error = 0.0f;
    float mean_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float err = std::abs(h_output[i] - h_output_ref[i]);
        max_error = std::max(max_error, err);
        mean_error += err;
    }
    mean_error /= (M * N);

    std::cout << "\n======================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Cosine similarity: " << cos_sim << std::endl;
    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Mean error: " << mean_error << std::endl;
    std::cout << "  Status: " << (cos_sim > 0.99f ? "PASS" : "FAIL") << std::endl;
    std::cout << "======================================" << std::endl;

    // Benchmark
    std::cout << "\nBenchmarking..." << std::endl;
    int warmup = 10;
    int iterations = 100;

    for (int i = 0; i < warmup; ++i) {
        quantizeActivationToFP8(d_activation_bf16, d_activation_fp8, d_activation_scale, M, K, stream);
        w4a8GemmForward(d_activation_fp8, d_activation_scale, d_weight_packed, d_weight_scale,
                        d_output, M, N, K, alpha, d_workspace, workspace_size, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        quantizeActivationToFP8(d_activation_bf16, d_activation_fp8, d_activation_scale, M, K, stream);
        w4a8GemmForward(d_activation_fp8, d_activation_scale, d_weight_packed, d_weight_scale,
                        d_output, M, N, K, alpha, d_workspace, workspace_size, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / iterations;
    double tflops = 2.0 * M * N * K / (avg_ms * 1e9);

    std::cout << "\n======================================" << std::endl;
    std::cout << "Performance:" << std::endl;
    std::cout << "  Average latency: " << avg_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << tflops << " TFLOPS" << std::endl;
    std::cout << "======================================" << std::endl;

    // Cleanup
    cudaFree(d_activation_bf16);
    cudaFree(d_weight_bf16);
    cudaFree(d_activation_fp8);
    cudaFree(d_activation_scale);
    cudaFree(d_weight_packed);
    cudaFree(d_weight_scale);
    cudaFree(d_output);
    cudaFree(d_workspace);
    cudaStreamDestroy(stream);

    return (cos_sim > 0.99f) ? 0 : 1;
}
