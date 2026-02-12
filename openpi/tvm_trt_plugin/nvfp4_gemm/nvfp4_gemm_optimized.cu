/**
 * 手写优化的 nvFP4 GEMV Kernel (M=1)
 *
 * 用于验证 TVM 优化的理论上限
 *
 * 优化清单:
 * 1. 寄存器累加 (不再每次循环读写 global memory)
 * 2. 循环展开
 * 3. Shared memory 缓存 A (所有线程共享)
 * 4. 向量化访问 (float4)
 *
 * Author: Claude Code
 * Date: 2026-02-10
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32  // nvFP4 scaling block size
#define THREADS_PER_BLOCK 256
#define TILE_K 256  // K 维度分块大小

// ============================================================================
// V1: 寄存器累加 (最小优化)
// ============================================================================
__global__ void nvfp4_gemv_v1_register_acc(
    const float* __restrict__ A,       // [1, K]
    const float* __restrict__ W,       // [N, K]
    const float* __restrict__ scale_A, // [1, num_blocks_k]
    const float* __restrict__ scale_W, // [N, num_blocks_k]
    float* __restrict__ C,             // [1, N]
    int N, int K
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N) {
        // 关键优化: 使用寄存器累加
        float acc = 0.0f;

        for (int k = 0; k < K; ++k) {
            int block_idx = k / BLOCK_SIZE;
            float a_val = A[k] * scale_A[block_idx];
            float w_val = W[j * K + k] * scale_W[j * (K / BLOCK_SIZE) + block_idx];
            acc += a_val * w_val;
        }

        // 只写一次 global memory
        C[j] = acc;
    }
}

// ============================================================================
// V2: 寄存器累加 + 8x 循环展开
// ============================================================================
__global__ void nvfp4_gemv_v2_unroll8(
    const float* __restrict__ A,
    const float* __restrict__ W,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    float* __restrict__ C,
    int N, int K
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < N) {
        float acc = 0.0f;
        int num_k_blocks = K / BLOCK_SIZE;

        // 展开循环，每次处理 8 个元素
        for (int k = 0; k < K; k += 8) {
            int block_idx = k / BLOCK_SIZE;
            float a_scale = scale_A[block_idx];
            float w_scale = scale_W[j * num_k_blocks + block_idx];

            int w_base = j * K + k;

            // 8x 展开
            acc += A[k + 0] * a_scale * W[w_base + 0] * w_scale;
            acc += A[k + 1] * a_scale * W[w_base + 1] * w_scale;
            acc += A[k + 2] * a_scale * W[w_base + 2] * w_scale;
            acc += A[k + 3] * a_scale * W[w_base + 3] * w_scale;
            acc += A[k + 4] * a_scale * W[w_base + 4] * w_scale;
            acc += A[k + 5] * a_scale * W[w_base + 5] * w_scale;
            acc += A[k + 6] * a_scale * W[w_base + 6] * w_scale;
            acc += A[k + 7] * a_scale * W[w_base + 7] * w_scale;
        }

        C[j] = acc;
    }
}

// ============================================================================
// V3: Shared Memory 缓存 A + 寄存器累加
// ============================================================================
__global__ void nvfp4_gemv_v3_shared_a(
    const float* __restrict__ A,
    const float* __restrict__ W,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    float* __restrict__ C,
    int N, int K
) {
    __shared__ float A_shared[TILE_K];
    __shared__ float scale_A_shared[TILE_K / BLOCK_SIZE + 1];

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float acc = 0.0f;
    int num_k_blocks = K / BLOCK_SIZE;

    // 遍历 K 维度的 tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // 协作加载 A 到 shared memory
        for (int load_idx = tid; load_idx < TILE_K; load_idx += blockDim.x) {
            int k_global = k_tile + load_idx;
            if (k_global < K) {
                A_shared[load_idx] = A[k_global];
            } else {
                A_shared[load_idx] = 0.0f;
            }
        }

        // 加载 scale_A
        if (tid < TILE_K / BLOCK_SIZE + 1) {
            int scale_idx = k_tile / BLOCK_SIZE + tid;
            if (scale_idx < (K / BLOCK_SIZE)) {
                scale_A_shared[tid] = scale_A[scale_idx];
            } else {
                scale_A_shared[tid] = 1.0f;
            }
        }

        __syncthreads();

        // 计算
        if (j < N) {
            int w_base = j * K + k_tile;
            int scale_w_base = j * num_k_blocks + k_tile / BLOCK_SIZE;

            for (int k_local = 0; k_local < TILE_K && (k_tile + k_local) < K; ++k_local) {
                int scale_local_idx = k_local / BLOCK_SIZE;
                float a_val = A_shared[k_local] * scale_A_shared[scale_local_idx];

                int k_global = k_tile + k_local;
                int global_block_idx = k_global / BLOCK_SIZE;
                float w_val = W[w_base + k_local] * scale_W[j * num_k_blocks + global_block_idx];

                acc += a_val * w_val;
            }
        }

        __syncthreads();
    }

    if (j < N) {
        C[j] = acc;
    }
}

// ============================================================================
// V4: Shared Memory + 向量化访问 (float4)
// ============================================================================
__global__ void nvfp4_gemv_v4_vectorized(
    const float* __restrict__ A,
    const float* __restrict__ W,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    float* __restrict__ C,
    int N, int K
) {
    __shared__ float4 A_shared[TILE_K / 4];
    __shared__ float scale_A_shared[TILE_K / BLOCK_SIZE + 1];

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float acc = 0.0f;
    int num_k_blocks = K / BLOCK_SIZE;

    // 遍历 K 维度的 tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // 协作加载 A 到 shared memory (float4 向量化)
        for (int load_idx = tid; load_idx < TILE_K / 4; load_idx += blockDim.x) {
            int k_global = k_tile + load_idx * 4;
            if (k_global + 3 < K) {
                A_shared[load_idx] = *reinterpret_cast<const float4*>(&A[k_global]);
            } else {
                // 边界处理
                float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int i = 0; i < 4 && k_global + i < K; ++i) {
                    ((float*)&tmp)[i] = A[k_global + i];
                }
                A_shared[load_idx] = tmp;
            }
        }

        // 加载 scale_A
        if (tid < TILE_K / BLOCK_SIZE + 1) {
            int scale_idx = k_tile / BLOCK_SIZE + tid;
            if (scale_idx < (K / BLOCK_SIZE)) {
                scale_A_shared[tid] = scale_A[scale_idx];
            } else {
                scale_A_shared[tid] = 1.0f;
            }
        }

        __syncthreads();

        // 计算 (使用 float4 加载 W)
        if (j < N) {
            int w_base = j * K + k_tile;

            for (int k_local = 0; k_local < TILE_K / 4 && (k_tile + k_local * 4) < K; ++k_local) {
                float4 a_vec = A_shared[k_local];
                int k_global_base = k_tile + k_local * 4;

                // 获取 scale
                int scale_local_idx = (k_local * 4) / BLOCK_SIZE;
                float a_scale = scale_A_shared[scale_local_idx];

                // W 向量化加载
                float4 w_vec;
                if (w_base + k_local * 4 + 3 < j * K + K) {
                    w_vec = *reinterpret_cast<const float4*>(&W[w_base + k_local * 4]);
                } else {
                    w_vec = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int i = 0; i < 4 && k_global_base + i < K; ++i) {
                        ((float*)&w_vec)[i] = W[w_base + k_local * 4 + i];
                    }
                }

                // W 的 scale
                int global_block_idx = k_global_base / BLOCK_SIZE;
                float w_scale = scale_W[j * num_k_blocks + global_block_idx];

                // 4 个乘加
                acc += a_vec.x * a_scale * w_vec.x * w_scale;
                acc += a_vec.y * a_scale * w_vec.y * w_scale;
                acc += a_vec.z * a_scale * w_vec.z * w_scale;
                acc += a_vec.w * a_scale * w_vec.w * w_scale;
            }
        }

        __syncthreads();
    }

    if (j < N) {
        C[j] = acc;
    }
}

// ============================================================================
// V5: 简单但有效 - 每线程处理多个输出
// ============================================================================
#define OUTPUTS_PER_THREAD 4

__global__ void nvfp4_gemv_v5_multi_output(
    const float* __restrict__ A,
    const float* __restrict__ W,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    float* __restrict__ C,
    int N, int K
) {
    int base_j = (blockIdx.x * blockDim.x + threadIdx.x) * OUTPUTS_PER_THREAD;

    float acc[OUTPUTS_PER_THREAD] = {0.0f};
    int num_k_blocks = K / BLOCK_SIZE;

    // 遍历 K
    for (int k = 0; k < K; ++k) {
        int block_idx = k / BLOCK_SIZE;
        float a_val = A[k] * scale_A[block_idx];

        // 每个线程处理 OUTPUTS_PER_THREAD 个输出
        #pragma unroll
        for (int i = 0; i < OUTPUTS_PER_THREAD; ++i) {
            int j = base_j + i;
            if (j < N) {
                float w_val = W[j * K + k] * scale_W[j * num_k_blocks + block_idx];
                acc[i] += a_val * w_val;
            }
        }
    }

    // 写出
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD; ++i) {
        int j = base_j + i;
        if (j < N) {
            C[j] = acc[i];
        }
    }
}

// ============================================================================
// 测试函数
// ============================================================================

void benchmark_kernel(const char* name, void (*kernel)(const float*, const float*,
    const float*, const float*, float*, int, int),
    float* d_A, float* d_W, float* d_scale_A, float* d_scale_W, float* d_C,
    int N, int K, int warmup, int runs)
{
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_A, d_W, d_scale_A, d_scale_W, d_C, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_A, d_W, d_scale_A, d_scale_W, d_C, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / runs;

    // TFLOPS
    double flops = 2.0 * N * K;
    double tflops = flops / (avg_ms / 1000.0) / 1e12;

    printf("%-25s: %.4f ms (%.4f TFLOPS)\n", name, avg_ms, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void benchmark_v5(float* d_A, float* d_W, float* d_scale_A, float* d_scale_W, float* d_C,
    int N, int K, int warmup, int runs)
{
    int threads = THREADS_PER_BLOCK;
    int num_blocks = (N + threads * OUTPUTS_PER_THREAD - 1) / (threads * OUTPUTS_PER_THREAD);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        nvfp4_gemv_v5_multi_output<<<num_blocks, threads>>>(d_A, d_W, d_scale_A, d_scale_W, d_C, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        nvfp4_gemv_v5_multi_output<<<num_blocks, threads>>>(d_A, d_W, d_scale_A, d_scale_W, d_C, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / runs;

    double flops = 2.0 * N * K;
    double tflops = flops / (avg_ms / 1000.0) / 1e12;

    printf("%-25s: %.4f ms (%.4f TFLOPS)\n", "V5_multi_output_4x", avg_ms, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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

    int num_blocks_k = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("======================================================================\n");
    printf("nvFP4 GEMV Optimized Kernels Benchmark\n");
    printf("======================================================================\n");
    printf("N=%d, K=%d, warmup=%d, runs=%d\n\n", N, K, warmup, runs);

    // Allocate
    float *h_A = (float*)malloc(K * sizeof(float));
    float *h_W = (float*)malloc(N * K * sizeof(float));
    float *h_scale_A = (float*)malloc(num_blocks_k * sizeof(float));
    float *h_scale_W = (float*)malloc(N * num_blocks_k * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    // Initialize
    for (int i = 0; i < K; ++i) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < N * K; ++i) h_W[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < num_blocks_k; ++i) h_scale_A[i] = 0.1f;
    for (int i = 0; i < N * num_blocks_k; ++i) h_scale_W[i] = 0.1f;

    float *d_A, *d_W, *d_scale_A, *d_scale_W, *d_C;
    cudaMalloc(&d_A, K * sizeof(float));
    cudaMalloc(&d_W, N * K * sizeof(float));
    cudaMalloc(&d_scale_A, num_blocks_k * sizeof(float));
    cudaMalloc(&d_scale_W, N * num_blocks_k * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_A, h_scale_A, num_blocks_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_W, h_scale_W, N * num_blocks_k * sizeof(float), cudaMemcpyHostToDevice);

    printf("Benchmarking kernels...\n\n");

    benchmark_kernel("V1_register_acc", nvfp4_gemv_v1_register_acc,
                     d_A, d_W, d_scale_A, d_scale_W, d_C, N, K, warmup, runs);

    benchmark_kernel("V2_unroll8", nvfp4_gemv_v2_unroll8,
                     d_A, d_W, d_scale_A, d_scale_W, d_C, N, K, warmup, runs);

    benchmark_kernel("V3_shared_a", nvfp4_gemv_v3_shared_a,
                     d_A, d_W, d_scale_A, d_scale_W, d_C, N, K, warmup, runs);

    benchmark_kernel("V4_vectorized", nvfp4_gemv_v4_vectorized,
                     d_A, d_W, d_scale_A, d_scale_W, d_C, N, K, warmup, runs);

    benchmark_v5(d_A, d_W, d_scale_A, d_scale_W, d_C, N, K, warmup, runs);

    printf("\n======================================================================\n");
    printf("TRT FP8 Baseline: 0.53 ms\n");
    printf("======================================================================\n");

    // Cleanup
    free(h_A); free(h_W); free(h_scale_A); free(h_scale_W); free(h_C);
    cudaFree(d_A); cudaFree(d_W); cudaFree(d_scale_A); cudaFree(d_scale_W); cudaFree(d_C);

    return 0;
}
