/**
 * Packed nvFP4 GEMV Kernel
 *
 * 使用真正的 4-bit packed 格式，而不是 float32 模拟
 *
 * 内存读取量对比:
 * - float32 模拟: N*K*4 = 3072*3072*4 = 36MB
 * - packed uint8:  N*K/2 = 3072*3072/2 = 4.5MB (8x 带宽节省!)
 *
 * Author: Claude Code
 * Date: 2026-02-10
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define BLOCK_SIZE 32  // nvFP4 scaling block size
#define THREADS_PER_BLOCK 256

// NVFP4 E2M1 解码表 (4-bit -> float32)
// Encoding: sign(1) | magnitude_index(3)
// magnitude_index: 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6
__constant__ float NVFP4_DECODE_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,   // positive
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // negative
};

// ============================================================================
// 辅助函数：解包 FP4
// ============================================================================
__device__ __forceinline__ float decode_nvfp4(uint8_t nibble) {
    return NVFP4_DECODE_TABLE[nibble & 0xF];
}

__device__ __forceinline__ float2 unpack_nvfp4_pair(uint8_t packed) {
    // Low nibble: even index, High nibble: odd index
    float2 result;
    result.x = NVFP4_DECODE_TABLE[packed & 0xF];
    result.y = NVFP4_DECODE_TABLE[(packed >> 4) & 0xF];
    return result;
}

// ============================================================================
// V1: Packed FP4 GEMV - 基础版本
// ============================================================================
__global__ void nvfp4_gemv_packed_v1(
    const float* __restrict__ A,              // [1, K] 激活 (float32)
    const uint8_t* __restrict__ W_packed,     // [N, K/2] packed FP4 权重
    const float* __restrict__ scale_A,        // [1, num_blocks_k]
    const float* __restrict__ scale_W,        // [N, num_blocks_k]
    float* __restrict__ C,                    // [1, N] 输出
    int N, int K
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks_k = K / BLOCK_SIZE;

    if (j < N) {
        float acc = 0.0f;

        // 遍历 K 维度，每次处理 2 个元素（1 个 packed byte）
        for (int k = 0; k < K; k += 2) {
            int block_idx = k / BLOCK_SIZE;
            float a_scale = scale_A[block_idx];
            float w_scale = scale_W[j * num_blocks_k + block_idx];

            // 读取 packed 权重 (1 byte = 2 个 FP4)
            uint8_t w_packed = W_packed[j * (K / 2) + k / 2];
            float2 w_vals = unpack_nvfp4_pair(w_packed);

            // 反量化并累加
            float a0 = A[k] * a_scale;
            float a1 = A[k + 1] * a_scale;
            float w0 = w_vals.x * w_scale;
            float w1 = w_vals.y * w_scale;

            acc += a0 * w0 + a1 * w1;
        }

        C[j] = acc;
    }
}

// ============================================================================
// V2: Packed FP4 GEMV - Shared Memory 缓存 A
// ============================================================================
#define TILE_K 256

__global__ void nvfp4_gemv_packed_v2_shared(
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    float* __restrict__ C,
    int N, int K
) {
    __shared__ float A_shared[TILE_K];
    __shared__ float scale_A_shared[TILE_K / BLOCK_SIZE + 1];

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int num_blocks_k = K / BLOCK_SIZE;

    float acc = 0.0f;

    // 遍历 K tiles
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
            if (scale_idx < num_blocks_k) {
                scale_A_shared[tid] = scale_A[scale_idx];
            } else {
                scale_A_shared[tid] = 1.0f;
            }
        }

        __syncthreads();

        // 计算
        if (j < N) {
            int w_base = j * (K / 2) + k_tile / 2;
            int scale_w_base = j * num_blocks_k + k_tile / BLOCK_SIZE;

            for (int k_local = 0; k_local < TILE_K && (k_tile + k_local) < K; k_local += 2) {
                int scale_local_idx = k_local / BLOCK_SIZE;
                float a_scale = scale_A_shared[scale_local_idx];

                int k_global = k_tile + k_local;
                int global_block_idx = k_global / BLOCK_SIZE;
                float w_scale = scale_W[j * num_blocks_k + global_block_idx];

                // 读取 packed 权重
                uint8_t w_packed = W_packed[w_base + k_local / 2];
                float2 w_vals = unpack_nvfp4_pair(w_packed);

                // 反量化并累加
                float a0 = A_shared[k_local] * a_scale;
                float a1 = A_shared[k_local + 1] * a_scale;
                float w0 = w_vals.x * w_scale;
                float w1 = w_vals.y * w_scale;

                acc += a0 * w0 + a1 * w1;
            }
        }

        __syncthreads();
    }

    if (j < N) {
        C[j] = acc;
    }
}

// ============================================================================
// V3: Packed FP4 GEMV - 向量化加载 (uint32 = 8 个 FP4)
// ============================================================================
__global__ void nvfp4_gemv_packed_v3_vectorized(
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    float* __restrict__ C,
    int N, int K
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks_k = K / BLOCK_SIZE;

    if (j < N) {
        float acc = 0.0f;

        // 每次处理 8 个元素（4 bytes = 1 个 uint32）
        const uint32_t* W_packed_u32 = reinterpret_cast<const uint32_t*>(W_packed + j * (K / 2));

        for (int k = 0; k < K; k += 8) {
            int block_idx = k / BLOCK_SIZE;
            float a_scale = scale_A[block_idx];
            float w_scale = scale_W[j * num_blocks_k + block_idx];

            // 向量化读取 4 bytes = 8 个 FP4
            uint32_t w_u32 = W_packed_u32[k / 8];

            // 解包 8 个 FP4 值
            float w_vals[8];
            w_vals[0] = NVFP4_DECODE_TABLE[(w_u32 >> 0) & 0xF];
            w_vals[1] = NVFP4_DECODE_TABLE[(w_u32 >> 4) & 0xF];
            w_vals[2] = NVFP4_DECODE_TABLE[(w_u32 >> 8) & 0xF];
            w_vals[3] = NVFP4_DECODE_TABLE[(w_u32 >> 12) & 0xF];
            w_vals[4] = NVFP4_DECODE_TABLE[(w_u32 >> 16) & 0xF];
            w_vals[5] = NVFP4_DECODE_TABLE[(w_u32 >> 20) & 0xF];
            w_vals[6] = NVFP4_DECODE_TABLE[(w_u32 >> 24) & 0xF];
            w_vals[7] = NVFP4_DECODE_TABLE[(w_u32 >> 28) & 0xF];

            // 展开累加
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc += A[k + i] * a_scale * w_vals[i] * w_scale;
            }
        }

        C[j] = acc;
    }
}

// ============================================================================
// V4: Packed FP4 GEMV - 多线程协作 + Warp Shuffle
// ============================================================================
#define WARP_SIZE 32
#define K_PER_WARP 1024  // 每个 warp 处理 1024 个 K 元素

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void nvfp4_gemv_packed_v4_warp_reduce(
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    float* __restrict__ C,
    int N, int K
) {
    // 每个 warp 处理一个输出元素
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_blocks_k = K / BLOCK_SIZE;

    if (warp_id < N) {
        int j = warp_id;
        float local_sum = 0.0f;

        // 每个 lane 处理 K/32 的部分
        int k_per_lane = K / WARP_SIZE;
        int k_start = lane_id * k_per_lane;
        int k_end = k_start + k_per_lane;

        for (int k = k_start; k < k_end; k += 2) {
            int block_idx = k / BLOCK_SIZE;
            float a_scale = scale_A[block_idx];
            float w_scale = scale_W[j * num_blocks_k + block_idx];

            uint8_t w_packed = W_packed[j * (K / 2) + k / 2];
            float2 w_vals = unpack_nvfp4_pair(w_packed);

            float a0 = A[k] * a_scale;
            float a1 = A[k + 1] * a_scale;

            local_sum += a0 * w_vals.x * w_scale + a1 * w_vals.y * w_scale;
        }

        // Warp reduce
        float total = warpReduceSum(local_sum);

        // Lane 0 写出结果
        if (lane_id == 0) {
            C[j] = total;
        }
    }
}

// ============================================================================
// 测试函数
// ============================================================================

// 打包 float32 数组到 uint8 packed
void pack_nvfp4(const float* input, uint8_t* packed, int N, int K) {
    // NVFP4 编码: 找到最近的可表示值
    auto encode_nvfp4 = [](float val) -> uint8_t {
        static const float values[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
        int sign = (val < 0) ? 8 : 0;
        float abs_val = fabsf(val);

        // 找最近值
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

void benchmark_packed_kernel(
    const char* name,
    void (*kernel)(const float*, const uint8_t*, const float*, const float*, float*, int, int),
    const float* d_A, const uint8_t* d_W_packed,
    const float* d_scale_A, const float* d_scale_W, float* d_C,
    int N, int K, int warmup, int runs, int blocks, int threads
) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        kernel<<<blocks, threads>>>(d_A, d_W_packed, d_scale_A, d_scale_W, d_C, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) {
        kernel<<<blocks, threads>>>(d_A, d_W_packed, d_scale_A, d_scale_W, d_C, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / runs;

    // TFLOPS
    double flops = 2.0 * N * K;
    double tflops = flops / (avg_ms / 1000.0) / 1e12;

    // Memory bandwidth (packed format)
    // Read: A[K] + W_packed[N*K/2] + scale_A[K/32] + scale_W[N*K/32]
    // Write: C[N]
    double read_bytes = K * 4.0 + N * K / 2.0 + (K / 32) * 4.0 + N * (K / 32) * 4.0;
    double write_bytes = N * 4.0;
    double bandwidth_gb = (read_bytes + write_bytes) / (avg_ms / 1000.0) / 1e9;

    printf("%-30s: %.4f ms | %.4f TFLOPS | %.2f GB/s\n", name, avg_ms, tflops, bandwidth_gb);

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
    printf("nvFP4 PACKED GEMV Kernels Benchmark\n");
    printf("======================================================================\n");
    printf("N=%d, K=%d, warmup=%d, runs=%d\n\n", N, K, warmup, runs);

    printf("Memory comparison:\n");
    printf("  float32 format: %.2f MB (W)\n", N * K * 4.0 / 1e6);
    printf("  packed uint8:   %.2f MB (W) [8x smaller!]\n", N * K / 2.0 / 1e6);
    printf("\n");

    // Allocate host memory
    float *h_A = (float*)malloc(K * sizeof(float));
    float *h_W = (float*)malloc(N * K * sizeof(float));
    uint8_t *h_W_packed = (uint8_t*)malloc(N * K / 2);
    float *h_scale_A = (float*)malloc(num_blocks_k * sizeof(float));
    float *h_scale_W = (float*)malloc(N * num_blocks_k * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    // Initialize
    srand(42);
    for (int i = 0; i < K; ++i) h_A[i] = (float)(rand() % 100) / 100.0f;
    // 使用 NVFP4 可表示的值
    static const float fp4_vals[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6};
    for (int i = 0; i < N * K; ++i) h_W[i] = fp4_vals[rand() % 15];
    for (int i = 0; i < num_blocks_k; ++i) h_scale_A[i] = 0.1f;
    for (int i = 0; i < N * num_blocks_k; ++i) h_scale_W[i] = 0.1f;

    // Pack weights
    pack_nvfp4(h_W, h_W_packed, N, K);

    // Allocate device memory
    float *d_A, *d_scale_A, *d_scale_W, *d_C;
    uint8_t *d_W_packed;
    cudaMalloc(&d_A, K * sizeof(float));
    cudaMalloc(&d_W_packed, N * K / 2);
    cudaMalloc(&d_scale_A, num_blocks_k * sizeof(float));
    cudaMalloc(&d_scale_W, N * num_blocks_k * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_packed, h_W_packed, N * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_A, h_scale_A, num_blocks_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_W, h_scale_W, N * num_blocks_k * sizeof(float), cudaMemcpyHostToDevice);

    printf("Benchmarking PACKED kernels...\n\n");

    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    benchmark_packed_kernel("V1_packed_basic",
        nvfp4_gemv_packed_v1,
        d_A, d_W_packed, d_scale_A, d_scale_W, d_C,
        N, K, warmup, runs, num_blocks, THREADS_PER_BLOCK);

    benchmark_packed_kernel("V2_packed_shared_A",
        nvfp4_gemv_packed_v2_shared,
        d_A, d_W_packed, d_scale_A, d_scale_W, d_C,
        N, K, warmup, runs, num_blocks, THREADS_PER_BLOCK);

    benchmark_packed_kernel("V3_packed_vectorized",
        nvfp4_gemv_packed_v3_vectorized,
        d_A, d_W_packed, d_scale_A, d_scale_W, d_C,
        N, K, warmup, runs, num_blocks, THREADS_PER_BLOCK);

    // V4 需要特殊的线程配置
    int warps_needed = N;
    int threads_v4 = WARP_SIZE;  // 每个 block 一个 warp
    int blocks_v4 = (warps_needed + (THREADS_PER_BLOCK / WARP_SIZE) - 1) / (THREADS_PER_BLOCK / WARP_SIZE);
    benchmark_packed_kernel("V4_packed_warp_reduce",
        nvfp4_gemv_packed_v4_warp_reduce,
        d_A, d_W_packed, d_scale_A, d_scale_W, d_C,
        N, K, warmup, runs, blocks_v4, THREADS_PER_BLOCK);

    printf("\n======================================================================\n");
    printf("TRT FP8 Baseline: 0.53 ms\n");
    printf("Target: < 0.53 ms to beat FP8\n");
    printf("======================================================================\n");

    // Cleanup
    free(h_A); free(h_W); free(h_W_packed);
    free(h_scale_A); free(h_scale_W); free(h_C);
    cudaFree(d_A); cudaFree(d_W_packed);
    cudaFree(d_scale_A); cudaFree(d_scale_W); cudaFree(d_C);

    return 0;
}
