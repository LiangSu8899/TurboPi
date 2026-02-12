/**
 * NVFP4 GEMM Kernel - 针对 M > 1 优化
 *
 * 使用 tiled GEMM 算法，支持任意 M 值。
 * 针对 Thor (SM110/SM89) 优化。
 *
 * Author: Claude Code
 * Date: 2026-02-10
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Tiling 参数
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32
#define WARP_SIZE 32
#define BLOCK_SIZE_SCALE 32

// NVFP4 解码表
__constant__ float NVFP4_DECODE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float decode_nvfp4(uint8_t packed, int idx) {
    int nibble = (idx & 1) ? (packed >> 4) : (packed & 0xF);
    return NVFP4_DECODE[nibble];
}

__device__ __forceinline__ float gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coef * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * NVFP4 GEMM Kernel - Tiled 版本
 *
 * C[M, N] = A[M, K] @ dequant(W_packed[N, K/2])
 *
 * 优化:
 * 1. 2D tiling with shared memory
 * 2. 向量化内存访问
 * 3. Register blocking
 */
template<int THREAD_M = 4, int THREAD_N = 4>
__global__ void nvfp4_gemm_tiled_kernel(
    const float* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ W_packed, // [N, K/2]
    const float* __restrict__ scale_W,    // [N, K/BLOCK_SIZE_SCALE]
    const float* __restrict__ bias,       // [N] or nullptr
    float* __restrict__ C,                // [M, N]
    int M, int N, int K,
    int activation_type  // 0=none, 1=gelu, 2=silu
) {
    // Block 索引
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Thread 索引
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Warp 内索引
    const int warp_id = (ty * blockDim.x + tx) / WARP_SIZE;
    const int lane_id = (ty * blockDim.x + tx) % WARP_SIZE;

    // Block 内 thread 数量
    const int THREADS_X = BLOCK_N / THREAD_N;
    const int THREADS_Y = BLOCK_M / THREAD_M;

    // Shared memory
    __shared__ float As[BLOCK_M][BLOCK_K + 1];  // +1 避免 bank conflict
    __shared__ float Ws[BLOCK_K][BLOCK_N + 1];

    // 累加器 (每个 thread 计算 THREAD_M x THREAD_N 个元素)
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // 全局起始位置
    const int m_start = by * BLOCK_M;
    const int n_start = bx * BLOCK_N;
    const int num_blocks_k = K / BLOCK_SIZE_SCALE;

    // 每个 thread 负责加载的元素数
    const int num_threads = blockDim.x * blockDim.y;
    const int thread_id = ty * blockDim.x + tx;

    // K 维度循环
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // =============================================
        // 加载 A tile 到 shared memory
        // =============================================
        #pragma unroll
        for (int i = thread_id; i < BLOCK_M * BLOCK_K; i += num_threads) {
            int m_local = i / BLOCK_K;
            int k_local = i % BLOCK_K;
            int m = m_start + m_local;
            int k = k0 + k_local;

            if (m < M && k < K) {
                As[m_local][k_local] = A[m * K + k];
            } else {
                As[m_local][k_local] = 0.0f;
            }
        }

        // =============================================
        // 加载 W tile 到 shared memory (同时反量化)
        // =============================================
        #pragma unroll
        for (int i = thread_id; i < BLOCK_K * BLOCK_N; i += num_threads) {
            int k_local = i / BLOCK_N;
            int n_local = i % BLOCK_N;
            int k = k0 + k_local;
            int n = n_start + n_local;

            if (k < K && n < N) {
                // 获取 packed 权重
                int packed_idx = k / 2;
                uint8_t packed = W_packed[n * (K / 2) + packed_idx];
                float w_val = decode_nvfp4(packed, k);

                // 获取 scale
                int block_idx = k / BLOCK_SIZE_SCALE;
                float scale = scale_W[n * num_blocks_k + block_idx];

                Ws[k_local][n_local] = w_val * scale;
            } else {
                Ws[k_local][n_local] = 0.0f;
            }
        }

        __syncthreads();

        // =============================================
        // 计算 GEMM tile
        // =============================================
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            // 加载 A 和 W 的值到寄存器
            float a_reg[THREAD_M];
            float w_reg[THREAD_N];

            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                a_reg[i] = As[ty * THREAD_M + i][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                w_reg[j] = Ws[k][tx * THREAD_N + j];
            }

            // 累加
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    acc[i][j] += a_reg[i] * w_reg[j];
                }
            }
        }

        __syncthreads();
    }

    // =============================================
    // 写回结果
    // =============================================
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int gm = m_start + ty * THREAD_M + i;
            int gn = n_start + tx * THREAD_N + j;

            if (gm < M && gn < N) {
                float val = acc[i][j];

                // 添加 bias
                if (bias != nullptr) {
                    val += bias[gn];
                }

                // 激活函数
                if (activation_type == 1) {
                    val = gelu_tanh(val);
                } else if (activation_type == 2) {
                    val = silu(val);
                }

                C[gm * N + gn] = val;
            }
        }
    }
}

/**
 * 针对小 M 的优化版本 (M <= 16)
 * 使用 GEMV 风格的 warp reduce
 */
__global__ void nvfp4_gemm_small_m_kernel(
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scale_W,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K,
    int activation_type
) {
    // 每个 warp 处理一行输出
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= M * N) return;

    int m = warp_id / N;
    int n = warp_id % N;

    int num_blocks_k = K / BLOCK_SIZE_SCALE;

    // 每个 lane 处理 K/32 个元素
    float local_sum = 0.0f;
    int k_per_lane = K / WARP_SIZE;

    for (int i = 0; i < k_per_lane; i++) {
        int k = lane_id * k_per_lane + i;

        // 获取 A 值
        float a_val = A[m * K + k];

        // 获取并反量化 W 值
        int packed_idx = k / 2;
        uint8_t packed = W_packed[n * (K / 2) + packed_idx];
        float w_val = decode_nvfp4(packed, k);

        // 获取 scale
        int block_idx = k / BLOCK_SIZE_SCALE;
        float scale = scale_W[n * num_blocks_k + block_idx];

        local_sum += a_val * w_val * scale;
    }

    // Warp reduce
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // 写回结果
    if (lane_id == 0) {
        float val = local_sum;

        if (bias != nullptr) {
            val += bias[n];
        }

        if (activation_type == 1) {
            val = gelu_tanh(val);
        } else if (activation_type == 2) {
            val = silu(val);
        }

        C[m * N + n] = val;
    }
}

// ============================================================================
// C 接口
// ============================================================================

extern "C" {

void nvfp4_gemm_cuda(
    const float* A,
    const uint8_t* W_packed,
    const float* scale_W,
    const float* bias,
    float* C,
    int M, int N, int K,
    int activation_type,
    cudaStream_t stream
) {
    if (M <= 16) {
        // 小 M: 使用 warp reduce 版本
        int num_warps = M * N;
        int threads_per_block = 256;
        int num_blocks = (num_warps * WARP_SIZE + threads_per_block - 1) / threads_per_block;

        nvfp4_gemm_small_m_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            A, W_packed, scale_W, bias, C, M, N, K, activation_type
        );
    } else {
        // 大 M: 使用 tiled GEMM
        const int THREAD_M = 4;
        const int THREAD_N = 4;

        dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

        nvfp4_gemm_tiled_kernel<THREAD_M, THREAD_N><<<grid, block, 0, stream>>>(
            A, W_packed, scale_W, bias, C, M, N, K, activation_type
        );
    }
}

}  // extern "C"
