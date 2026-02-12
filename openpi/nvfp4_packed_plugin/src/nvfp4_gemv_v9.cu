/**
 * NVFP4 GEMV V9 - 最终优化版本
 *
 * 核心优化:
 * 1. 使用 shared memory 缓存权重 tile，实现 coalesced 全局内存访问
 * 2. 把 K 维度分成多个 tile，每个 tile 使用 shared memory
 * 3. 激活复用 - 一次加载，多次使用
 * 4. 向量化内存访问
 *
 * 访问模式分析:
 * - 权重存储: W[N, K/2] (packed FP4)
 * - 直接访问: 每个线程访问不同的 N，相邻线程间距离 K/2 bytes -> 非 coalesced
 * - 优化方案: 使用 shared memory 做 transpose，实现 coalesced 访问
 *
 * Author: Claude Code
 * Date: 2026-02-10
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE_SCALE 32
#define WARP_SIZE 32

// NVFP4 decode table
__constant__ float NVFP4_DECODE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

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

// ============================================================================
// V9: Tiled GEMV with Coalesced Memory Access
//
// 策略:
// - 每个 block 处理 N_TILE 个输出
// - K 分成多个 K_TILE
// - 使用 shared memory 缓存:
//   1. 激活 tile: A[K_TILE]
//   2. 权重 tile: W[N_TILE, K_TILE/2] (packed)
//   3. Scale tile: scale[N_TILE]
// ============================================================================

#define N_TILE 128      // 每个 block 处理的 N 数量
#define K_TILE 256      // 每次处理的 K 数量
#define THREADS 256

__global__ void nvfp4_gemv_v9_kernel(
    const float* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ W_packed, // [N, K/2]
    const float* __restrict__ scale_W,    // [N, num_blocks_k]
    const float* __restrict__ bias,
    float* __restrict__ C,                // [M, N]
    int M, int N, int K,
    int activation_type
) {
    const int block_n_start = blockIdx.x * N_TILE;
    const int m = blockIdx.y;

    if (m >= M) return;

    const int tid = threadIdx.x;
    const int num_blocks_k = K / BLOCK_SIZE_SCALE;
    const int K_half = K / 2;

    // Shared memory
    __shared__ float A_tile[K_TILE];                    // 激活 tile
    __shared__ uint8_t W_tile[N_TILE][K_TILE / 2 + 1];  // 权重 tile (+1 避免 bank conflict)
    __shared__ float scale_tile[N_TILE];                // 当前 scale

    // 每个线程的累加器
    float acc[N_TILE / THREADS + 1];
    const int n_per_thread = (N_TILE + THREADS - 1) / THREADS;
    #pragma unroll
    for (int i = 0; i < n_per_thread; i++) {
        acc[i] = 0.0f;
    }

    // K 维度分 tile 处理
    for (int k_start = 0; k_start < K; k_start += K_TILE) {
        int k_tile_size = min(K_TILE, K - k_start);

        // =========================================================
        // Phase 1: 协作加载激活到 shared memory (coalesced)
        // =========================================================
        for (int k = tid; k < k_tile_size; k += THREADS) {
            A_tile[k] = A[m * K + k_start + k];
        }

        // =========================================================
        // Phase 2: 协作加载权重到 shared memory
        // 这是关键优化点 - 使用 coalesced 访问模式
        // =========================================================
        // 每个线程负责加载一部分权重
        // 加载模式: 线程 t 加载 W[n, k_start + t*2 : k_start + t*2 + 2]
        // 然后 transpose 存到 shared memory

        int packed_tile_size = k_tile_size / 2;
        int total_packed_elements = N_TILE * packed_tile_size;

        for (int idx = tid; idx < total_packed_elements; idx += THREADS) {
            int local_n = idx / packed_tile_size;
            int local_k_packed = idx % packed_tile_size;
            int global_n = block_n_start + local_n;

            if (global_n < N) {
                W_tile[local_n][local_k_packed] = W_packed[global_n * K_half + (k_start / 2) + local_k_packed];
            } else {
                W_tile[local_n][local_k_packed] = 0;
            }
        }

        // =========================================================
        // Phase 3: 加载 scales (每个 K_TILE 可能有多个 scale block)
        // =========================================================
        int scale_block_start = k_start / BLOCK_SIZE_SCALE;

        for (int local_n = tid; local_n < N_TILE; local_n += THREADS) {
            int global_n = block_n_start + local_n;
            if (global_n < N) {
                scale_tile[local_n] = scale_W[global_n * num_blocks_k + scale_block_start];
            }
        }

        __syncthreads();

        // =========================================================
        // Phase 4: 计算 - 每个线程处理多个 N 输出
        // =========================================================
        for (int i = 0; i < n_per_thread; i++) {
            int local_n = tid + i * THREADS;
            if (local_n >= N_TILE) continue;

            int global_n = block_n_start + local_n;
            if (global_n >= N) continue;

            float local_sum = 0.0f;
            float w_scale = scale_tile[local_n];

            // 内层 K 循环
            #pragma unroll 8
            for (int local_k = 0; local_k < k_tile_size; local_k += 2) {
                // 更新 scale (如果跨越了 scale block 边界)
                int global_k = k_start + local_k;
                if ((global_k % BLOCK_SIZE_SCALE == 0) && (local_k > 0)) {
                    int new_block = global_k / BLOCK_SIZE_SCALE;
                    w_scale = scale_W[global_n * num_blocks_k + new_block];
                }

                uint8_t packed = W_tile[local_n][local_k / 2];
                float w0 = NVFP4_DECODE[packed & 0xF] * w_scale;
                float w1 = NVFP4_DECODE[(packed >> 4) & 0xF] * w_scale;

                float a0 = A_tile[local_k];
                float a1 = A_tile[local_k + 1];

                local_sum += a0 * w0 + a1 * w1;
            }

            acc[i] += local_sum;
        }

        __syncthreads();
    }

    // =========================================================
    // Phase 5: 写回结果
    // =========================================================
    for (int i = 0; i < n_per_thread; i++) {
        int local_n = tid + i * THREADS;
        if (local_n >= N_TILE) continue;

        int global_n = block_n_start + local_n;
        if (global_n >= N) continue;

        float val = acc[i];

        if (bias != nullptr) {
            val += bias[global_n];
        }

        if (activation_type == 1) {
            val = gelu_tanh(val);
        } else if (activation_type == 2) {
            val = silu(val);
        }

        C[m * N + global_n] = val;
    }
}

// ============================================================================
// V10: 进一步优化 - 使用寄存器 tiling 和向量化
// ============================================================================

#define N_TILE_V10 256
#define K_TILE_V10 128
#define THREADS_V10 256

__global__ void nvfp4_gemv_v10_kernel(
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scale_W,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K,
    int activation_type
) {
    const int block_n_start = blockIdx.x * N_TILE_V10;
    const int m = blockIdx.y;

    if (m >= M) return;

    const int tid = threadIdx.x;
    const int num_blocks_k = K / BLOCK_SIZE_SCALE;
    const int K_half = K / 2;

    // 每个线程处理一个 N 输出
    const int local_n = tid;
    const int global_n = block_n_start + local_n;

    // 激活缓存 (shared memory)
    __shared__ float A_shared[2048];

    // 协作加载激活
    for (int k = tid; k < K; k += THREADS_V10) {
        A_shared[k] = A[m * K + k];
    }
    __syncthreads();

    if (global_n >= N) return;

    // 累加器
    float acc = 0.0f;

    // 预取 scale
    int current_block = 0;
    float w_scale = scale_W[global_n * num_blocks_k + 0];

    // K 循环 - 向量化处理
    const uint32_t* W_vec = reinterpret_cast<const uint32_t*>(W_packed + global_n * K_half);

    for (int k = 0; k < K; k += 8) {
        // 更新 scale
        int block_idx = k / BLOCK_SIZE_SCALE;
        if (block_idx != current_block) {
            current_block = block_idx;
            w_scale = scale_W[global_n * num_blocks_k + block_idx];
        }

        // 加载 8 个 FP4 值 (4 bytes)
        uint32_t packed4 = W_vec[k / 8];

        // 解包
        float w0 = NVFP4_DECODE[(packed4) & 0xF] * w_scale;
        float w1 = NVFP4_DECODE[(packed4 >> 4) & 0xF] * w_scale;
        float w2 = NVFP4_DECODE[(packed4 >> 8) & 0xF] * w_scale;
        float w3 = NVFP4_DECODE[(packed4 >> 12) & 0xF] * w_scale;
        float w4 = NVFP4_DECODE[(packed4 >> 16) & 0xF] * w_scale;
        float w5 = NVFP4_DECODE[(packed4 >> 20) & 0xF] * w_scale;
        float w6 = NVFP4_DECODE[(packed4 >> 24) & 0xF] * w_scale;
        float w7 = NVFP4_DECODE[(packed4 >> 28) & 0xF] * w_scale;

        // 计算
        acc += A_shared[k] * w0 + A_shared[k+1] * w1;
        acc += A_shared[k+2] * w2 + A_shared[k+3] * w3;
        acc += A_shared[k+4] * w4 + A_shared[k+5] * w5;
        acc += A_shared[k+6] * w6 + A_shared[k+7] * w7;
    }

    // 写回
    float val = acc;
    if (bias != nullptr) {
        val += bias[global_n];
    }
    if (activation_type == 1) {
        val = gelu_tanh(val);
    } else if (activation_type == 2) {
        val = silu(val);
    }
    C[m * N + global_n] = val;
}

// ============================================================================
// V11: 最简单但可能最快 - 直接全局内存访问 + 大量 unrolling
// 适合 K 较小的情况
// ============================================================================

__global__ void nvfp4_gemv_v11_kernel(
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scale_W,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K,
    int activation_type
) {
    // 每个线程处理一个 N 输出
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (m >= M || n >= N) return;

    const int num_blocks_k = K / BLOCK_SIZE_SCALE;
    const int K_half = K / 2;

    // 直接从全局内存读取激活和权重
    const float* A_row = A + m * K;
    const uint8_t* W_row = W_packed + n * K_half;

    float acc = 0.0f;

    // K 循环
    int k = 0;

    // 处理每个 scale block
    for (int block_idx = 0; block_idx < num_blocks_k; block_idx++) {
        float w_scale = scale_W[n * num_blocks_k + block_idx];

        // 每个 block 32 个元素，16 个 packed bytes
        #pragma unroll 4
        for (int i = 0; i < 16; i++) {
            uint8_t packed = W_row[block_idx * 16 + i];
            float w0 = NVFP4_DECODE[packed & 0xF] * w_scale;
            float w1 = NVFP4_DECODE[(packed >> 4) & 0xF] * w_scale;

            float a0 = A_row[k];
            float a1 = A_row[k + 1];
            k += 2;

            acc += a0 * w0 + a1 * w1;
        }
    }

    // 激活函数
    if (bias != nullptr) {
        acc += bias[n];
    }
    if (activation_type == 1) {
        acc = gelu_tanh(acc);
    } else if (activation_type == 2) {
        acc = silu(acc);
    }

    C[m * N + n] = acc;
}

// ============================================================================
// PyTorch Interface
// ============================================================================

torch::Tensor nvfp4_gemv_v9(
    torch::Tensor activation,
    torch::Tensor weight_packed,
    torch::Tensor scale_W,
    c10::optional<torch::Tensor> bias,
    int M, int N, int K,
    int activation_type
) {
    activation = activation.to(torch::kFloat32).contiguous();
    weight_packed = weight_packed.contiguous();
    scale_W = scale_W.to(torch::kFloat32).contiguous();

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(activation.device());
    torch::Tensor output = torch::zeros({M, N}, options);

    const float* A_ptr = activation.data_ptr<float>();
    const uint8_t* W_ptr = weight_packed.data_ptr<uint8_t>();
    const float* scale_W_ptr = scale_W.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().numel() > 0) {
        auto b = bias.value().to(torch::kFloat32);
        bias_ptr = b.data_ptr<float>();
    }
    float* C_ptr = output.data_ptr<float>();

    dim3 block(THREADS);
    dim3 grid((N + N_TILE - 1) / N_TILE, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_v9_kernel<<<grid, block, 0, stream>>>(
        A_ptr, W_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

torch::Tensor nvfp4_gemv_v10(
    torch::Tensor activation,
    torch::Tensor weight_packed,
    torch::Tensor scale_W,
    c10::optional<torch::Tensor> bias,
    int M, int N, int K,
    int activation_type
) {
    activation = activation.to(torch::kFloat32).contiguous();
    weight_packed = weight_packed.contiguous();
    scale_W = scale_W.to(torch::kFloat32).contiguous();

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(activation.device());
    torch::Tensor output = torch::zeros({M, N}, options);

    const float* A_ptr = activation.data_ptr<float>();
    const uint8_t* W_ptr = weight_packed.data_ptr<uint8_t>();
    const float* scale_W_ptr = scale_W.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().numel() > 0) {
        auto b = bias.value().to(torch::kFloat32);
        bias_ptr = b.data_ptr<float>();
    }
    float* C_ptr = output.data_ptr<float>();

    dim3 block(THREADS_V10);
    dim3 grid((N + N_TILE_V10 - 1) / N_TILE_V10, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_v10_kernel<<<grid, block, 0, stream>>>(
        A_ptr, W_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

torch::Tensor nvfp4_gemv_v11(
    torch::Tensor activation,
    torch::Tensor weight_packed,
    torch::Tensor scale_W,
    c10::optional<torch::Tensor> bias,
    int M, int N, int K,
    int activation_type
) {
    activation = activation.to(torch::kFloat32).contiguous();
    weight_packed = weight_packed.contiguous();
    scale_W = scale_W.to(torch::kFloat32).contiguous();

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(activation.device());
    torch::Tensor output = torch::zeros({M, N}, options);

    const float* A_ptr = activation.data_ptr<float>();
    const uint8_t* W_ptr = weight_packed.data_ptr<uint8_t>();
    const float* scale_W_ptr = scale_W.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().numel() > 0) {
        auto b = bias.value().to(torch::kFloat32);
        bias_ptr = b.data_ptr<float>();
    }
    float* C_ptr = output.data_ptr<float>();

    constexpr int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_v11_kernel<<<grid, block, 0, stream>>>(
        A_ptr, W_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_gemv_v9", &nvfp4_gemv_v9,
          "NVFP4 GEMV v9 (tiled with shared memory)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("activation_type"));

    m.def("nvfp4_gemv_v10", &nvfp4_gemv_v10,
          "NVFP4 GEMV v10 (vectorized)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("activation_type"));

    m.def("nvfp4_gemv_v11", &nvfp4_gemv_v11,
          "NVFP4 GEMV v11 (simple direct)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("activation_type"));
}
