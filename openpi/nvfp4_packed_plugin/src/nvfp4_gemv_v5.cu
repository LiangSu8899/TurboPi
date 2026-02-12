/**
 * NVFP4 GEMV V5 - 高效版本
 *
 * 核心优化:
 * 1. 每个 block 处理多个 N 输出 (N_PER_BLOCK)
 * 2. 使用 shared memory 缓存激活
 * 3. 向量化内存访问 (float4 + uint32)
 * 4. 最大化内存带宽利用
 *
 * 目标: 接近 memory bandwidth 理论上限
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

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
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

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ============================================================================
// V5: 每个 block 处理多个 N 输出
// 关键: 把权重加载到 shared memory，然后让所有 thread 复用
// ============================================================================
template<int THREADS = 256, int N_PER_BLOCK = 8>
__global__ void nvfp4_gemv_v5_kernel(
    const float* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ W_packed, // [N, K/2]
    const float* __restrict__ scale_W,    // [N, num_blocks_k]
    const float* __restrict__ bias,
    float* __restrict__ C,                // [M, N]
    int M, int N, int K,
    int activation_type
) {
    const int block_n_start = blockIdx.x * N_PER_BLOCK;
    const int m = blockIdx.y;

    if (m >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = THREADS / WARP_SIZE;

    const int num_blocks_k = K / BLOCK_SIZE_SCALE;
    const int K_half = K / 2;

    // 每个 warp 负责一个 N 输出
    // 每个 block 有 num_warps 个 warps，处理 N_PER_BLOCK 个输出
    // 需要多轮处理如果 N_PER_BLOCK > num_warps

    // Shared memory: 缓存激活
    __shared__ float A_shared[2048];  // Max K = 2048

    // 协作加载激活
    for (int k = tid; k < K; k += THREADS) {
        A_shared[k] = A[m * K + k];
    }
    __syncthreads();

    // 每个 warp 处理一个 N 输出
    for (int local_n = warp_id; local_n < N_PER_BLOCK; local_n += num_warps) {
        int n = block_n_start + local_n;
        if (n >= N) continue;

        float local_sum = 0.0f;

        // 每个 lane 处理 K/32 个元素
        int k_per_lane = K / WARP_SIZE;
        int k_start = lane_id * k_per_lane;

        // 预加载 scale (可能跨多个 block)
        #pragma unroll 2
        for (int k = k_start; k < k_start + k_per_lane; k += 2) {
            int block_idx = k / BLOCK_SIZE_SCALE;
            float w_scale = scale_W[n * num_blocks_k + block_idx];

            // 加载 packed 权重
            uint8_t packed = W_packed[n * K_half + k / 2];
            float w0 = NVFP4_DECODE[packed & 0xF] * w_scale;
            float w1 = NVFP4_DECODE[(packed >> 4) & 0xF] * w_scale;

            // 从 shared memory 读取激活
            float a0 = A_shared[k];
            float a1 = A_shared[k + 1];

            local_sum += a0 * w0 + a1 * w1;
        }

        // Warp reduce
        float total = warpReduceSum(local_sum);

        // 写回结果
        if (lane_id == 0) {
            float val = total;
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
}

// ============================================================================
// V6: 更激进的优化 - 每个 warp 处理多个 N，使用寄存器累加
// ============================================================================
template<int THREADS = 256, int N_PER_WARP = 4>
__global__ void nvfp4_gemv_v6_kernel(
    const float* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ W_packed, // [N, K/2]
    const float* __restrict__ scale_W,    // [N, num_blocks_k]
    const float* __restrict__ bias,
    float* __restrict__ C,                // [M, N]
    int M, int N, int K,
    int activation_type
) {
    const int num_warps = THREADS / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // 每个 block 处理 num_warps * N_PER_WARP 个 N 输出
    const int N_PER_BLOCK = num_warps * N_PER_WARP;
    const int block_n_start = blockIdx.x * N_PER_BLOCK;
    const int m = blockIdx.y;

    if (m >= M) return;

    const int num_blocks_k = K / BLOCK_SIZE_SCALE;
    const int K_half = K / 2;

    // Shared memory: 激活 (最大支持 K=8192, 需要32KB)
    // Thor有48KB shared memory per block，足够用
    __shared__ float A_shared[8192];

    // 协作加载激活
    for (int k = threadIdx.x; k < K; k += THREADS) {
        A_shared[k] = A[m * K + k];
    }
    __syncthreads();

    // 每个 warp 处理 N_PER_WARP 个输出
    float local_sums[N_PER_WARP] = {0.0f};

    // 每个 lane 处理 K/32 个元素
    int k_per_lane = K / WARP_SIZE;
    int k_start = lane_id * k_per_lane;

    // 计算这个 warp 负责的 N 起始位置
    int warp_n_start = block_n_start + warp_id * N_PER_WARP;

    // K 循环
    #pragma unroll 2
    for (int k = k_start; k < k_start + k_per_lane; k += 2) {
        int block_idx = k / BLOCK_SIZE_SCALE;

        // 加载激活 (复用给所有 N)
        float a0 = A_shared[k];
        float a1 = A_shared[k + 1];

        // 处理 N_PER_WARP 个输出
        #pragma unroll
        for (int i = 0; i < N_PER_WARP; i++) {
            int n = warp_n_start + i;
            if (n >= N) continue;

            float w_scale = scale_W[n * num_blocks_k + block_idx];

            uint8_t packed = W_packed[n * K_half + k / 2];
            float w0 = NVFP4_DECODE[packed & 0xF] * w_scale;
            float w1 = NVFP4_DECODE[(packed >> 4) & 0xF] * w_scale;

            local_sums[i] += a0 * w0 + a1 * w1;
        }
    }

    // Warp reduce 每个输出
    #pragma unroll
    for (int i = 0; i < N_PER_WARP; i++) {
        float total = warpReduceSum(local_sums[i]);

        if (lane_id == 0) {
            int n = warp_n_start + i;
            if (n < N) {
                float val = total;
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
    }
}

// ============================================================================
// V7: 最激进 - 每个线程处理一个完整的 N 输出
// 适合 K 较小的情况
// ============================================================================
template<int THREADS = 256>
__global__ void nvfp4_gemv_v7_kernel(
    const float* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ W_packed, // [N, K/2]
    const float* __restrict__ scale_W,    // [N, num_blocks_k]
    const float* __restrict__ bias,
    float* __restrict__ C,                // [M, N]
    int M, int N, int K,
    int activation_type
) {
    // 每个 block 处理 THREADS 个 N 输出
    const int block_n_start = blockIdx.x * THREADS;
    const int m = blockIdx.y;

    if (m >= M) return;

    const int n = block_n_start + threadIdx.x;
    if (n >= N) return;

    const int num_blocks_k = K / BLOCK_SIZE_SCALE;
    const int K_half = K / 2;

    // Shared memory: 激活
    __shared__ float A_shared[2048];

    // 协作加载激活
    for (int k = threadIdx.x; k < K; k += THREADS) {
        A_shared[k] = A[m * K + k];
    }
    __syncthreads();

    // 每个线程处理整个 K 维度
    float local_sum = 0.0f;

    int block_idx = 0;
    float w_scale = scale_W[n * num_blocks_k + 0];

    #pragma unroll 8
    for (int k = 0; k < K; k += 2) {
        // 更新 scale (每 32 个元素)
        int new_block_idx = k / BLOCK_SIZE_SCALE;
        if (new_block_idx != block_idx) {
            block_idx = new_block_idx;
            w_scale = scale_W[n * num_blocks_k + block_idx];
        }

        uint8_t packed = W_packed[n * K_half + k / 2];
        float w0 = NVFP4_DECODE[packed & 0xF] * w_scale;
        float w1 = NVFP4_DECODE[(packed >> 4) & 0xF] * w_scale;

        float a0 = A_shared[k];
        float a1 = A_shared[k + 1];

        local_sum += a0 * w0 + a1 * w1;
    }

    // 直接写回 (无需 reduce)
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

// ============================================================================
// V8: 向量化版本 - float4 + uint32
// ============================================================================
template<int THREADS = 256>
__global__ void nvfp4_gemv_v8_kernel(
    const float4* __restrict__ A4,        // [M, K/4]
    const uint32_t* __restrict__ W4,      // [N, K/8]
    const float* __restrict__ scale_W,    // [N, num_blocks_k]
    const float* __restrict__ bias,
    float* __restrict__ C,                // [M, N]
    int M, int N, int K,
    int activation_type
) {
    const int block_n_start = blockIdx.x * THREADS;
    const int m = blockIdx.y;

    if (m >= M) return;

    const int n = block_n_start + threadIdx.x;
    if (n >= N) return;

    const int num_blocks_k = K / BLOCK_SIZE_SCALE;
    const int K_div_8 = K / 8;
    const int K_div_4 = K / 4;

    // Shared memory: 激活 (as float4)
    __shared__ float4 A4_shared[512];  // Max K/4 = 512

    // 协作加载激活
    for (int i = threadIdx.x; i < K_div_4; i += THREADS) {
        A4_shared[i] = A4[m * K_div_4 + i];
    }
    __syncthreads();

    float local_sum = 0.0f;

    // 每 8 个 k 一组处理
    #pragma unroll 4
    for (int g = 0; g < K_div_8; g++) {
        int k = g * 8;
        int block_idx = k / BLOCK_SIZE_SCALE;
        float w_scale = scale_W[n * num_blocks_k + block_idx];

        // 加载 8 个 FP4
        uint32_t w_packed = W4[n * K_div_8 + g];

        // 加载 8 个激活 (2 个 float4)
        float4 a_lo = A4_shared[k / 4];
        float4 a_hi = A4_shared[k / 4 + 1];

        // 解包并计算
        float w0 = NVFP4_DECODE[(w_packed) & 0xF] * w_scale;
        float w1 = NVFP4_DECODE[(w_packed >> 4) & 0xF] * w_scale;
        float w2 = NVFP4_DECODE[(w_packed >> 8) & 0xF] * w_scale;
        float w3 = NVFP4_DECODE[(w_packed >> 12) & 0xF] * w_scale;
        float w4 = NVFP4_DECODE[(w_packed >> 16) & 0xF] * w_scale;
        float w5 = NVFP4_DECODE[(w_packed >> 20) & 0xF] * w_scale;
        float w6 = NVFP4_DECODE[(w_packed >> 24) & 0xF] * w_scale;
        float w7 = NVFP4_DECODE[(w_packed >> 28) & 0xF] * w_scale;

        local_sum += a_lo.x * w0 + a_lo.y * w1 + a_lo.z * w2 + a_lo.w * w3;
        local_sum += a_hi.x * w4 + a_hi.y * w5 + a_hi.z * w6 + a_hi.w * w7;
    }

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

// ============================================================================
// PyTorch Interface
// ============================================================================

torch::Tensor nvfp4_gemv_v5(
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

    torch::Tensor bias_tensor;
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().numel() > 0) {
        bias_tensor = bias.value().to(torch::kFloat32).contiguous();
        bias_ptr = bias_tensor.data_ptr<float>();
    }
    float* C_ptr = output.data_ptr<float>();

    constexpr int THREADS = 256;
    constexpr int N_PER_BLOCK = 8;
    dim3 block(THREADS);
    dim3 grid((N + N_PER_BLOCK - 1) / N_PER_BLOCK, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_v5_kernel<THREADS, N_PER_BLOCK><<<grid, block, 0, stream>>>(
        A_ptr, W_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

torch::Tensor nvfp4_gemv_v6(
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

    // 保持 bias tensor 在作用域内，防止指针失效
    torch::Tensor bias_tensor;
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().numel() > 0) {
        bias_tensor = bias.value().to(torch::kFloat32).contiguous();
        bias_ptr = bias_tensor.data_ptr<float>();
    }
    float* C_ptr = output.data_ptr<float>();

    constexpr int THREADS = 256;
    constexpr int N_PER_WARP = 4;
    constexpr int num_warps = THREADS / WARP_SIZE;
    constexpr int N_PER_BLOCK = num_warps * N_PER_WARP;

    dim3 block(THREADS);
    dim3 grid((N + N_PER_BLOCK - 1) / N_PER_BLOCK, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_v6_kernel<THREADS, N_PER_WARP><<<grid, block, 0, stream>>>(
        A_ptr, W_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

torch::Tensor nvfp4_gemv_v7(
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

    torch::Tensor bias_tensor;
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().numel() > 0) {
        bias_tensor = bias.value().to(torch::kFloat32).contiguous();
        bias_ptr = bias_tensor.data_ptr<float>();
    }
    float* C_ptr = output.data_ptr<float>();

    constexpr int THREADS = 256;
    dim3 block(THREADS);
    dim3 grid((N + THREADS - 1) / THREADS, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_v7_kernel<THREADS><<<grid, block, 0, stream>>>(
        A_ptr, W_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

torch::Tensor nvfp4_gemv_v8(
    torch::Tensor activation,
    torch::Tensor weight_packed,
    torch::Tensor scale_W,
    c10::optional<torch::Tensor> bias,
    int M, int N, int K,
    int activation_type
) {
    TORCH_CHECK(K % 8 == 0, "K must be divisible by 8 for v8 kernel");

    activation = activation.to(torch::kFloat32).contiguous();
    weight_packed = weight_packed.contiguous();
    scale_W = scale_W.to(torch::kFloat32).contiguous();

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(activation.device());
    torch::Tensor output = torch::zeros({M, N}, options);

    const float4* A4_ptr = reinterpret_cast<const float4*>(activation.data_ptr<float>());
    const uint32_t* W4_ptr = reinterpret_cast<const uint32_t*>(weight_packed.data_ptr<uint8_t>());
    const float* scale_W_ptr = scale_W.data_ptr<float>();

    torch::Tensor bias_tensor;
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().numel() > 0) {
        bias_tensor = bias.value().to(torch::kFloat32).contiguous();
        bias_ptr = bias_tensor.data_ptr<float>();
    }
    float* C_ptr = output.data_ptr<float>();

    constexpr int THREADS = 256;
    dim3 block(THREADS);
    dim3 grid((N + THREADS - 1) / THREADS, M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_v8_kernel<THREADS><<<grid, block, 0, stream>>>(
        A4_ptr, W4_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_gemv_v5", &nvfp4_gemv_v5,
          "NVFP4 GEMV v5 (N per block)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("activation_type"));

    m.def("nvfp4_gemv_v6", &nvfp4_gemv_v6,
          "NVFP4 GEMV v6 (N per warp)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("activation_type"));

    m.def("nvfp4_gemv_v7", &nvfp4_gemv_v7,
          "NVFP4 GEMV v7 (thread per N)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("activation_type"));

    m.def("nvfp4_gemv_v8", &nvfp4_gemv_v8,
          "NVFP4 GEMV v8 (float4 + uint32 vectorized)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("activation_type"));
}
