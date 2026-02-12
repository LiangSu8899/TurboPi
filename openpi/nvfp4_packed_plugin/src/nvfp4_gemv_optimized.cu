/**
 * NVFP4 Optimized GEMV Kernel
 *
 * 针对 bs=1 场景深度优化的 GEMV kernel。
 *
 * 优化策略:
 * 1. 每个 block 处理多个输出元素 (减少 kernel launch 开销)
 * 2. 向量化内存访问 (uint4 = 8 bytes = 16 个 FP4 值)
 * 3. 使用 shared memory 缓存激活
 * 4. Loop unrolling
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
// 优化版本 1: 一个 block 处理一行输出 (N 个元素)
// 适合 M=1 的情况
// ============================================================================
template<int THREADS = 256, int OUTPUTS_PER_BLOCK = 4>
__global__ void nvfp4_gemv_opt_v1_kernel(
    const float* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ W_packed, // [N, K/2]
    const float* __restrict__ scale_W,    // [N, num_blocks_k]
    const float* __restrict__ bias,
    float* __restrict__ C,                // [M, N]
    int M, int N, int K,
    int activation_type
) {
    // 每个 block 处理 OUTPUTS_PER_BLOCK 个输出
    const int n_start = blockIdx.x * OUTPUTS_PER_BLOCK;
    const int m = blockIdx.y;

    if (m >= M) return;

    const int num_blocks_k = K / BLOCK_SIZE_SCALE;
    const int tid = threadIdx.x;
    const int K_half = K / 2;

    // Shared memory 缓存激活
    __shared__ float A_shared[2048];  // 最大 K=2048

    // 协作加载激活到 shared memory
    for (int k = tid; k < K; k += THREADS) {
        A_shared[k] = A[m * K + k];
    }
    __syncthreads();

    // 每个线程处理多个 k 值
    const int k_per_thread = K / THREADS;

    #pragma unroll
    for (int out_idx = 0; out_idx < OUTPUTS_PER_BLOCK; out_idx++) {
        int n = n_start + out_idx;
        if (n >= N) continue;

        float local_sum = 0.0f;

        // 每个线程处理 k_per_thread 个 K 元素
        int k_start = tid * k_per_thread;
        int k_end = k_start + k_per_thread;

        // 向量化处理 (每次处理 8 个 k，即 4 个 packed bytes)
        #pragma unroll 4
        for (int k = k_start; k < k_end; k += 8) {
            int block_idx = k / BLOCK_SIZE_SCALE;
            float w_scale = scale_W[n * num_blocks_k + block_idx];

            // 加载 4 个 packed bytes = 8 个 FP4 值
            int packed_idx = k / 2;

            // 展开处理 8 个元素
            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                uint8_t packed = W_packed[n * K_half + packed_idx + i/2];
                float w0 = NVFP4_DECODE[packed & 0xF] * w_scale;
                float w1 = NVFP4_DECODE[(packed >> 4) & 0xF] * w_scale;

                local_sum += A_shared[k + i] * w0;
                local_sum += A_shared[k + i + 1] * w1;
            }
        }

        // Warp reduce
        local_sum = warpReduceSum(local_sum);

        // 跨 warp reduce
        __shared__ float warp_sums[8];  // THREADS/WARP_SIZE
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
            warp_sums[warp_id] = local_sum;
        }
        __syncthreads();

        // 第一个 warp 做最终 reduce
        if (tid < THREADS / WARP_SIZE) {
            local_sum = warp_sums[tid];
        } else {
            local_sum = 0.0f;
        }
        local_sum = warpReduceSum(local_sum);

        // 写回结果
        if (tid == 0) {
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
        __syncthreads();
    }
}

// ============================================================================
// 优化版本 2: 每个 warp 处理一个输出，向量化加载
// ============================================================================
template<int THREADS = 256>
__global__ void nvfp4_gemv_opt_v2_kernel(
    const float* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ W_packed, // [N, K/2]
    const float* __restrict__ scale_W,    // [N, num_blocks_k]
    const float* __restrict__ bias,
    float* __restrict__ C,                // [M, N]
    int M, int N, int K,
    int activation_type
) {
    const int warps_per_block = THREADS / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int global_warp = blockIdx.x * warps_per_block + warp_id;
    const int m = global_warp / N;
    const int n = global_warp % N;

    if (m >= M || n >= N) return;

    const int num_blocks_k = K / BLOCK_SIZE_SCALE;
    const int K_half = K / 2;

    // 每个 lane 处理 K/32 个元素
    const int elements_per_lane = K / WARP_SIZE;

    float local_sum = 0.0f;

    // 使用 uint4 向量化加载 (16 bytes = 32 个 FP4 值)
    const int k_start = lane_id * elements_per_lane;

    // 预加载 scale (一个 lane 可能跨多个 scale block)
    #pragma unroll 2
    for (int k = k_start; k < k_start + elements_per_lane; k += 16) {
        int block_idx = k / BLOCK_SIZE_SCALE;
        float w_scale = scale_W[n * num_blocks_k + block_idx];

        // 加载 8 bytes = 16 个 FP4 值
        int packed_base = n * K_half + k / 2;

        // 手动展开以获得更好的指令级并行
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            uint8_t packed = W_packed[packed_base + i/2];
            float w0 = NVFP4_DECODE[packed & 0xF] * w_scale;
            float w1 = NVFP4_DECODE[(packed >> 4) & 0xF] * w_scale;

            float a0 = A[m * K + k + i];
            float a1 = A[m * K + k + i + 1];

            local_sum += a0 * w0 + a1 * w1;
        }
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

// ============================================================================
// 优化版本 3: Block-level 处理，最大化内存带宽利用
// 每个 block 处理一个 (m, n) 输出，全部线程协作
// ============================================================================
template<int THREADS = 256>
__global__ void nvfp4_gemv_opt_v3_kernel(
    const float* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ W_packed, // [N, K/2]
    const float* __restrict__ scale_W,    // [N, num_blocks_k]
    const float* __restrict__ bias,
    float* __restrict__ C,                // [M, N]
    int M, int N, int K,
    int activation_type
) {
    const int global_idx = blockIdx.x;
    const int m = global_idx / N;
    const int n = global_idx % N;

    if (m >= M) return;

    const int num_blocks_k = K / BLOCK_SIZE_SCALE;
    const int K_half = K / 2;
    const int tid = threadIdx.x;

    // 每个线程处理 K/THREADS 个元素
    const int elements_per_thread = K / THREADS;
    const int k_start = tid * elements_per_thread;

    float local_sum = 0.0f;

    // 预取第一个 scale
    int block_idx = k_start / BLOCK_SIZE_SCALE;
    float w_scale = scale_W[n * num_blocks_k + block_idx];

    #pragma unroll 4
    for (int k = k_start; k < k_start + elements_per_thread; k += 2) {
        // 检查是否需要更新 scale
        int new_block_idx = k / BLOCK_SIZE_SCALE;
        if (new_block_idx != block_idx) {
            block_idx = new_block_idx;
            w_scale = scale_W[n * num_blocks_k + block_idx];
        }

        // 加载 packed 权重
        uint8_t packed = W_packed[n * K_half + k / 2];
        float w0 = NVFP4_DECODE[packed & 0xF] * w_scale;
        float w1 = NVFP4_DECODE[(packed >> 4) & 0xF] * w_scale;

        // 加载激活
        float a0 = A[m * K + k];
        float a1 = A[m * K + k + 1];

        local_sum += a0 * w0 + a1 * w1;
    }

    // Block-level reduce
    __shared__ float shared_sum[THREADS];
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduction
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // 写回结果
    if (tid == 0) {
        float val = shared_sum[0];
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
// 优化版本 4: Coalesced memory access + float4 加载激活
// ============================================================================
template<int THREADS = 256>
__global__ void nvfp4_gemv_opt_v4_kernel(
    const float4* __restrict__ A4,        // [M, K/4] as float4
    const uint32_t* __restrict__ W4,      // [N, K/8] as uint32 (8 FP4 values)
    const float* __restrict__ scale_W,    // [N, num_blocks_k]
    const float* __restrict__ bias,
    float* __restrict__ C,                // [M, N]
    int M, int N, int K,
    int activation_type
) {
    const int global_idx = blockIdx.x;
    const int m = global_idx / N;
    const int n = global_idx % N;

    if (m >= M) return;

    const int num_blocks_k = K / BLOCK_SIZE_SCALE;
    const int K_div_8 = K / 8;
    const int tid = threadIdx.x;

    // 每个线程处理 K/THREADS 个元素 (按 8 个一组)
    const int groups_per_thread = K_div_8 / THREADS;

    float local_sum = 0.0f;

    for (int g = 0; g < groups_per_thread; g++) {
        int group_idx = tid + g * THREADS;
        int k = group_idx * 8;

        int block_idx = k / BLOCK_SIZE_SCALE;
        float w_scale = scale_W[n * num_blocks_k + block_idx];

        // 加载 8 个 FP4 值 (1 个 uint32)
        uint32_t w_packed = W4[n * K_div_8 + group_idx];

        // 加载 8 个激活值 (2 个 float4)
        float4 a_lo = A4[m * (K/4) + k/4];
        float4 a_hi = A4[m * (K/4) + k/4 + 1];

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

    // Block-level reduce
    __shared__ float shared_sum[THREADS];
    shared_sum[tid] = local_sum;
    __syncthreads();

    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float val = shared_sum[0];
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
// PyTorch Interface - 多版本对比测试
// ============================================================================

torch::Tensor nvfp4_gemv_v2(
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

    constexpr int THREADS = 256;
    int warps_per_block = THREADS / WARP_SIZE;
    int num_blocks = (M * N + warps_per_block - 1) / warps_per_block;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_opt_v2_kernel<THREADS><<<num_blocks, THREADS, 0, stream>>>(
        A_ptr, W_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

torch::Tensor nvfp4_gemv_v3(
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

    constexpr int THREADS = 256;
    int num_blocks = M * N;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_opt_v3_kernel<THREADS><<<num_blocks, THREADS, 0, stream>>>(
        A_ptr, W_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

torch::Tensor nvfp4_gemv_v4(
    torch::Tensor activation,
    torch::Tensor weight_packed,
    torch::Tensor scale_W,
    c10::optional<torch::Tensor> bias,
    int M, int N, int K,
    int activation_type
) {
    // 需要 K % 8 == 0
    TORCH_CHECK(K % 8 == 0, "K must be divisible by 8 for v4 kernel");

    activation = activation.to(torch::kFloat32).contiguous();
    weight_packed = weight_packed.contiguous();
    scale_W = scale_W.to(torch::kFloat32).contiguous();

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(activation.device());
    torch::Tensor output = torch::zeros({M, N}, options);

    // 重新解释为 float4 和 uint32
    const float4* A4_ptr = reinterpret_cast<const float4*>(activation.data_ptr<float>());
    const uint32_t* W4_ptr = reinterpret_cast<const uint32_t*>(weight_packed.data_ptr<uint8_t>());
    const float* scale_W_ptr = scale_W.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().numel() > 0) {
        auto b = bias.value().to(torch::kFloat32);
        bias_ptr = b.data_ptr<float>();
    }
    float* C_ptr = output.data_ptr<float>();

    constexpr int THREADS = 256;
    int num_blocks = M * N;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_opt_v4_kernel<THREADS><<<num_blocks, THREADS, 0, stream>>>(
        A4_ptr, W4_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_gemv_v2", &nvfp4_gemv_v2,
          "NVFP4 GEMV v2 (vectorized warp)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("activation_type"));

    m.def("nvfp4_gemv_v3", &nvfp4_gemv_v3,
          "NVFP4 GEMV v3 (block-level reduce)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("activation_type"));

    m.def("nvfp4_gemv_v4", &nvfp4_gemv_v4,
          "NVFP4 GEMV v4 (float4 + uint32 vectorized)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("activation_type"));
}
