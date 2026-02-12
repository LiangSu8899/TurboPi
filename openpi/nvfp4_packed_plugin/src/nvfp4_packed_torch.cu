/**
 * NVFP4 Packed GEMV - PyTorch Extension
 *
 * PyTorch bindings for the NVFP4 packed kernel.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#define BLOCK_SIZE_SCALE 32
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256

// NVFP4 decode table
__constant__ float NVFP4_DECODE_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float2 unpack_nvfp4_pair(uint8_t packed) {
    float2 result;
    result.x = NVFP4_DECODE_TABLE[packed & 0xF];
    result.y = NVFP4_DECODE_TABLE[(packed >> 4) & 0xF];
    return result;
}

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
// W4A4 GEMV Kernel - 激活量化版本
// ============================================================================
__global__ void nvfp4_gemv_w4a4_kernel(
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K,
    int activation_type
) {
    int num_blocks_k = K / BLOCK_SIZE_SCALE;
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int total_outputs = M * N;
    if (global_warp_id >= total_outputs) return;

    int m = global_warp_id / N;
    int n = global_warp_id % N;

    float local_sum = 0.0f;
    int k_per_lane = K / WARP_SIZE;
    int k_start = (lane_id * k_per_lane / 2) * 2;
    int k_end = k_start + k_per_lane;

    for (int k = k_start; k < k_end; k += 2) {
        int block_idx = k / BLOCK_SIZE_SCALE;
        float a_scale = scale_A[m * num_blocks_k + block_idx];
        float w_scale = scale_W[n * num_blocks_k + block_idx];

        uint8_t w_packed = W_packed[n * (K / 2) + k / 2];
        float2 w_vals = unpack_nvfp4_pair(w_packed);

        float a0 = A[m * K + k];
        float a1 = A[m * K + k + 1];

        local_sum += a0 * a_scale * w_vals.x * w_scale;
        local_sum += a1 * a_scale * w_vals.y * w_scale;
    }

    float total = warpReduceSum(local_sum);

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
// W4A16 GEMV Kernel - 激活不量化，直接用 float
// ============================================================================
__global__ void nvfp4_gemv_w4a16_kernel(
    const float* __restrict__ A,          // [M, K] 原始 float 激活
    const uint8_t* __restrict__ W_packed, // [N, K/2] packed FP4 权重
    const float* __restrict__ scale_W,    // [N, num_blocks] 权重 scale
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K,
    int activation_type
) {
    int num_blocks_k = K / BLOCK_SIZE_SCALE;
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int total_outputs = M * N;
    if (global_warp_id >= total_outputs) return;

    int m = global_warp_id / N;
    int n = global_warp_id % N;

    float local_sum = 0.0f;
    int k_per_lane = K / WARP_SIZE;
    int k_start = (lane_id * k_per_lane / 2) * 2;
    int k_end = k_start + k_per_lane;

    for (int k = k_start; k < k_end; k += 2) {
        int block_idx = k / BLOCK_SIZE_SCALE;
        float w_scale = scale_W[n * num_blocks_k + block_idx];

        uint8_t w_packed = W_packed[n * (K / 2) + k / 2];
        float2 w_vals = unpack_nvfp4_pair(w_packed);

        // 激活直接使用，只有权重需要 scale
        float a0 = A[m * K + k];
        float a1 = A[m * K + k + 1];

        local_sum += a0 * w_vals.x * w_scale;
        local_sum += a1 * w_vals.y * w_scale;
    }

    float total = warpReduceSum(local_sum);

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
// PyTorch Interface
// ============================================================================

// W4A4 版本: 激活和权重都量化
torch::Tensor nvfp4_gemv(
    torch::Tensor activation,      // [M, K] float32 (量化后的值)
    torch::Tensor weight_packed,   // [N, K/2] uint8
    torch::Tensor scale_A,         // [M, num_blocks] float32
    torch::Tensor scale_W,         // [N, num_blocks] float32
    c10::optional<torch::Tensor> bias,  // [N] float32 optional
    int M, int N, int K,
    int activation_type
) {
    // Convert to float32 if needed
    if (activation.scalar_type() != torch::kFloat32) {
        activation = activation.to(torch::kFloat32);
    }
    if (scale_A.scalar_type() != torch::kFloat32) {
        scale_A = scale_A.to(torch::kFloat32);
    }
    if (scale_W.scalar_type() != torch::kFloat32) {
        scale_W = scale_W.to(torch::kFloat32);
    }

    // Ensure contiguous
    activation = activation.contiguous();
    weight_packed = weight_packed.contiguous();
    scale_A = scale_A.contiguous();
    scale_W = scale_W.contiguous();

    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(activation.device());
    torch::Tensor output = torch::zeros({M, N}, options);

    // Get pointers
    const float* A_ptr = activation.data_ptr<float>();
    const uint8_t* W_ptr = reinterpret_cast<const uint8_t*>(weight_packed.data_ptr());
    const float* scale_A_ptr = scale_A.data_ptr<float>();
    const float* scale_W_ptr = scale_W.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().numel() > 0) {
        auto bias_tensor = bias.value();
        if (bias_tensor.scalar_type() != torch::kFloat32) {
            bias_tensor = bias_tensor.to(torch::kFloat32);
        }
        bias_ptr = bias_tensor.data_ptr<float>();
    }
    float* C_ptr = output.data_ptr<float>();

    // Launch kernel
    int num_warps = M * N;
    int warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;
    int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_w4a4_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        A_ptr, W_ptr, scale_A_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

// W4A16 版本: 只有权重量化，激活保持 float
torch::Tensor nvfp4_gemv_w4a16(
    torch::Tensor activation,      // [M, K] float32 (原始激活)
    torch::Tensor weight_packed,   // [N, K/2] uint8
    torch::Tensor scale_W,         // [N, num_blocks] float32
    c10::optional<torch::Tensor> bias,  // [N] float32 optional
    int M, int N, int K,
    int activation_type
) {
    // Convert to float32 if needed
    if (activation.scalar_type() != torch::kFloat32) {
        activation = activation.to(torch::kFloat32);
    }
    if (scale_W.scalar_type() != torch::kFloat32) {
        scale_W = scale_W.to(torch::kFloat32);
    }

    // Ensure contiguous
    activation = activation.contiguous();
    weight_packed = weight_packed.contiguous();
    scale_W = scale_W.contiguous();

    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(activation.device());
    torch::Tensor output = torch::zeros({M, N}, options);

    // Get pointers
    const float* A_ptr = activation.data_ptr<float>();
    const uint8_t* W_ptr = reinterpret_cast<const uint8_t*>(weight_packed.data_ptr());
    const float* scale_W_ptr = scale_W.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().numel() > 0) {
        auto bias_tensor = bias.value();
        if (bias_tensor.scalar_type() != torch::kFloat32) {
            bias_tensor = bias_tensor.to(torch::kFloat32);
        }
        bias_ptr = bias_tensor.data_ptr<float>();
    }
    float* C_ptr = output.data_ptr<float>();

    // Launch kernel
    int num_warps = M * N;
    int warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;
    int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_gemv_w4a16_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        A_ptr, W_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type
    );

    return output;
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_gemv", &nvfp4_gemv,
          "NVFP4 Packed GEMV (W4A4)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_A"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"),
          py::arg("N"),
          py::arg("K"),
          py::arg("activation_type"));

    m.def("nvfp4_gemv_w4a16", &nvfp4_gemv_w4a16,
          "NVFP4 Packed GEMV (W4A16 - faster, activation not quantized)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"),
          py::arg("N"),
          py::arg("K"),
          py::arg("activation_type"));
}
