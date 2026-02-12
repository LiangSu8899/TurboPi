/**
 * NVFP4 GEMM - PyTorch Extension
 *
 * 针对 M > 1 优化的 GEMM kernel。
 *
 * Author: Claude Code
 * Date: 2026-02-10
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// 引入 kernel 实现
#include "nvfp4_gemm_kernel.cu"

torch::Tensor nvfp4_gemm(
    torch::Tensor activation,      // [M, K] float32
    torch::Tensor weight_packed,   // [N, K/2] uint8
    torch::Tensor scale_W,         // [N, num_blocks] float32
    c10::optional<torch::Tensor> bias,  // [N] float32 optional
    int M, int N, int K,
    int activation_type  // 0=none, 1=gelu, 2=silu
) {
    // 确保输入是 float32
    if (activation.scalar_type() != torch::kFloat32) {
        activation = activation.to(torch::kFloat32);
    }
    if (scale_W.scalar_type() != torch::kFloat32) {
        scale_W = scale_W.to(torch::kFloat32);
    }

    // 确保 contiguous
    activation = activation.contiguous();
    weight_packed = weight_packed.contiguous();
    scale_W = scale_W.contiguous();

    // 创建输出张量
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(activation.device());
    torch::Tensor output = torch::zeros({M, N}, options);

    // 获取指针
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

    // 获取 CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 调用 kernel
    nvfp4_gemm_cuda(
        A_ptr, W_ptr, scale_W_ptr, bias_ptr, C_ptr,
        M, N, K, activation_type, stream
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_gemm", &nvfp4_gemm,
          "NVFP4 GEMM (optimized for any M)",
          py::arg("activation"),
          py::arg("weight_packed"),
          py::arg("scale_W"),
          py::arg("bias"),
          py::arg("M"),
          py::arg("N"),
          py::arg("K"),
          py::arg("activation_type"));
}
