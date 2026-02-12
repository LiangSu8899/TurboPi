/**
 * PyTorch C++ Extension for NVFP4 Persistent MLP Kernel.
 *
 * This wraps the CUDA kernel for easy calling from Python.
 *
 * Build with:
 *   python setup_persistent.py build_ext --inplace
 *
 * Author: Claude Code
 * Date: 2026-02-10
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations from CUDA file
struct LayerWeights {
    const uint8_t* gate_packed;
    const float* gate_scale;
    const uint8_t* up_packed;
    const float* up_scale;
    const uint8_t* down_packed;
    const float* down_scale;
};

extern "C" {
    void launch_4layer_persistent_mlp(
        const float* input, float* output,
        const LayerWeights* layers, cudaStream_t stream);

    void launch_6layer_persistent_mlp(
        const float* input, float* output,
        const LayerWeights* layers, cudaStream_t stream);

    void launch_8layer_persistent_mlp(
        const float* input, float* output,
        const LayerWeights* layers, cudaStream_t stream);

    void launch_18layer_persistent_mlp(
        const float* input, float* output,
        const LayerWeights* layers, cudaStream_t stream);

    int get_smem_size();
    void print_kernel_info();
}

// Python interface
torch::Tensor nvfp4_persistent_mlp_forward(
    torch::Tensor input,
    std::vector<torch::Tensor> gate_packed_list,
    std::vector<torch::Tensor> gate_scale_list,
    std::vector<torch::Tensor> up_packed_list,
    std::vector<torch::Tensor> up_scale_list,
    std::vector<torch::Tensor> down_packed_list,
    std::vector<torch::Tensor> down_scale_list,
    int num_layers
) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D [batch, hidden]");
    TORCH_CHECK(input.size(0) == 1, "Batch size must be 1");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    const int hidden_size = input.size(1);
    TORCH_CHECK(hidden_size == 2048, "Hidden size must be 2048");

    // Validate list lengths
    TORCH_CHECK(static_cast<int>(gate_packed_list.size()) == num_layers, "gate_packed_list length mismatch");
    TORCH_CHECK(static_cast<int>(gate_scale_list.size()) == num_layers, "gate_scale_list length mismatch");
    TORCH_CHECK(static_cast<int>(up_packed_list.size()) == num_layers, "up_packed_list length mismatch");
    TORCH_CHECK(static_cast<int>(up_scale_list.size()) == num_layers, "up_scale_list length mismatch");
    TORCH_CHECK(static_cast<int>(down_packed_list.size()) == num_layers, "down_packed_list length mismatch");
    TORCH_CHECK(static_cast<int>(down_scale_list.size()) == num_layers, "down_scale_list length mismatch");

    // Allocate output
    auto output = torch::empty_like(input);

    // Build LayerWeights array on host
    std::vector<LayerWeights> layers_vec(num_layers);
    for (int i = 0; i < num_layers; i++) {
        TORCH_CHECK(gate_packed_list[i].is_cuda(), "gate_packed must be CUDA tensor");
        TORCH_CHECK(gate_scale_list[i].is_cuda(), "gate_scale must be CUDA tensor");
        TORCH_CHECK(up_packed_list[i].is_cuda(), "up_packed must be CUDA tensor");
        TORCH_CHECK(up_scale_list[i].is_cuda(), "up_scale must be CUDA tensor");
        TORCH_CHECK(down_packed_list[i].is_cuda(), "down_packed must be CUDA tensor");
        TORCH_CHECK(down_scale_list[i].is_cuda(), "down_scale must be CUDA tensor");

        layers_vec[i].gate_packed = gate_packed_list[i].data_ptr<uint8_t>();
        layers_vec[i].gate_scale = gate_scale_list[i].data_ptr<float>();
        layers_vec[i].up_packed = up_packed_list[i].data_ptr<uint8_t>();
        layers_vec[i].up_scale = up_scale_list[i].data_ptr<float>();
        layers_vec[i].down_packed = down_packed_list[i].data_ptr<uint8_t>();
        layers_vec[i].down_scale = down_scale_list[i].data_ptr<float>();
    }

    // Allocate device memory for layers array
    LayerWeights* d_layers;
    cudaError_t err = cudaMalloc(&d_layers, num_layers * sizeof(LayerWeights));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc failed: ", cudaGetErrorString(err));

    // Copy to device (async)
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    err = cudaMemcpyAsync(d_layers, layers_vec.data(), num_layers * sizeof(LayerWeights),
                          cudaMemcpyHostToDevice, stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));

    // Launch kernel based on layer count
    if (num_layers == 4) {
        launch_4layer_persistent_mlp(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            d_layers,
            stream
        );
    } else if (num_layers == 6) {
        launch_6layer_persistent_mlp(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            d_layers,
            stream
        );
    } else if (num_layers == 8) {
        launch_8layer_persistent_mlp(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            d_layers,
            stream
        );
    } else if (num_layers == 18) {
        launch_18layer_persistent_mlp(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            d_layers,
            stream
        );
    } else {
        cudaFree(d_layers);
        TORCH_CHECK(false, "Only 4, 6, 8, and 18 layer versions implemented");
    }

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_layers);
        TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(err));
    }

    // Free device memory after stream completes (enqueue on stream)
    // Note: For production, we'd want a memory pool to avoid alloc/free overhead
    cudaFreeAsync(d_layers, stream);

    return output;
}

// Get kernel info
int get_persistent_smem_size() {
    return get_smem_size();
}

void print_persistent_kernel_info() {
    print_kernel_info();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &nvfp4_persistent_mlp_forward,
          "NVFP4 Persistent MLP forward pass",
          py::arg("input"),
          py::arg("gate_packed_list"),
          py::arg("gate_scale_list"),
          py::arg("up_packed_list"),
          py::arg("up_scale_list"),
          py::arg("down_packed_list"),
          py::arg("down_scale_list"),
          py::arg("num_layers"));

    m.def("get_smem_size", &get_persistent_smem_size,
          "Get shared memory size used by kernel");

    m.def("print_kernel_info", &print_persistent_kernel_info,
          "Print kernel register and memory info");
}
