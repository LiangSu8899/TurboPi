#!/usr/bin/env python3
"""
Build W4A16 TVM Kernel and Export to .so

This script:
1. Creates optimized W4A16 GEMV kernels using TVM TensorIR
2. Exports them as a shared library (.so)
3. Generates C header for TRT Plugin integration

The exported .so can be loaded directly from C++ without Python overhead.

Usage:
    python build_tvm_kernel.py --output-dir /path/to/output

Author: Claude Code
Date: 2026-02-11
"""

import os
import sys
import argparse
import numpy as np

# Add TVM and workspace to path
sys.path.insert(0, "/workspace/external/tvm/python")
sys.path.insert(0, "/workspace/src")

import tvm
from tvm import te, tir
from tvm.script import tir as T
from tvm.contrib import cc

# Constants
BLOCK_SIZE = 32  # nvFP4 scaling block size
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 16384


def create_w4a16_gate_up_kernel(H: int, I: int, block_size: int = BLOCK_SIZE):
    """
    Fused gate_proj + up_proj kernel.

    Computes:
        gate_out[1, I] = x[1, H] @ gate_W[I, H].T  (with FP4 decode)
        up_out[1, I] = x[1, H] @ up_W[I, H].T      (with FP4 decode)

    Both outputs computed in single kernel (shared input load).
    """
    num_blocks_H = (H + block_size - 1) // block_size
    H_packed = H // 2

    THREADS = 256
    num_blocks = (I + THREADS - 1) // THREADS

    @T.prim_func
    def kernel(
        x: T.Buffer((1, H), "float32"),
        gate_W: T.Buffer((I, H_packed), "uint8"),
        gate_S: T.Buffer((I, num_blocks_H), "float32"),
        up_W: T.Buffer((I, H_packed), "uint8"),
        up_S: T.Buffer((I, num_blocks_H), "float32"),
        gate_out: T.Buffer((1, I), "float32"),
        up_out: T.Buffer((1, I), "float32"),
    ):
        T.func_attr({
            "global_symbol": "w4a16_gate_up",
            "tir.noalias": True,
        })

        lut = T.alloc_buffer((16,), "float32", scope="shared")
        x_shared = T.alloc_buffer((H,), "float32", scope="shared")

        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):

                # Init LUT
                if tx < 16:
                    lut[tx] = T.Select(
                        tx < 8,
                        T.Cast("float32", tx) * T.float32(0.5) + T.Select(tx >= 4, T.float32(tx - 2), T.float32(0)),
                        T.float32(0) - (T.Cast("float32", tx - 8) * T.float32(0.5) + T.Select(tx >= 12, T.float32(tx - 10), T.float32(0)))
                    )
                # Hardcode LUT values for clarity
                if tx == 0: lut[0] = T.float32(0.0)
                if tx == 1: lut[1] = T.float32(0.5)
                if tx == 2: lut[2] = T.float32(1.0)
                if tx == 3: lut[3] = T.float32(1.5)
                if tx == 4: lut[4] = T.float32(2.0)
                if tx == 5: lut[5] = T.float32(3.0)
                if tx == 6: lut[6] = T.float32(4.0)
                if tx == 7: lut[7] = T.float32(6.0)
                if tx == 8: lut[8] = T.float32(0.0)
                if tx == 9: lut[9] = T.float32(-0.5)
                if tx == 10: lut[10] = T.float32(-1.0)
                if tx == 11: lut[11] = T.float32(-1.5)
                if tx == 12: lut[12] = T.float32(-2.0)
                if tx == 13: lut[13] = T.float32(-3.0)
                if tx == 14: lut[14] = T.float32(-4.0)
                if tx == 15: lut[15] = T.float32(-6.0)

                # Load x to shared (cooperative)
                for load_idx in T.serial((H + THREADS - 1) // THREADS):
                    idx = tx + load_idx * THREADS
                    if idx < H:
                        x_shared[idx] = x[0, idx]

                T.tvm_storage_sync("shared")

                j = bx * THREADS + tx

                if j < I:
                    gate_acc = T.float32(0)
                    up_acc = T.float32(0)

                    for k in T.serial(H):
                        a_val = x_shared[k]
                        byte_idx = k // 2
                        is_high = k % 2
                        blk = k // block_size

                        # Gate weight
                        gp = gate_W[j, byte_idx]
                        gf = T.if_then_else(is_high == 0, gp & T.uint8(0xF), (gp >> 4) & T.uint8(0xF))
                        gw = lut[T.Cast("int32", gf)] * gate_S[j, blk]
                        gate_acc = gate_acc + a_val * gw

                        # Up weight
                        up = up_W[j, byte_idx]
                        uf = T.if_then_else(is_high == 0, up & T.uint8(0xF), (up >> 4) & T.uint8(0xF))
                        uw = lut[T.Cast("int32", uf)] * up_S[j, blk]
                        up_acc = up_acc + a_val * uw

                    gate_out[0, j] = gate_acc
                    up_out[0, j] = up_acc

    return kernel


def create_w4a16_gelu_mul_kernel(I: int):
    """
    Fused GeLU(gate) * up kernel.

    Computes: out[1, I] = GeLU(gate[1, I]) * up[1, I]
    """
    THREADS = 256
    num_blocks = (I + THREADS - 1) // THREADS

    @T.prim_func
    def kernel(
        gate: T.Buffer((1, I), "float32"),
        up: T.Buffer((1, I), "float32"),
        out: T.Buffer((1, I), "float32"),
    ):
        T.func_attr({
            "global_symbol": "w4a16_gelu_mul",
            "tir.noalias": True,
        })

        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                j = bx * THREADS + tx

                if j < I:
                    g = gate[0, j]

                    # GeLU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    sqrt_2_pi = T.float32(0.7978845608)
                    c = T.float32(0.044715)
                    inner = sqrt_2_pi * (g + c * g * g * g)

                    # Tanh approximation
                    tanh_v = T.if_then_else(
                        inner > T.float32(4.0), T.float32(1.0),
                        T.if_then_else(
                            inner < T.float32(-4.0), T.float32(-1.0),
                            inner * (T.float32(27.0) + inner * inner) /
                            (T.float32(27.0) + T.float32(9.0) * inner * inner)
                        )
                    )

                    gelu_g = g * T.float32(0.5) * (T.float32(1.0) + tanh_v)
                    out[0, j] = gelu_g * up[0, j]

    return kernel


def create_w4a16_down_proj_kernel(H: int, I: int, block_size: int = BLOCK_SIZE):
    """
    Down projection kernel.

    Computes: out[1, H] = intermediate[1, I] @ down_W[H, I].T (with FP4 decode)
    """
    num_blocks_I = (I + block_size - 1) // block_size
    I_packed = I // 2

    THREADS = 256
    num_blocks = (H + THREADS - 1) // THREADS

    @T.prim_func
    def kernel(
        intermediate: T.Buffer((1, I), "float32"),
        down_W: T.Buffer((H, I_packed), "uint8"),
        down_S: T.Buffer((H, num_blocks_I), "float32"),
        out: T.Buffer((1, H), "float32"),
    ):
        T.func_attr({
            "global_symbol": "w4a16_down_proj",
            "tir.noalias": True,
        })

        lut = T.alloc_buffer((16,), "float32", scope="shared")

        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):

                # Init LUT
                if tx == 0: lut[0] = T.float32(0.0)
                if tx == 1: lut[1] = T.float32(0.5)
                if tx == 2: lut[2] = T.float32(1.0)
                if tx == 3: lut[3] = T.float32(1.5)
                if tx == 4: lut[4] = T.float32(2.0)
                if tx == 5: lut[5] = T.float32(3.0)
                if tx == 6: lut[6] = T.float32(4.0)
                if tx == 7: lut[7] = T.float32(6.0)
                if tx == 8: lut[8] = T.float32(0.0)
                if tx == 9: lut[9] = T.float32(-0.5)
                if tx == 10: lut[10] = T.float32(-1.0)
                if tx == 11: lut[11] = T.float32(-1.5)
                if tx == 12: lut[12] = T.float32(-2.0)
                if tx == 13: lut[13] = T.float32(-3.0)
                if tx == 14: lut[14] = T.float32(-4.0)
                if tx == 15: lut[15] = T.float32(-6.0)

                T.tvm_storage_sync("shared")

                j = bx * THREADS + tx

                if j < H:
                    acc = T.float32(0)

                    for k in T.serial(I):
                        a_val = intermediate[0, k]
                        byte_idx = k // 2
                        is_high = k % 2
                        blk = k // block_size

                        dp = down_W[j, byte_idx]
                        df = T.if_then_else(is_high == 0, dp & T.uint8(0xF), (dp >> 4) & T.uint8(0xF))
                        dw = lut[T.Cast("int32", df)] * down_S[j, blk]
                        acc = acc + a_val * dw

                    out[0, j] = acc

    return kernel


def build_and_export(output_dir: str, hidden_size: int, intermediate_size: int):
    """Build all kernels and export to .so"""

    os.makedirs(output_dir, exist_ok=True)

    H = hidden_size
    I = intermediate_size
    target = "cuda -arch=sm_110"

    print(f"Building W4A16 TVM kernels...")
    print(f"  hidden_size={H}, intermediate_size={I}")
    print(f"  target={target}")

    # Build kernels
    kernels = {}

    print("\n[1/3] Building gate_up kernel...")
    gate_up_func = create_w4a16_gate_up_kernel(H, I)

    print("[2/3] Building gelu_mul kernel...")
    gelu_mul_func = create_w4a16_gelu_mul_kernel(I)

    print("[3/3] Building down_proj kernel...")
    down_proj_func = create_w4a16_down_proj_kernel(H, I)

    # Build modules
    target_obj = tvm.target.Target(target)

    with tvm.transform.PassContext(opt_level=3):
        gate_up_mod = tvm.build(gate_up_func, target=target_obj)
        gelu_mul_mod = tvm.build(gelu_mul_func, target=target_obj)
        down_proj_mod = tvm.build(down_proj_func, target=target_obj)

    # Export to .so files
    gate_up_so = os.path.join(output_dir, "libw4a16_gate_up.so")
    gelu_mul_so = os.path.join(output_dir, "libw4a16_gelu_mul.so")
    down_proj_so = os.path.join(output_dir, "libw4a16_down_proj.so")

    gate_up_mod.export_library(gate_up_so)
    gelu_mul_mod.export_library(gelu_mul_so)
    down_proj_mod.export_library(down_proj_so)

    print(f"\nExported .so files:")
    print(f"  {gate_up_so}")
    print(f"  {gelu_mul_so}")
    print(f"  {down_proj_so}")

    # Generate C header
    num_blocks_H = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_I = (I + BLOCK_SIZE - 1) // BLOCK_SIZE

    header = f'''/*
 * W4A16 TVM Kernels for TensorRT Plugin
 * Auto-generated - DO NOT EDIT
 *
 * hidden_size={H}, intermediate_size={I}
 * block_size={BLOCK_SIZE}
 */

#ifndef W4A16_TVM_KERNELS_H
#define W4A16_TVM_KERNELS_H

#include <cuda_runtime.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

// Dimensions
constexpr int W4A16_HIDDEN_SIZE = {H};
constexpr int W4A16_INTERMEDIATE_SIZE = {I};
constexpr int W4A16_BLOCK_SIZE = {BLOCK_SIZE};
constexpr int W4A16_NUM_BLOCKS_H = {num_blocks_H};
constexpr int W4A16_NUM_BLOCKS_I = {num_blocks_I};

// Packed dimensions
constexpr int W4A16_H_PACKED = {H // 2};
constexpr int W4A16_I_PACKED = {I // 2};

// Weight memory sizes (bytes)
constexpr size_t W4A16_GATE_W_SIZE = {I * (H // 2)};  // uint8
constexpr size_t W4A16_GATE_S_SIZE = {I * num_blocks_H} * sizeof(float);
constexpr size_t W4A16_UP_W_SIZE = {I * (H // 2)};    // uint8
constexpr size_t W4A16_UP_S_SIZE = {I * num_blocks_H} * sizeof(float);
constexpr size_t W4A16_DOWN_W_SIZE = {H * (I // 2)};  // uint8
constexpr size_t W4A16_DOWN_S_SIZE = {H * num_blocks_I} * sizeof(float);

namespace turbo_pi {{

class W4A16TVMKernels {{
public:
    W4A16TVMKernels(const std::string& lib_dir);
    ~W4A16TVMKernels();

    // Execute fused gate + up projection
    // Input:  x [1, H]
    // Output: gate_out [1, I], up_out [1, I]
    void gateUp(
        const float* x,
        const uint8_t* gate_W, const float* gate_S,
        const uint8_t* up_W, const float* up_S,
        float* gate_out, float* up_out,
        cudaStream_t stream
    );

    // Execute GeLU(gate) * up
    // Input:  gate [1, I], up [1, I]
    // Output: out [1, I]
    void geluMul(
        const float* gate, const float* up,
        float* out,
        cudaStream_t stream
    );

    // Execute down projection
    // Input:  intermediate [1, I]
    // Output: out [1, H]
    void downProj(
        const float* intermediate,
        const uint8_t* down_W, const float* down_S,
        float* out,
        cudaStream_t stream
    );

    // Execute full MLP (convenience method)
    // Input:  x [1, H]
    // Output: out [1, H]
    // Workspace: [2 * I] floats for intermediate
    void fullMLP(
        const float* x,
        const uint8_t* gate_W, const float* gate_S,
        const uint8_t* up_W, const float* up_S,
        const uint8_t* down_W, const float* down_S,
        float* out,
        float* workspace,
        cudaStream_t stream
    );

private:
    tvm::runtime::Module gate_up_mod_;
    tvm::runtime::Module gelu_mul_mod_;
    tvm::runtime::Module down_proj_mod_;

    tvm::runtime::PackedFunc gate_up_func_;
    tvm::runtime::PackedFunc gelu_mul_func_;
    tvm::runtime::PackedFunc down_proj_func_;
}};

}}  // namespace turbo_pi

#endif  // W4A16_TVM_KERNELS_H
'''

    header_path = os.path.join(output_dir, "w4a16_tvm_kernels.h")
    with open(header_path, "w") as f:
        f.write(header)

    print(f"  {header_path}")

    # Generate C++ implementation
    cpp_impl = f'''/*
 * W4A16 TVM Kernels Implementation
 * Auto-generated - DO NOT EDIT
 */

#include "w4a16_tvm_kernels.h"
#include <dlpack/dlpack.h>
#include <tvm/runtime/device_api.h>

namespace turbo_pi {{

W4A16TVMKernels::W4A16TVMKernels(const std::string& lib_dir) {{
    // Load TVM modules
    gate_up_mod_ = tvm::runtime::Module::LoadFromFile(lib_dir + "/libw4a16_gate_up.so");
    gelu_mul_mod_ = tvm::runtime::Module::LoadFromFile(lib_dir + "/libw4a16_gelu_mul.so");
    down_proj_mod_ = tvm::runtime::Module::LoadFromFile(lib_dir + "/libw4a16_down_proj.so");

    // Get kernel functions
    gate_up_func_ = gate_up_mod_.GetFunction("w4a16_gate_up");
    gelu_mul_func_ = gelu_mul_mod_.GetFunction("w4a16_gelu_mul");
    down_proj_func_ = down_proj_mod_.GetFunction("w4a16_down_proj");
}}

W4A16TVMKernels::~W4A16TVMKernels() = default;

namespace {{

// Helper to create DLTensor from CUDA pointer
DLTensor makeDLTensor(void* data, int64_t* shape, int ndim, DLDataType dtype) {{
    DLTensor t;
    t.data = data;
    t.device = {{kDLCUDA, 0}};
    t.ndim = ndim;
    t.dtype = dtype;
    t.shape = shape;
    t.strides = nullptr;
    t.byte_offset = 0;
    return t;
}}

}}  // anonymous namespace

void W4A16TVMKernels::gateUp(
    const float* x,
    const uint8_t* gate_W, const float* gate_S,
    const uint8_t* up_W, const float* up_S,
    float* gate_out, float* up_out,
    cudaStream_t stream
) {{
    constexpr int H = W4A16_HIDDEN_SIZE;
    constexpr int I = W4A16_INTERMEDIATE_SIZE;
    constexpr int H_packed = W4A16_H_PACKED;
    constexpr int num_blocks_H = W4A16_NUM_BLOCKS_H;

    // Create DLTensors
    int64_t x_shape[] = {{1, H}};
    int64_t gate_W_shape[] = {{I, H_packed}};
    int64_t gate_S_shape[] = {{I, num_blocks_H}};
    int64_t up_W_shape[] = {{I, H_packed}};
    int64_t up_S_shape[] = {{I, num_blocks_H}};
    int64_t out_shape[] = {{1, I}};

    DLDataType f32 = {{kDLFloat, 32, 1}};
    DLDataType u8 = {{kDLUInt, 8, 1}};

    DLTensor x_t = makeDLTensor((void*)x, x_shape, 2, f32);
    DLTensor gate_W_t = makeDLTensor((void*)gate_W, gate_W_shape, 2, u8);
    DLTensor gate_S_t = makeDLTensor((void*)gate_S, gate_S_shape, 2, f32);
    DLTensor up_W_t = makeDLTensor((void*)up_W, up_W_shape, 2, u8);
    DLTensor up_S_t = makeDLTensor((void*)up_S, up_S_shape, 2, f32);
    DLTensor gate_out_t = makeDLTensor(gate_out, out_shape, 2, f32);
    DLTensor up_out_t = makeDLTensor(up_out, out_shape, 2, f32);

    // Call TVM function
    gate_up_func_(&x_t, &gate_W_t, &gate_S_t, &up_W_t, &up_S_t, &gate_out_t, &up_out_t);
}}

void W4A16TVMKernels::geluMul(
    const float* gate, const float* up,
    float* out,
    cudaStream_t stream
) {{
    constexpr int I = W4A16_INTERMEDIATE_SIZE;

    int64_t shape[] = {{1, I}};
    DLDataType f32 = {{kDLFloat, 32, 1}};

    DLTensor gate_t = makeDLTensor((void*)gate, shape, 2, f32);
    DLTensor up_t = makeDLTensor((void*)up, shape, 2, f32);
    DLTensor out_t = makeDLTensor(out, shape, 2, f32);

    gelu_mul_func_(&gate_t, &up_t, &out_t);
}}

void W4A16TVMKernels::downProj(
    const float* intermediate,
    const uint8_t* down_W, const float* down_S,
    float* out,
    cudaStream_t stream
) {{
    constexpr int H = W4A16_HIDDEN_SIZE;
    constexpr int I = W4A16_INTERMEDIATE_SIZE;
    constexpr int I_packed = W4A16_I_PACKED;
    constexpr int num_blocks_I = W4A16_NUM_BLOCKS_I;

    int64_t in_shape[] = {{1, I}};
    int64_t down_W_shape[] = {{H, I_packed}};
    int64_t down_S_shape[] = {{H, num_blocks_I}};
    int64_t out_shape[] = {{1, H}};

    DLDataType f32 = {{kDLFloat, 32, 1}};
    DLDataType u8 = {{kDLUInt, 8, 1}};

    DLTensor in_t = makeDLTensor((void*)intermediate, in_shape, 2, f32);
    DLTensor down_W_t = makeDLTensor((void*)down_W, down_W_shape, 2, u8);
    DLTensor down_S_t = makeDLTensor((void*)down_S, down_S_shape, 2, f32);
    DLTensor out_t = makeDLTensor(out, out_shape, 2, f32);

    down_proj_func_(&in_t, &down_W_t, &down_S_t, &out_t);
}}

void W4A16TVMKernels::fullMLP(
    const float* x,
    const uint8_t* gate_W, const float* gate_S,
    const uint8_t* up_W, const float* up_S,
    const uint8_t* down_W, const float* down_S,
    float* out,
    float* workspace,
    cudaStream_t stream
) {{
    constexpr int I = W4A16_INTERMEDIATE_SIZE;

    // workspace layout: [gate_out (I), up_out (I), gelu_out (I)]
    float* gate_out = workspace;
    float* up_out = workspace + I;
    float* gelu_out = workspace;  // Reuse gate_out after gelu_mul

    // Step 1: gate_proj + up_proj
    gateUp(x, gate_W, gate_S, up_W, up_S, gate_out, up_out, stream);

    // Step 2: GeLU(gate) * up
    geluMul(gate_out, up_out, gelu_out, stream);

    // Step 3: down_proj
    downProj(gelu_out, down_W, down_S, out, stream);
}}

}}  // namespace turbo_pi
'''

    cpp_path = os.path.join(output_dir, "w4a16_tvm_kernels.cpp")
    with open(cpp_path, "w") as f:
        f.write(cpp_impl)

    print(f"  {cpp_path}")

    # Generate CMakeLists.txt
    cmake = f'''cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(W4A16TVMPlugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "110")

# Find TVM
if(NOT DEFINED TVM_HOME)
    set(TVM_HOME "/workspace/external/tvm")
endif()

find_path(TVM_INCLUDE_DIR tvm/runtime/packed_func.h
    HINTS ${{TVM_HOME}}/include
)

find_library(TVM_RUNTIME_LIB tvm_runtime
    HINTS ${{TVM_HOME}}/build
)

if(NOT TVM_INCLUDE_DIR OR NOT TVM_RUNTIME_LIB)
    message(FATAL_ERROR "TVM not found. Please set TVM_HOME.")
endif()

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include /usr/local/include
)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib /usr/local/lib
)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# W4A16 TVM Kernels library
add_library(w4a16_tvm_kernels SHARED
    w4a16_tvm_kernels.cpp
)

target_include_directories(w4a16_tvm_kernels PUBLIC
    ${{CMAKE_CURRENT_SOURCE_DIR}}
    ${{TVM_INCLUDE_DIR}}
    ${{TVM_INCLUDE_DIR}}/../3rdparty/dlpack/include
    ${{CUDAToolkit_INCLUDE_DIRS}}
)

target_link_libraries(w4a16_tvm_kernels
    ${{TVM_RUNTIME_LIB}}
    CUDA::cudart
)

# TensorRT Plugin (optional, if TensorRT found)
if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    add_library(w4a16_trt_plugin SHARED
        w4a16_trt_plugin.cpp
        w4a16_tvm_kernels.cpp
    )

    target_include_directories(w4a16_trt_plugin PUBLIC
        ${{CMAKE_CURRENT_SOURCE_DIR}}
        ${{TVM_INCLUDE_DIR}}
        ${{TVM_INCLUDE_DIR}}/../3rdparty/dlpack/include
        ${{TENSORRT_INCLUDE_DIR}}
        ${{CUDAToolkit_INCLUDE_DIRS}}
    )

    target_link_libraries(w4a16_trt_plugin
        ${{TVM_RUNTIME_LIB}}
        ${{TENSORRT_LIBRARY}}
        CUDA::cudart
    )

    message(STATUS "TensorRT Plugin will be built")
else()
    message(STATUS "TensorRT not found, skipping TRT plugin")
endif()

# Install
install(TARGETS w4a16_tvm_kernels LIBRARY DESTINATION lib)
install(FILES w4a16_tvm_kernels.h DESTINATION include)
install(FILES
    libw4a16_gate_up.so
    libw4a16_gelu_mul.so
    libw4a16_down_proj.so
    DESTINATION lib
)

message(STATUS "")
message(STATUS "W4A16 TVM Plugin Configuration:")
message(STATUS "  TVM: ${{TVM_INCLUDE_DIR}}")
message(STATUS "  TVM Runtime: ${{TVM_RUNTIME_LIB}}")
message(STATUS "  Output: ${{CMAKE_CURRENT_SOURCE_DIR}}")
message(STATUS "")
'''

    cmake_path = os.path.join(output_dir, "CMakeLists.txt")
    with open(cmake_path, "w") as f:
        f.write(cmake)

    print(f"  {cmake_path}")

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\nTo build C++ library:")
    print(f"  cd {output_dir}")
    print(f"  mkdir build && cd build")
    print(f"  cmake ..")
    print(f"  make")

    return {
        "gate_up_so": gate_up_so,
        "gelu_mul_so": gelu_mul_so,
        "down_proj_so": down_proj_so,
        "header": header_path,
    }


def benchmark(output_dir: str):
    """Benchmark the exported kernels"""
    import time

    print("\n" + "=" * 60)
    print("BENCHMARK")
    print("=" * 60)

    H = HIDDEN_SIZE
    I = INTERMEDIATE_SIZE

    # Load modules
    gate_up_mod = tvm.runtime.load_module(os.path.join(output_dir, "libw4a16_gate_up.so"))
    gelu_mul_mod = tvm.runtime.load_module(os.path.join(output_dir, "libw4a16_gelu_mul.so"))
    down_proj_mod = tvm.runtime.load_module(os.path.join(output_dir, "libw4a16_down_proj.so"))

    gate_up_func = gate_up_mod.get_function("w4a16_gate_up")
    gelu_mul_func = gelu_mul_mod.get_function("w4a16_gelu_mul")
    down_proj_func = down_proj_mod.get_function("w4a16_down_proj")

    device = tvm.runtime.cuda(0)

    # Allocate test buffers
    np.random.seed(42)

    x = tvm.runtime.empty((1, H), "float32", device)
    x.copyfrom(np.random.randn(1, H).astype(np.float32))

    gate_W = tvm.runtime.empty((I, H // 2), "uint8", device)
    gate_S = tvm.runtime.empty((I, (H + BLOCK_SIZE - 1) // BLOCK_SIZE), "float32", device)
    up_W = tvm.runtime.empty((I, H // 2), "uint8", device)
    up_S = tvm.runtime.empty((I, (H + BLOCK_SIZE - 1) // BLOCK_SIZE), "float32", device)
    down_W = tvm.runtime.empty((H, I // 2), "uint8", device)
    down_S = tvm.runtime.empty((H, (I + BLOCK_SIZE - 1) // BLOCK_SIZE), "float32", device)

    gate_W.copyfrom(np.random.randint(0, 256, (I, H // 2), dtype=np.uint8))
    gate_S.copyfrom(np.random.rand(I, (H + BLOCK_SIZE - 1) // BLOCK_SIZE).astype(np.float32) * 0.1)
    up_W.copyfrom(np.random.randint(0, 256, (I, H // 2), dtype=np.uint8))
    up_S.copyfrom(np.random.rand(I, (H + BLOCK_SIZE - 1) // BLOCK_SIZE).astype(np.float32) * 0.1)
    down_W.copyfrom(np.random.randint(0, 256, (H, I // 2), dtype=np.uint8))
    down_S.copyfrom(np.random.rand(H, (I + BLOCK_SIZE - 1) // BLOCK_SIZE).astype(np.float32) * 0.1)

    gate_out = tvm.runtime.empty((1, I), "float32", device)
    up_out = tvm.runtime.empty((1, I), "float32", device)
    gelu_out = tvm.runtime.empty((1, I), "float32", device)
    out = tvm.runtime.empty((1, H), "float32", device)

    # Warmup
    print("\nWarming up...")
    for _ in range(20):
        gate_up_func(x, gate_W, gate_S, up_W, up_S, gate_out, up_out)
        gelu_mul_func(gate_out, up_out, gelu_out)
        down_proj_func(gelu_out, down_W, down_S, out)
    device.sync()

    # Benchmark individual kernels
    runs = 100

    # gate_up
    device.sync()
    start = time.time()
    for _ in range(runs):
        gate_up_func(x, gate_W, gate_S, up_W, up_S, gate_out, up_out)
    device.sync()
    gate_up_ms = (time.time() - start) / runs * 1000

    # gelu_mul
    device.sync()
    start = time.time()
    for _ in range(runs):
        gelu_mul_func(gate_out, up_out, gelu_out)
    device.sync()
    gelu_mul_ms = (time.time() - start) / runs * 1000

    # down_proj
    device.sync()
    start = time.time()
    for _ in range(runs):
        down_proj_func(gelu_out, down_W, down_S, out)
    device.sync()
    down_proj_ms = (time.time() - start) / runs * 1000

    # Full MLP
    device.sync()
    start = time.time()
    for _ in range(runs):
        gate_up_func(x, gate_W, gate_S, up_W, up_S, gate_out, up_out)
        gelu_mul_func(gate_out, up_out, gelu_out)
        down_proj_func(gelu_out, down_W, down_S, out)
    device.sync()
    full_mlp_ms = (time.time() - start) / runs * 1000

    print(f"\nKernel Timing (single layer):")
    print(f"  gate_up:   {gate_up_ms:.3f} ms")
    print(f"  gelu_mul:  {gelu_mul_ms:.3f} ms")
    print(f"  down_proj: {down_proj_ms:.3f} ms")
    print(f"  ---------------------")
    print(f"  Full MLP:  {full_mlp_ms:.3f} ms")

    print(f"\n18-layer Projection:")
    print(f"  W4A16 TVM: {full_mlp_ms * 18:.1f} ms")
    print(f"  TRT FP8:   12.39 ms (baseline)")
    speedup = 12.39 / (full_mlp_ms * 18)
    if speedup > 1:
        print(f"  Speedup:   {speedup:.2f}x FASTER")
    else:
        print(f"  Slowdown:  {1/speedup:.2f}x slower")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build W4A16 TVM Kernels")
    parser.add_argument("--output-dir", type=str,
                        default="/home/heima-thor/suliang/Turbo-Pi/openpi/w4a16_tvm_plugin/lib",
                        help="Output directory")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--intermediate-size", type=int, default=INTERMEDIATE_SIZE)
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after build")

    args = parser.parse_args()

    results = build_and_export(args.output_dir, args.hidden_size, args.intermediate_size)

    if args.benchmark:
        benchmark(args.output_dir)
