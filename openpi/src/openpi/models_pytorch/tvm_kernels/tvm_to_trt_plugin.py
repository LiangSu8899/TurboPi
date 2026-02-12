#!/usr/bin/env python3
"""
TVM to TensorRT Plugin Generator

This module provides utilities to:
1. Define kernels using TVM TensorIR
2. Export CUDA source code
3. Generate TensorRT Plugin wrapper code

Usage:
    python tvm_to_trt_plugin.py --kernel nvfp4_gemm --output /path/to/output
"""

import os
import sys
from typing import Tuple, Optional
from pathlib import Path


def check_tvm_environment() -> bool:
    """Check if TVM is available and configured correctly."""
    try:
        import tvm
        print(f"TVM version: {tvm.__version__}")
        return True
    except ImportError:
        print("ERROR: TVM not found. Please set up TVM environment:")
        print("  export TVM_HOME=/path/to/tvm")
        print("  export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH")
        print("  export LD_LIBRARY_PATH=$TVM_HOME/build:$LD_LIBRARY_PATH")
        return False


def create_nvfp4_gemm_kernel(M: int, N: int, K: int, block_size: int = 32):
    """
    Create nvFP4 GEMM kernel with fused dequantization.

    Args:
        M: Batch size (number of tokens)
        N: Output features
        K: Input features
        block_size: Block size for scaling (default 32)

    Returns:
        TVM TensorIR function
    """
    import tvm
    from tvm.script import tir as T

    num_blocks_k = (K + block_size - 1) // block_size
    THREADS_PER_BLOCK = min(256, N)  # Adjust based on N
    total_elements = M * N
    num_blocks_x = (total_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    @T.prim_func
    def nvfp4_gemm_func(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "nvfp4_gemm", "tir.noalias": True})

        for bx in T.thread_binding(num_blocks_x, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):
                idx = bx * THREADS_PER_BLOCK + tx
                if idx < total_elements:
                    i = idx // N
                    j = idx % N

                    # Initialize output
                    C[i, j] = T.float32(0)

                    # Fused dequant + GEMM
                    for k in T.serial(K):
                        block_idx = k // block_size
                        a_val = A[i, k] * scale_A[i, block_idx]
                        w_val = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + a_val * w_val

    return nvfp4_gemm_func


def create_w4a8_gemm_kernel(M: int, N: int, K: int, block_size: int = 32):
    """
    Create W4A8 GEMM kernel (FP8 activation, nvFP4 weight) with fused dequantization.

    Args:
        M: Batch size (number of tokens)
        N: Output features
        K: Input features
        block_size: Block size for scaling (default 32)

    Returns:
        TVM TensorIR function
    """
    import tvm
    from tvm.script import tir as T

    num_blocks_k = (K + block_size - 1) // block_size
    THREADS_PER_BLOCK = min(256, N)
    total_elements = M * N
    num_blocks_x = (total_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    @T.prim_func
    def w4a8_gemm_func(
        A: T.Buffer((M, K), "float32"),      # FP8 activation (stored as float32)
        W: T.Buffer((N, K), "float32"),      # nvFP4 weight (stored as float32)
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a8_gemm", "tir.noalias": True})

        for bx in T.thread_binding(num_blocks_x, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):
                idx = bx * THREADS_PER_BLOCK + tx
                if idx < total_elements:
                    i = idx // N
                    j = idx % N

                    # Initialize output
                    C[i, j] = T.float32(0)

                    # Fused dequant + GEMM
                    for k in T.serial(K):
                        block_idx = k // block_size
                        a_val = A[i, k] * scale_A[i, block_idx]
                        w_val = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + a_val * w_val

    return w4a8_gemm_func


def create_w4a16_gemm_kernel(M: int, N: int, K: int, block_size: int = 32):
    """
    Create W4A16 GEMM kernel (FP16/FP32 activation, nvFP4 weight) with fused dequantization.

    W4A16: Weight is 4-bit nvFP4, Activation is 16-bit (BF16/FP16) or FP32
    Only weight needs dequantization, activation is already high precision.

    Args:
        M: Batch size (number of tokens)
        N: Output features
        K: Input features
        block_size: Block size for weight scaling (default 32)

    Returns:
        TVM TensorIR function
    """
    import tvm
    from tvm.script import tir as T

    num_blocks_k = (K + block_size - 1) // block_size
    THREADS_PER_BLOCK = min(256, N)
    total_elements = M * N
    num_blocks_x = (total_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    @T.prim_func
    def w4a16_gemm_func(
        A: T.Buffer((M, K), "float32"),      # FP16/FP32 activation (no scale needed)
        W: T.Buffer((N, K), "float32"),      # nvFP4 weight (stored as float32)
        scale_W: T.Buffer((N, num_blocks_k), "float32"),  # Only weight scales
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_gemm", "tir.noalias": True})

        for bx in T.thread_binding(num_blocks_x, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):
                idx = bx * THREADS_PER_BLOCK + tx
                if idx < total_elements:
                    i = idx // N
                    j = idx % N

                    # Initialize output
                    C[i, j] = T.float32(0)

                    # Fused dequant + GEMM (only weight dequant)
                    for k in T.serial(K):
                        block_idx = k // block_size
                        w_dequant = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + A[i, k] * w_dequant

    return w4a16_gemm_func


def export_cuda_source(
    tir_func,
    target: str = "cuda -arch=sm_110",
    opt_level: int = 3,
) -> Tuple[str, str]:
    """
    Export TVM TensorIR function to CUDA source code.

    Args:
        tir_func: TVM TensorIR function
        target: TVM target string
        opt_level: Optimization level

    Returns:
        Tuple of (cuda_source, llvm_ir)
    """
    import tvm

    target_obj = tvm.target.Target(target)

    with tvm.transform.PassContext(opt_level=opt_level):
        mod = tvm.build(tir_func, target=target_obj)

    # Extract CUDA source from imported modules
    cuda_source = ""
    if hasattr(mod, 'imports_') and len(mod.imports_) > 0:
        cuda_mod = mod.imports_[0]
        if hasattr(cuda_mod, 'inspect_source'):
            cuda_source = cuda_mod.inspect_source()

    # Get host LLVM IR
    llvm_ir = ""
    if hasattr(mod, 'inspect_source'):
        llvm_ir = mod.inspect_source()

    return cuda_source, llvm_ir


def generate_trt_plugin_wrapper(
    kernel_name: str,
    cuda_source: str,
    M: int, N: int, K: int,
    output_dir: str,
    has_activation_scale: bool = True,
) -> None:
    """
    Generate TensorRT Plugin wrapper for the CUDA kernel.

    Args:
        kernel_name: Name of the kernel (e.g., "nvfp4_gemm")
        cuda_source: CUDA source code
        M, N, K: Matrix dimensions
        output_dir: Output directory for generated files
        has_activation_scale: Whether kernel needs activation scale (False for W4A16)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save CUDA source
    cuda_file = os.path.join(output_dir, f"{kernel_name}_kernel.cu")
    with open(cuda_file, "w") as f:
        f.write(cuda_source)
    print(f"[Saved] {cuda_file}")

    # Generate header file based on kernel type
    if has_activation_scale:
        header_content = f'''/*
 * {kernel_name.upper()} TensorRT Plugin - TVM Generated
 *
 * Auto-generated from TVM TensorIR
 * Target: CUDA SM110 (Jetson Thor)
 *
 * Matrix dimensions: M={M}, N={N}, K={K}
 */

#ifndef {kernel_name.upper()}_TVM_PLUGIN_H
#define {kernel_name.upper()}_TVM_PLUGIN_H

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

namespace turbo_pi {{

// Kernel launcher (W4A4 or W4A8)
void launch_{kernel_name}(
    const float* A,
    const float* W,
    const float* scale_A,
    const float* scale_W,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
);

}}  // namespace turbo_pi

#endif  // {kernel_name.upper()}_TVM_PLUGIN_H
'''
    else:
        # W4A16: no activation scale
        header_content = f'''/*
 * {kernel_name.upper()} TensorRT Plugin - TVM Generated
 *
 * Auto-generated from TVM TensorIR
 * Target: CUDA SM110 (Jetson Thor)
 *
 * Matrix dimensions: M={M}, N={N}, K={K}
 * Note: W4A16 - only weight has scale, activation is full precision
 */

#ifndef {kernel_name.upper()}_TVM_PLUGIN_H
#define {kernel_name.upper()}_TVM_PLUGIN_H

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

namespace turbo_pi {{

// Kernel launcher (W4A16 - no activation scale)
void launch_{kernel_name}(
    const float* A,      // [M, K] full precision activation
    const float* W,      // [N, K] nvFP4 weight
    const float* scale_W, // [N, num_blocks_k] weight scales only
    float* C,            // [M, N] output
    int M, int N, int K,
    cudaStream_t stream
);

}}  // namespace turbo_pi

#endif  // {kernel_name.upper()}_TVM_PLUGIN_H
'''
    header_file = os.path.join(output_dir, f"{kernel_name}_tvm_plugin.h")
    with open(header_file, "w") as f:
        f.write(header_content)
    print(f"[Saved] {header_file}")

    # Generate launcher implementation based on kernel type
    if has_activation_scale:
        launcher_content = f'''/*
 * {kernel_name.upper()} Kernel Launcher - TVM Generated
 */

#include "{kernel_name}_tvm_plugin.h"
#include <cuda.h>

namespace turbo_pi {{

// Include TVM-generated kernel
{cuda_source}

void launch_{kernel_name}(
    const float* A,
    const float* W,
    const float* scale_A,
    const float* scale_W,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {{
    // Calculate grid dimensions
    const int THREADS_PER_BLOCK = {min(256, N)};
    const int total_elements = M * N;
    const int num_blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch TVM-generated kernel
    {kernel_name}_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        const_cast<float*>(A),
        C,
        const_cast<float*>(W),
        const_cast<float*>(scale_A),
        const_cast<float*>(scale_W)
    );
}}

}}  // namespace turbo_pi
'''
    else:
        # W4A16: only weight scale
        launcher_content = f'''/*
 * {kernel_name.upper()} Kernel Launcher - TVM Generated (W4A16)
 */

#include "{kernel_name}_tvm_plugin.h"
#include <cuda.h>

namespace turbo_pi {{

// Include TVM-generated kernel
{cuda_source}

void launch_{kernel_name}(
    const float* A,
    const float* W,
    const float* scale_W,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {{
    // Calculate grid dimensions
    const int THREADS_PER_BLOCK = {min(256, N)};
    const int total_elements = M * N;
    const int num_blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch TVM-generated kernel (W4A16: no activation scale)
    {kernel_name}_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        const_cast<float*>(A),
        C,
        const_cast<float*>(W),
        const_cast<float*>(scale_W)
    );
}}

}}  // namespace turbo_pi
'''
    launcher_file = os.path.join(output_dir, f"{kernel_name}_launcher.cu")
    with open(launcher_file, "w") as f:
        f.write(launcher_content)
    print(f"[Saved] {launcher_file}")

    # Generate CMakeLists.txt snippet
    cmake_content = f'''# {kernel_name.upper()} TVM Plugin - CMake Configuration

# Add TVM-generated kernel
add_library({kernel_name}_tvm_plugin SHARED
    {kernel_name}_kernel.cu
    {kernel_name}_launcher.cu
)

target_include_directories({kernel_name}_tvm_plugin PUBLIC
    ${{CMAKE_CURRENT_SOURCE_DIR}}
    ${{TENSORRT_INCLUDE_DIR}}
    ${{CUDA_INCLUDE_DIRS}}
)

target_link_libraries({kernel_name}_tvm_plugin
    ${{TENSORRT_LIBRARIES}}
    ${{CUDA_LIBRARIES}}
)

set_target_properties({kernel_name}_tvm_plugin PROPERTIES
    CUDA_ARCHITECTURES "110"  # Jetson Thor SM110
)
'''
    cmake_file = os.path.join(output_dir, "CMakeLists_snippet.txt")
    with open(cmake_file, "w") as f:
        f.write(cmake_content)
    print(f"[Saved] {cmake_file}")


def generate_all_kernels(
    M: int = 1,
    N: int = 3072,
    K: int = 3072,
    block_size: int = 32,
    target: str = "cuda -arch=sm_110",
    output_base: str = "/tmp/tvm_trt_plugins",
):
    """
    Generate TensorRT Plugins for all three kernel types (W4A4, W4A8, W4A16).

    Returns dict with kernel names and their output directories.
    """
    results = {}

    kernels = [
        ("nvfp4_gemm", create_nvfp4_gemm_kernel, True),   # W4A4
        ("w4a8_gemm", create_w4a8_gemm_kernel, True),     # W4A8
        ("w4a16_gemm", create_w4a16_gemm_kernel, False),  # W4A16 (no activation scale)
    ]

    for kernel_name, create_func, has_act_scale in kernels:
        print(f"\n{'='*60}")
        print(f"Generating {kernel_name}")
        print(f"{'='*60}")

        output_dir = os.path.join(output_base, kernel_name)

        # Create kernel
        tir_func = create_func(M, N, K, block_size)

        # Export CUDA source
        cuda_source, llvm_ir = export_cuda_source(tir_func, target)

        if not cuda_source:
            print(f"ERROR: Failed to export CUDA source for {kernel_name}")
            continue

        print(f"  CUDA source: {len(cuda_source)} bytes")

        # Generate TRT plugin wrapper
        generate_trt_plugin_wrapper(
            kernel_name, cuda_source, M, N, K, output_dir,
            has_activation_scale=has_act_scale
        )

        results[kernel_name] = {
            "output_dir": output_dir,
            "cuda_bytes": len(cuda_source),
        }

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TVM to TensorRT Plugin Generator")
    parser.add_argument("--kernel", type=str, default="nvfp4_gemm",
                        choices=["nvfp4_gemm", "w4a8_gemm", "w4a16_gemm", "all"],
                        help="Kernel type to generate ('all' for all three)")
    parser.add_argument("--M", type=int, default=1, help="Batch size")
    parser.add_argument("--N", type=int, default=3072, help="Output features")
    parser.add_argument("--K", type=int, default=3072, help="Input features")
    parser.add_argument("--block-size", type=int, default=32, help="Block size for scaling")
    parser.add_argument("--target", type=str, default="cuda -arch=sm_110",
                        help="TVM target")
    parser.add_argument("--output", type=str, default="/tmp/tvm_trt_plugin",
                        help="Output directory")
    parser.add_argument("--test", action="store_true", help="Run test after generation")

    args = parser.parse_args()

    if not check_tvm_environment():
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"TVM to TensorRT Plugin Generator")
    print(f"  M={args.M}, N={args.N}, K={args.K}")
    print(f"  Block size: {args.block_size}")
    print(f"  Target: {args.target}")
    print(f"{'='*60}\n")

    # Generate all kernels if requested
    if args.kernel == "all":
        results = generate_all_kernels(
            args.M, args.N, args.K, args.block_size,
            args.target, args.output
        )
        print(f"\n{'='*60}")
        print("SUMMARY - All Kernels Generated")
        print(f"{'='*60}")
        for name, info in results.items():
            print(f"  {name}: {info['output_dir']}")
        return

    # Create single kernel
    has_act_scale = args.kernel != "w4a16_gemm"

    if args.kernel == "nvfp4_gemm":
        tir_func = create_nvfp4_gemm_kernel(args.M, args.N, args.K, args.block_size)
    elif args.kernel == "w4a8_gemm":
        tir_func = create_w4a8_gemm_kernel(args.M, args.N, args.K, args.block_size)
    else:  # w4a16_gemm
        tir_func = create_w4a16_gemm_kernel(args.M, args.N, args.K, args.block_size)

    # Export CUDA source
    print("Building and exporting CUDA source...")
    cuda_source, llvm_ir = export_cuda_source(tir_func, args.target)

    if not cuda_source:
        print("ERROR: Failed to export CUDA source")
        sys.exit(1)

    print(f"  CUDA source: {len(cuda_source)} bytes")
    print(f"  LLVM IR: {len(llvm_ir)} bytes")

    # Generate TRT plugin wrapper
    print("\nGenerating TensorRT Plugin wrapper...")
    generate_trt_plugin_wrapper(
        args.kernel, cuda_source, args.M, args.N, args.K, args.output,
        has_activation_scale=has_act_scale
    )

    print(f"\n{'='*60}")
    print(f"SUCCESS! Files generated in: {args.output}")
    print(f"{'='*60}")

    # Show CUDA kernel preview
    print("\n--- CUDA Kernel Preview ---")
    lines = cuda_source.split('\n')
    kernel_start = next((i for i, l in enumerate(lines) if '__global__' in l), 0)
    kernel_end = min(kernel_start + 25, len(lines))
    print('\n'.join(lines[kernel_start:kernel_end]))

    if args.test:
        print("\n--- Running Test ---")
        # TODO: Add test code
        pass


if __name__ == "__main__":
    main()
