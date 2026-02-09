#!/usr/bin/env python3
"""
Setup script for NVFP4 GEMM PyTorch Extension

Build:
    cd nvfp4_extension
    python setup.py install

Or with pip:
    pip install -e .

Requirements:
    - CUDA 12.8+
    - PyTorch with CUDA support
    - CUTLASS (included in this repo)
    - SM110 GPU (Thor)
"""

import os
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this script
THIS_DIR = Path(__file__).parent.absolute()

# CUTLASS include path (in Docker container)
# Try multiple possible paths
CUTLASS_CANDIDATES = [
    Path("/workspace/external/cutlass_nvfp4_build/include"),
    Path("/workspace/external/cutlass_sm110_build"),
    THIS_DIR.parent.parent.parent.parent.parent / "external" / "cutlass_nvfp4_build" / "include",
]

CUTLASS_INCLUDE = None
for path in CUTLASS_CANDIDATES:
    if path.exists():
        CUTLASS_INCLUDE = path
        break

if CUTLASS_INCLUDE is None:
    CUTLASS_INCLUDE = Path("/workspace/external/cutlass_nvfp4_build/include")
    print(f"Warning: CUTLASS not found, using default: {CUTLASS_INCLUDE}")

CUTE_INCLUDE = CUTLASS_INCLUDE

print(f"CUTLASS include: {CUTLASS_INCLUDE}")
print(f"Building NVFP4 GEMM extension...")

# NVCC flags for SM110a (Thor)
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-gencode=arch=compute_110a,code=sm_110a",  # SM110a (Thor - must match GPU cc 11.0)
    "-DCUTLASS_ARCH_MMA_SM110_SUPPORTED=1",
    "-DCUTLASS_ENABLE_SM100_INSTRUCTIONS=1",
    "-DCUTLASS_ENABLE_SM110_INSTRUCTIONS=1",
]

# C++ compiler flags
cxx_flags = [
    "-O3",
    "-std=c++17",
]

# Include directories
CUTLASS_TOOLS_INCLUDE = CUTLASS_INCLUDE.parent / "tools" / "util" / "include"
include_dirs = [
    str(CUTLASS_INCLUDE),
    str(CUTE_INCLUDE),
    str(CUTLASS_TOOLS_INCLUDE),
]

# Define the extension
ext_modules = [
    CUDAExtension(
        name="nvfp4_gemm",
        sources=["nvfp4_gemm.cu"],
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        libraries=["cublas", "cublasLt"],
    )
]

setup(
    name="nvfp4_gemm",
    version="0.1.0",
    author="OpenPi Team",
    description="NVFP4 GEMM Extension for Thor SM110",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.10",
)
