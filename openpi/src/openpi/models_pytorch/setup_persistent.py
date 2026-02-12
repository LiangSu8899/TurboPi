#!/usr/bin/env python3
"""
Setup script for NVFP4 Persistent MLP PyTorch Extension.

Build with:
    python setup_persistent.py build_ext --inplace

Author: Claude Code
Date: 2026-02-10
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Get CUDA compute capability
cuda_version = torch.version.cuda
print(f"CUDA version: {cuda_version}")
print(f"PyTorch version: {torch.__version__}")

# Thor is SM 8.7 (not SM 11.0 as previously thought)
# Check actual GPU
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    cc = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {cc[0]}.{cc[1]}")
    arch_flag = f"-arch=sm_{cc[0]}{cc[1]}"
else:
    # Default to SM 8.7 for Thor
    arch_flag = "-arch=sm_87"
    print(f"No GPU detected, using default {arch_flag}")

print(f"Using: {arch_flag}")

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='nvfp4_persistent',
    ext_modules=[
        CUDAExtension(
            name='nvfp4_persistent',
            sources=[
                os.path.join(current_dir, 'nvfp4_persistent_extension.cpp'),
                os.path.join(current_dir, 'nvfp4_nlayer_persistent.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    arch_flag,
                    '-std=c++17',
                    '-Xptxas=-v',  # Verbose PTX assembly
                    '-lineinfo',   # Line info for profiling
                ],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
