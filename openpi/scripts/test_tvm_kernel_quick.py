#!/usr/bin/env python3
"""Quick test for W4A16 TVM kernel."""

import sys
sys.path.insert(0, "/workspace/src")
sys.path.insert(0, "/workspace/external/tvm/python")

import numpy as np
import time

print("=" * 60)
print("W4A16 TVM Kernel Quick Test")
print("=" * 60)

from openpi.models_pytorch.tvm_kernels.w4a16_gemv import (
    create_w4a16_gemv_fast,
    build_kernel,
    quantize_to_nvfp4_packed,
    NVFP4_LUT,
    BLOCK_SIZE,
)

import tvm

# Test dimensions
N, K = 16384, 2048
print(f"\nTest: N={N}, K={K}")

# Generate test data
np.random.seed(42)
A_np = np.random.randn(1, K).astype(np.float32)
W_np = np.random.randn(N, K).astype(np.float32) * 0.1

# Quantize
W_packed, scales = quantize_to_nvfp4_packed(W_np)
num_blocks_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

print("Building TVM kernel...")
kernel_func = create_w4a16_gemv_fast(N, K)
mod = build_kernel(kernel_func, target="cuda -arch=sm_110")
func = mod["w4a16_gemv_fast"]
print("Kernel built successfully!")

# Setup TVM arrays
device = tvm.runtime.cuda(0)
A_tvm = tvm.runtime.empty((1, K), dtype="float32", device=device)
W_packed_tvm = tvm.runtime.empty((N, K // 2), dtype="uint8", device=device)
scales_tvm = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
out_tvm = tvm.runtime.empty((1, N), dtype="float32", device=device)

A_tvm.copyfrom(A_np)
W_packed_tvm.copyfrom(W_packed)
scales_tvm.copyfrom(scales)

# Warmup
for _ in range(20):
    func(A_tvm, W_packed_tvm, scales_tvm, out_tvm)
device.sync()

# Benchmark
runs = 100
device.sync()
start = time.time()
for _ in range(runs):
    func(A_tvm, W_packed_tvm, scales_tvm, out_tvm)
device.sync()
elapsed_ms = (time.time() - start) / runs * 1000

print(f"\nPerformance: {elapsed_ms:.3f} ms per call")
print(f"Throughput:  {1000/elapsed_ms:.1f} calls/sec")
print("=" * 60)
print("TVM Kernel Test PASSED!")
print("=" * 60)
