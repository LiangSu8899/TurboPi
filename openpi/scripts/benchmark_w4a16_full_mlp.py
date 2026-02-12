#!/usr/bin/env python3
"""
Benchmark W4A16 Full MLP Performance

Tests the complete MLP pipeline:
- gate_proj + up_proj (fused)
- GeLU activation
- element-wise multiply
- down_proj

Compares against TRT FP8 baseline (12.39ms for 18 layers).

Author: Claude Code
Date: 2026-02-11
"""

import sys
sys.path.insert(0, "/workspace/src")
sys.path.insert(0, "/workspace/external/tvm/python")

import numpy as np
import time
import torch
import torch.nn.functional as F

import tvm
import tvm.runtime

from openpi.models_pytorch.tvm_kernels.w4a16_gemv import (
    create_w4a16_gemv_fast,
    build_kernel,
    quantize_to_nvfp4_packed,
    BLOCK_SIZE,
)

# Dimensions
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 16384


def benchmark_full_mlp():
    """Benchmark complete MLP (gate + up + GeLU + mul + down)"""
    print("=" * 70)
    print("W4A16 Full MLP Benchmark")
    print("=" * 70)
    print(f"\nDimensions:")
    print(f"  hidden_size = {HIDDEN_SIZE}")
    print(f"  intermediate_size = {INTERMEDIATE_SIZE}")

    H = HIDDEN_SIZE
    I = INTERMEDIATE_SIZE
    num_blocks_H = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_I = (I + BLOCK_SIZE - 1) // BLOCK_SIZE

    device = tvm.runtime.cuda(0)
    torch_device = torch.device("cuda:0")

    # Build kernels
    print("\n[1/3] Building gate/up kernel (N=16384, K=2048)...")
    gate_up_kernel = create_w4a16_gemv_fast(I, H)
    gate_up_mod = build_kernel(gate_up_kernel, target="cuda -arch=sm_110")
    gate_up_func = gate_up_mod["w4a16_gemv_fast"]

    print("[2/3] Building down kernel (N=2048, K=16384)...")
    down_kernel = create_w4a16_gemv_fast(H, I)
    down_mod = build_kernel(down_kernel, target="cuda -arch=sm_110")
    down_func = down_mod["w4a16_gemv_fast"]

    print("[3/3] Preparing test data...")

    # Generate weights
    np.random.seed(42)
    gate_W_np = np.random.randn(I, H).astype(np.float32) * 0.1
    up_W_np = np.random.randn(I, H).astype(np.float32) * 0.1
    down_W_np = np.random.randn(H, I).astype(np.float32) * 0.1

    # Quantize
    gate_W_packed, gate_scales = quantize_to_nvfp4_packed(gate_W_np)
    up_W_packed, up_scales = quantize_to_nvfp4_packed(up_W_np)
    down_W_packed, down_scales = quantize_to_nvfp4_packed(down_W_np)

    # TVM arrays
    x_tvm = tvm.runtime.empty((1, H), "float32", device)

    gate_W_tvm = tvm.runtime.empty((I, H // 2), "uint8", device)
    gate_W_tvm.copyfrom(gate_W_packed)
    gate_scales_tvm = tvm.runtime.empty((I, num_blocks_H), "float32", device)
    gate_scales_tvm.copyfrom(gate_scales)

    up_W_tvm = tvm.runtime.empty((I, H // 2), "uint8", device)
    up_W_tvm.copyfrom(up_W_packed)
    up_scales_tvm = tvm.runtime.empty((I, num_blocks_H), "float32", device)
    up_scales_tvm.copyfrom(up_scales)

    down_W_tvm = tvm.runtime.empty((H, I // 2), "uint8", device)
    down_W_tvm.copyfrom(down_W_packed)
    down_scales_tvm = tvm.runtime.empty((H, num_blocks_I), "float32", device)
    down_scales_tvm.copyfrom(down_scales)

    gate_out_tvm = tvm.runtime.empty((1, I), "float32", device)
    up_out_tvm = tvm.runtime.empty((1, I), "float32", device)
    down_out_tvm = tvm.runtime.empty((1, H), "float32", device)

    # PyTorch tensors for GeLU (shared memory with TVM via DLPack)
    gate_out_torch = torch.from_dlpack(gate_out_tvm)
    up_out_torch = torch.from_dlpack(up_out_tvm)
    gelu_mul_torch = torch.empty(1, I, device=torch_device, dtype=torch.float32)

    # Input
    x_np = np.random.randn(1, H).astype(np.float32)
    x_tvm.copyfrom(x_np)

    def run_full_mlp():
        """Run one complete MLP forward pass"""
        # gate_proj
        gate_up_func(x_tvm, gate_W_tvm, gate_scales_tvm, gate_out_tvm)
        # up_proj
        gate_up_func(x_tvm, up_W_tvm, up_scales_tvm, up_out_tvm)
        # GeLU(gate) * up using PyTorch (in-place)
        torch.mul(F.gelu(gate_out_torch, approximate='tanh'), up_out_torch, out=gelu_mul_torch)
        # down_proj (read from gelu_mul_torch via shared memory)
        gelu_mul_tvm = tvm.runtime.from_dlpack(gelu_mul_torch)
        down_func(gelu_mul_tvm, down_W_tvm, down_scales_tvm, down_out_tvm)

    # Warmup
    print("\nWarming up...")
    for _ in range(20):
        run_full_mlp()
    device.sync()

    # Benchmark
    runs = 100
    print(f"Benchmarking ({runs} iterations)...")

    # Individual kernels
    device.sync()
    start = time.time()
    for _ in range(runs):
        gate_up_func(x_tvm, gate_W_tvm, gate_scales_tvm, gate_out_tvm)
    device.sync()
    gate_ms = (time.time() - start) / runs * 1000

    device.sync()
    start = time.time()
    for _ in range(runs):
        gate_up_func(x_tvm, up_W_tvm, up_scales_tvm, up_out_tvm)
    device.sync()
    up_ms = (time.time() - start) / runs * 1000

    device.sync()
    start = time.time()
    for _ in range(runs):
        torch.mul(F.gelu(gate_out_torch, approximate='tanh'), up_out_torch, out=gelu_mul_torch)
    torch.cuda.synchronize()
    gelu_mul_ms = (time.time() - start) / runs * 1000

    gelu_mul_tvm = tvm.runtime.from_dlpack(gelu_mul_torch)
    device.sync()
    start = time.time()
    for _ in range(runs):
        down_func(gelu_mul_tvm, down_W_tvm, down_scales_tvm, down_out_tvm)
    device.sync()
    down_ms = (time.time() - start) / runs * 1000

    # Full MLP
    device.sync()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        run_full_mlp()
    device.sync()
    torch.cuda.synchronize()
    full_ms = (time.time() - start) / runs * 1000

    # Results
    print("\n" + "=" * 70)
    print("RESULTS: Single Layer")
    print("=" * 70)
    print(f"\n{'Operation':<20} {'Time (ms)':<12} {'Notes'}")
    print("-" * 50)
    print(f"{'gate_proj':<20} {gate_ms:<12.3f} N=16384, K=2048")
    print(f"{'up_proj':<20} {up_ms:<12.3f} N=16384, K=2048")
    print(f"{'GeLU * up':<20} {gelu_mul_ms:<12.3f} PyTorch, I=16384")
    print(f"{'down_proj':<20} {down_ms:<12.3f} N=2048, K=16384")
    print("-" * 50)
    component_sum = gate_ms + up_ms + gelu_mul_ms + down_ms
    print(f"{'Sum':<20} {component_sum:<12.3f}")
    print(f"{'Full MLP (measured)':<20} {full_ms:<12.3f}")
    overhead = full_ms - component_sum
    print(f"{'Overhead':<20} {overhead:<12.3f}")

    # 18-layer projection
    print("\n" + "=" * 70)
    print("18-LAYER PROJECTION")
    print("=" * 70)

    trt_fp8_baseline = 12.39  # ms
    w4a16_18layers = full_ms * 18

    print(f"\n{'Method':<25} {'Time (ms)':<12} {'vs TRT FP8'}")
    print("-" * 50)
    print(f"{'TRT FP8 (baseline)':<25} {trt_fp8_baseline:<12.2f} 1.00x")
    print(f"{'W4A16 TVM':<25} {w4a16_18layers:<12.2f} ", end="")

    if w4a16_18layers < trt_fp8_baseline:
        speedup = trt_fp8_baseline / w4a16_18layers
        print(f"{speedup:.2f}x FASTER")
    else:
        slowdown = w4a16_18layers / trt_fp8_baseline
        print(f"{slowdown:.2f}x slower")

    print("=" * 70)

    return {
        "gate_ms": gate_ms,
        "up_ms": up_ms,
        "gelu_mul_ms": gelu_mul_ms,
        "down_ms": down_ms,
        "full_ms": full_ms,
        "18layer_ms": w4a16_18layers,
    }


if __name__ == "__main__":
    benchmark_full_mlp()
