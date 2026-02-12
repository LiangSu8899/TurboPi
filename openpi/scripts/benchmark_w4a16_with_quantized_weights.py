#!/usr/bin/env python3
"""
Benchmark W4A16 TVM with Pre-quantized Weights

This script benchmarks W4A16 MLP performance using:
1. Pre-quantized nvFP4 weights (loaded from safetensors)
2. TVM kernels (compiled for SM110)

This represents the actual performance we'll get when integrated into the inference pipeline.

Usage:
    In Docker container (turbo_pi_eval):
    python /workspace/scripts/benchmark_w4a16_with_quantized_weights.py
"""

import sys
import os
import numpy as np
import time

# Add paths
sys.path.insert(0, "/workspace/src")
sys.path.insert(0, "/workspace/external/tvm/python")

# Check TVM availability
try:
    import tvm
    print(f"TVM version: {tvm.__version__}")
except ImportError as e:
    print(f"TVM not available: {e}")
    sys.exit(1)

from safetensors import safe_open

# Constants
HIDDEN_SIZE = 2048
MLP_DIM = 16384
BLOCK_SIZE = 32
NUM_LAYERS = 18

QUANTIZED_WEIGHTS = "/workspace/quantized_weights/mlp_weights_nvfp4.safetensors"


def benchmark_mlp_layers():
    """Benchmark MLP layers with pre-quantized weights."""
    from openpi.models_pytorch.tvm_kernels.w4a16_gemv import (
        create_w4a16_gemv_fast,
        build_kernel,
        BLOCK_SIZE,
    )

    H = HIDDEN_SIZE
    I = MLP_DIM
    num_blocks_H = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_I = (I + BLOCK_SIZE - 1) // BLOCK_SIZE

    device = tvm.runtime.cuda(0)

    # Build TVM kernels
    print("\n[Step 1] Building TVM kernels...")
    gate_up_kernel = create_w4a16_gemv_fast(I, H)
    gate_up_mod = build_kernel(gate_up_kernel, target="cuda -arch=sm_110")
    gate_up_func = gate_up_mod["w4a16_gemv_fast"]

    down_kernel = create_w4a16_gemv_fast(H, I)
    down_mod = build_kernel(down_kernel, target="cuda -arch=sm_110")
    down_func = down_mod["w4a16_gemv_fast"]
    print("  Done")

    # Load pre-quantized weights
    print("\n[Step 2] Loading pre-quantized weights...")
    print(f"  File: {QUANTIZED_WEIGHTS}")

    layer_data = []
    with safe_open(QUANTIZED_WEIGHTS, framework="numpy") as f:
        for layer_idx in range(NUM_LAYERS):
            gate_W = f.get_tensor(f"layer.{layer_idx}.gate_proj.weight_packed")
            gate_S = f.get_tensor(f"layer.{layer_idx}.gate_proj.scales")
            up_W = f.get_tensor(f"layer.{layer_idx}.up_proj.weight_packed")
            up_S = f.get_tensor(f"layer.{layer_idx}.up_proj.scales")
            down_W = f.get_tensor(f"layer.{layer_idx}.down_proj.weight_packed")
            down_S = f.get_tensor(f"layer.{layer_idx}.down_proj.scales")

            # Create TVM arrays
            gate_W_tvm = tvm.runtime.empty((I, H // 2), "uint8", device)
            gate_W_tvm.copyfrom(gate_W)
            gate_S_tvm = tvm.runtime.empty((I, num_blocks_H), "float32", device)
            gate_S_tvm.copyfrom(gate_S)

            up_W_tvm = tvm.runtime.empty((I, H // 2), "uint8", device)
            up_W_tvm.copyfrom(up_W)
            up_S_tvm = tvm.runtime.empty((I, num_blocks_H), "float32", device)
            up_S_tvm.copyfrom(up_S)

            down_W_tvm = tvm.runtime.empty((H, I // 2), "uint8", device)
            down_W_tvm.copyfrom(down_W)
            down_S_tvm = tvm.runtime.empty((H, num_blocks_I), "float32", device)
            down_S_tvm.copyfrom(down_S)

            layer_data.append({
                "gate_W": gate_W_tvm,
                "gate_S": gate_S_tvm,
                "up_W": up_W_tvm,
                "up_S": up_S_tvm,
                "down_W": down_W_tvm,
                "down_S": down_S_tvm,
            })

    print(f"  Loaded {len(layer_data)} layers")

    # Create input/output buffers
    x_tvm = tvm.runtime.empty((1, H), "float32", device)
    x_np = np.random.randn(1, H).astype(np.float32)
    x_tvm.copyfrom(x_np)

    gate_out_tvm = tvm.runtime.empty((1, I), "float32", device)
    up_out_tvm = tvm.runtime.empty((1, I), "float32", device)
    down_out_tvm = tvm.runtime.empty((1, H), "float32", device)

    # Warmup
    print("\n[Step 3] Warming up...")
    for _ in range(20):
        for layer in layer_data:
            gate_up_func(x_tvm, layer["gate_W"], layer["gate_S"], gate_out_tvm)
            gate_up_func(x_tvm, layer["up_W"], layer["up_S"], up_out_tvm)
            down_func(gate_out_tvm, layer["down_W"], layer["down_S"], down_out_tvm)
    device.sync()
    print("  Done")

    # Benchmark
    runs = 100
    print(f"\n[Step 4] Benchmarking ({runs} runs)...")

    device.sync()
    start = time.time()
    for _ in range(runs):
        for layer in layer_data:
            # gate_proj
            gate_up_func(x_tvm, layer["gate_W"], layer["gate_S"], gate_out_tvm)
            # up_proj
            gate_up_func(x_tvm, layer["up_W"], layer["up_S"], up_out_tvm)
            # down_proj (GeLU*up is skipped for pure MLP timing)
            down_func(gate_out_tvm, layer["down_W"], layer["down_S"], down_out_tvm)
    device.sync()
    elapsed = (time.time() - start) / runs * 1000

    # Results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"W4A16 TVM MLP (18 layers):     {elapsed:.3f} ms")
    print(f"Per-layer time:                {elapsed / NUM_LAYERS:.4f} ms")
    print()
    print("Comparison:")
    print(f"  TRT FP8 baseline:            12.39 ms")
    print(f"  W4A16 TVM (Python):          {elapsed:.2f} ms")
    print(f"  Python overhead (~0.094ms/layer): ~{0.094 * NUM_LAYERS:.2f} ms")
    print(f"  Expected C++ TRT Plugin:     ~{elapsed - 0.094 * NUM_LAYERS:.2f} ms")
    print()

    if elapsed < 12.39:
        print(f"  *** W4A16 TVM is {12.39 - elapsed:.2f}ms FASTER than TRT FP8! ***")
    else:
        gap = elapsed - 12.39
        expected_cpp = elapsed - 0.094 * NUM_LAYERS
        print(f"  Current gap vs TRT FP8:      +{gap:.2f} ms")
        if expected_cpp < 12.39:
            print(f"  With C++ plugin (no Python overhead): {12.39 - expected_cpp:.2f}ms FASTER")
        else:
            print(f"  Even with C++ plugin:        +{expected_cpp - 12.39:.2f} ms")

    print()
    print("Memory footprint:")
    fp16_mb = (I * H * 2 + I * H * 2 + H * I * 2) * NUM_LAYERS / 1024 / 1024
    nvfp4_mb = (I * H // 2 + I * num_blocks_H * 4 + I * H // 2 + I * num_blocks_H * 4 + H * I // 2 + H * num_blocks_I * 4) * NUM_LAYERS / 1024 / 1024
    print(f"  FP16 weights:    {fp16_mb:.1f} MB")
    print(f"  nvFP4 weights:   {nvfp4_mb:.1f} MB ({fp16_mb / nvfp4_mb:.1f}x compression)")

    return elapsed


def main():
    print("=" * 60)
    print("W4A16 TVM Benchmark with Pre-quantized Weights")
    print("=" * 60)
    print(f"Model: Pi0.5 / PaliGemma")
    print(f"Hidden Size: {HIDDEN_SIZE}")
    print(f"MLP Dim: {MLP_DIM}")
    print(f"Num Layers: {NUM_LAYERS}")
    print()

    if not os.path.exists(QUANTIZED_WEIGHTS):
        print(f"ERROR: Quantized weights not found: {QUANTIZED_WEIGHTS}")
        print("Run quantize_mlp_weights_nvfp4.py first.")
        return

    benchmark_mlp_layers()


if __name__ == "__main__":
    main()
