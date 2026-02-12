#!/usr/bin/env python3
"""
Benchmark MLP Layers - Accurate Performance Comparison
=======================================================

Compare W4A16 TVM vs BF16 vs TRT FP8 baseline for MLP layers.
Ensures weights are pre-quantized (one-time cost) before benchmarking.

Usage:
    python /workspace/scripts/benchmark_mlp_layers.py

Author: Claude Code
Date: 2026-02-11
"""

import sys
import os
import time
import argparse
import pathlib

# Setup paths
script_dir = pathlib.Path(__file__).parent
for path in [
    script_dir.parent / "src",
    "/workspace/src",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# TVM paths
TVM_HOME = os.environ.get("TVM_HOME", "/workspace/external/tvm")
if TVM_HOME and os.path.exists(TVM_HOME):
    sys.path.insert(0, os.path.join(TVM_HOME, "python"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

# Model constants
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 16384
NUM_LAYERS = 18


class BF16MLP(nn.Module):
    """Standard BF16 MLP layer."""
    def __init__(self, hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.gelu(self.gate_proj(x), approximate='tanh')
        return self.down_proj(gate * self.up_proj(x))


def benchmark_bf16(device, iterations=100, warmup=20):
    """Benchmark BF16 MLP layers."""
    print("\n[BF16] Creating 18 MLP layers...")
    mlps = [BF16MLP().to(device).to(torch.bfloat16) for _ in range(NUM_LAYERS)]

    x = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    # Warmup
    print(f"[BF16] Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        out = x
        for mlp in mlps:
            out = mlp(out)
    torch.cuda.synchronize()

    # Benchmark
    print(f"[BF16] Benchmarking ({iterations} iterations)...")
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = x
        for mlp in mlps:
            out = mlp(out)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'per_layer_ms': np.mean(times) / NUM_LAYERS,
    }


def benchmark_w4a16_pytorch(device, iterations=100, warmup=20):
    """Benchmark W4A16 PyTorch (dequantize + matmul) MLP layers."""
    from openpi.models_pytorch.w4a16_mlp import W4A16MLP

    print("\n[W4A16 PyTorch] Creating 18 MLP layers...")
    mlps = []
    for i in range(NUM_LAYERS):
        mlp = W4A16MLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, use_tvm=False).to(device)
        mlps.append(mlp)
        if (i + 1) % 6 == 0:
            print(f"  Created {i + 1}/{NUM_LAYERS} layers")

    x = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    # Force quantization
    print("[W4A16 PyTorch] Pre-quantizing weights...")
    with torch.no_grad():
        for mlp in mlps:
            mlp.gate_proj.quantize_weights()
            mlp.up_proj.quantize_weights()
            mlp.down_proj.quantize_weights()

    # Warmup
    print(f"[W4A16 PyTorch] Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        out = x
        for mlp in mlps:
            out = mlp(out)
    torch.cuda.synchronize()

    # Benchmark
    print(f"[W4A16 PyTorch] Benchmarking ({iterations} iterations)...")
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = x
        for mlp in mlps:
            out = mlp(out)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'per_layer_ms': np.mean(times) / NUM_LAYERS,
    }


def benchmark_w4a16_tvm(device, iterations=100, warmup=20):
    """Benchmark W4A16 TVM (packed FP4 kernel) MLP layers."""
    from openpi.models_pytorch.w4a16_mlp import W4A16MLP, _tvm_available

    if not _tvm_available:
        print("[W4A16 TVM] TVM not available, skipping...")
        return None

    print("\n[W4A16 TVM] Creating 18 MLP layers...")
    mlps = []
    for i in range(NUM_LAYERS):
        mlp = W4A16MLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, use_tvm=True).to(device)
        mlps.append(mlp)
        if (i + 1) % 6 == 0:
            print(f"  Created {i + 1}/{NUM_LAYERS} layers")

    x = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    # Force quantization and TVM kernel caching
    print("[W4A16 TVM] Pre-quantizing weights and caching TVM kernels...")
    with torch.no_grad():
        for mlp in mlps:
            mlp.gate_proj.quantize_weights()
            mlp.up_proj.quantize_weights()
            mlp.down_proj.quantize_weights()
        # First forward pass to cache TVM kernels and weights
        out = x
        for mlp in mlps:
            out = mlp(out)
    torch.cuda.synchronize()

    # Warmup
    print(f"[W4A16 TVM] Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        out = x
        for mlp in mlps:
            out = mlp(out)
    torch.cuda.synchronize()

    # Benchmark
    print(f"[W4A16 TVM] Benchmarking ({iterations} iterations)...")
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = x
        for mlp in mlps:
            out = mlp(out)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'per_layer_ms': np.mean(times) / NUM_LAYERS,
    }


def benchmark_tvm_direct(device, iterations=100, warmup=20):
    """Benchmark TVM kernels directly (bypassing Python overhead)."""
    try:
        import tvm
        from openpi.models_pytorch.tvm_kernels.w4a16_gemv import (
            create_w4a16_gemv_fast,
            build_kernel,
            BLOCK_SIZE,
        )
    except ImportError:
        print("[TVM Direct] TVM not available, skipping...")
        return None

    print("\n[TVM Direct] Building TVM kernels...")

    H = HIDDEN_SIZE
    I = INTERMEDIATE_SIZE
    num_blocks_H = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_I = (I + BLOCK_SIZE - 1) // BLOCK_SIZE

    tvm_device = tvm.runtime.cuda(0)

    # Build kernels
    gate_up_kernel = create_w4a16_gemv_fast(I, H)
    gate_up_mod = build_kernel(gate_up_kernel, target="cuda -arch=sm_110")
    gate_up_func = gate_up_mod["w4a16_gemv_fast"]

    down_kernel = create_w4a16_gemv_fast(H, I)
    down_mod = build_kernel(down_kernel, target="cuda -arch=sm_110")
    down_func = down_mod["w4a16_gemv_fast"]

    print("[TVM Direct] Pre-quantizing weights for 18 layers...")

    # nvFP4 quantization
    NVFP4_LUT = np.array([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=np.float32)

    def quantize_to_nvfp4_packed(weight):
        N, K = weight.shape
        num_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

        weight_blocked = weight.reshape(N, num_blocks, BLOCK_SIZE)
        block_max = np.abs(weight_blocked).max(axis=2)
        scales = np.where(block_max > 0, block_max / 6.0, 1.0).astype(np.float32)

        scales_expanded = scales[:, :, np.newaxis]
        normalized = weight_blocked / scales_expanded
        normalized = normalized.reshape(N, K)

        diffs = np.abs(normalized[:, :, np.newaxis] - NVFP4_LUT[np.newaxis, np.newaxis, :])
        W_quant = np.argmin(diffs, axis=2).astype(np.uint8)

        W_packed = (W_quant[:, 0::2] & 0xF) | ((W_quant[:, 1::2] & 0xF) << 4)
        return W_packed, scales

    # Create and quantize weights for 18 layers
    np.random.seed(42)
    layer_data = []
    for layer_idx in range(NUM_LAYERS):
        gate_W_np = np.random.randn(I, H).astype(np.float32) * 0.1
        up_W_np = np.random.randn(I, H).astype(np.float32) * 0.1
        down_W_np = np.random.randn(H, I).astype(np.float32) * 0.1

        gate_W_packed, gate_scales = quantize_to_nvfp4_packed(gate_W_np)
        up_W_packed, up_scales = quantize_to_nvfp4_packed(up_W_np)
        down_W_packed, down_scales = quantize_to_nvfp4_packed(down_W_np)

        # TVM arrays
        gate_W_tvm = tvm.runtime.empty((I, H // 2), "uint8", tvm_device)
        gate_W_tvm.copyfrom(gate_W_packed)
        gate_scales_tvm = tvm.runtime.empty((I, num_blocks_H), "float32", tvm_device)
        gate_scales_tvm.copyfrom(gate_scales)

        up_W_tvm = tvm.runtime.empty((I, H // 2), "uint8", tvm_device)
        up_W_tvm.copyfrom(up_W_packed)
        up_scales_tvm = tvm.runtime.empty((I, num_blocks_H), "float32", tvm_device)
        up_scales_tvm.copyfrom(up_scales)

        down_W_tvm = tvm.runtime.empty((H, I // 2), "uint8", tvm_device)
        down_W_tvm.copyfrom(down_W_packed)
        down_scales_tvm = tvm.runtime.empty((H, num_blocks_I), "float32", tvm_device)
        down_scales_tvm.copyfrom(down_scales)

        layer_data.append({
            "gate_W": gate_W_tvm, "gate_S": gate_scales_tvm,
            "up_W": up_W_tvm, "up_S": up_scales_tvm,
            "down_W": down_W_tvm, "down_S": down_scales_tvm,
        })

    # I/O buffers
    x_tvm = tvm.runtime.empty((1, H), "float32", tvm_device)
    x_np = np.random.randn(1, H).astype(np.float32)
    x_tvm.copyfrom(x_np)

    gate_out_tvm = tvm.runtime.empty((1, I), "float32", tvm_device)
    up_out_tvm = tvm.runtime.empty((1, I), "float32", tvm_device)
    down_out_tvm = tvm.runtime.empty((1, H), "float32", tvm_device)

    # Warmup
    print(f"[TVM Direct] Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        for layer in layer_data:
            gate_up_func(x_tvm, layer["gate_W"], layer["gate_S"], gate_out_tvm)
            gate_up_func(x_tvm, layer["up_W"], layer["up_S"], up_out_tvm)
            down_func(gate_out_tvm, layer["down_W"], layer["down_S"], down_out_tvm)
    tvm_device.sync()

    # Benchmark
    print(f"[TVM Direct] Benchmarking ({iterations} iterations)...")
    times = []
    for _ in range(iterations):
        tvm_device.sync()
        start = time.perf_counter()
        for layer in layer_data:
            gate_up_func(x_tvm, layer["gate_W"], layer["gate_S"], gate_out_tvm)
            gate_up_func(x_tvm, layer["up_W"], layer["up_S"], up_out_tvm)
            down_func(gate_out_tvm, layer["down_W"], layer["down_S"], down_out_tvm)
        tvm_device.sync()
        times.append((time.perf_counter() - start) * 1000)

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'per_layer_ms': np.mean(times) / NUM_LAYERS,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLP Layers")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    print("=" * 70)
    print("MLP Layer Benchmark - Accurate Performance Comparison")
    print("=" * 70)
    print(f"Model: Pi0.5 / PaliGemma")
    print(f"Hidden Size: {HIDDEN_SIZE}")
    print(f"Intermediate Size: {INTERMEDIATE_SIZE}")
    print(f"Num Layers: {NUM_LAYERS}")
    print(f"Batch Size: 1")
    print(f"Iterations: {args.iterations}")
    print("=" * 70)

    device = torch.device('cuda')

    results = {}

    # 1. BF16 Baseline
    results['BF16'] = benchmark_bf16(device, args.iterations, args.warmup)

    # 2. W4A16 PyTorch (dequantize + matmul)
    results['W4A16 PyTorch'] = benchmark_w4a16_pytorch(device, args.iterations, args.warmup)

    # 3. W4A16 TVM (via Python interface)
    results['W4A16 TVM (Python)'] = benchmark_w4a16_tvm(device, args.iterations, args.warmup)

    # 4. TVM Direct (bypassing PyTorch)
    results['W4A16 TVM (Direct)'] = benchmark_tvm_direct(device, args.iterations, args.warmup)

    # Results Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (18 MLP Layers, batch=1)")
    print("=" * 70)
    print(f"\n{'Method':<25} {'Total (ms)':>12} {'Per-Layer':>12} {'vs TRT FP8':>12}")
    print("-" * 70)

    trt_fp8_baseline = 12.39  # ms for 18 layers

    for method, data in results.items():
        if data is None:
            continue
        total = data['mean_ms']
        per_layer = data['per_layer_ms']
        ratio = total / trt_fp8_baseline
        print(f"{method:<25} {total:>9.2f} ms {per_layer:>9.3f} ms {ratio:>9.2f}x")

    print("-" * 70)
    print(f"{'TRT FP8 (baseline)':<25} {trt_fp8_baseline:>9.2f} ms {trt_fp8_baseline/NUM_LAYERS:>9.3f} ms {'1.00x':>12}")
    print("-" * 70)

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if results.get('W4A16 TVM (Direct)'):
        tvm_direct = results['W4A16 TVM (Direct)']['mean_ms']
        bf16 = results['BF16']['mean_ms']
        python_overhead = results['W4A16 TVM (Python)']['mean_ms'] - tvm_direct if results.get('W4A16 TVM (Python)') else 0

        print(f"""
Performance Breakdown:
  BF16 (PyTorch):           {bf16:.2f} ms
  TRT FP8 (baseline):       {trt_fp8_baseline:.2f} ms
  W4A16 TVM (Direct):       {tvm_direct:.2f} ms
  Python call overhead:     ~{python_overhead:.2f} ms (if using Python interface)

Comparisons:
  W4A16 TVM vs BF16:        {bf16/tvm_direct:.2f}x speedup
  W4A16 TVM vs TRT FP8:     {tvm_direct/trt_fp8_baseline:.2f}x ({"faster" if tvm_direct < trt_fp8_baseline else "slower"})

Memory Savings (W4A16 nvFP4 vs FP16):
  FP16 MLP weights:         {(INTERMEDIATE_SIZE * HIDDEN_SIZE * 2 + INTERMEDIATE_SIZE * HIDDEN_SIZE * 2 + HIDDEN_SIZE * INTERMEDIATE_SIZE * 2) * NUM_LAYERS / 1024 / 1024:.1f} MB
  nvFP4 MLP weights:        {(INTERMEDIATE_SIZE * HIDDEN_SIZE // 2 + INTERMEDIATE_SIZE * HIDDEN_SIZE // 2 + HIDDEN_SIZE * INTERMEDIATE_SIZE // 2) * NUM_LAYERS / 1024 / 1024:.1f} MB
  Compression:              ~3.2x
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
