#!/usr/bin/env python3
"""
Benchmark W4A16 TVM MLP Performance

This script benchmarks the W4A16 TVM GEMV kernels for MLP layers,
comparing against TRT FP8 baseline.

Performance targets:
- TRT FP8 (18 layers): 12.39 ms
- W4A16 TVM (Python): ~13.87 ms (with Python overhead)
- W4A16 TVM (C++ Plugin): ~12.09 ms (expected, eliminating overhead)

Usage:
    In Docker container (turbo_pi_eval):
    python /workspace/scripts/benchmark_w4a16_mlp.py
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

# Constants
HIDDEN_SIZE = 2048
MLP_DIM = 16384
BLOCK_SIZE = 32
NUM_LAYERS = 18
SEQ_LEN = 968

# nvFP4 E2M1 lookup table
NVFP4_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # positive
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0  # negative
], dtype=np.float32)


def quantize_to_nvfp4_packed(weight: np.ndarray, block_size: int = BLOCK_SIZE):
    """Quantize weight to packed nvFP4 format."""
    N, K = weight.shape
    num_blocks = (K + block_size - 1) // block_size

    # Compute scales per block
    scales = np.zeros((N, num_blocks), dtype=np.float32)
    for n in range(N):
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, K)
            block_max = np.abs(weight[n, start:end]).max()
            scales[n, b] = block_max / 6.0 if block_max > 0 else 1.0

    # Quantize to nvFP4 indices
    W_quant = np.zeros((N, K), dtype=np.int32)
    for n in range(N):
        for k in range(K):
            block_idx = k // block_size
            scaled_val = weight[n, k] / scales[n, block_idx]

            # Find closest nvFP4 value
            best_idx = 0
            best_diff = abs(scaled_val - NVFP4_LUT[0])
            for i in range(1, 16):
                diff = abs(scaled_val - NVFP4_LUT[i])
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            W_quant[n, k] = best_idx

    # Pack to uint8 (2 FP4 per byte)
    K_packed = K // 2
    W_packed = np.zeros((N, K_packed), dtype=np.uint8)
    for n in range(N):
        for k in range(0, K, 2):
            low = W_quant[n, k] & 0xF
            high = W_quant[n, k + 1] & 0xF
            W_packed[n, k // 2] = low | (high << 4)

    return W_packed, scales


def benchmark_single_token_mlp():
    """Benchmark W4A16 MLP for single token (batch=1, seq=1)."""
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

    # Build kernels
    print("\n[Building TVM kernels...]")
    gate_up_kernel = create_w4a16_gemv_fast(I, H)
    gate_up_mod = build_kernel(gate_up_kernel, target="cuda -arch=sm_110")
    gate_up_func = gate_up_mod["w4a16_gemv_fast"]

    down_kernel = create_w4a16_gemv_fast(H, I)
    down_mod = build_kernel(down_kernel, target="cuda -arch=sm_110")
    down_func = down_mod["w4a16_gemv_fast"]

    # Prepare data for 18 layers
    print("\n[Preparing quantized weights for 18 layers...]")
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

        layer_data.append({
            "gate_W": gate_W_tvm,
            "gate_S": gate_scales_tvm,
            "up_W": up_W_tvm,
            "up_S": up_scales_tvm,
            "down_W": down_W_tvm,
            "down_S": down_scales_tvm,
        })

    # Input and output buffers
    x_tvm = tvm.runtime.empty((1, H), "float32", device)
    x_np = np.random.randn(1, H).astype(np.float32)
    x_tvm.copyfrom(x_np)

    gate_out_tvm = tvm.runtime.empty((1, I), "float32", device)
    up_out_tvm = tvm.runtime.empty((1, I), "float32", device)
    down_out_tvm = tvm.runtime.empty((1, H), "float32", device)

    # Warmup
    print("\n[Warming up...]")
    for _ in range(20):
        for layer in layer_data:
            gate_up_func(x_tvm, layer["gate_W"], layer["gate_S"], gate_out_tvm)
            gate_up_func(x_tvm, layer["up_W"], layer["up_S"], up_out_tvm)
            down_func(gate_out_tvm, layer["down_W"], layer["down_S"], down_out_tvm)
    device.sync()

    # Benchmark
    runs = 100
    print(f"\n[Benchmarking ({runs} runs)...]")

    device.sync()
    start = time.time()
    for _ in range(runs):
        for layer in layer_data:
            gate_up_func(x_tvm, layer["gate_W"], layer["gate_S"], gate_out_tvm)
            gate_up_func(x_tvm, layer["up_W"], layer["up_S"], up_out_tvm)
            down_func(gate_out_tvm, layer["down_W"], layer["down_S"], down_out_tvm)
    device.sync()
    elapsed = (time.time() - start) / runs * 1000

    print(f"\n{'='*60}")
    print(f"W4A16 TVM MLP Benchmark (Single Token, 18 Layers)")
    print(f"{'='*60}")
    print(f"Total time for 18 layers: {elapsed:.3f} ms")
    print(f"Per-layer time: {elapsed / NUM_LAYERS:.3f} ms")
    print(f"")
    print(f"Comparison:")
    print(f"  TRT FP8 baseline:  12.39 ms")
    print(f"  W4A16 TVM Python:  {elapsed:.2f} ms")
    print(f"  Python overhead:   ~0.094 ms/layer = ~1.7 ms total")
    print(f"  Expected C++:      ~{elapsed - 1.7:.2f} ms")
    print(f"")
    print(f"Memory Savings (W4A16 vs FP16):")
    print(f"  FP16 MLP weights: {(I * H * 2 + I * H * 2 + H * I * 2) * NUM_LAYERS / 1024 / 1024:.1f} MB")
    print(f"  W4A16 weights:    {(I * H // 2 + I * H // 2 + H * I // 2) * NUM_LAYERS / 1024 / 1024:.1f} MB (4x compression)")

    return elapsed


def benchmark_full_sequence_mlp():
    """Benchmark W4A16 MLP for full sequence (batch=1, seq=968)."""
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

    # Build kernels
    print("\n[Building TVM kernels for full sequence...]")
    gate_up_kernel = create_w4a16_gemv_fast(I, H)
    gate_up_mod = build_kernel(gate_up_kernel, target="cuda -arch=sm_110")
    gate_up_func = gate_up_mod["w4a16_gemv_fast"]

    down_kernel = create_w4a16_gemv_fast(H, I)
    down_mod = build_kernel(down_kernel, target="cuda -arch=sm_110")
    down_func = down_mod["w4a16_gemv_fast"]

    # Prepare data for 1 layer
    print("\n[Preparing quantized weights...]")
    np.random.seed(42)

    gate_W_np = np.random.randn(I, H).astype(np.float32) * 0.1
    up_W_np = np.random.randn(I, H).astype(np.float32) * 0.1
    down_W_np = np.random.randn(H, I).astype(np.float32) * 0.1

    gate_W_packed, gate_scales = quantize_to_nvfp4_packed(gate_W_np)
    up_W_packed, up_scales = quantize_to_nvfp4_packed(up_W_np)
    down_W_packed, down_scales = quantize_to_nvfp4_packed(down_W_np)

    # TVM arrays
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

    # Input and output buffers (single token - GEMV)
    x_tvm = tvm.runtime.empty((1, H), "float32", device)
    gate_out_tvm = tvm.runtime.empty((1, I), "float32", device)
    up_out_tvm = tvm.runtime.empty((1, I), "float32", device)
    down_out_tvm = tvm.runtime.empty((1, H), "float32", device)

    # Warmup
    print("\n[Warming up...]")
    x_np = np.random.randn(1, H).astype(np.float32)
    x_tvm.copyfrom(x_np)
    for _ in range(20):
        gate_up_func(x_tvm, gate_W_tvm, gate_scales_tvm, gate_out_tvm)
        gate_up_func(x_tvm, up_W_tvm, up_scales_tvm, up_out_tvm)
        down_func(gate_out_tvm, down_W_tvm, down_scales_tvm, down_out_tvm)
    device.sync()

    # Benchmark: Process full sequence token by token
    runs = 10
    print(f"\n[Benchmarking full sequence ({SEQ_LEN} tokens, {runs} runs)...]")

    device.sync()
    start = time.time()
    for _ in range(runs):
        for token_idx in range(SEQ_LEN):
            # In real inference, we'd process each token
            gate_up_func(x_tvm, gate_W_tvm, gate_scales_tvm, gate_out_tvm)
            gate_up_func(x_tvm, up_W_tvm, up_scales_tvm, up_out_tvm)
            down_func(gate_out_tvm, down_W_tvm, down_scales_tvm, down_out_tvm)
    device.sync()
    elapsed = (time.time() - start) / runs * 1000

    print(f"\n{'='*60}")
    print(f"W4A16 TVM MLP Benchmark (Full Sequence: {SEQ_LEN} tokens)")
    print(f"{'='*60}")
    print(f"Total time for {SEQ_LEN} tokens (1 layer): {elapsed:.3f} ms")
    print(f"Per-token time: {elapsed / SEQ_LEN:.4f} ms")
    print(f"")
    print(f"Note: Full sequence is processed token-by-token (GEMV, not GEMM)")
    print(f"For batched inference, a GEMM kernel would be more efficient.")

    return elapsed


def main():
    print("="*60)
    print("W4A16 TVM MLP Performance Benchmark")
    print("="*60)
    print(f"Model: Pi0.5 / PaliGemma")
    print(f"Hidden Size: {HIDDEN_SIZE}")
    print(f"MLP Dim: {MLP_DIM}")
    print(f"Num Layers: {NUM_LAYERS}")
    print(f"Seq Len: {SEQ_LEN}")

    # Single token benchmark (what TRT uses for static graph)
    print("\n" + "="*60)
    print("Test 1: Single Token MLP (18 Layers)")
    print("="*60)
    single_token_time = benchmark_single_token_mlp()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"W4A16 TVM (18 layers, single token): {single_token_time:.2f} ms")
    print(f"TRT FP8 baseline:                    12.39 ms")
    print(f"")
    if single_token_time > 12.39:
        overhead = single_token_time - 12.39
        print(f"Current overhead: {overhead:.2f} ms (Python call overhead)")
        print(f"C++ Plugin expected: ~{single_token_time - 1.7:.2f} ms (eliminating Python overhead)")
    else:
        print(f"W4A16 TVM is {12.39 - single_token_time:.2f} ms faster than TRT FP8!")


if __name__ == "__main__":
    main()
