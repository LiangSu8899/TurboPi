#!/usr/bin/env python3
"""
Test NVFP4 Persistent MLP Extension.

This script tests the compiled PyTorch C++ extension.

Usage:
    python test_persistent_extension.py

Author: Claude Code
Date: 2026-02-10
"""

import torch
import torch.nn.functional as F
import time
import sys

# ============================================================================
# Constants
# ============================================================================

HIDDEN_SIZE = 2048
MLP_DIM = 16384
BLOCK_SIZE = 32

NVFP4_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
NVFP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ============================================================================
# NVFP4 Quantization
# ============================================================================

def quantize_to_nvfp4(weight: torch.Tensor, block_size: int = 32):
    """Quantize FP32 weight to NVFP4 format."""
    N, K = weight.shape
    device = weight.device
    weight = weight.to(torch.float32)

    num_blocks = K // block_size
    weight_blocked = weight.view(N, num_blocks, block_size)

    scales = weight_blocked.abs().amax(dim=-1) / 6.0
    scales = scales.clamp(min=1e-8)

    weight_normalized = weight_blocked / scales.unsqueeze(-1)

    nvfp4_magnitudes = torch.tensor(NVFP4_MAGNITUDES, device=device, dtype=torch.float32)
    signs = (weight_normalized < 0).to(torch.uint8) * 8
    abs_vals = weight_normalized.abs()

    diffs = (abs_vals.unsqueeze(-1) - nvfp4_magnitudes).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)

    fp4_vals = (signs + indices).view(N, K)

    low = fp4_vals[:, 0::2]
    high = fp4_vals[:, 1::2]
    packed = (high << 4) | low

    return packed.to(torch.uint8), scales.to(torch.float32)


def dequantize_from_nvfp4(packed: torch.Tensor, scales: torch.Tensor, block_size: int = 32):
    """Dequantize NVFP4 back to FP32."""
    N = packed.shape[0]
    K = packed.shape[1] * 2
    device = packed.device

    full_lut = torch.tensor(NVFP4_LUT, device=device, dtype=torch.float32)

    low = packed & 0xF
    high = (packed >> 4) & 0xF

    fp4_vals = torch.zeros(N, K, dtype=torch.uint8, device=device)
    fp4_vals[:, 0::2] = low
    fp4_vals[:, 1::2] = high

    decoded = full_lut[fp4_vals.to(torch.int64)]

    num_blocks = K // block_size
    decoded_blocked = decoded.view(N, num_blocks, block_size)
    weight = (decoded_blocked * scales.unsqueeze(-1)).view(N, K)

    return weight


# ============================================================================
# Reference Implementation
# ============================================================================

def reference_mlp(x, layers_data):
    """Reference MLP using PyTorch dequantization."""
    activation = x.to(torch.float32)

    for ld in layers_data:
        gate = dequantize_from_nvfp4(ld['gate_packed'], ld['gate_scale'])
        up = dequantize_from_nvfp4(ld['up_packed'], ld['up_scale'])
        down = dequantize_from_nvfp4(ld['down_packed'], ld['down_scale'])

        g = F.linear(activation, gate)
        u = F.linear(activation, up)
        inter = F.silu(g) * u
        activation = F.linear(inter, down)

    return activation


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_bf16_mlp(num_layers: int, warmup: int = 50, runs: int = 200):
    """Benchmark BF16 cuBLAS MLP."""
    device = torch.device('cuda')

    layers = []
    for _ in range(num_layers):
        gate = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        up = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        down = torch.randn(HIDDEN_SIZE, MLP_DIM, device=device, dtype=torch.bfloat16)
        layers.append((gate, up, down))

    x = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    def forward(x):
        activation = x
        for gate, up, down in layers:
            g = F.linear(activation, gate)
            u = F.linear(activation, up)
            inter = F.silu(g) * u
            activation = F.linear(inter, down)
        return activation

    for _ in range(warmup):
        _ = forward(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        _ = forward(x)
    torch.cuda.synchronize()

    return (time.time() - start) / runs * 1000


def benchmark_persistent_kernel(nvfp4_ext, num_layers: int, warmup: int = 50, runs: int = 200):
    """Benchmark persistent kernel."""
    device = torch.device('cuda')

    # Create and quantize weights
    gate_packed_list = []
    gate_scale_list = []
    up_packed_list = []
    up_scale_list = []
    down_packed_list = []
    down_scale_list = []

    for _ in range(num_layers):
        gate = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32)
        up = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32)
        down = torch.randn(HIDDEN_SIZE, MLP_DIM, device=device, dtype=torch.float32)

        gate_p, gate_s = quantize_to_nvfp4(gate)
        up_p, up_s = quantize_to_nvfp4(up)
        down_p, down_s = quantize_to_nvfp4(down)

        gate_packed_list.append(gate_p)
        gate_scale_list.append(gate_s)
        up_packed_list.append(up_p)
        up_scale_list.append(up_s)
        down_packed_list.append(down_p)
        down_scale_list.append(down_s)

    x = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        _ = nvfp4_ext.forward(
            x,
            gate_packed_list, gate_scale_list,
            up_packed_list, up_scale_list,
            down_packed_list, down_scale_list,
            num_layers
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(runs):
        _ = nvfp4_ext.forward(
            x,
            gate_packed_list, gate_scale_list,
            up_packed_list, up_scale_list,
            down_packed_list, down_scale_list,
            num_layers
        )
    torch.cuda.synchronize()

    return (time.time() - start) / runs * 1000


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("NVFP4 Persistent MLP Extension Test")
    print("=" * 70)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Try to import extension
    try:
        import nvfp4_persistent
        print("\n[OK] Extension loaded successfully")
        print(f"Shared memory size: {nvfp4_persistent.get_smem_size()} bytes")
        print("\nKernel info:")
        nvfp4_persistent.print_kernel_info()
    except ImportError as e:
        print(f"\n[ERROR] Failed to import extension: {e}")
        print("Make sure to run: python setup_persistent.py build_ext --inplace")
        sys.exit(1)

    # ========================================================================
    # Test correctness (simple sanity check)
    # ========================================================================

    print("\n" + "=" * 70)
    print("Correctness Test")
    print("=" * 70)

    device = torch.device('cuda')
    num_layers = 6

    # Create test data
    gate_packed_list = []
    gate_scale_list = []
    up_packed_list = []
    up_scale_list = []
    down_packed_list = []
    down_scale_list = []
    layers_data = []

    for _ in range(num_layers):
        gate = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32)
        up = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32)
        down = torch.randn(HIDDEN_SIZE, MLP_DIM, device=device, dtype=torch.float32)

        gate_p, gate_s = quantize_to_nvfp4(gate)
        up_p, up_s = quantize_to_nvfp4(up)
        down_p, down_s = quantize_to_nvfp4(down)

        gate_packed_list.append(gate_p)
        gate_scale_list.append(gate_s)
        up_packed_list.append(up_p)
        up_scale_list.append(up_s)
        down_packed_list.append(down_p)
        down_scale_list.append(down_s)

        layers_data.append({
            'gate_packed': gate_p,
            'gate_scale': gate_s,
            'up_packed': up_p,
            'up_scale': up_s,
            'down_packed': down_p,
            'down_scale': down_s,
        })

    x = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.float32)

    # Run extension
    ext_output = nvfp4_persistent.forward(
        x,
        gate_packed_list, gate_scale_list,
        up_packed_list, up_scale_list,
        down_packed_list, down_scale_list,
        num_layers
    )
    torch.cuda.synchronize()

    # Run reference
    ref_output = reference_mlp(x, layers_data)
    torch.cuda.synchronize()

    # Compare
    diff = (ext_output - ref_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {ext_output.shape}")
    print(f"Max diff: {max_diff:.6e}")
    print(f"Mean diff: {mean_diff:.6e}")

    if max_diff < 1.0:
        print("[OK] Correctness test passed (diff < 1.0)")
    else:
        print("[WARN] Large difference detected - kernel may have issues")

    # ========================================================================
    # Performance Benchmark
    # ========================================================================

    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)

    trt_fp8_baseline = 20.39  # From previous measurements

    results = {}

    for n_layers in [4, 6, 8, 18]:
        print(f"\n--- {n_layers} layers ---")

        # BF16 baseline
        bf16_time = benchmark_bf16_mlp(n_layers)
        print(f"  BF16 cuBLAS: {bf16_time:.2f} ms")

        # Persistent kernel
        try:
            persistent_time = benchmark_persistent_kernel(nvfp4_persistent, n_layers)
            print(f"  NVFP4 Persistent: {persistent_time:.2f} ms")
            speedup = bf16_time / persistent_time
            print(f"  Speedup vs BF16: {speedup:.2f}x")
        except Exception as e:
            print(f"  NVFP4 Persistent: ERROR - {e}")
            persistent_time = None

        results[n_layers] = {
            'bf16': bf16_time,
            'persistent': persistent_time,
        }

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY - 18 Layers")
    print("=" * 70)

    if 18 in results and results[18]['persistent'] is not None:
        bf16_18 = results[18]['bf16']
        persistent_18 = results[18]['persistent']

        # Theoretical (from memory traffic analysis)
        thor_bandwidth = 122.8  # GB/s
        fp4_traffic_gb = 1.13  # GB for 18 layers
        theoretical_time = fp4_traffic_gb / thor_bandwidth * 1000

        print(f"\n{'Approach':<40} {'Time (ms)':<12} {'vs TRT FP8':<12}")
        print("-" * 65)
        print(f"{'TRT FP8 Baseline':<40} {trt_fp8_baseline:<12.2f} {'1.00x':<12}")
        print(f"{'BF16 cuBLAS':<40} {bf16_18:<12.2f} {trt_fp8_baseline/bf16_18:.2f}x")
        print(f"{'NVFP4 Persistent (actual)':<40} {persistent_18:<12.2f} {trt_fp8_baseline/persistent_18:.2f}x")
        print(f"{'NVFP4 Persistent (theoretical)':<40} {theoretical_time:<12.2f} {trt_fp8_baseline/theoretical_time:.2f}x")

        print(f"\n{'Speedup vs BF16:':<40} {bf16_18/persistent_18:.2f}x")
        print(f"{'Theoretical speedup:':<40} {bf16_18/theoretical_time:.2f}x")
        print(f"{'Efficiency:':<40} {theoretical_time/persistent_18*100:.1f}%")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
