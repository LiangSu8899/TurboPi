#!/usr/bin/env python3
"""
Debug Script: Verify W4A16 Packer and Kernel Bit-Shifting Logic

This script:
1. Creates a deterministic small weight matrix
2. Manually calculates expected uint32 values
3. Compares with packer output
4. Verifies single-layer accuracy (target > 0.99)
5. Benchmarks wrapper overhead

Author: Claude Code
Date: 2026-02-11
"""

import sys
import os
import time
import numpy as np

_test_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_test_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from openpi.utils.w4a16_packer import W4A16Packer, W4A16PackerFast, QUANT_BLOCK, INT4_OFFSET
from openpi.ops.w4a16_gemv import w4a16_gemv, precompile_kernels
from openpi.modules.w4a16_linear import W4A16Linear


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_bit_shifting_logic():
    """Test 1: Verify bit-shifting logic with deterministic values."""
    print_section("Test 1: Bit-Shifting Logic Verification")

    # Create a simple 4x32 weight (4 output features, 32 input = 1 quant block)
    N, K = 4, 32

    # Create deterministic weight: each element is its index mod 15 - 7 (range [-7, 7])
    weight_np = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            # Values in range [-7, 7]
            weight_np[n, k] = ((n * K + k) % 15) - 7

    print(f"Weight shape: {weight_np.shape}")
    print(f"Weight range: [{weight_np.min()}, {weight_np.max()}]")
    print(f"First row: {weight_np[0, :8]}")

    # Pack manually
    packer = W4A16Packer()
    weight_torch = torch.from_numpy(weight_np).to(torch.float32)
    packed = packer.pack(weight_torch)

    # Manual calculation for first element
    # For n=0, qb=0:
    # - scale = max(|weight[0, :]|) / 7
    # - quantized[k] = round(weight[0, k] / scale) + 8, clamped to [0, 15]

    block_values = weight_np[0, :]  # First row, first block
    scale_manual = np.max(np.abs(block_values)) / 7
    print(f"\nManual calculation for n=0, qb=0:")
    print(f"  Max abs: {np.max(np.abs(block_values))}")
    print(f"  Scale: {scale_manual}")

    # Check packer's scale
    packer_scale = packed.scales[0, 0].item()
    print(f"  Packer scale: {packer_scale}")

    # Quantize first 8 values (first uint32)
    quantized_manual = []
    for i in range(8):
        q = int(np.round(block_values[i] / scale_manual) + 8)
        q = max(0, min(15, q))
        quantized_manual.append(q)
    print(f"  Quantized [0:8]: {quantized_manual}")

    # Calculate expected uint32
    expected_uint32 = 0
    for i in range(8):
        expected_uint32 |= (quantized_manual[i] << (i * 4))
    print(f"  Expected uint32[0]: 0x{expected_uint32:08X} ({expected_uint32})")

    # Check packer's uint32
    packer_uint32 = packed.weight_packed[0, 0, 0].item()
    # Convert signed int32 to unsigned for display
    if packer_uint32 < 0:
        packer_uint32_unsigned = packer_uint32 + 2**32
    else:
        packer_uint32_unsigned = packer_uint32
    print(f"  Packer uint32[0]:  0x{packer_uint32_unsigned:08X} ({packer_uint32})")

    # Verify
    match = (expected_uint32 == packer_uint32_unsigned) or (expected_uint32 == packer_uint32)
    print(f"  Match: {'PASS' if match else 'FAIL'}")

    # Unpack and verify round-trip
    unpacked = packer.unpack(packed)

    # Compare
    cos_sim = F.cosine_similarity(
        weight_torch.flatten().unsqueeze(0),
        unpacked.flatten().float().unsqueeze(0)
    ).item()
    print(f"\nRound-trip verification:")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Status: {'PASS' if cos_sim > 0.99 else 'FAIL'}")

    return cos_sim > 0.99


def test_single_layer_accuracy():
    """Test 2: Verify single-layer accuracy with GPU kernel."""
    print_section("Test 2: Single-Layer Accuracy (GPU Kernel)")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return True

    device = 'cuda'
    N, K = 16384, 2048

    print(f"Testing W4A16Linear({K} -> {N})")

    # Create reference Linear layer (FP16 to match W4A16)
    linear_ref = nn.Linear(K, N, bias=False).to(device).to(torch.float16)
    nn.init.normal_(linear_ref.weight, std=0.02)  # Small std for stability

    # Create W4A16Linear from reference
    w4a16_layer = W4A16Linear.from_linear(linear_ref)

    # Test with random input
    num_samples = 10
    cos_sims = []

    for _ in range(num_samples):
        x = torch.randn(1, K, dtype=torch.float16, device=device)

        with torch.no_grad():
            # Reference output (FP16 matmul)
            ref_out = F.linear(x, linear_ref.weight)

            # W4A16 output
            w4a16_out = w4a16_layer(x)

        cos_sim = F.cosine_similarity(
            ref_out.flatten().float().unsqueeze(0),
            w4a16_out.flatten().float().unsqueeze(0)
        ).item()
        cos_sims.append(cos_sim)

    mean_cos = sum(cos_sims) / len(cos_sims)
    min_cos = min(cos_sims)

    print(f"  Samples: {num_samples}")
    print(f"  Mean cosine similarity: {mean_cos:.6f}")
    print(f"  Min cosine similarity:  {min_cos:.6f}")
    print(f"  Status: {'PASS' if mean_cos > 0.99 else 'FAIL'}")

    if mean_cos < 0.99:
        print("\n  DIAGNOSING LOW ACCURACY:")

        # Check if packer round-trip is accurate
        packer = W4A16Packer()
        packed = packer.pack(linear_ref.weight.data.float().cpu())
        unpacked = packer.unpack(packed)

        packer_cos = F.cosine_similarity(
            linear_ref.weight.data.float().cpu().flatten().unsqueeze(0),
            unpacked.flatten().float().unsqueeze(0)
        ).item()
        print(f"  Packer round-trip cos_sim: {packer_cos:.6f}")

        # Check kernel directly
        x_test = torch.randn(1, K, dtype=torch.float16, device=device)

        # Manual dequantize and matmul
        w_dequant = w4a16_layer._dequantize_weights()
        manual_out = F.linear(x_test, w_dequant)

        # Kernel output
        kernel_out = w4a16_gemv(x_test, w4a16_layer.weight_packed, w4a16_layer.scales)

        kernel_cos = F.cosine_similarity(
            manual_out.flatten().float().unsqueeze(0),
            kernel_out.flatten().float().unsqueeze(0)
        ).item()
        print(f"  Kernel vs dequant cos_sim: {kernel_cos:.6f}")

        if kernel_cos < 0.99:
            print("  --> KERNEL BUG DETECTED!")
        elif packer_cos < 0.99:
            print("  --> PACKER BUG DETECTED!")

    return mean_cos > 0.99


def test_wrapper_overhead():
    """Test 3: Benchmark wrapper overhead."""
    print_section("Test 3: Wrapper Overhead Analysis")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    device = 'cuda'
    N, K = 16384, 2048
    num_scale_blocks = K // QUANT_BLOCK
    warmup, runs = 50, 200

    # Create test data
    x = torch.randn(1, K, dtype=torch.float16, device=device)
    weight_packed = torch.randint(0, 2**31 - 1, (num_scale_blocks, N, 4),
                                   dtype=torch.int32, device=device)
    scales = torch.randn(num_scale_blocks, N, dtype=torch.float16, device=device).abs() + 0.1

    # Test 1: Raw kernel (via w4a16_gemv function)
    print("\n1. Raw w4a16_gemv() function:")

    # Warmup
    for _ in range(warmup):
        _ = w4a16_gemv(x, weight_packed, scales)
    torch.cuda.synchronize()

    # Benchmark with CUDA Events (accurate GPU timing)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(runs):
        _ = w4a16_gemv(x, weight_packed, scales)
    end_event.record()
    torch.cuda.synchronize()

    raw_kernel_ms = start_event.elapsed_time(end_event) / runs
    print(f"   Latency: {raw_kernel_ms:.4f} ms")

    # Test 2: W4A16Linear forward
    print("\n2. W4A16Linear.forward():")

    linear_ref = nn.Linear(K, N, bias=False).cuda().half()
    w4a16_layer = W4A16Linear.from_linear(linear_ref)

    # Warmup
    for _ in range(warmup):
        _ = w4a16_layer(x)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(runs):
        _ = w4a16_layer(x)
    end_event.record()
    torch.cuda.synchronize()

    w4a16linear_ms = start_event.elapsed_time(end_event) / runs
    print(f"   Latency: {w4a16linear_ms:.4f} ms")

    # Test 3: F.linear baseline
    print("\n3. F.linear (BF16) baseline:")

    linear_fp16 = nn.Linear(K, N, bias=False).cuda().half()
    x_half = x.half()

    for _ in range(warmup):
        _ = linear_fp16(x_half)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(runs):
        _ = linear_fp16(x_half)
    end_event.record()
    torch.cuda.synchronize()

    flinear_ms = start_event.elapsed_time(end_event) / runs
    print(f"   Latency: {flinear_ms:.4f} ms")

    # Analysis
    print("\n" + "-" * 50)
    print("ANALYSIS:")
    overhead = w4a16linear_ms - raw_kernel_ms
    print(f"  Raw kernel:     {raw_kernel_ms:.4f} ms")
    print(f"  W4A16Linear:    {w4a16linear_ms:.4f} ms")
    print(f"  Wrapper overhead: {overhead:.4f} ms ({overhead/raw_kernel_ms*100:.1f}%)")
    print(f"  F.linear BF16:  {flinear_ms:.4f} ms")
    print(f"  Speedup vs F.linear: {flinear_ms/w4a16linear_ms:.2f}x")

    if overhead > 0.1:
        print("\n  WARNING: High wrapper overhead detected!")
        print("  Possible causes:")
        print("    - dtype conversion in forward()")
        print("    - non-contiguous tensor handling")
        print("    - output buffer allocation")


def test_kernel_vs_manual_dequant():
    """Test 4: Compare kernel output with manual dequantization."""
    print_section("Test 4: Kernel vs Manual Dequantization")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return True

    device = 'cuda'
    N, K = 256, 64  # Small for debugging

    # Create known weight
    weight = torch.randn(N, K, dtype=torch.float32) * 0.1

    # Pack
    packer = W4A16Packer()
    packed = packer.pack(weight)

    # Move to GPU
    weight_packed = packed.weight_packed.to(device)
    scales = packed.scales.to(device)

    # Random input
    x = torch.randn(1, K, dtype=torch.float16, device=device)

    # Method 1: Manual dequantize then matmul
    weight_dequant = packer.unpack(packed).to(device).half()
    manual_out = F.linear(x, weight_dequant)

    # Method 2: Kernel
    precompile_kernels([(N, K)])
    kernel_out = w4a16_gemv(x, weight_packed, scales)

    # Compare
    cos_sim = F.cosine_similarity(
        manual_out.flatten().float().unsqueeze(0),
        kernel_out.flatten().float().unsqueeze(0)
    ).item()

    mse = F.mse_loss(manual_out.float(), kernel_out.float()).item()
    max_diff = (manual_out - kernel_out).abs().max().item()

    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  MSE: {mse:.8f}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Status: {'PASS' if cos_sim > 0.999 else 'FAIL'}")

    if cos_sim < 0.999:
        # Debug: print first few values
        print("\n  DEBUG - First 10 values:")
        print(f"  Manual: {manual_out[0, :10].tolist()}")
        print(f"  Kernel: {kernel_out[0, :10].tolist()}")

        # Check if scales match
        print(f"\n  Scales shape: {scales.shape}")
        print(f"  Scales [0, :5]: {scales[0, :5].tolist()}")

    return cos_sim > 0.999


def main():
    print("=" * 70)
    print("W4A16 Accuracy & Performance Debug Suite")
    print("=" * 70)

    results = []

    # Test 1: Bit-shifting
    try:
        pass1 = test_bit_shifting_logic()
        results.append(("Bit-shifting logic", pass1))
    except Exception as e:
        print(f"Test 1 FAILED with exception: {e}")
        results.append(("Bit-shifting logic", False))

    # Test 2: Single-layer accuracy
    try:
        pass2 = test_single_layer_accuracy()
        results.append(("Single-layer accuracy", pass2))
    except Exception as e:
        print(f"Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Single-layer accuracy", False))

    # Test 3: Wrapper overhead
    try:
        test_wrapper_overhead()
        results.append(("Wrapper overhead", True))  # Informational
    except Exception as e:
        print(f"Test 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Wrapper overhead", False))

    # Test 4: Kernel vs manual
    try:
        pass4 = test_kernel_vs_manual_dequant()
        results.append(("Kernel vs dequant", pass4))
    except Exception as e:
        print(f"Test 4 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Kernel vs dequant", False))

    # Summary
    print_section("SUMMARY")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
