#!/usr/bin/env python3
"""
W4A16 Fused Dequant + GEMM using Triton

Key Strategy:
1. Fuse INT4 dequantization with GEMM computation
2. Dequantize in registers/shared memory -> FP16
3. Use Triton's tl.dot which maps to Tensor Cores

This should achieve close to cuBLAS FP16 performance while saving memory.

Author: Claude Code
Date: 2026-02-11
"""

import torch
import triton
import triton.language as tl
import numpy as np
from typing import Tuple


# ==============================================================================
# Quantization Utilities
# ==============================================================================

def quantize_to_int4_packed(weight: torch.Tensor, block_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight to packed INT4 format.

    Args:
        weight: [N, K] float weight matrix
        block_size: number of elements per scale block

    Returns:
        packed: [N, K//2] uint8, packed INT4 values
        scales: [N, num_blocks] float16, scale factors
    """
    N, K = weight.shape
    num_blocks = (K + block_size - 1) // block_size

    # Pad K to multiple of block_size
    K_padded = num_blocks * block_size
    if K < K_padded:
        weight = torch.nn.functional.pad(weight, (0, K_padded - K))

    # Reshape for block processing
    weight_blocked = weight.view(N, num_blocks, block_size)

    # Compute scales (symmetric quantization, range -7 to 7)
    max_abs = weight_blocked.abs().max(dim=-1).values
    scales = max_abs / 7.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    scales = scales.to(torch.float16)

    # Quantize (with zero point at 8)
    weight_scaled = weight_blocked / scales.unsqueeze(-1)
    quantized = torch.clamp(torch.round(weight_scaled + 8), 0, 15).to(torch.uint8)

    # Flatten back
    quantized = quantized.view(N, K_padded)

    # Pack pairs of INT4 into bytes
    packed = (quantized[:, 1::2] << 4) | quantized[:, ::2]

    return packed[:, :K//2].contiguous(), scales


def dequantize_int4_packed(packed: torch.Tensor, scales: torch.Tensor,
                           K: int, block_size: int = 32) -> torch.Tensor:
    """Dequantize packed INT4 to float16."""
    N = packed.shape[0]
    num_blocks = scales.shape[1]

    # Unpack
    low = (packed & 0xF).to(torch.int8) - 8
    high = ((packed >> 4) & 0xF).to(torch.int8) - 8

    # Interleave
    unpacked = torch.zeros(N, K, dtype=torch.float16, device=packed.device)
    unpacked[:, ::2] = low[:, :K//2].to(torch.float16)
    unpacked[:, 1::2] = high[:, :K//2].to(torch.float16)

    # Apply scales
    unpacked = unpacked.view(N, num_blocks, block_size)
    unpacked = unpacked * scales.unsqueeze(-1)
    unpacked = unpacked.view(N, num_blocks * block_size)[:, :K]

    return unpacked


# ==============================================================================
# Triton W4A16 Fused Dequant + GEMM
# ==============================================================================

@triton.jit
def w4a16_gemm_kernel(
    # Pointers
    a_ptr, w_packed_ptr, scales_ptr, c_ptr,
    # Strides
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_sn, stride_sb,
    stride_cm, stride_cn,
    # Dimensions
    M, N, K,
    # Block size for quantization
    BLOCK_SIZE_QUANT: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Grouping
    GROUP_M: tl.constexpr,
):
    """
    W4A16 Fused Dequant + GEMM Kernel

    C[M, N] = A[M, K] @ W_dequant[N, K]^T

    where W_dequant = dequant(W_packed, scales)

    Uses Triton's tl.dot for Tensor Core acceleration.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Base pointers for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_packed_ptr + offs_n[:, None] * stride_wn  # Will compute k offset in loop

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Number of K blocks for quantization
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_QUANT)

    # Main loop over K
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k

        # Load A tile
        a_mask = (offs_m[:, None] < M) & (k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        # Load and dequantize W tile
        # W is packed: 2 INT4 per byte
        # k // 2 gives byte offset
        k_byte = k // 2
        is_high = (k % 2).to(tl.int32)

        w_byte_ptrs = w_packed_ptr + offs_n[:, None] * stride_wn + k_byte[None, :]
        w_mask = (offs_n[:, None] < N) & (k_byte[None, :] < K // 2)
        w_packed = tl.load(w_byte_ptrs, mask=w_mask, other=0).to(tl.uint8)

        # Extract INT4 values
        # is_high = 0: low nibble, is_high = 1: high nibble
        int4_val = tl.where(
            is_high[None, :] == 0,
            w_packed & 0xF,
            (w_packed >> 4) & 0xF
        )

        # Convert to signed (zero point at 8)
        signed_val = int4_val.to(tl.float16) - 8.0

        # Get scales
        k_block_idx = k // BLOCK_SIZE_QUANT
        scale_ptrs = scales_ptr + offs_n[:, None] * stride_sn + k_block_idx[None, :] * stride_sb
        scale_mask = (offs_n[:, None] < N) & (k_block_idx[None, :] < num_k_blocks)
        scale = tl.load(scale_ptrs, mask=scale_mask, other=1.0).to(tl.float16)

        # Dequantize
        w = signed_val * scale

        # Compute: C += A @ W^T
        # A: [BLOCK_M, BLOCK_K], W: [BLOCK_N, BLOCK_K]
        # We need A @ W^T = A @ W.T
        acc += tl.dot(a, tl.trans(w))

        # Advance A pointer
        a_ptrs += BLOCK_K * stride_ak

    # Store result
    c = acc.to(tl.float32)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w4a16_gemm_triton(
    A: torch.Tensor,  # [M, K] float16
    W_packed: torch.Tensor,  # [N, K//2] uint8
    scales: torch.Tensor,  # [N, num_blocks] float16
    block_size_quant: int = 32,
) -> torch.Tensor:
    """
    W4A16 GEMM using Triton.

    C[M, N] = A[M, K] @ W_dequant[N, K]^T
    """
    M, K = A.shape
    N = W_packed.shape[0]

    assert A.dtype == torch.float16
    assert W_packed.dtype == torch.uint8
    assert scales.dtype == torch.float16

    # Output
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Block sizes for Tensor Core (must be multiples of 16 for MMA)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    w4a16_gemm_kernel[grid](
        A, W_packed, scales, C,
        A.stride(0), A.stride(1),
        W_packed.stride(0), W_packed.stride(1),
        scales.stride(0), scales.stride(1),
        C.stride(0), C.stride(1),
        M, N, K,
        BLOCK_SIZE_QUANT=block_size_quant,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )

    return C


# ==============================================================================
# Alternative: Simple Dequant + cuBLAS (for comparison)
# ==============================================================================

def w4a16_gemm_dequant_cublas(
    A: torch.Tensor,  # [M, K] float16
    W_packed: torch.Tensor,  # [N, K//2] uint8
    scales: torch.Tensor,  # [N, num_blocks] float16
    block_size: int = 32,
) -> torch.Tensor:
    """
    W4A16 GEMM using explicit dequant + cuBLAS.

    This is the fallback approach.
    """
    N = W_packed.shape[0]
    K = A.shape[1]

    # Dequantize
    W = dequantize_int4_packed(W_packed, scales, K, block_size)

    # GEMM
    return torch.mm(A, W.T).float()


# ==============================================================================
# Benchmark
# ==============================================================================

def benchmark_w4a16(M=712, N=16384, K=2048, warmup=50, runs=200):
    """Benchmark W4A16 implementations."""
    import time

    print(f"\n{'='*70}")
    print(f"W4A16 Triton GEMM Benchmark")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*70}")

    device = torch.device("cuda")
    torch.manual_seed(42)

    # Generate test data
    A = torch.randn(M, K, dtype=torch.float16, device=device)
    W = torch.randn(N, K, dtype=torch.float32, device=device)

    # Quantize
    print("Quantizing weights...")
    W_packed, scales = quantize_to_int4_packed(W)
    W_packed = W_packed.to(device)
    scales = scales.to(device)

    print(f"  Original: {W.numel() * 4 / 1e6:.2f} MB")
    print(f"  Packed:   {(W_packed.numel() + scales.numel() * 2) / 1e6:.2f} MB")

    # Reference
    W_dequant = dequantize_int4_packed(W_packed, scales, K)
    C_ref = torch.mm(A, W_dequant.T).float()

    results = []

    # Test Triton kernel
    print("\n--- Triton W4A16 GEMM ---")
    try:
        # Warmup
        for _ in range(warmup):
            C = w4a16_gemm_triton(A, W_packed, scales)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(runs):
            C = w4a16_gemm_triton(A, W_packed, scales)
        torch.cuda.synchronize()

        avg_ms = (time.time() - start) / runs * 1000

        # Verify
        max_diff = (C - C_ref).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            C.flatten().unsqueeze(0),
            C_ref.flatten().unsqueeze(0)
        ).item()

        flops = 2.0 * M * N * K
        tflops = flops / (avg_ms / 1000) / 1e12

        print(f"  Time:    {avg_ms:.4f} ms")
        print(f"  TFLOPS:  {tflops:.4f}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Cos sim:  {cos_sim:.6f}")

        results.append(("Triton W4A16", avg_ms, tflops, cos_sim > 0.99))
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Test dequant + cuBLAS
    print("\n--- Dequant + cuBLAS ---")
    try:
        for _ in range(warmup):
            C = w4a16_gemm_dequant_cublas(A, W_packed, scales)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(runs):
            C = w4a16_gemm_dequant_cublas(A, W_packed, scales)
        torch.cuda.synchronize()

        avg_ms = (time.time() - start) / runs * 1000

        max_diff = (C - C_ref).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            C.flatten().unsqueeze(0),
            C_ref.flatten().unsqueeze(0)
        ).item()

        flops = 2.0 * M * N * K
        tflops = flops / (avg_ms / 1000) / 1e12

        print(f"  Time:    {avg_ms:.4f} ms")
        print(f"  TFLOPS:  {tflops:.4f}")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Cos sim:  {cos_sim:.6f}")

        results.append(("Dequant+cuBLAS", avg_ms, tflops, cos_sim > 0.99))
    except Exception as e:
        print(f"  Error: {e}")

    # cuBLAS BF16 baseline
    print("\n--- cuBLAS BF16 Baseline ---")
    W_bf16 = W_dequant.to(torch.bfloat16)
    A_bf16 = A.to(torch.bfloat16)

    for _ in range(warmup):
        C = torch.mm(A_bf16, W_bf16.T)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        C = torch.mm(A_bf16, W_bf16.T)
    torch.cuda.synchronize()

    avg_ms = (time.time() - start) / runs * 1000
    tflops = flops / (avg_ms / 1000) / 1e12

    print(f"  Time:    {avg_ms:.4f} ms")
    print(f"  TFLOPS:  {tflops:.4f}")

    results.append(("cuBLAS BF16", avg_ms, tflops, True))

    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    bf16_ms = results[-1][1] if results else 1.0

    for name, time_ms, tflops, correct in results:
        status = "✅" if correct else "❌"
        speedup = bf16_ms / time_ms
        print(f"  {name:<20} {time_ms:.4f}ms  {tflops:.2f} TFLOPS  vs BF16: {speedup:.2f}x  {status}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=712)
    parser.add_argument("--N", type=int, default=16384)
    parser.add_argument("--K", type=int, default=2048)

    args = parser.parse_args()

    benchmark_w4a16(args.M, args.N, args.K)
