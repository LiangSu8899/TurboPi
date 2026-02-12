#!/usr/bin/env python3
"""
W4A16 Optimized Triton GEMM v2

Key Optimizations:
1. Pre-dequantize weight tile in shared memory
2. Use proper tl.dot for Tensor Core (16x16 tiles)
3. Better memory coalescing

Author: Claude Code
Date: 2026-02-11
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


# ==============================================================================
# Quantization Utilities
# ==============================================================================

def quantize_to_int4_packed(weight: torch.Tensor, block_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight to packed INT4 format."""
    N, K = weight.shape
    num_blocks = (K + block_size - 1) // block_size
    K_padded = num_blocks * block_size

    if K < K_padded:
        weight = torch.nn.functional.pad(weight, (0, K_padded - K))

    weight_blocked = weight.view(N, num_blocks, block_size)
    max_abs = weight_blocked.abs().max(dim=-1).values
    scales = max_abs / 7.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    scales = scales.to(torch.float16)

    weight_scaled = weight_blocked / scales.unsqueeze(-1)
    quantized = torch.clamp(torch.round(weight_scaled + 8), 0, 15).to(torch.uint8)
    quantized = quantized.view(N, K_padded)

    packed = (quantized[:, 1::2] << 4) | quantized[:, ::2]

    return packed[:, :K//2].contiguous(), scales


def dequantize_int4_packed(packed: torch.Tensor, scales: torch.Tensor,
                           K: int, block_size: int = 32) -> torch.Tensor:
    """Dequantize packed INT4 to float16."""
    N = packed.shape[0]
    num_blocks = scales.shape[1]

    low = (packed & 0xF).to(torch.int8) - 8
    high = ((packed >> 4) & 0xF).to(torch.int8) - 8

    unpacked = torch.zeros(N, K, dtype=torch.float16, device=packed.device)
    unpacked[:, ::2] = low[:, :K//2].to(torch.float16)
    unpacked[:, 1::2] = high[:, :K//2].to(torch.float16)

    unpacked = unpacked.view(N, num_blocks, block_size)
    unpacked = unpacked * scales.unsqueeze(-1)
    unpacked = unpacked.view(N, num_blocks * block_size)[:, :K]

    return unpacked


# ==============================================================================
# Triton W4A16 Optimized Kernel v2
# ==============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def w4a16_gemm_kernel_v2(
    a_ptr, w_dequant_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Standard FP16 GEMM kernel using pre-dequantized weights.

    C[M, N] = A[M, K] @ W[N, K]^T

    Uses Triton's autotune to find optimal config.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = w_dequant_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, tl.trans(b))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_wk

    c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w4a16_gemm_v2(
    A: torch.Tensor,  # [M, K] float16
    W_packed: torch.Tensor,  # [N, K//2] uint8
    scales: torch.Tensor,  # [N, num_blocks] float16
    block_size: int = 32,
) -> torch.Tensor:
    """
    W4A16 GEMM: Dequant first, then use optimized Triton GEMM.
    """
    M, K = A.shape
    N = W_packed.shape[0]

    # Dequantize weight
    W = dequantize_int4_packed(W_packed, scales, K, block_size)

    # Output
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    w4a16_gemm_kernel_v2[grid](
        A, W, C,
        M, N, K,
        A.stride(0), A.stride(1),
        W.stride(0), W.stride(1),
        C.stride(0), C.stride(1),
    )

    return C


# ==============================================================================
# Fused Dequant + GEMM (Alternative approach)
# ==============================================================================

@triton.jit
def w4a16_fused_kernel(
    a_ptr, w_packed_ptr, scales_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_sn, stride_sb,
    stride_cm, stride_cn,
    BLOCK_SIZE_QUANT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W4A16 kernel with in-register dequantization.

    Key optimization: Dequantize weights in shared memory before tl.dot.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_QUANT)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        # Load and dequant W
        k_byte = offs_k // 2
        is_high = offs_k % 2

        w_byte_ptrs = w_packed_ptr + offs_n[:, None] * stride_wn + k_byte[None, :]
        w_mask = (offs_n[:, None] < N) & (k_byte[None, :] < K // 2)
        w_packed = tl.load(w_byte_ptrs, mask=w_mask, other=0).to(tl.uint8)

        int4_val = tl.where(is_high[None, :] == 0, w_packed & 0xF, (w_packed >> 4) & 0xF)
        signed_val = int4_val.to(tl.float16) - 8.0

        # Scale
        k_block = offs_k // BLOCK_SIZE_QUANT
        scale_ptrs = scales_ptr + offs_n[:, None] * stride_sn + k_block[None, :] * stride_sb
        scale_mask = (offs_n[:, None] < N) & (k_block[None, :] < num_k_blocks)
        scale = tl.load(scale_ptrs, mask=scale_mask, other=1.0).to(tl.float16)

        w = signed_val * scale

        # GEMM
        accumulator += tl.dot(a, tl.trans(w))

    # Store
    c = accumulator.to(tl.float32)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w4a16_gemm_fused(
    A: torch.Tensor,
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """Fused W4A16 GEMM."""
    M, K = A.shape
    N = W_packed.shape[0]

    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    w4a16_fused_kernel[grid](
        A, W_packed, scales, C,
        M, N, K,
        A.stride(0), A.stride(1),
        W_packed.stride(0), W_packed.stride(1),
        scales.stride(0), scales.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_QUANT=block_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return C


# ==============================================================================
# Benchmark
# ==============================================================================

def benchmark_w4a16_v2(M=712, N=16384, K=2048, warmup=50, runs=200):
    """Benchmark optimized W4A16 implementations."""
    import time

    print(f"\n{'='*70}")
    print(f"W4A16 Triton GEMM v2 Benchmark")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*70}")

    device = torch.device("cuda")
    torch.manual_seed(42)

    A = torch.randn(M, K, dtype=torch.float16, device=device)
    W = torch.randn(N, K, dtype=torch.float32, device=device)

    print("Quantizing weights...")
    W_packed, scales = quantize_to_int4_packed(W)
    W_packed = W_packed.to(device)
    scales = scales.to(device)

    W_dequant = dequantize_int4_packed(W_packed, scales, K)
    C_ref = torch.mm(A, W_dequant.T).float()

    results = []

    # Test methods
    methods = [
        ("Dequant + Triton GEMM", lambda: w4a16_gemm_v2(A, W_packed, scales)),
        ("Fused W4A16", lambda: w4a16_gemm_fused(A, W_packed, scales)),
        ("Dequant + cuBLAS", lambda: torch.mm(A, dequantize_int4_packed(W_packed, scales, K).T).float()),
        ("cuBLAS BF16", lambda: torch.mm(A.bfloat16(), W_dequant.bfloat16().T).float()),
    ]

    for name, fn in methods:
        print(f"\n--- {name} ---")
        try:
            # Warmup
            for _ in range(warmup):
                C = fn()
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(runs):
                C = fn()
            torch.cuda.synchronize()

            avg_ms = (time.time() - start) / runs * 1000

            max_diff = (C - C_ref).abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                C.flatten().unsqueeze(0), C_ref.flatten().unsqueeze(0)
            ).item()

            flops = 2.0 * M * N * K
            tflops = flops / (avg_ms / 1000) / 1e12

            print(f"  Time:    {avg_ms:.4f} ms")
            print(f"  TFLOPS:  {tflops:.4f}")
            print(f"  Cos sim: {cos_sim:.6f}")

            results.append((name, avg_ms, tflops, cos_sim > 0.99))
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    bf16_ms = [r[1] for r in results if 'BF16' in r[0]][0] if results else 1.0

    for name, time_ms, tflops, correct in results:
        status = "✅" if correct else "❌"
        speedup = bf16_ms / time_ms
        print(f"  {name:<25} {time_ms:.4f}ms  {tflops:.2f} TFLOPS  vs BF16: {speedup:.2f}x  {status}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=712)
    parser.add_argument("--N", type=int, default=16384)
    parser.add_argument("--K", type=int, default=2048)

    args = parser.parse_args()

    benchmark_w4a16_v2(args.M, args.N, args.K)
