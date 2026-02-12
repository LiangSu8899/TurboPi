#!/usr/bin/env python3
"""
Prefill Phase Benchmark

测试prefill phase (M > 1) 的 QKV pipeline性能.
这是FP4真正可能发挥作用的场景.
"""

import sys
import os
import time

sys.path.insert(0, '/workspace/kv_fuse_plugin/python')

import torch
import torch.nn.functional as F
import numpy as np

from kv_fuse import FusedQKVProjection, quantize_to_fp4


def benchmark_prefill_qkv(
    hidden_size: int = 2048,
    num_heads: int = 8,
    num_kv_heads: int = 1,
    head_dim: int = 256,
    seq_lengths: list = [1, 16, 64, 128, 256, 455, 512],
    warmup: int = 50,
    runs: int = 200,
):
    """Benchmark QKV projection for different sequence lengths (prefill)."""

    device = torch.device('cuda')
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    print(f"\n{'='*70}")
    print(f"Prefill Phase Benchmark: QKV Projection")
    print(f"{'='*70}")
    print(f"hidden_size={hidden_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}")
    print(f"head_dim={head_dim}")
    print(f"\n{'seq_len':>8} | {'cuBLAS BF16':>12} | {'FP4 Fused':>12} | {'Speedup':>10}")
    print("-" * 50)

    # 创建权重
    Wq = torch.randn(q_dim, hidden_size, device=device, dtype=torch.bfloat16)
    Wk = torch.randn(kv_dim, hidden_size, device=device, dtype=torch.bfloat16)
    Wv = torch.randn(kv_dim, hidden_size, device=device, dtype=torch.bfloat16)

    # FP4量化权重
    Wq_packed, scale_Wq = quantize_to_fp4(Wq.float())
    Wk_packed, scale_Wk = quantize_to_fp4(Wk.float())
    Wv_packed, scale_Wv = quantize_to_fp4(Wv.float())

    results = []

    for seq_len in seq_lengths:
        # 创建输入
        x_bf16 = torch.randn(seq_len, hidden_size, device=device, dtype=torch.bfloat16)
        x_f32 = x_bf16.float()

        # =====================================================================
        # cuBLAS BF16 baseline
        # =====================================================================
        for _ in range(warmup):
            Q = F.linear(x_bf16, Wq)
            K = F.linear(x_bf16, Wk)
            V = F.linear(x_bf16, Wv)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            Q = F.linear(x_bf16, Wq)
            K = F.linear(x_bf16, Wk)
            V = F.linear(x_bf16, Wv)
        torch.cuda.synchronize()
        cublas_time = (time.perf_counter() - start) / runs * 1000

        # =====================================================================
        # FP4 (反量化 + GEMM, 模拟fused kernel的理论上限)
        # =====================================================================
        # 这里我们测试反量化后用cuBLAS,作为FP4 fused kernel的理论上限
        # 真正的fused kernel应该比这个更快(避免了反量化的中间存储)

        # 反量化权重
        nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)

        def dequantize_fp4(packed, scales, N, K, block_size=32):
            K_half = K // 2
            unpacked = torch.zeros(N, K, dtype=torch.long, device=device)
            unpacked[:, 0::2] = packed & 0xF
            unpacked[:, 1::2] = (packed >> 4) & 0xF

            signs = (unpacked >= 8).float()
            indices = (unpacked % 8).long()
            values = nvfp4_values[indices.flatten()].view(N, K)
            values = values * (1 - 2 * signs)

            num_blocks = K // block_size
            values_blocked = values.view(N, num_blocks, block_size)
            scales_expanded = scales.unsqueeze(-1)
            values_blocked = values_blocked * scales_expanded
            return values_blocked.view(N, K).to(torch.bfloat16)

        Wq_dequant = dequantize_fp4(Wq_packed, scale_Wq, q_dim, hidden_size)
        Wk_dequant = dequantize_fp4(Wk_packed, scale_Wk, kv_dim, hidden_size)
        Wv_dequant = dequantize_fp4(Wv_packed, scale_Wv, kv_dim, hidden_size)

        for _ in range(warmup):
            Q = F.linear(x_bf16, Wq_dequant)
            K = F.linear(x_bf16, Wk_dequant)
            V = F.linear(x_bf16, Wv_dequant)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            Q = F.linear(x_bf16, Wq_dequant)
            K = F.linear(x_bf16, Wk_dequant)
            V = F.linear(x_bf16, Wv_dequant)
        torch.cuda.synchronize()
        fp4_dequant_time = (time.perf_counter() - start) / runs * 1000

        # =====================================================================
        # 结果
        # =====================================================================
        speedup = cublas_time / fp4_dequant_time

        print(f"{seq_len:>8} | {cublas_time:>10.4f} ms | {fp4_dequant_time:>10.4f} ms | {speedup:>8.2f}x")

        results.append({
            'seq_len': seq_len,
            'cublas': cublas_time,
            'fp4_dequant': fp4_dequant_time,
            'speedup': speedup,
        })

    # =========================================================================
    # 理论分析
    # =========================================================================
    print("\n" + "=" * 70)
    print("Memory Analysis")
    print("=" * 70)

    # 权重大小
    wq_bf16 = q_dim * hidden_size * 2
    wk_bf16 = kv_dim * hidden_size * 2
    wv_bf16 = kv_dim * hidden_size * 2
    total_bf16 = wq_bf16 + wk_bf16 + wv_bf16

    wq_fp4 = q_dim * hidden_size // 2 + q_dim * (hidden_size // 32) * 4
    wk_fp4 = kv_dim * hidden_size // 2 + kv_dim * (hidden_size // 32) * 4
    wv_fp4 = kv_dim * hidden_size // 2 + kv_dim * (hidden_size // 32) * 4
    total_fp4 = wq_fp4 + wk_fp4 + wv_fp4

    print(f"BF16 weight size: {total_bf16 / 1024 / 1024:.2f} MB")
    print(f"FP4 weight size:  {total_fp4 / 1024 / 1024:.2f} MB")
    print(f"Memory reduction: {total_bf16 / total_fp4:.2f}x")

    # Roofline
    memory_bw = 122.8  # GB/s
    print(f"\nTheoretical minimum time (memory-bound at {memory_bw} GB/s):")
    print(f"  BF16: {total_bf16 / (memory_bw * 1e9) * 1000:.4f} ms")
    print(f"  FP4:  {total_fp4 / (memory_bw * 1e9) * 1000:.4f} ms")

    return results


def benchmark_memory_bound_check():
    """Check if QKV projection is memory-bound or compute-bound."""

    print("\n" + "=" * 70)
    print("Memory vs Compute Bound Analysis")
    print("=" * 70)

    device = torch.device('cuda')
    hidden_size = 2048
    q_dim = 2048
    kv_dim = 256

    # 权重
    Wq = torch.randn(q_dim, hidden_size, device=device, dtype=torch.bfloat16)
    Wk = torch.randn(kv_dim, hidden_size, device=device, dtype=torch.bfloat16)
    Wv = torch.randn(kv_dim, hidden_size, device=device, dtype=torch.bfloat16)

    # 测试不同seq_len下的arithmetic intensity
    seq_lengths = [1, 16, 64, 256, 512, 1024]

    print(f"\n{'seq_len':>8} | {'Arith.Int.':>12} | {'Time(ms)':>10} | {'TFLOPS':>10} | {'GB/s':>10}")
    print("-" * 60)

    for seq_len in seq_lengths:
        x = torch.randn(seq_len, hidden_size, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(50):
            Q = F.linear(x, Wq)
            K = F.linear(x, Wk)
            V = F.linear(x, Wv)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(200):
            Q = F.linear(x, Wq)
            K = F.linear(x, Wk)
            V = F.linear(x, Wv)
        torch.cuda.synchronize()
        time_ms = (time.perf_counter() - start) / 200 * 1000

        # FLOPS: 2 * M * N * K for each GEMM
        flops_q = 2 * seq_len * q_dim * hidden_size
        flops_k = 2 * seq_len * kv_dim * hidden_size
        flops_v = 2 * seq_len * kv_dim * hidden_size
        total_flops = flops_q + flops_k + flops_v

        # Memory: weights + input + output
        mem_weights = (q_dim + kv_dim + kv_dim) * hidden_size * 2  # BF16
        mem_input = seq_len * hidden_size * 2
        mem_output = seq_len * (q_dim + kv_dim + kv_dim) * 2
        total_mem = mem_weights + mem_input + mem_output

        # Arithmetic intensity
        arith_int = total_flops / total_mem

        # Achieved performance
        tflops = total_flops / (time_ms / 1000) / 1e12
        gb_s = total_mem / (time_ms / 1000) / 1e9

        print(f"{seq_len:>8} | {arith_int:>10.1f} | {time_ms:>10.4f} | {tflops:>8.2f} | {gb_s:>8.1f}")

    # Thor specs
    print("\nThor approximate specs:")
    print("  Memory BW: ~123 GB/s")
    print("  BF16 TFLOPS: ~50 TFLOPS (estimated)")
    print("  Roofline knee: ~400 FLOPS/byte")


if __name__ == "__main__":
    # PaLiGemma
    print("\n" + "=" * 70)
    print("PaLiGemma (Gemma 2B)")
    print("=" * 70)
    benchmark_prefill_qkv(
        hidden_size=2048,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        seq_lengths=[1, 16, 64, 128, 256, 455, 512],
    )

    # Memory bound analysis
    benchmark_memory_bound_check()
