#!/usr/bin/env python3
"""
Complete KV Pipeline Benchmark

比较完整的KV pipeline:
1. Q/K/V projection
2. K/V layout transpose
3. KV cache write

这才是真正的KV pipeline latency,包括所有memory operations.
"""

import sys
import os
import time

# Setup paths
for path in [
    os.path.join(os.path.dirname(__file__), "..", "python"),
    "/workspace/kv_fuse_plugin/python",
]:
    if os.path.exists(path):
        sys.path.insert(0, path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def benchmark_complete_kv_pipeline(
    hidden_size: int = 2048,
    num_heads: int = 8,
    num_kv_heads: int = 1,
    head_dim: int = 256,
    batch_size: int = 1,
    max_seq_len: int = 512,
    warmup: int = 100,
    runs: int = 500,
):
    """Benchmark complete KV pipeline including cache writes."""

    device = torch.device('cuda')
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"Complete KV Pipeline Benchmark")
    print(f"{'='*60}")
    print(f"hidden_size={hidden_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}")
    print(f"head_dim={head_dim}, batch_size={batch_size}, max_seq_len={max_seq_len}")
    print(f"q_dim={q_dim}, kv_dim={kv_dim}")

    # 创建权重
    Wq = torch.randn(q_dim, hidden_size, device=device, dtype=torch.bfloat16)
    Wk = torch.randn(kv_dim, hidden_size, device=device, dtype=torch.bfloat16)
    Wv = torch.randn(kv_dim, hidden_size, device=device, dtype=torch.bfloat16)

    # 创建输入
    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    # 创建KV cache - 标准layout: [B, num_kv_heads, max_seq, head_dim]
    K_cache = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=torch.bfloat16)
    V_cache = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=torch.bfloat16)

    # =========================================================================
    # Baseline: Separate cuBLAS + KV cache write
    # =========================================================================
    print("\n--- Baseline: Separate cuBLAS + KV cache write ---")

    def baseline_qkv_pipeline(x, Wq, Wk, Wv, K_cache, V_cache, cache_pos):
        # Q projection
        Q = F.linear(x, Wq)  # [B, q_dim]

        # K projection + cache write
        K = F.linear(x, Wk)  # [B, kv_dim]
        K_reshaped = K.view(batch_size, num_kv_heads, head_dim)  # [B, num_kv_heads, head_dim]
        K_cache[:, :, cache_pos, :] = K_reshaped

        # V projection + cache write
        V = F.linear(x, Wv)  # [B, kv_dim]
        V_reshaped = V.view(batch_size, num_kv_heads, head_dim)
        V_cache[:, :, cache_pos, :] = V_reshaped

        return Q, K_cache, V_cache

    # Warmup
    for i in range(warmup):
        Q, _, _ = baseline_qkv_pipeline(x, Wq, Wk, Wv, K_cache, V_cache, i % max_seq_len)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for i in range(runs):
        Q, _, _ = baseline_qkv_pipeline(x, Wq, Wk, Wv, K_cache, V_cache, i % max_seq_len)
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) / runs * 1000
    print(f"Baseline (cuBLAS + cache write): {baseline_time:.4f} ms")

    # =========================================================================
    # Breakdown: Measure each component
    # =========================================================================
    print("\n--- Breakdown ---")

    # Q projection only
    for _ in range(warmup):
        Q = F.linear(x, Wq)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        Q = F.linear(x, Wq)
    torch.cuda.synchronize()
    q_proj_time = (time.perf_counter() - start) / runs * 1000
    print(f"  Q projection: {q_proj_time:.4f} ms")

    # K projection only
    for _ in range(warmup):
        K = F.linear(x, Wk)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        K = F.linear(x, Wk)
    torch.cuda.synchronize()
    k_proj_time = (time.perf_counter() - start) / runs * 1000
    print(f"  K projection: {k_proj_time:.4f} ms")

    # V projection only
    for _ in range(warmup):
        V = F.linear(x, Wv)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        V = F.linear(x, Wv)
    torch.cuda.synchronize()
    v_proj_time = (time.perf_counter() - start) / runs * 1000
    print(f"  V projection: {v_proj_time:.4f} ms")

    # K cache write only
    K = F.linear(x, Wk)
    K_reshaped = K.view(batch_size, num_kv_heads, head_dim)
    for i in range(warmup):
        K_cache[:, :, i % max_seq_len, :] = K_reshaped
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(runs):
        K_cache[:, :, i % max_seq_len, :] = K_reshaped
    torch.cuda.synchronize()
    k_write_time = (time.perf_counter() - start) / runs * 1000
    print(f"  K cache write: {k_write_time:.4f} ms")

    # V cache write only
    V = F.linear(x, Wv)
    V_reshaped = V.view(batch_size, num_kv_heads, head_dim)
    for i in range(warmup):
        V_cache[:, :, i % max_seq_len, :] = V_reshaped
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(runs):
        V_cache[:, :, i % max_seq_len, :] = V_reshaped
    torch.cuda.synchronize()
    v_write_time = (time.perf_counter() - start) / runs * 1000
    print(f"  V cache write: {v_write_time:.4f} ms")

    total_breakdown = q_proj_time + k_proj_time + v_proj_time + k_write_time + v_write_time
    print(f"  Total breakdown: {total_breakdown:.4f} ms")

    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n--- Analysis ---")

    proj_total = q_proj_time + k_proj_time + v_proj_time
    cache_total = k_write_time + v_write_time

    print(f"Projection total: {proj_total:.4f} ms ({proj_total/baseline_time*100:.1f}%)")
    print(f"Cache write total: {cache_total:.4f} ms ({cache_total/baseline_time*100:.1f}%)")

    # 理论分析
    print("\n--- Memory Analysis ---")

    # 权重大小
    wq_size = q_dim * hidden_size * 2  # BF16
    wk_size = kv_dim * hidden_size * 2
    wv_size = kv_dim * hidden_size * 2
    total_weight_size = wq_size + wk_size + wv_size

    # FP4权重大小
    wq_fp4_size = q_dim * hidden_size // 2 + q_dim * (hidden_size // 32) * 4
    wk_fp4_size = kv_dim * hidden_size // 2 + kv_dim * (hidden_size // 32) * 4
    wv_fp4_size = kv_dim * hidden_size // 2 + kv_dim * (hidden_size // 32) * 4
    total_fp4_size = wq_fp4_size + wk_fp4_size + wv_fp4_size

    print(f"BF16 weight size: {total_weight_size / 1024:.1f} KB")
    print(f"FP4 weight size: {total_fp4_size / 1024:.1f} KB")
    print(f"Memory reduction: {total_weight_size / total_fp4_size:.2f}x")

    # KV cache write size per token
    kv_write_size = 2 * batch_size * num_kv_heads * head_dim * 2  # K + V, BF16
    print(f"KV cache write per token: {kv_write_size} bytes")

    # 带宽计算
    memory_bw = 122.8  # GB/s for Thor
    theoretical_time = total_weight_size / (memory_bw * 1e9) * 1000
    print(f"\nTheoretical projection time (memory-bound): {theoretical_time:.4f} ms")
    print(f"Actual projection time: {proj_total:.4f} ms")
    print(f"Bandwidth utilization: {theoretical_time / proj_total * 100:.1f}%")

    return baseline_time, proj_total, cache_total


if __name__ == "__main__":
    # PaLiGemma
    print("\n" + "=" * 70)
    print("PaLiGemma (Gemma 2B)")
    print("=" * 70)
    benchmark_complete_kv_pipeline(
        hidden_size=2048,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    )

    # Action Expert
    print("\n" + "=" * 70)
    print("Action Expert (Gemma 300M)")
    print("=" * 70)
    benchmark_complete_kv_pipeline(
        hidden_size=1024,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    )
