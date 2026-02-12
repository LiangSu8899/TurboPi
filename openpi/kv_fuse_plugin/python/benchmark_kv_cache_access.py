#!/usr/bin/env python3
"""
KV Cache Access Pattern Analysis

分析 Attention 中 KV cache 访问的内存瓶颈.

关键发现:
- KV cache 大小: 2 × 18层 × B × num_kv_heads × seq_len × head_dim × 2 bytes
- 对于 PaLiGemma (seq=1018, kv_heads=1, head_dim=256):
  - 每层 KV cache: 2 × 1 × 1 × 1018 × 256 × 2 = 1,042,432 bytes ≈ 1 MB/层
  - 18层总计: 约 18 MB

Attention 计算的内存访问:
1. Q @ K^T: 读取 K cache [B, kv_heads, seq, head_dim]
2. Softmax: 在 attention weights 上操作
3. attn @ V: 读取 V cache [B, kv_heads, seq, head_dim]

关键优化方向:
1. KV cache 量化 (BF16 -> FP8/INT8) - 减少 50% 内存带宽
2. FlashDecoding - 更好的内存访问模式
3. PagedAttention - 更好的内存管理
"""

import sys
import os
import time

sys.path.insert(0, '/workspace/kv_fuse_plugin/python')

import torch
import torch.nn.functional as F
import numpy as np


def analyze_kv_cache_memory():
    """理论分析 KV cache 的内存访问."""

    print("=" * 70)
    print("KV Cache Memory Access Analysis")
    print("=" * 70)

    # PaLiGemma 配置
    configs = [
        {
            'name': 'PaLiGemma (Gemma 2B)',
            'num_layers': 18,
            'num_heads': 8,
            'num_kv_heads': 1,
            'head_dim': 256,
            'prefix_len': 968,  # image tokens + system prompt
            'suffix_len': 50,   # action tokens (denoising)
            'batch_size': 1,
        },
        {
            'name': 'Action Expert (Gemma 300M)',
            'num_layers': 18,
            'num_heads': 8,
            'num_kv_heads': 1,
            'head_dim': 256,
            'prefix_len': 1,     # latent token
            'suffix_len': 50,
            'batch_size': 1,
        }
    ]

    memory_bw = 122.8  # GB/s Thor

    for config in configs:
        name = config['name']
        L = config['num_layers']
        H = config['num_heads']
        KV_H = config['num_kv_heads']
        D = config['head_dim']
        prefix = config['prefix_len']
        suffix = config['suffix_len']
        B = config['batch_size']
        total_seq = prefix + suffix

        print(f"\n--- {name} ---")
        print(f"Layers: {L}, Heads: {H}, KV Heads: {KV_H}, Head Dim: {D}")
        print(f"Sequence: prefix={prefix} + suffix={suffix} = {total_seq}")

        # KV cache size per layer
        kv_per_layer = 2 * B * KV_H * total_seq * D * 2  # K + V, BF16
        kv_total = kv_per_layer * L

        print(f"\nKV Cache Size:")
        print(f"  Per layer: {kv_per_layer / 1024:.1f} KB")
        print(f"  Total (18 layers): {kv_total / 1024 / 1024:.2f} MB")

        # Attention 计算中的内存访问
        # 对于 decode (suffix 生成), 每个 token 需要读取整个 KV cache

        # 单步 attention 内存访问:
        # - Read K cache: B × KV_H × seq × D × 2 bytes
        # - Read V cache: B × KV_H × seq × D × 2 bytes
        # - Write attention output: B × H × 1 × D × 2 bytes (小)

        k_read = B * KV_H * total_seq * D * 2
        v_read = B * KV_H * total_seq * D * 2
        kv_read_per_layer = k_read + v_read
        kv_read_per_step = kv_read_per_layer * L  # 18 layers

        print(f"\nMemory Access per Denoising Step:")
        print(f"  K cache read: {k_read / 1024:.1f} KB/layer")
        print(f"  V cache read: {v_read / 1024:.1f} KB/layer")
        print(f"  Total per layer: {kv_read_per_layer / 1024:.1f} KB")
        print(f"  Total per step (18 layers): {kv_read_per_step / 1024 / 1024:.2f} MB")

        # 10 denoising steps
        kv_read_10_steps = kv_read_per_step * 10
        print(f"  Total 10 steps: {kv_read_10_steps / 1024 / 1024:.2f} MB")

        # 理论时间 (memory-bound)
        time_per_step_ms = kv_read_per_step / (memory_bw * 1e9) * 1000
        time_10_steps_ms = kv_read_10_steps / (memory_bw * 1e9) * 1000

        print(f"\nTheoretical Time (memory-bound at {memory_bw} GB/s):")
        print(f"  Per denoising step: {time_per_step_ms:.2f} ms")
        print(f"  10 steps: {time_10_steps_ms:.2f} ms")

        # FP8 KV cache 优化后
        kv_read_fp8 = kv_read_per_step // 2  # FP8 = 1 byte
        time_fp8_ms = kv_read_fp8 / (memory_bw * 1e9) * 1000

        print(f"\nWith FP8 KV Cache:")
        print(f"  Memory access: {kv_read_fp8 / 1024 / 1024:.2f} MB/step")
        print(f"  Theoretical time: {time_fp8_ms:.2f} ms/step")
        print(f"  Speedup: {time_per_step_ms / time_fp8_ms:.2f}x")


def benchmark_kv_cache_read(
    batch_size: int = 1,
    num_kv_heads: int = 1,
    seq_len: int = 1018,
    head_dim: int = 256,
    warmup: int = 100,
    runs: int = 500,
):
    """Benchmark raw KV cache memory read."""

    device = torch.device('cuda')

    print("\n" + "=" * 70)
    print("KV Cache Read Benchmark")
    print("=" * 70)
    print(f"batch={batch_size}, kv_heads={num_kv_heads}, seq={seq_len}, head_dim={head_dim}")

    # Create KV cache
    K_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim,
                          device=device, dtype=torch.bfloat16)
    V_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim,
                          device=device, dtype=torch.bfloat16)

    # 模拟 attention 中的 KV cache 读取
    # 简单读取操作
    def read_kv_cache():
        k = K_cache.contiguous()
        v = V_cache.contiguous()
        return k, v

    # Warmup
    for _ in range(warmup):
        k, v = read_kv_cache()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        k, v = read_kv_cache()
    torch.cuda.synchronize()
    time_ms = (time.perf_counter() - start) / runs * 1000

    # 计算带宽
    bytes_read = 2 * batch_size * num_kv_heads * seq_len * head_dim * 2
    bandwidth_gb_s = bytes_read / (time_ms / 1000) / 1e9

    print(f"\nResults:")
    print(f"  Time: {time_ms:.4f} ms")
    print(f"  Bytes read: {bytes_read / 1024:.1f} KB")
    print(f"  Bandwidth: {bandwidth_gb_s:.1f} GB/s")

    # 对比理论
    memory_bw = 122.8
    theoretical_time = bytes_read / (memory_bw * 1e9) * 1000
    print(f"\n  Theoretical time: {theoretical_time:.4f} ms")
    print(f"  Efficiency: {theoretical_time / time_ms * 100:.1f}%")

    return time_ms, bandwidth_gb_s


def benchmark_attention_kv_access(
    batch_size: int = 1,
    num_heads: int = 8,
    num_kv_heads: int = 1,
    head_dim: int = 256,
    prefix_len: int = 968,
    suffix_len: int = 50,
    warmup: int = 100,
    runs: int = 200,
):
    """Benchmark attention with KV cache access."""

    device = torch.device('cuda')
    total_len = prefix_len + suffix_len

    print("\n" + "=" * 70)
    print("Attention with KV Cache Benchmark")
    print("=" * 70)
    print(f"batch={batch_size}, heads={num_heads}, kv_heads={num_kv_heads}")
    print(f"head_dim={head_dim}, prefix={prefix_len}, suffix={suffix_len}")

    # Create tensors
    # Query: only suffix tokens
    Q = torch.randn(batch_size, num_heads, suffix_len, head_dim,
                    device=device, dtype=torch.bfloat16)

    # K, V: full sequence (KV cache)
    K = torch.randn(batch_size, num_kv_heads, total_len, head_dim,
                    device=device, dtype=torch.bfloat16)
    V = torch.randn(batch_size, num_kv_heads, total_len, head_dim,
                    device=device, dtype=torch.bfloat16)

    # GQA expansion
    num_groups = num_heads // num_kv_heads
    K_expanded = K[:, :, None, :, :].expand(
        batch_size, num_kv_heads, num_groups, total_len, head_dim
    ).reshape(batch_size, num_heads, total_len, head_dim)
    V_expanded = V[:, :, None, :, :].expand(
        batch_size, num_kv_heads, num_groups, total_len, head_dim
    ).reshape(batch_size, num_heads, total_len, head_dim)

    scaling = head_dim ** -0.5

    # =========================================================================
    # 1. SDPA
    # =========================================================================
    print("\n--- SDPA ---")

    def sdpa_attention():
        return F.scaled_dot_product_attention(
            Q, K_expanded, V_expanded,
            is_causal=False,  # Not causal for cross-attention with prefix
            scale=scaling,
        )

    for _ in range(warmup):
        _ = sdpa_attention()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        out = sdpa_attention()
    torch.cuda.synchronize()
    sdpa_time = (time.perf_counter() - start) / runs * 1000
    print(f"SDPA: {sdpa_time:.3f} ms")

    # =========================================================================
    # 2. Eager attention (for comparison)
    # =========================================================================
    print("\n--- Eager Attention ---")

    def eager_attention():
        attn_weights = torch.matmul(Q, K_expanded.transpose(2, 3)) * scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
        return torch.matmul(attn_weights, V_expanded)

    for _ in range(warmup):
        _ = eager_attention()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        out = eager_attention()
    torch.cuda.synchronize()
    eager_time = (time.perf_counter() - start) / runs * 1000
    print(f"Eager: {eager_time:.3f} ms")

    # =========================================================================
    # 3. Analysis
    # =========================================================================
    print("\n--- Analysis ---")

    # Memory access
    q_bytes = batch_size * num_heads * suffix_len * head_dim * 2
    kv_bytes = 2 * batch_size * num_kv_heads * total_len * head_dim * 2  # K + V

    # With GQA expansion (what actually happens)
    kv_expanded_bytes = 2 * batch_size * num_heads * total_len * head_dim * 2

    print(f"Memory access:")
    print(f"  Q: {q_bytes / 1024:.1f} KB")
    print(f"  K+V (original): {kv_bytes / 1024:.1f} KB")
    print(f"  K+V (expanded for GQA): {kv_expanded_bytes / 1024:.1f} KB")

    # Compute-to-memory ratio
    # Attention: Q @ K^T + softmax + attn @ V
    # FLOPS: 2 * suffix * total * head_dim * num_heads (Q@K) +
    #        2 * suffix * total * head_dim * num_heads (attn@V)
    attn_flops = 4 * batch_size * num_heads * suffix_len * total_len * head_dim

    print(f"\nCompute: {attn_flops / 1e9:.3f} GFLOPS")
    print(f"Arithmetic intensity: {attn_flops / kv_expanded_bytes:.1f} FLOPS/byte")

    # Roofline
    memory_bw = 122.8  # GB/s
    compute_peak = 50   # TFLOPS (estimated for Thor)

    theoretical_mem_time = kv_expanded_bytes / (memory_bw * 1e9) * 1000
    theoretical_compute_time = attn_flops / (compute_peak * 1e12) * 1000

    print(f"\nTheoretical time:")
    print(f"  Memory-bound: {theoretical_mem_time:.3f} ms")
    print(f"  Compute-bound: {theoretical_compute_time:.3f} ms")
    print(f"  Bottleneck: {'Memory' if theoretical_mem_time > theoretical_compute_time else 'Compute'}")

    # Speedup with FP8 KV cache
    print(f"\nWith FP8 KV Cache:")
    kv_fp8_bytes = kv_expanded_bytes // 2
    theoretical_fp8_time = kv_fp8_bytes / (memory_bw * 1e9) * 1000
    print(f"  Memory access: {kv_fp8_bytes / 1024:.1f} KB")
    print(f"  Theoretical time: {theoretical_fp8_time:.3f} ms")
    print(f"  Potential speedup: {theoretical_mem_time / theoretical_fp8_time:.2f}x")

    return sdpa_time, eager_time


if __name__ == "__main__":
    # 理论分析
    analyze_kv_cache_memory()

    # Benchmark
    benchmark_kv_cache_read()
    benchmark_attention_kv_access()
