#!/usr/bin/env python3
"""
Complete Decoder Layer Benchmark

分析 transformer decoder layer 的每个组件耗时.

目标: 找出真正的 57% KV latency 来自哪里.
"""

import sys
import os
import time

sys.path.insert(0, '/workspace/kv_fuse_plugin/python')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RMSNorm(nn.Module):
    """RMS Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def benchmark_decoder_layer_components(
    hidden_size: int = 2048,
    num_heads: int = 8,
    num_kv_heads: int = 1,
    head_dim: int = 256,
    mlp_dim: int = 16384,
    batch_size: int = 1,
    seq_len: int = 1,  # Decode mode
    kv_seq_len: int = 1018,  # Full KV cache length
    warmup: int = 100,
    runs: int = 500,
):
    """Benchmark each component of a decoder layer."""

    device = torch.device('cuda')
    dtype = torch.bfloat16

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    print(f"\n{'='*70}")
    print(f"Decoder Layer Component Benchmark")
    print(f"{'='*70}")
    print(f"hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}")
    print(f"head_dim={head_dim}, mlp_dim={mlp_dim}")
    print(f"batch={batch_size}, seq={seq_len}, kv_seq={kv_seq_len}")

    # Create weights
    # Attention
    Wq = torch.randn(q_dim, hidden_size, device=device, dtype=dtype)
    Wk = torch.randn(kv_dim, hidden_size, device=device, dtype=dtype)
    Wv = torch.randn(kv_dim, hidden_size, device=device, dtype=dtype)
    Wo = torch.randn(hidden_size, q_dim, device=device, dtype=dtype)

    # MLP
    Wgate = torch.randn(mlp_dim, hidden_size, device=device, dtype=dtype)
    Wup = torch.randn(mlp_dim, hidden_size, device=device, dtype=dtype)
    Wdown = torch.randn(hidden_size, mlp_dim, device=device, dtype=dtype)

    # LayerNorm
    ln1 = RMSNorm(hidden_size).to(device, dtype)
    ln2 = RMSNorm(hidden_size).to(device, dtype)

    # Rotary embeddings
    cos = torch.randn(kv_seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.randn(kv_seq_len, head_dim, device=device, dtype=dtype)

    # KV Cache
    K_cache = torch.randn(batch_size, num_kv_heads, kv_seq_len, head_dim,
                          device=device, dtype=dtype)
    V_cache = torch.randn(batch_size, num_kv_heads, kv_seq_len, head_dim,
                          device=device, dtype=dtype)

    # Input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    results = {}

    def benchmark(name, fn, warmup_iters=warmup, run_iters=runs):
        for _ in range(warmup_iters):
            _ = fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(run_iters):
            out = fn()
        torch.cuda.synchronize()
        time_ms = (time.perf_counter() - start) / run_iters * 1000
        results[name] = time_ms
        return time_ms

    # =========================================================================
    # 1. Input LayerNorm
    # =========================================================================
    print("\n--- LayerNorm ---")
    t = benchmark("ln1", lambda: ln1(x))
    print(f"  Input LN: {t:.4f} ms")

    t = benchmark("ln2", lambda: ln2(x))
    print(f"  Post-attn LN: {t:.4f} ms")

    # =========================================================================
    # 2. QKV Projection
    # =========================================================================
    print("\n--- QKV Projection ---")

    x_flat = x.view(batch_size * seq_len, hidden_size)
    t = benchmark("q_proj", lambda: F.linear(x_flat, Wq))
    print(f"  Q projection: {t:.4f} ms")

    t = benchmark("k_proj", lambda: F.linear(x_flat, Wk))
    print(f"  K projection: {t:.4f} ms")

    t = benchmark("v_proj", lambda: F.linear(x_flat, Wv))
    print(f"  V projection: {t:.4f} ms")

    # =========================================================================
    # 3. RoPE
    # =========================================================================
    print("\n--- RoPE ---")

    Q = F.linear(x_flat, Wq).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    K = F.linear(x_flat, Wk).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Get position-specific cos/sin
    pos = kv_seq_len - 1  # Last position for decode
    cos_pos = cos[pos:pos+1].unsqueeze(0).unsqueeze(0)  # [1, 1, 1, head_dim]
    sin_pos = sin[pos:pos+1].unsqueeze(0).unsqueeze(0)

    def apply_rope():
        q_rot = (Q * cos_pos) + (rotate_half(Q) * sin_pos)
        k_rot = (K * cos_pos) + (rotate_half(K) * sin_pos)
        return q_rot, k_rot

    t = benchmark("rope", apply_rope)
    print(f"  RoPE: {t:.4f} ms")

    # =========================================================================
    # 4. KV Cache Update
    # =========================================================================
    print("\n--- KV Cache Update ---")

    K_new = F.linear(x_flat, Wk).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    V_new = F.linear(x_flat, Wv).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    def update_kv_cache():
        K_cache[:, :, -seq_len:, :] = K_new
        V_cache[:, :, -seq_len:, :] = V_new
        return K_cache, V_cache

    t = benchmark("kv_update", update_kv_cache)
    print(f"  KV Cache Update: {t:.4f} ms")

    # =========================================================================
    # 5. Attention (SDPA)
    # =========================================================================
    print("\n--- Attention ---")

    Q_attn = F.linear(x_flat, Wq).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # GQA expansion
    num_groups = num_heads // num_kv_heads
    K_expanded = K_cache[:, :, None, :, :].expand(
        batch_size, num_kv_heads, num_groups, kv_seq_len, head_dim
    ).reshape(batch_size, num_heads, kv_seq_len, head_dim)
    V_expanded = V_cache[:, :, None, :, :].expand(
        batch_size, num_kv_heads, num_groups, kv_seq_len, head_dim
    ).reshape(batch_size, num_heads, kv_seq_len, head_dim)

    def sdpa_attention():
        return F.scaled_dot_product_attention(
            Q_attn, K_expanded, V_expanded,
            is_causal=False,
        )

    t = benchmark("sdpa", sdpa_attention)
    print(f"  SDPA: {t:.4f} ms")

    # GQA expansion cost
    def gqa_expansion():
        k_exp = K_cache[:, :, None, :, :].expand(
            batch_size, num_kv_heads, num_groups, kv_seq_len, head_dim
        ).reshape(batch_size, num_heads, kv_seq_len, head_dim)
        v_exp = V_cache[:, :, None, :, :].expand(
            batch_size, num_kv_heads, num_groups, kv_seq_len, head_dim
        ).reshape(batch_size, num_heads, kv_seq_len, head_dim)
        return k_exp, v_exp

    t = benchmark("gqa_expand", gqa_expansion)
    print(f"  GQA Expansion: {t:.4f} ms")

    # =========================================================================
    # 6. Output Projection
    # =========================================================================
    print("\n--- Output Projection ---")

    attn_out = sdpa_attention()
    attn_out_flat = attn_out.transpose(1, 2).reshape(batch_size * seq_len, q_dim)

    t = benchmark("o_proj", lambda: F.linear(attn_out_flat, Wo))
    print(f"  Output projection: {t:.4f} ms")

    # =========================================================================
    # 7. MLP
    # =========================================================================
    print("\n--- MLP ---")

    x_mlp = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=dtype)

    t = benchmark("gate_proj", lambda: F.linear(x_mlp, Wgate))
    print(f"  Gate projection: {t:.4f} ms")

    t = benchmark("up_proj", lambda: F.linear(x_mlp, Wup))
    print(f"  Up projection: {t:.4f} ms")

    gate_out = F.linear(x_mlp, Wgate)
    up_out = F.linear(x_mlp, Wup)
    mlp_mid = F.silu(gate_out) * up_out

    t = benchmark("down_proj", lambda: F.linear(mlp_mid, Wdown))
    print(f"  Down projection: {t:.4f} ms")

    t = benchmark("silu_mul", lambda: F.silu(gate_out) * up_out)
    print(f"  SiLU + Mul: {t:.4f} ms")

    # =========================================================================
    # 8. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # Group by category
    attn_related = ['ln1', 'q_proj', 'k_proj', 'v_proj', 'rope', 'kv_update',
                    'gqa_expand', 'sdpa', 'o_proj']
    mlp_related = ['ln2', 'gate_proj', 'up_proj', 'down_proj', 'silu_mul']

    attn_total = sum(results.get(k, 0) for k in attn_related)
    mlp_total = sum(results.get(k, 0) for k in mlp_related)
    total = attn_total + mlp_total

    print(f"\nAttention Block: {attn_total:.4f} ms ({attn_total/total*100:.1f}%)")
    for k in attn_related:
        if k in results:
            print(f"  {k}: {results[k]:.4f} ms ({results[k]/total*100:.1f}%)")

    print(f"\nMLP Block: {mlp_total:.4f} ms ({mlp_total/total*100:.1f}%)")
    for k in mlp_related:
        if k in results:
            print(f"  {k}: {results[k]:.4f} ms ({results[k]/total*100:.1f}%)")

    print(f"\nTotal per layer: {total:.4f} ms")
    print(f"18 layers: {total * 18:.4f} ms")
    print(f"10 steps × 18 layers: {total * 18 * 10:.2f} ms")

    # Identify bottleneck
    print("\n--- Bottleneck Analysis ---")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 time consumers:")
    for name, t in sorted_results[:5]:
        print(f"  {name}: {t:.4f} ms ({t/total*100:.1f}%)")

    return results


if __name__ == "__main__":
    # PaLiGemma configuration
    print("\n" + "=" * 70)
    print("PaLiGemma (Gemma 2B) - Decode Mode")
    print("=" * 70)
    benchmark_decoder_layer_components(
        hidden_size=2048,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        mlp_dim=16384,
        seq_len=1,       # Decode: 1 token
        kv_seq_len=1018, # Full KV cache
    )

    # Prefill mode
    print("\n" + "=" * 70)
    print("PaLiGemma (Gemma 2B) - Prefill Mode (suffix=50)")
    print("=" * 70)
    benchmark_decoder_layer_components(
        hidden_size=2048,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        mlp_dim=16384,
        seq_len=50,      # Prefill: 50 tokens
        kv_seq_len=1018,
    )

    # Action Expert
    print("\n" + "=" * 70)
    print("Action Expert (Gemma 300M) - Decode Mode")
    print("=" * 70)
    benchmark_decoder_layer_components(
        hidden_size=1024,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        mlp_dim=4096,
        seq_len=1,
        kv_seq_len=51,
    )
