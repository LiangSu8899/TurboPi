#!/usr/bin/env python3
"""
Benchmark attention implementations for denoising.

Compares:
1. Eager attention (manual matmul + softmax)
2. SDPA with explicit mask
3. FlashAttention 2 with causal=True

This benchmarks just the attention operation, not the full denoising step.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

device = "cuda"
dtype = torch.bfloat16

# Configuration matching Pi0.5
batch_size = 1
num_heads = 8
num_kv_heads = 1
head_dim = 256
prefix_len = 968
suffix_len = 50
total_len = prefix_len + suffix_len
num_iterations = 100
warmup_iterations = 20

print("=" * 60)
print("Attention Benchmark for Denoising")
print("=" * 60)
print(f"Config: batch={batch_size}, heads={num_heads}, kv_heads={num_kv_heads}")
print(f"        head_dim={head_dim}, prefix={prefix_len}, suffix={suffix_len}")
print(f"Iterations: {num_iterations} (warmup: {warmup_iterations})")
print()

# Create tensors
torch.manual_seed(42)

# Query (suffix only)
q = torch.randn(batch_size, num_heads, suffix_len, head_dim, device=device, dtype=dtype)

# K, V for prefix and suffix (not expanded - GQA)
prefix_k = torch.randn(batch_size, num_kv_heads, prefix_len, head_dim, device=device, dtype=dtype)
prefix_v = torch.randn(batch_size, num_kv_heads, prefix_len, head_dim, device=device, dtype=dtype)
suffix_k = torch.randn(batch_size, num_kv_heads, suffix_len, head_dim, device=device, dtype=dtype)
suffix_v = torch.randn(batch_size, num_kv_heads, suffix_len, head_dim, device=device, dtype=dtype)

# Concatenated K, V
full_k = torch.cat([prefix_k, suffix_k], dim=2)
full_v = torch.cat([prefix_v, suffix_v], dim=2)

# GQA expanded K, V for SDPA/Eager
num_kv_groups = num_heads // num_kv_heads
full_k_expanded = full_k[:, :, None, :, :].expand(
    batch_size, num_kv_heads, num_kv_groups, total_len, head_dim
).reshape(batch_size, num_heads, total_len, head_dim)
full_v_expanded = full_v[:, :, None, :, :].expand(
    batch_size, num_kv_heads, num_kv_groups, total_len, head_dim
).reshape(batch_size, num_heads, total_len, head_dim)

# Pre-compute attention mask
attn_mask = torch.zeros(suffix_len, total_len, device=device, dtype=dtype)
attn_mask[:, :prefix_len] = 0
suffix_mask = torch.triu(torch.ones(suffix_len, suffix_len, device=device), diagonal=1) * -1e9
attn_mask[:, prefix_len:] = suffix_mask.to(dtype)
attn_mask_4d = attn_mask[None, None, :, :]

# FlashAttention layout
if HAS_FLASH_ATTN:
    q_fa = q.transpose(1, 2).contiguous()
    k_fa = full_k.transpose(1, 2).contiguous()
    v_fa = full_v.transpose(1, 2).contiguous()

scaling = head_dim ** -0.5

results = {}

# 1. Eager attention
def eager_attention():
    attn_weights = torch.matmul(q, full_k_expanded.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attn_mask_4d
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
    return torch.matmul(attn_weights, full_v_expanded)

print("Warming up Eager...")
for _ in range(warmup_iterations):
    _ = eager_attention()
torch.cuda.synchronize()

times = []
for _ in range(num_iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = eager_attention()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)
results["eager"] = {"mean": np.mean(times), "std": np.std(times), "output": out}
print(f"Eager: {results['eager']['mean']:.3f} ± {results['eager']['std']:.3f} ms")

# 2. SDPA with mask
def sdpa_attention():
    return F.scaled_dot_product_attention(
        q, full_k_expanded, full_v_expanded,
        attn_mask=attn_mask_4d,
        scale=scaling,
    )

print("Warming up SDPA...")
for _ in range(warmup_iterations):
    _ = sdpa_attention()
torch.cuda.synchronize()

times = []
for _ in range(num_iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = sdpa_attention()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)
results["sdpa"] = {"mean": np.mean(times), "std": np.std(times), "output": out}
print(f"SDPA: {results['sdpa']['mean']:.3f} ± {results['sdpa']['std']:.3f} ms")

# 3. FlashAttention
if HAS_FLASH_ATTN:
    def flash_attention():
        out = flash_attn_func(q_fa, k_fa, v_fa, causal=True, softmax_scale=scaling)
        return out.transpose(1, 2).contiguous()

    print("Warming up FlashAttention...")
    for _ in range(warmup_iterations):
        _ = flash_attention()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = flash_attention()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    results["flash"] = {"mean": np.mean(times), "std": np.std(times), "output": out}
    print(f"FlashAttention: {results['flash']['mean']:.3f} ± {results['flash']['std']:.3f} ms")

# 4. FlashAttention without transpose (to isolate attention cost)
if HAS_FLASH_ATTN:
    # Pre-transposed tensors
    q_fa_pre = q.transpose(1, 2).contiguous()
    k_fa_pre = full_k.transpose(1, 2).contiguous()
    v_fa_pre = full_v.transpose(1, 2).contiguous()

    def flash_attention_notranspose():
        return flash_attn_func(q_fa_pre, k_fa_pre, v_fa_pre, causal=True, softmax_scale=scaling)

    print("Warming up FlashAttention (no transpose)...")
    for _ in range(warmup_iterations):
        _ = flash_attention_notranspose()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = flash_attention_notranspose()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    results["flash_notranspose"] = {"mean": np.mean(times), "std": np.std(times), "output": out.transpose(1, 2)}
    print(f"FlashAttention (no transpose): {results['flash_notranspose']['mean']:.3f} ± {results['flash_notranspose']['std']:.3f} ms")

# Precision check
print("\n" + "=" * 60)
print("Precision Check")
print("=" * 60)

baseline = results["eager"]["output"].float()
for name, data in results.items():
    if name == "eager":
        continue
    out = data["output"].float()
    max_diff = (baseline - out).abs().max().item()
    cos_sim = F.cosine_similarity(baseline.flatten(), out.flatten(), dim=0).item()
    status = "✅" if max_diff < 0.01 and cos_sim > 0.999 else "⚠️"
    print(f"{name}: max_diff={max_diff:.4e}, cos_sim={cos_sim:.6f} {status}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

baseline_time = results["eager"]["mean"]
print(f"\n{'Method':<25} {'Latency (ms)':<15} {'Speedup':<10}")
print("-" * 50)
for name, data in results.items():
    speedup = baseline_time / data["mean"]
    print(f"{name:<25} {data['mean']:.3f} ± {data['std']:.3f}  {speedup:.2f}x")

# Per-layer estimate
print("\n" + "=" * 60)
print("Full Denoising Estimate (18 layers, 10 steps)")
print("=" * 60)

for name, data in results.items():
    # 18 layers * 10 steps = 180 attention operations
    total_ms = data["mean"] * 18 * 10
    freq = 1000 / total_ms
    print(f"{name}: {total_ms:.1f} ms ({freq:.1f} Hz) - attention only")

print("\nNote: Actual denoising includes MLP, LayerNorm, projections etc.")
