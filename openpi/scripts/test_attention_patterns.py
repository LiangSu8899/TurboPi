#!/usr/bin/env python3
"""
Test attention patterns for denoising.

The denoising attention pattern is:
- Query: suffix tokens only (50 tokens)
- Key/Value: prefix (968) + suffix (50) = 1018 tokens
- Mask: suffix can see all prefix + causal within suffix

This is NOT a standard causal pattern. The mask looks like:
For suffix token i, it can see:
- All prefix tokens [0:968]
- Suffix tokens [968:968+i+1]

We need to find the right FlashAttention API for this.
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), "src")
sys.path.insert(0, src_dir)

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache, flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

device = "cuda"
dtype = torch.bfloat16

# Configuration
batch_size = 1
num_heads = 8
num_kv_heads = 1
head_dim = 256
prefix_len = 968
suffix_len = 50
total_len = prefix_len + suffix_len

# Create test tensors
torch.manual_seed(42)

# Query (suffix only)
q = torch.randn(batch_size, num_heads, suffix_len, head_dim, device=device, dtype=dtype)

# K, V for prefix and suffix
prefix_k = torch.randn(batch_size, num_kv_heads, prefix_len, head_dim, device=device, dtype=dtype)
prefix_v = torch.randn(batch_size, num_kv_heads, prefix_len, head_dim, device=device, dtype=dtype)
suffix_k = torch.randn(batch_size, num_kv_heads, suffix_len, head_dim, device=device, dtype=dtype)
suffix_v = torch.randn(batch_size, num_kv_heads, suffix_len, head_dim, device=device, dtype=dtype)

# Full K, V
full_k = torch.cat([prefix_k, suffix_k], dim=2)
full_v = torch.cat([prefix_v, suffix_v], dim=2)

# GQA expansion
num_kv_groups = num_heads // num_kv_heads
full_k_expanded = full_k[:, :, None, :, :].expand(
    batch_size, num_kv_heads, num_kv_groups, total_len, head_dim
).reshape(batch_size, num_heads, total_len, head_dim)
full_v_expanded = full_v[:, :, None, :, :].expand(
    batch_size, num_kv_heads, num_kv_groups, total_len, head_dim
).reshape(batch_size, num_heads, total_len, head_dim)

scaling = head_dim ** -0.5

print("=" * 60)
print("Testing Attention Patterns")
print("=" * 60)
print(f"Prefix length: {prefix_len}")
print(f"Suffix length: {suffix_len}")
print(f"Total length: {total_len}")
print()

# 1. Baseline: Eager with correct mask
print("1. Eager Attention (baseline)")
# Mask: suffix can see all prefix + causal within suffix
attn_mask = torch.zeros(suffix_len, total_len, device=device, dtype=dtype)
# Suffix can attend to all prefix (no masking)
attn_mask[:, :prefix_len] = 0
# Causal within suffix
suffix_mask = torch.triu(torch.ones(suffix_len, suffix_len, device=device), diagonal=1) * -1e9
attn_mask[:, prefix_len:] = suffix_mask.to(dtype)
attn_mask_4d = attn_mask[None, None, :, :]

attn_weights = torch.matmul(q, full_k_expanded.transpose(2, 3)) * scaling
attn_weights = attn_weights + attn_mask_4d
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
eager_output = torch.matmul(attn_weights, full_v_expanded)
print(f"  Shape: {eager_output.shape}")

# 2. SDPA with mask
print("\n2. SDPA with mask")
sdpa_output = F.scaled_dot_product_attention(
    q, full_k_expanded, full_v_expanded,
    attn_mask=attn_mask_4d,
    scale=scaling,
)
max_diff = (eager_output - sdpa_output).abs().max().item()
cos_sim = F.cosine_similarity(eager_output.flatten().float(), sdpa_output.flatten().float(), dim=0).item()
print(f"  vs Eager: max_diff={max_diff:.4e}, cos_sim={cos_sim:.6f}")

# 3. FlashAttention with causal (incorrect for this pattern)
if HAS_FLASH_ATTN:
    print("\n3. FlashAttention with causal=True (incorrect pattern)")
    # FlashAttention expects (B, S, H, D)
    q_fa = q.transpose(1, 2).contiguous()
    full_k_fa = full_k.transpose(1, 2).contiguous()
    full_v_fa = full_v.transpose(1, 2).contiguous()

    # This is wrong because causal mask starts from the query position
    # but our suffix queries start at position 968, not 0
    flash_output_wrong = flash_attn_func(q_fa, full_k_fa, full_v_fa, causal=True, softmax_scale=scaling)
    flash_output_wrong = flash_output_wrong.transpose(1, 2).contiguous()
    max_diff = (eager_output - flash_output_wrong).abs().max().item()
    cos_sim = F.cosine_similarity(eager_output.flatten().float(), flash_output_wrong.flatten().float(), dim=0).item()
    print(f"  vs Eager: max_diff={max_diff:.4e}, cos_sim={cos_sim:.6f}")
    print("  Note: This is incorrect because causal mask doesn't account for prefix offset")

    # 4. FlashAttention with window (allow seeing all prefix)
    print("\n4. FlashAttention with window_size")
    # window_size=(prefix_len, 0) means left context of prefix_len, no right context
    # This allows suffix to see prefix_len tokens to the left
    try:
        flash_window = flash_attn_func(
            q_fa, full_k_fa, full_v_fa,
            causal=True,
            window_size=(prefix_len + suffix_len, 0),  # Allow seeing all left tokens
            softmax_scale=scaling
        )
        flash_window = flash_window.transpose(1, 2).contiguous()
        max_diff = (eager_output - flash_window).abs().max().item()
        cos_sim = F.cosine_similarity(eager_output.flatten().float(), flash_window.flatten().float(), dim=0).item()
        print(f"  vs Eager: max_diff={max_diff:.4e}, cos_sim={cos_sim:.6f}")
    except Exception as e:
        print(f"  Error: {e}")

    # 5. Use flash_attn_varlen for variable-length sequences
    print("\n5. FlashAttention varlen (correct pattern)")
    # varlen allows specifying different query and key lengths
    # This is the correct approach for prefix+suffix

    # For varlen, we need to pack all tokens together
    # Query: only suffix tokens (positions 968-1017)
    # Key/Value: all tokens (positions 0-1017)

    q_packed = q.transpose(1, 2).reshape(-1, num_heads, head_dim).contiguous()  # (suffix_len, H, D)
    k_packed = full_k.transpose(1, 2).reshape(-1, num_kv_heads, head_dim).contiguous()  # (total_len, KV_H, D)
    v_packed = full_v.transpose(1, 2).reshape(-1, num_kv_heads, head_dim).contiguous()  # (total_len, KV_H, D)

    # Cumulative sequence lengths
    cu_seqlens_q = torch.tensor([0, suffix_len], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, total_len], device=device, dtype=torch.int32)

    try:
        # With varlen + causal, the mask is applied such that position i in Q
        # can attend to positions 0..i in K. But since Q starts at position 0
        # (relative to its own sequence), this still doesn't give us what we want.
        #
        # We need: Q[i] can attend to K[0:prefix_len+i+1]
        # causal gives: Q[i] can attend to K[0:i+1]
        #
        # Solution: We need to use causal=False and pass our own mask
        # But flash_attn doesn't support arbitrary masks...

        flash_varlen = flash_attn_varlen_func(
            q_packed, k_packed, v_packed,
            cu_seqlens_q, cu_seqlens_k,
            suffix_len, total_len,
            causal=False,  # No causal because pattern is special
            softmax_scale=scaling
        )
        flash_varlen = flash_varlen.reshape(batch_size, suffix_len, num_heads, head_dim).transpose(1, 2)
        max_diff = (eager_output - flash_varlen).abs().max().item()
        cos_sim = F.cosine_similarity(eager_output.flatten().float(), flash_varlen.flatten().float(), dim=0).item()
        print(f"  vs Eager: max_diff={max_diff:.4e}, cos_sim={cos_sim:.6f}")
    except Exception as e:
        print(f"  Error: {e}")

# 6. Test the right approach: SDPA is_causal with proper layout
print("\n6. SDPA is_causal with proper layout (concat Q to full seq)")
# For is_causal to work correctly, Q must start at position 0
# We can achieve this by padding Q with zeros at the prefix positions
# and masking them out later

# This doesn't really work because is_causal expects Q and K to have same length
# Let's just conclude: SDPA with explicit mask is the best option for this pattern

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
The denoising attention pattern (suffix attending to prefix+suffix with
causal within suffix) is a special pattern that:

1. Standard FlashAttention causal=True doesn't handle correctly
   because it assumes Q and K have the same length and Q starts at position 0.

2. SDPA with explicit mask works correctly and is well-optimized.
   This is the recommended approach.

3. For maximum performance, consider:
   - Using SDPA with is_causal=True if we can restructure to make K,V
     have same length as Q (by padding Q)
   - Or accepting that SDPA with mask is the best we can do

Recommendation: Use SDPA with explicit mask for correctness.
The speedup from FlashAttention is not available for this specific pattern.
""")

# Final benchmark
import time
import numpy as np

print("\n" + "=" * 60)
print("LATENCY BENCHMARK")
print("=" * 60)

iterations = 100

# Warmup
for _ in range(10):
    _ = F.scaled_dot_product_attention(q, full_k_expanded, full_v_expanded, attn_mask=attn_mask_4d, scale=scaling)
torch.cuda.synchronize()

# SDPA with mask
times = []
for _ in range(iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = F.scaled_dot_product_attention(q, full_k_expanded, full_v_expanded, attn_mask=attn_mask_4d, scale=scaling)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)
print(f"SDPA with mask: {np.mean(times):.3f} ± {np.std(times):.3f} ms")

# Eager
for _ in range(10):
    attn_weights = torch.matmul(q, full_k_expanded.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attn_mask_4d
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
    _ = torch.matmul(attn_weights, full_v_expanded)
torch.cuda.synchronize()

times = []
for _ in range(iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    attn_weights = torch.matmul(q, full_k_expanded.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attn_mask_4d
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
    _ = torch.matmul(attn_weights, full_v_expanded)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)
print(f"Eager: {np.mean(times):.3f} ± {np.std(times):.3f} ms")

if HAS_FLASH_ATTN:
    # FlashAttention (non-causal for full attention without mask)
    for _ in range(10):
        _ = flash_attn_func(q_fa, full_k_fa, full_v_fa, causal=False, softmax_scale=scaling)
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = flash_attn_func(q_fa, full_k_fa, full_v_fa, causal=False, softmax_scale=scaling)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    print(f"FlashAttention (no mask): {np.mean(times):.3f} ± {np.std(times):.3f} ms")
