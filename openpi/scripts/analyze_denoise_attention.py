#!/usr/bin/env python3
"""
Analyze the attention pattern in denoise_step.

Goal: Determine if we can use is_causal=True for Flash Attention.

Author: Turbo-Pi Team
Date: 2026-02-13
"""

import sys
import torch

# Check attention mask structure
def analyze_denoise_attention_mask():
    """Analyze the attention mask structure in denoise_step."""

    print("=" * 60)
    print("Analyzing Denoise Step Attention Pattern")
    print("=" * 60)

    # Simulate denoise_step attention mask construction
    # From pi0_pytorch.py:878-888

    batch_size = 1
    prefix_len = 1050  # Typical prefix length (vision + language tokens)
    suffix_len = 83    # Action horizon + state + time tokens

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create masks as in pi0_pytorch.py
    prefix_pad_masks = torch.ones(batch_size, prefix_len, device=device, dtype=torch.bool)
    suffix_pad_masks = torch.ones(batch_size, suffix_len, device=device, dtype=torch.bool)
    suffix_att_masks = torch.ones(batch_size, suffix_len, suffix_len, device=device, dtype=torch.bool)

    # Line 878-882
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = suffix_att_masks  # (B, suffix_len, suffix_len) - should be causal for DiT

    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    print(f"\nMask shapes:")
    print(f"  prefix_pad_2d_masks: {prefix_pad_2d_masks.shape}")
    print(f"  suffix_att_2d_masks: {suffix_att_2d_masks.shape}")
    print(f"  full_att_2d_masks:   {full_att_2d_masks.shape}")

    print(f"\nAttention pattern:")
    print(f"  - Query length: {suffix_len} (suffix only)")
    print(f"  - Key/Value length: {prefix_len + suffix_len} (KV cache + suffix)")

    # Analyze the pattern
    print(f"\nMask analysis:")
    print(f"  - suffix→prefix: ALL TRUE (suffix attends to all prefix tokens)")
    print(f"  - suffix→suffix: {suffix_att_2d_masks[0].sum().item()} / {suffix_len * suffix_len}")

    # Check if suffix→suffix is causal
    causal_mask = torch.tril(torch.ones(suffix_len, suffix_len, device=device, dtype=torch.bool))
    is_suffix_causal = torch.all(suffix_att_2d_masks[0] == causal_mask).item()

    print(f"  - suffix→suffix is causal: {is_suffix_causal}")

    print("\n" + "-" * 60)
    print("CONCLUSION")
    print("-" * 60)

    if is_suffix_causal:
        print("""
  The denoise attention pattern is:
    [suffix queries] attend to [prefix KV cache | suffix KV]

  This is NOT pure causal - suffix can attend to ALL prefix tokens.

  However, we can optimize using:
    1. SDPA with causal mask on (suffix→suffix) portion
    2. Full attention on (suffix→prefix) portion

  OR use Flash Attention 2 with:
    - flash_attn_varlen_func() with custom cu_seqlens
""")
    else:
        print("""
  The suffix→suffix attention is NOT purely causal.
  Need to investigate the actual mask structure.
""")

    return full_att_2d_masks


def test_sdpa_with_kv_cache():
    """Test SDPA performance with KV cache pattern."""

    print("\n" + "=" * 60)
    print("Testing SDPA with KV Cache Pattern")
    print("=" * 60)

    import torch.nn.functional as F
    import time

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch_size = 1
    num_heads = 8
    head_dim = 256
    prefix_len = 1050  # KV cache from prefix
    suffix_len = 83    # Current queries

    # Q: only suffix
    query = torch.randn(batch_size, num_heads, suffix_len, head_dim, device=device, dtype=dtype)

    # K,V: prefix (from KV cache) + suffix
    kv_len = prefix_len + suffix_len
    key = torch.randn(batch_size, num_heads, kv_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, kv_len, head_dim, device=device, dtype=dtype)

    # Attention mask: suffix attends to all prefix + causal suffix
    # Shape: (suffix_len, kv_len)
    attn_mask = torch.zeros(suffix_len, kv_len, device=device, dtype=dtype)
    # suffix→prefix: all allowed (zeros = no masking)
    # suffix→suffix: causal
    suffix_start = prefix_len
    for i in range(suffix_len):
        # Can attend to all prefix + suffix[:i+1]
        attn_mask[i, suffix_start + i + 1:] = float("-inf")

    # Expand to 4D
    attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0)

    warmup, runs = 20, 100

    # 1. SDPA with explicit mask
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask_4d)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask_4d)
    torch.cuda.synchronize()
    sdpa_mask_ms = (time.perf_counter() - start) / runs * 1000

    print(f"\n  SDPA (with mask): {sdpa_mask_ms:.4f} ms")

    # 2. Eager attention (baseline)
    def eager_attention(q, k, v, mask):
        scale = head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    for _ in range(warmup):
        _ = eager_attention(query, key, value, attn_mask_4d)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = eager_attention(query, key, value, attn_mask_4d)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - start) / runs * 1000

    print(f"  Eager attention:  {eager_ms:.4f} ms")
    print(f"\n  SDPA vs Eager: {eager_ms / sdpa_mask_ms:.2f}x")

    # Estimate for 18 layers × 10 steps
    print(f"\n  18 layers × 10 steps:")
    print(f"    SDPA:  {sdpa_mask_ms * 18 * 10:.2f} ms")
    print(f"    Eager: {eager_ms * 18 * 10:.2f} ms")
    print(f"    Savings: {(eager_ms - sdpa_mask_ms) * 18 * 10:.2f} ms")


def main():
    analyze_denoise_attention_mask()
    test_sdpa_with_kv_cache()


if __name__ == "__main__":
    main()
