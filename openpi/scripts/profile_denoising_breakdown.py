#!/usr/bin/env python3
"""
Profile Denoising Latency Breakdown.

Measures individual components:
1. QKV projections (GEMM)
2. RoPE computation
3. Attention (eager vs SDPA vs FlashAttention)
4. Output projection
5. MLP (gate + up + down)
6. LayerNorm

This helps identify optimization priorities.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent dir to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), "src")
sys.path.insert(0, src_dir)


@dataclass
class ProfileResult:
    """Profile result for a component."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    count: int

    def __str__(self):
        return f"{self.name}: {self.mean_ms:.3f} ± {self.std_ms:.3f} ms (min={self.min_ms:.3f}, max={self.max_ms:.3f})"


def benchmark_component(fn, warmup=10, iterations=100, name="Component"):
    """Benchmark a component function."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return ProfileResult(
        name=name,
        mean_ms=np.mean(times),
        std_ms=np.std(times),
        min_ms=np.min(times),
        max_ms=np.max(times),
        count=len(times),
    )


def profile_attention_variants(batch_size=1, seq_len=970, head_dim=256, num_heads=8, num_kv_heads=1):
    """Profile different attention implementations."""
    device = "cuda"
    dtype = torch.bfloat16

    logger.info("=" * 60)
    logger.info("Profiling Attention Variants")
    logger.info(f"Config: batch={batch_size}, seq={seq_len}, heads={num_heads}, kv_heads={num_kv_heads}, dim={head_dim}")
    logger.info("=" * 60)

    # Create test tensors
    # Q: (batch, num_heads, seq, head_dim)
    # K/V: (batch, num_kv_heads, seq, head_dim) - needs expansion for GQA
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

    # GQA expansion (1 -> 8 heads)
    num_kv_groups = num_heads // num_kv_heads
    k_expanded = k[:, :, None, :, :].expand(batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim)
    k_expanded = k_expanded.reshape(batch_size, num_heads, seq_len, head_dim)
    v_expanded = v[:, :, None, :, :].expand(batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim)
    v_expanded = v_expanded.reshape(batch_size, num_heads, seq_len, head_dim)

    # Attention mask (causal)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1) * -1e9
    causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq, seq)

    results = []

    # 1. Eager Attention (manual implementation)
    def eager_attention():
        scaling = head_dim ** -0.5
        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * scaling
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(attn_weights, v_expanded)

    results.append(benchmark_component(eager_attention, name="1. Eager Attention"))

    # 2. SDPA (PyTorch native)
    def sdpa_attention():
        return F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=head_dim ** -0.5,
        )

    results.append(benchmark_component(sdpa_attention, name="2. SDPA"))

    # 3. SDPA with is_causal (no explicit mask)
    def sdpa_causal():
        return F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=head_dim ** -0.5,
        )

    results.append(benchmark_component(sdpa_causal, name="3. SDPA (is_causal=True)"))

    # 4. FlashAttention 2 (if available)
    try:
        from flash_attn import flash_attn_func

        # FlashAttention expects (batch, seq, heads, dim)
        q_fa = q.transpose(1, 2).contiguous()  # (B, S, H, D)
        k_fa = k.transpose(1, 2).contiguous()  # (B, S, KV_H, D) - NOT expanded
        v_fa = v.transpose(1, 2).contiguous()  # (B, S, KV_H, D)

        def flash_attention():
            return flash_attn_func(q_fa, k_fa, v_fa, causal=True)

        results.append(benchmark_component(flash_attention, name="4. FlashAttention 2"))
    except ImportError:
        logger.warning("FlashAttention not available")

    # 5. FlashAttention with KV cache simulation
    try:
        from flash_attn import flash_attn_with_kvcache

        # Simulate prefix (cached) + suffix (new) attention
        prefix_len = 900
        suffix_len = 50

        q_suffix = torch.randn(batch_size, suffix_len, num_heads, head_dim, device=device, dtype=dtype)
        k_cache = torch.randn(batch_size, prefix_len + suffix_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size, prefix_len + suffix_len, num_kv_heads, head_dim, device=device, dtype=dtype)

        def flash_kv_cache():
            return flash_attn_with_kvcache(
                q_suffix, k_cache, v_cache,
                cache_seqlens=torch.tensor([prefix_len + suffix_len], device=device, dtype=torch.int32),
                causal=True,
            )

        results.append(benchmark_component(flash_kv_cache, name="5. FlashAttention KV Cache"))
    except (ImportError, Exception) as e:
        logger.warning(f"FlashAttention KV cache not available: {e}")

    return results


def profile_mlp(batch_size=1, seq_len=50, hidden_size=1024, mlp_dim=4096):
    """Profile MLP components."""
    device = "cuda"
    dtype = torch.bfloat16

    logger.info("=" * 60)
    logger.info("Profiling MLP Components")
    logger.info(f"Config: batch={batch_size}, seq={seq_len}, hidden={hidden_size}, mlp_dim={mlp_dim}")
    logger.info("=" * 60)

    # Create layers
    gate_proj = nn.Linear(hidden_size, mlp_dim, bias=False).to(device=device, dtype=dtype)
    up_proj = nn.Linear(hidden_size, mlp_dim, bias=False).to(device=device, dtype=dtype)
    down_proj = nn.Linear(mlp_dim, hidden_size, bias=False).to(device=device, dtype=dtype)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    results = []

    # 1. Gate projection
    def gate_fn():
        return gate_proj(x)
    results.append(benchmark_component(gate_fn, name="1. Gate Projection"))

    # 2. Up projection
    def up_fn():
        return up_proj(x)
    results.append(benchmark_component(up_fn, name="2. Up Projection"))

    # 3. Activation (GELU)
    gate_out = gate_proj(x)
    def gelu_fn():
        return F.gelu(gate_out, approximate='tanh')
    results.append(benchmark_component(gelu_fn, name="3. GELU Activation"))

    # 4. Element-wise multiplication
    up_out = up_proj(x)
    gelu_out = F.gelu(gate_out, approximate='tanh')
    def mul_fn():
        return gelu_out * up_out
    results.append(benchmark_component(mul_fn, name="4. Element-wise Mul"))

    # 5. Down projection
    combined = gelu_out * up_out
    def down_fn():
        return down_proj(combined)
    results.append(benchmark_component(down_fn, name="5. Down Projection"))

    # 6. Full MLP
    def full_mlp():
        g = F.gelu(gate_proj(x), approximate='tanh')
        u = up_proj(x)
        return down_proj(g * u)
    results.append(benchmark_component(full_mlp, name="6. Full MLP"))

    return results


def profile_full_denoise_step(checkpoint_dir: Optional[str] = None):
    """Profile a full denoising step if model is available."""
    device = "cuda"

    if checkpoint_dir is None:
        checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/checkpoints/pi05_1b_libero")

    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint not found at {checkpoint_dir}, skipping full denoise profile")
        return []

    logger.info("=" * 60)
    logger.info("Profiling Full Denoise Step")
    logger.info(f"Checkpoint: {checkpoint_dir}")
    logger.info("=" * 60)

    try:
        from openpi.models_pytorch.pi0_pytorch import Pi0PaligemmaPytorch, Pi0Config
        from openpi.models_pytorch.gemma_pytorch import load_pretrained_weights

        # Load model
        config = Pi0Config(
            action_dim=7,
            action_horizon=50,
            max_token_len=200,
            max_state_dim=32,
        )
        model = Pi0PaligemmaPytorch(config)
        load_pretrained_weights(model.paligemma_with_expert, checkpoint_dir)
        model = model.to(device).eval()

        # Create dummy inputs
        batch_size = 1
        prefix_len = 920
        action_horizon = 50
        action_dim = 7

        # Pre-compute prefix KV cache
        prefix_embs = torch.randn(batch_size, prefix_len, 2048, device=device, dtype=torch.bfloat16)
        prefix_pad_masks = torch.ones(batch_size, prefix_len, device=device, dtype=torch.bool)
        prefix_att_masks = torch.ones(batch_size, prefix_len, prefix_len, device=device, dtype=torch.bool)
        prefix_att_masks = torch.tril(prefix_att_masks)

        # Compute KV cache once
        with torch.no_grad():
            prefix_kv_cache = model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)

        # Create suffix inputs
        state = torch.randn(batch_size, 32, device=device, dtype=torch.float32)
        x_t = torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=torch.bfloat16)
        timestep = torch.tensor([0.5], device=device, dtype=torch.float32)

        results = []

        # Profile single denoise step
        def denoise_step():
            return model.denoise_step_with_cache(state, prefix_kv_cache, prefix_pad_masks, x_t, timestep)

        results.append(benchmark_component(denoise_step, warmup=5, iterations=50, name="Full Denoise Step"))

        # Profile denoising loop (10 steps)
        def denoise_loop():
            x = x_t.clone()
            dt = -1.0 / 10
            t = 1.0
            for _ in range(10):
                v = model.denoise_step_with_cache(state, prefix_kv_cache, prefix_pad_masks, x, torch.tensor([t], device=device))
                x = x + dt * v
                t += dt
            return x

        results.append(benchmark_component(denoise_loop, warmup=3, iterations=20, name="Denoise Loop (10 steps)"))

        return results

    except Exception as e:
        logger.error(f"Error profiling full denoise step: {e}")
        import traceback
        traceback.print_exc()
        return []


def profile_qkv_projections(batch_size=1, seq_len=50, hidden_size=1024, num_heads=8, num_kv_heads=1, head_dim=256):
    """Profile QKV projections."""
    device = "cuda"
    dtype = torch.bfloat16

    logger.info("=" * 60)
    logger.info("Profiling QKV Projections")
    logger.info(f"Config: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")
    logger.info("=" * 60)

    q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False).to(device=device, dtype=dtype)
    k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False).to(device=device, dtype=dtype)
    v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False).to(device=device, dtype=dtype)
    o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(device=device, dtype=dtype)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    results = []

    def q_fn():
        return q_proj(x)
    results.append(benchmark_component(q_fn, name="Q Projection"))

    def k_fn():
        return k_proj(x)
    results.append(benchmark_component(k_fn, name="K Projection"))

    def v_fn():
        return v_proj(x)
    results.append(benchmark_component(v_fn, name="V Projection"))

    def o_fn():
        return o_proj(torch.randn(batch_size, seq_len, num_heads * head_dim, device=device, dtype=dtype))
    results.append(benchmark_component(o_fn, name="O Projection"))

    def all_qkv():
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)
        return q, k, v
    results.append(benchmark_component(all_qkv, name="All QKV"))

    return results


def print_results(results: List[ProfileResult], title: str):
    """Print profile results."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    total = sum(r.mean_ms for r in results)
    for r in results:
        pct = (r.mean_ms / total * 100) if total > 0 else 0
        print(f"{r.name}: {r.mean_ms:.3f} ms ({pct:.1f}%)")

    print(f"{'=' * 60}")
    print(f"Total: {total:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="Profile Denoising Breakdown")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("DENOISING PERFORMANCE PROFILE")
    print("=" * 80)

    # Profile attention variants
    attn_results = profile_attention_variants()
    print_results(attn_results, "ATTENTION VARIANTS")

    # Profile MLP
    mlp_results = profile_mlp()
    print_results(mlp_results, "MLP COMPONENTS")

    # Profile QKV
    qkv_results = profile_qkv_projections()
    print_results(qkv_results, "QKV PROJECTIONS")

    # Profile full denoise step
    denoise_results = profile_full_denoise_step(args.checkpoint)
    if denoise_results:
        print_results(denoise_results, "FULL DENOISE STEP")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if attn_results:
        eager = next((r for r in attn_results if "Eager" in r.name), None)
        sdpa = next((r for r in attn_results if "SDPA" in r.name and "causal" not in r.name.lower()), None)
        flash = next((r for r in attn_results if "Flash" in r.name and "KV" not in r.name), None)

        if eager and sdpa:
            print(f"SDPA speedup over Eager: {eager.mean_ms / sdpa.mean_ms:.2f}x")
        if eager and flash:
            print(f"FlashAttn speedup over Eager: {eager.mean_ms / flash.mean_ms:.2f}x")
        if sdpa and flash:
            print(f"FlashAttn speedup over SDPA: {sdpa.mean_ms / flash.mean_ms:.2f}x")

    print("\n✅ Profile complete!")


if __name__ == "__main__":
    main()
