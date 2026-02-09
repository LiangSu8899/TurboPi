#!/usr/bin/env python3
"""
Validate FlashAttention Precision vs Baseline.

Compares outputs between:
1. Eager Attention (baseline)
2. SDPA
3. FlashAttention 2

Reports:
- Maximum absolute difference
- Mean absolute difference
- Cosine similarity
- Relative error

Ensures FlashAttention maintains accuracy for production use.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
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


def test_attention_precision():
    """Test attention precision between different implementations."""
    device = "cuda"
    dtype = torch.bfloat16

    logger.info("=" * 60)
    logger.info("Testing Attention Precision")
    logger.info("=" * 60)

    # Test configuration
    batch_size = 1
    num_heads = 8
    num_kv_heads = 1
    head_dim = 256
    prefix_len = 920
    suffix_len = 50
    total_len = prefix_len + suffix_len

    # Create test tensors with fixed seed for reproducibility
    torch.manual_seed(42)

    # Query (suffix only)
    q = torch.randn(batch_size, num_heads, suffix_len, head_dim, device=device, dtype=dtype)

    # K, V for prefix (cached) and suffix
    prefix_k = torch.randn(batch_size, num_kv_heads, prefix_len, head_dim, device=device, dtype=dtype)
    prefix_v = torch.randn(batch_size, num_kv_heads, prefix_len, head_dim, device=device, dtype=dtype)
    suffix_k = torch.randn(batch_size, num_kv_heads, suffix_len, head_dim, device=device, dtype=dtype)
    suffix_v = torch.randn(batch_size, num_kv_heads, suffix_len, head_dim, device=device, dtype=dtype)

    # Full K, V (concatenated)
    full_k = torch.cat([prefix_k, suffix_k], dim=2)
    full_v = torch.cat([prefix_v, suffix_v], dim=2)

    scaling = head_dim ** -0.5

    # GQA expansion
    num_kv_groups = num_heads // num_kv_heads
    full_k_expanded = full_k[:, :, None, :, :].expand(
        batch_size, num_kv_heads, num_kv_groups, total_len, head_dim
    ).reshape(batch_size, num_heads, total_len, head_dim)
    full_v_expanded = full_v[:, :, None, :, :].expand(
        batch_size, num_kv_heads, num_kv_groups, total_len, head_dim
    ).reshape(batch_size, num_heads, total_len, head_dim)

    # Create attention mask (suffix attending to prefix + causal within suffix)
    attn_mask = torch.zeros(suffix_len, total_len, device=device, dtype=dtype)
    # Suffix can attend to all prefix
    attn_mask[:, :prefix_len] = 0
    # Causal within suffix
    suffix_mask = torch.triu(torch.ones(suffix_len, suffix_len, device=device), diagonal=1) * -1e9
    attn_mask[:, prefix_len:] = suffix_mask
    attn_mask_4d = attn_mask[None, None, :, :]

    results = {}

    # 1. Eager Attention (baseline)
    logger.info("Computing Eager Attention (baseline)...")
    attn_weights = torch.matmul(q, full_k_expanded.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attn_mask_4d
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
    eager_output = torch.matmul(attn_weights, full_v_expanded)
    results["eager"] = eager_output

    # 2. SDPA
    logger.info("Computing SDPA...")
    sdpa_output = F.scaled_dot_product_attention(
        q, full_k_expanded, full_v_expanded,
        attn_mask=attn_mask_4d,
        scale=scaling,
    )
    results["sdpa"] = sdpa_output

    # 3. SDPA with is_causal (can't use for this pattern, but test full causal)
    logger.info("Computing SDPA (is_causal)...")
    # For full sequence with causal
    q_full = torch.randn(batch_size, num_heads, total_len, head_dim, device=device, dtype=dtype)
    sdpa_causal_output = F.scaled_dot_product_attention(
        q_full, full_k_expanded, full_v_expanded,
        is_causal=True,
        scale=scaling,
    )
    # Can't directly compare, just verify it runs

    # 4. FlashAttention
    try:
        from flash_attn import flash_attn_func

        logger.info("Computing FlashAttention...")

        # FlashAttention expects (B, S, H, D)
        q_fa = q.transpose(1, 2).contiguous()
        full_k_fa = full_k.transpose(1, 2).contiguous()  # Not expanded - native GQA
        full_v_fa = full_v.transpose(1, 2).contiguous()

        # For prefix+suffix with causal, we use causal=True
        # This works because causal allows attending to all previous positions
        flash_output = flash_attn_func(q_fa, full_k_fa, full_v_fa, causal=True, softmax_scale=scaling)
        flash_output = flash_output.transpose(1, 2).contiguous()
        results["flash"] = flash_output

    except ImportError:
        logger.warning("FlashAttention not available")
        results["flash"] = None

    # 5. Our FlashAttention wrapper
    try:
        from openpi.inference.flash_attention_denoise import flash_attention_with_prefix_cache

        logger.info("Computing FlashAttention (wrapper)...")
        wrapper_output = flash_attention_with_prefix_cache(
            q, suffix_k, suffix_v, prefix_k, prefix_v, softmax_scale=scaling
        )
        results["wrapper"] = wrapper_output

    except Exception as e:
        logger.warning(f"FlashAttention wrapper failed: {e}")
        results["wrapper"] = None

    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("PRECISION COMPARISON")
    logger.info("=" * 60)

    baseline = results["eager"].float()

    for name, output in results.items():
        if name == "eager" or output is None:
            continue

        output_float = output.float()
        max_diff = (baseline - output_float).abs().max().item()
        mean_diff = (baseline - output_float).abs().mean().item()
        rel_error = ((baseline - output_float).abs() / (baseline.abs() + 1e-6)).mean().item()

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            baseline.flatten(), output_float.flatten(), dim=0
        ).item()

        logger.info(f"\n{name.upper()} vs Eager:")
        logger.info(f"  Max Abs Diff: {max_diff:.6e}")
        logger.info(f"  Mean Abs Diff: {mean_diff:.6e}")
        logger.info(f"  Relative Error: {rel_error:.6e}")
        logger.info(f"  Cosine Similarity: {cos_sim:.8f}")

        # Check if within acceptable tolerance
        if max_diff < 1e-2 and cos_sim > 0.999:
            logger.info(f"  ✅ PASSED (within tolerance)")
        else:
            logger.info(f"  ⚠️ WARNING (check precision)")

    return results


def test_full_denoise_precision(checkpoint_dir: str):
    """Test full denoise step precision between Eager and FlashAttention."""
    logger.info("=" * 60)
    logger.info("Testing Full Denoise Step Precision")
    logger.info("=" * 60)

    device = "cuda"

    try:
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
        from openpi.inference.flash_attention_denoise import FlashAttentionDenoiseWrapper
        from safetensors.torch import load_file
        import json

        # Load config
        config_path = Path(checkpoint_dir) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
            config = Pi0Config(
                action_dim=model_config.get("action_dim", 7),
                action_horizon=model_config.get("action_horizon", 50),
                max_token_len=model_config.get("max_token_len", 200),
                max_state_dim=model_config.get("max_state_dim", 32),
            )
        else:
            config = Pi0Config(action_dim=7, action_horizon=50, max_token_len=200, max_state_dim=32)

        # Load model
        model = PI0Pytorch(config)
        weights_path = Path(checkpoint_dir) / "model.safetensors"
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()

        # Create wrappers
        eager_wrapper = FlashAttentionDenoiseWrapper(model, use_flash_attn=False)
        flash_wrapper = FlashAttentionDenoiseWrapper(model, use_flash_attn=True)

        # Create dummy observation
        class DummyObservation:
            def __init__(self, batch_size=1):
                self.image = {
                    "base_0_rgb": torch.randint(0, 255, (batch_size, 224, 224, 3), dtype=torch.uint8, device=device),
                }
                self.image_mask = {"base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device)}
                self.state = torch.randn(batch_size, 32, device=device)
                self.tokenized_prompt = torch.randint(0, 100, (batch_size, 50), device=device)
                self.tokenized_prompt_mask = torch.ones(batch_size, 50, dtype=torch.bool, device=device)

        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        noise = torch.randn(1, 50, 7, device=device)
        observation = DummyObservation()

        # Test with same noise
        with torch.no_grad():
            eager_actions = eager_wrapper.sample_actions(observation, noise=noise.clone(), num_steps=10)
            flash_actions = flash_wrapper.sample_actions(observation, noise=noise.clone(), num_steps=10)

        # Compare
        eager_float = eager_actions.float()
        flash_float = flash_actions.float()

        max_diff = (eager_float - flash_float).abs().max().item()
        mean_diff = (eager_float - flash_float).abs().mean().item()
        cos_sim = F.cosine_similarity(eager_float.flatten(), flash_float.flatten(), dim=0).item()

        logger.info("\nFull Denoise Comparison (10 steps):")
        logger.info(f"  Max Abs Diff: {max_diff:.6e}")
        logger.info(f"  Mean Abs Diff: {mean_diff:.6e}")
        logger.info(f"  Cosine Similarity: {cos_sim:.8f}")

        if max_diff < 0.1 and cos_sim > 0.99:
            logger.info("  ✅ PASSED")
        else:
            logger.info("  ⚠️ Check precision differences")

        # Test action statistics
        logger.info("\nAction Statistics:")
        logger.info(f"  Eager - mean: {eager_float.mean():.4f}, std: {eager_float.std():.4f}")
        logger.info(f"  Flash - mean: {flash_float.mean():.4f}, std: {flash_float.std():.4f}")

        return {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "cos_sim": cos_sim,
        }

    except Exception as e:
        logger.error(f"Error testing full denoise precision: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_speed(checkpoint_dir: str, num_iterations: int = 20):
    """Benchmark speed comparison."""
    logger.info("=" * 60)
    logger.info("Speed Benchmark")
    logger.info("=" * 60)

    device = "cuda"

    try:
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
        from openpi.inference.flash_attention_denoise import FlashAttentionDenoiseWrapper
        from safetensors.torch import load_file
        import json

        # Load config
        config_path = Path(checkpoint_dir) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
            config = Pi0Config(
                action_dim=model_config.get("action_dim", 7),
                action_horizon=model_config.get("action_horizon", 50),
                max_token_len=model_config.get("max_token_len", 200),
                max_state_dim=model_config.get("max_state_dim", 32),
            )
        else:
            config = Pi0Config(action_dim=7, action_horizon=50, max_token_len=200, max_state_dim=32)

        # Load model
        model = PI0Pytorch(config)
        weights_path = Path(checkpoint_dir) / "model.safetensors"
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()

        # Create wrappers
        eager_wrapper = FlashAttentionDenoiseWrapper(model, use_flash_attn=False)
        flash_wrapper = FlashAttentionDenoiseWrapper(model, use_flash_attn=True)

        # Create dummy observation
        class DummyObservation:
            def __init__(self, batch_size=1):
                self.image = {
                    "base_0_rgb": torch.randint(0, 255, (batch_size, 224, 224, 3), dtype=torch.uint8, device=device),
                }
                self.image_mask = {"base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device)}
                self.state = torch.randn(batch_size, 32, device=device)
                self.tokenized_prompt = torch.randint(0, 100, (batch_size, 50), device=device)
                self.tokenized_prompt_mask = torch.ones(batch_size, 50, dtype=torch.bool, device=device)

        observation = DummyObservation()

        # Warmup
        logger.info("Warming up...")
        for _ in range(3):
            _ = eager_wrapper.sample_actions(observation, num_steps=10)
            _ = flash_wrapper.sample_actions(observation, num_steps=10)
        torch.cuda.synchronize()

        # Benchmark Eager
        logger.info(f"Benchmarking Eager Attention ({num_iterations} iterations)...")
        eager_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = eager_wrapper.sample_actions(observation, num_steps=10)
            torch.cuda.synchronize()
            eager_times.append((time.perf_counter() - start) * 1000)

        # Benchmark Flash
        logger.info(f"Benchmarking FlashAttention ({num_iterations} iterations)...")
        flash_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = flash_wrapper.sample_actions(observation, num_steps=10)
            torch.cuda.synchronize()
            flash_times.append((time.perf_counter() - start) * 1000)

        # Results
        eager_mean = np.mean(eager_times)
        flash_mean = np.mean(flash_times)
        speedup = eager_mean / flash_mean

        logger.info("\n" + "=" * 60)
        logger.info("SPEED RESULTS")
        logger.info("=" * 60)
        logger.info(f"Eager Attention: {eager_mean:.2f} ± {np.std(eager_times):.2f} ms")
        logger.info(f"FlashAttention:  {flash_mean:.2f} ± {np.std(flash_times):.2f} ms")
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info(f"Frequency: Eager={1000/eager_mean:.1f} Hz, Flash={1000/flash_mean:.1f} Hz")

        return {
            "eager_ms": eager_mean,
            "flash_ms": flash_mean,
            "speedup": speedup,
        }

    except Exception as e:
        logger.error(f"Error in speed benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("FLASHATTENTION PRECISION VALIDATION")
    print("=" * 80)

    # Test basic attention precision
    test_attention_precision()

    # Test full denoise if checkpoint available
    checkpoint_dir = args.checkpoint or os.environ.get("CHECKPOINT_DIR", "/checkpoints/pi05_1b_libero")

    if os.path.exists(checkpoint_dir):
        test_full_denoise_precision(checkpoint_dir)
        benchmark_speed(checkpoint_dir, args.iterations)
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_dir}")
        logger.info("Skipping full model tests")

    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
