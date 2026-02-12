#!/usr/bin/env python3
"""Diagnose Denoise Performance Gap.

Investigates why denoise takes 40ms in hybrid sampler vs expected 29ms.

Potential causes:
1. KV Cache update overhead (copy from TRT/eager to graph buffers)
2. CUDA Graph not being replayed correctly
3. W4A16 kernel underperforming
4. Different prefix lengths affecting attention computation

Author: Claude Code
Date: 2026-02-12
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import json
from dataclasses import dataclass


def load_model_and_apply_w4a16():
    """Load PI0Pytorch model and apply W4A16 MLP quantization."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from openpi.utils.model_patcher import patch_paligemma_decode_path, patch_expert_decode_path
    from safetensors.torch import load_file

    device = 'cuda'
    checkpoint_path = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()

    with open(checkpoint_path / "config.json") as f:
        model_config = json.load(f)

    max_token_len = model_config.get("tokenizer_max_length", 200)

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=model_config.get("action_dim", 32),
        action_horizon=model_config.get("action_horizon", 50),
        max_token_len=max_token_len,
        pi05=True,
        dtype="bfloat16",
    )

    print("Loading model...")
    model = PI0Pytorch(pi0_config)
    state_dict = load_file(checkpoint_path / "model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("Applying W4A16 MLP quantization...")
    pg_stats = patch_paligemma_decode_path(model, quantize_attention=False, verbose=False)
    ex_stats = patch_expert_decode_path(model, quantize_attention=False, verbose=False)
    print(f"  PaliGemma: {pg_stats['replaced']} layers, Expert: {ex_stats['replaced']} layers")

    return model, pi0_config, device, max_token_len


def create_test_observation(device, max_token_len):
    """Create a test observation."""
    from openpi.models_pytorch.pi0_pytorch import Observation

    return Observation(
        images={
            "base_0_rgb": torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device=device),
            "left_wrist_0_rgb": torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device=device),
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, dtype=torch.bfloat16, device=device),
        },
        image_masks={
            "base_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
        },
        state=torch.randn(1, 32, dtype=torch.bfloat16, device=device),
        tokenized_prompt=torch.randint(0, 1000, (1, max_token_len), dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.ones(1, max_token_len, dtype=torch.bool, device=device),
    )


def benchmark_denoise_isolated(model, device, prefix_kv_cache, prefix_pad_masks, state, num_steps=3, num_iters=50):
    """Benchmark denoise loop in isolation (no KV cache update)."""
    from openpi.modules.static_denoise import StaticDenoiseLoop

    print("\n" + "=" * 70)
    print("TEST 1: StaticDenoiseLoop (Isolated, No KV Update)")
    print("=" * 70)

    # Create denoise loop
    graphed_denoise = StaticDenoiseLoop(
        model=model,
        prefix_kv_cache=prefix_kv_cache,
        prefix_pad_masks=prefix_pad_masks,
        num_steps=num_steps,
        batch_size=1,
        device=device,
        dtype=torch.bfloat16,
    )

    # Capture graph
    print("  Capturing CUDA Graph...")
    graphed_denoise.capture_graph(warmup_iters=5)

    # Create noise
    noise = model.sample_noise((1, model.config.action_horizon, model.config.action_dim), device)
    noise = noise.to(torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = graphed_denoise(state, noise)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = graphed_denoise(state, noise)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"\n  Denoise (Isolated):")
    print(f"    Mean: {mean_ms:.2f} ms")
    print(f"    Std:  {std_ms:.2f} ms")
    print(f"    Hz:   {1000/mean_ms:.1f}")

    return mean_ms, graphed_denoise


def benchmark_denoise_with_kv_update(graphed_denoise, prefix_kv_cache, state, noise, num_iters=50):
    """Benchmark denoise with KV cache update (simulating hybrid flow)."""
    print("\n" + "=" * 70)
    print("TEST 2: StaticDenoiseLoop (With KV Cache Update)")
    print("=" * 70)

    # Warmup
    for _ in range(10):
        # Simulate KV cache update
        for layer_idx in range(len(prefix_kv_cache)):
            new_k, new_v = prefix_kv_cache[layer_idx]
            old_k, old_v = graphed_denoise._denoise_step.prefix_kv_cache[layer_idx]
            old_k.copy_(new_k)
            old_v.copy_(new_v)
        _ = graphed_denoise(state, noise)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        # KV cache update
        for layer_idx in range(len(prefix_kv_cache)):
            new_k, new_v = prefix_kv_cache[layer_idx]
            old_k, old_v = graphed_denoise._denoise_step.prefix_kv_cache[layer_idx]
            old_k.copy_(new_k)
            old_v.copy_(new_v)
        # Denoise
        _ = graphed_denoise(state, noise)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"\n  Denoise (With KV Update):")
    print(f"    Mean: {mean_ms:.2f} ms")
    print(f"    Std:  {std_ms:.2f} ms")
    print(f"    Hz:   {1000/mean_ms:.1f}")

    return mean_ms


def benchmark_kv_update_only(prefix_kv_cache, graphed_denoise, num_iters=100):
    """Benchmark KV cache update overhead."""
    print("\n" + "=" * 70)
    print("TEST 3: KV Cache Update Only (Overhead Measurement)")
    print("=" * 70)

    # Warmup
    for _ in range(10):
        for layer_idx in range(len(prefix_kv_cache)):
            new_k, new_v = prefix_kv_cache[layer_idx]
            old_k, old_v = graphed_denoise._denoise_step.prefix_kv_cache[layer_idx]
            old_k.copy_(new_k)
            old_v.copy_(new_v)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for layer_idx in range(len(prefix_kv_cache)):
            new_k, new_v = prefix_kv_cache[layer_idx]
            old_k, old_v = graphed_denoise._denoise_step.prefix_kv_cache[layer_idx]
            old_k.copy_(new_k)
            old_v.copy_(new_v)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"\n  KV Update Only:")
    print(f"    Mean: {mean_ms:.2f} ms")
    print(f"    Std:  {std_ms:.2f} ms")
    print(f"    Layers: {len(prefix_kv_cache)}")

    # Print KV cache sizes
    total_bytes = 0
    for k, v in prefix_kv_cache:
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()
    print(f"    Total KV size: {total_bytes / 1024 / 1024:.2f} MB")

    return mean_ms


def benchmark_w4a16_kernel(model, device, num_iters=100):
    """Benchmark W4A16 kernel performance."""
    print("\n" + "=" * 70)
    print("TEST 4: W4A16 MLP Kernel Performance")
    print("=" * 70)

    paligemma_lm = model.paligemma_with_expert.paligemma.language_model
    mlp = paligemma_lm.layers[0].mlp

    # Test input (batch=1, seq=1)
    x = torch.randn(1, 1, 2048, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(20):
        _ = mlp(x)
    torch.cuda.synchronize()

    # Benchmark single MLP
    times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = mlp(x)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    print(f"\n  Single W4A16 MLP (seq=1):")
    print(f"    Mean: {mean_ms:.3f} ms")
    print(f"    Target: 0.125 ms")
    print(f"    Overhead: {(mean_ms - 0.125) / 0.125 * 100:.1f}%")

    # Benchmark all MLPs (18 PaliGemma + 18 Expert = 36 per step)
    all_mlps = []
    for layer in paligemma_lm.layers:
        all_mlps.append(layer.mlp)

    expert = model.paligemma_with_expert.gemma_expert.model
    x_expert = torch.randn(1, 1, 1024, dtype=torch.bfloat16, device=device)
    for layer in expert.layers:
        all_mlps.append((layer.mlp, x_expert))

    # Time 18 PaliGemma MLPs
    times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for mlp in paligemma_lm.layers:
            _ = mlp.mlp(x)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    print(f"\n  18 PaliGemma MLPs (seq=1):")
    print(f"    Mean: {mean_ms:.2f} ms")
    print(f"    Per layer: {mean_ms/18:.3f} ms")

    return mean_ms


def benchmark_eager_denoise(model, device, prefix_kv_cache, prefix_pad_masks, state, num_steps=3, num_iters=20):
    """Benchmark eager (non-graph) denoise for comparison."""
    print("\n" + "=" * 70)
    print("TEST 5: Eager Denoise (No CUDA Graph)")
    print("=" * 70)

    noise = model.sample_noise((1, model.config.action_horizon, model.config.action_dim), device)
    noise = noise.to(torch.bfloat16)

    # Warmup
    for _ in range(5):
        actions = noise.clone()
        for step in range(num_steps):
            timestep = torch.tensor([step / num_steps], dtype=torch.float32, device=device)
            actions = model.denoise_step_with_cache(
                prefix_kv_cache, prefix_pad_masks, state, actions, timestep
            )
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        actions = noise.clone()
        start.record()
        for step in range(num_steps):
            timestep = torch.tensor([step / num_steps], dtype=torch.float32, device=device)
            actions = model.denoise_step_with_cache(
                prefix_kv_cache, prefix_pad_masks, state, actions, timestep
            )
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"\n  Eager Denoise ({num_steps} steps):")
    print(f"    Mean: {mean_ms:.2f} ms")
    print(f"    Std:  {std_ms:.2f} ms")
    print(f"    Hz:   {1000/mean_ms:.1f}")

    return mean_ms


def run_diagnosis():
    """Run complete diagnosis."""
    print("=" * 70)
    print("DENOISE PERFORMANCE DIAGNOSIS")
    print("Expected: ~29ms, Observed: ~40ms in Hybrid Sampler")
    print("=" * 70)
    print()

    # Load model
    model, config, device, max_token_len = load_model_and_apply_w4a16()
    observation = create_test_observation(device, max_token_len)

    # Compute prefix KV cache
    print("\nComputing prefix KV cache...")
    import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
    observation_proc = _preprocessing.preprocess_observation_pytorch(observation, train=False)

    with torch.no_grad():
        images = list(observation_proc.images.values())
        img_masks = list(observation_proc.image_masks.values())
        lang_tokens = observation_proc.tokenized_prompt
        lang_masks = observation_proc.tokenized_prompt_mask

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_kv_cache = model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)

    prefix_len = prefix_embs.shape[1]
    print(f"  Prefix length: {prefix_len}")
    print(f"  KV cache layers: {len(prefix_kv_cache)}")
    print(f"  K shape: {prefix_kv_cache[0][0].shape}")

    state = observation.state

    # Run tests
    isolated_ms, graphed_denoise = benchmark_denoise_isolated(
        model, device, prefix_kv_cache, prefix_pad_masks, state
    )

    noise = model.sample_noise((1, model.config.action_horizon, model.config.action_dim), device)
    noise = noise.to(torch.bfloat16)

    with_update_ms = benchmark_denoise_with_kv_update(
        graphed_denoise, prefix_kv_cache, state, noise
    )

    kv_update_ms = benchmark_kv_update_only(
        prefix_kv_cache, graphed_denoise
    )

    w4a16_ms = benchmark_w4a16_kernel(model, device)

    eager_ms = benchmark_eager_denoise(
        model, device, prefix_kv_cache, prefix_pad_masks, state
    )

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print(f"""
   +------------------------------------+----------+
   | Test                               | Time(ms) |
   +------------------------------------+----------+
   | Denoise (Isolated, Graph)          | {isolated_ms:7.2f}  |
   | Denoise (With KV Update)           | {with_update_ms:7.2f}  |
   | KV Cache Update Only               | {kv_update_ms:7.2f}  |
   | 18 PaliGemma MLPs (seq=1)          | {w4a16_ms:7.2f}  |
   | Denoise (Eager, No Graph)          | {eager_ms:7.2f}  |
   +------------------------------------+----------+

   Analysis:
   - CUDA Graph savings: {eager_ms - isolated_ms:.1f}ms ({(eager_ms - isolated_ms)/eager_ms*100:.0f}% reduction)
   - KV Update overhead: {kv_update_ms:.2f}ms
   - Expected total: {isolated_ms + kv_update_ms:.1f}ms
   - Observed in Hybrid: ~40ms

   Root Cause:
""")

    # Determine root cause
    if kv_update_ms > 5:
        print("   - KV Cache update is significant overhead!")
    if isolated_ms > 35:
        print("   - CUDA Graph denoise itself is slower than expected")
    if with_update_ms - isolated_ms > kv_update_ms + 2:
        print("   - Additional overhead beyond KV update (sync issues?)")

    print()

    return {
        'isolated_ms': isolated_ms,
        'with_update_ms': with_update_ms,
        'kv_update_ms': kv_update_ms,
        'w4a16_ms': w4a16_ms,
        'eager_ms': eager_ms,
    }


if __name__ == "__main__":
    results = run_diagnosis()
