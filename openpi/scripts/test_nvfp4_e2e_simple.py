#!/usr/bin/env python3
"""
简化的 NVFP4 端到端测试
直接比较 BF16 和 NVFP4 Packed 的推理精度和延迟
"""

import sys
import os
import time
import pathlib
import json

# Setup paths
for path in [
    os.path.join(os.path.dirname(__file__), "..", "src"),
    os.path.join(os.path.dirname(__file__), "..", "nvfp4_packed_plugin", "python"),
]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.nn.functional as F

# Disable cuDNN
torch.backends.cudnn.enabled = False

def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def load_model(checkpoint_dir):
    """Load PI0 model."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from safetensors.torch import load_file

    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()

    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
    else:
        model_config = {}

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

    model = PI0Pytorch(pi0_config)
    weights_path = checkpoint_path / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    return model, pi0_config


def create_dummy_observation(device):
    """Create dummy observation for testing."""
    from openpi.models_pytorch.pi0_pytorch import Observation

    batch_size = 1
    img_size = 224

    return Observation(
        images={
            "base_0_rgb": torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=torch.bfloat16),
            "left_wrist_0_rgb": torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=torch.bfloat16),
            "right_wrist_0_rgb": torch.zeros(batch_size, 3, img_size, img_size, device=device, dtype=torch.bfloat16),
        },
        image_masks={
            "base_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(batch_size, device=device, dtype=torch.bool),
        },
        state=torch.randn(batch_size, 32, device=device, dtype=torch.bfloat16),
        tokenized_prompt=torch.randint(0, 1000, (batch_size, 200), device=device, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(batch_size, 200, device=device, dtype=torch.bool),
    )


def benchmark_model(model, observation, device, name, warmup=10, runs=30):
    """Benchmark model inference."""
    flush_print(f"\n{'='*60}")
    flush_print(f"Benchmarking: {name}")
    flush_print(f"{'='*60}")

    # Warmup
    flush_print(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        with torch.no_grad():
            actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        if (i + 1) % 5 == 0:
            flush_print(f"  Warmup {i+1}/{warmup}")
    torch.cuda.synchronize()

    # Benchmark
    flush_print(f"Benchmarking ({runs} iterations)...")
    latencies = []
    actions_list = []
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        actions_list.append(actions.clone())
        if (i + 1) % 10 == 0:
            flush_print(f"  Run {i+1}/{runs}: {latencies[-1]:.2f} ms")

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    hz = 1000 / avg_latency

    flush_print(f"\nResults:")
    flush_print(f"  Average latency: {avg_latency:.2f} ms (std: {std_latency:.2f})")
    flush_print(f"  Throughput: {hz:.2f} Hz")

    return avg_latency, hz, actions_list


def main():
    flush_print("=" * 60)
    flush_print("NVFP4 Packed End-to-End Test (Simplified)")
    flush_print("=" * 60)

    device = torch.device('cuda')
    checkpoint_dir = "/root/.cache/openpi/checkpoints/pi05_libero"

    # Create observation once
    flush_print("\nCreating test observation...")
    observation = create_dummy_observation(device)

    # =========================================================================
    # Test 1: BF16 Baseline
    # =========================================================================
    flush_print("\n\n>>> Loading BF16 model...")
    model_bf16, _ = load_model(checkpoint_dir)
    bf16_latency, bf16_hz, bf16_actions = benchmark_model(
        model_bf16, observation, device, "BF16 Baseline", warmup=10, runs=30
    )
    del model_bf16
    torch.cuda.empty_cache()

    # =========================================================================
    # Test 2: NVFP4 Packed
    # =========================================================================
    flush_print("\n\n>>> Loading NVFP4 Packed model...")
    model_nvfp4, _ = load_model(checkpoint_dir)

    flush_print("Applying NVFP4 Packed quantization...")
    from nvfp4_packed import replace_mlp_with_nvfp4_packed
    replaced = replace_mlp_with_nvfp4_packed(model_nvfp4)
    flush_print(f"Replaced {replaced} MLP layers")

    nvfp4_latency, nvfp4_hz, nvfp4_actions = benchmark_model(
        model_nvfp4, observation, device, "NVFP4 Packed", warmup=10, runs=30
    )

    # =========================================================================
    # Compare Outputs
    # =========================================================================
    flush_print("\n" + "=" * 60)
    flush_print("Output Comparison (Last 5 runs)")
    flush_print("=" * 60)

    cos_sims = []
    for i in range(-5, 0):
        bf16_act = bf16_actions[i].flatten().float()
        nvfp4_act = nvfp4_actions[i].flatten().float()
        cos_sim = F.cosine_similarity(bf16_act.unsqueeze(0), nvfp4_act.unsqueeze(0)).item()
        max_diff = (bf16_actions[i] - nvfp4_actions[i]).abs().max().item()
        cos_sims.append(cos_sim)
        flush_print(f"  Run {i}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}")

    avg_cos_sim = np.mean(cos_sims)
    flush_print(f"\nAverage cosine similarity: {avg_cos_sim:.6f}")

    # =========================================================================
    # Summary
    # =========================================================================
    flush_print("\n" + "=" * 60)
    flush_print("SUMMARY")
    flush_print("=" * 60)
    flush_print(f"{'Method':<20} {'Latency (ms)':<15} {'Hz':<10}")
    flush_print("-" * 45)
    flush_print(f"{'BF16':<20} {bf16_latency:<15.2f} {bf16_hz:<10.2f}")
    flush_print(f"{'NVFP4 Packed':<20} {nvfp4_latency:<15.2f} {nvfp4_hz:<10.2f}")
    flush_print("-" * 45)

    if nvfp4_latency > 0:
        speedup = bf16_latency / nvfp4_latency
        if speedup > 1:
            flush_print(f"NVFP4 Speedup: {speedup:.2f}x FASTER")
        else:
            flush_print(f"NVFP4 Slowdown: {1/speedup:.2f}x SLOWER")

    flush_print(f"\nAction Output Similarity: {avg_cos_sim:.4f}")
    if avg_cos_sim > 0.99:
        flush_print("[PASS] High accuracy maintained")
    elif avg_cos_sim > 0.95:
        flush_print("[OK] Acceptable accuracy")
    else:
        flush_print("[WARN] Accuracy degradation detected")

    flush_print("\n" + "=" * 60)
    flush_print("Test Complete")
    flush_print("=" * 60)


if __name__ == "__main__":
    main()
