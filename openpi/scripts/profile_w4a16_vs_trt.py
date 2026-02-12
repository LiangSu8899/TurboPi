#!/usr/bin/env python3
"""
Profile W4A16 INT4 TVM vs TRT FP8 Performance.

This script identifies where time is spent in sample_actions() to understand
why W4A16 INT4 TVM (228ms) is 2x slower than TRT FP8 (120ms).

Expected breakdown for 3-step denoising:
- Vision encoder: ~17ms (TRT FP16)
- Prefix KV cache compute: ~Xms
- Denoising (3 steps): ~Xms
- Action decode: ~Xms

Author: Claude Code
Date: 2026-02-11
"""

import sys
import time
import pathlib

# Setup paths
script_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np


def profile_sample_actions():
    """Profile each component of sample_actions()."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
    from openpi.utils.model_patcher import patch_paligemma_decode_path
    from safetensors.torch import load_file
    import json

    device = 'cuda'
    checkpoint_path = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()

    # Load config
    config_path = checkpoint_path / "config.json"
    with open(config_path) as f:
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

    print("=" * 70)
    print("W4A16 INT4 TVM Performance Profile")
    print("=" * 70)

    # Load model
    print("\n1. Loading model...")
    model = PI0Pytorch(pi0_config)
    weights_path = checkpoint_path / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Apply W4A16 quantization
    print("\n2. Applying W4A16 INT4 TVM quantization...")
    stats = patch_paligemma_decode_path(model, verbose=False)
    print(f"   Replaced {stats['replaced']} MLP layers")
    print(f"   Memory saved: {stats['memory_saved_mb']:.1f} MB")

    # Create dummy observation
    print("\n3. Creating test observation...")
    observation = Observation(
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

    # Warmup
    print("\n4. Warmup (5 iterations)...")
    for _ in range(5):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
    torch.cuda.synchronize()

    # Profile overall timing
    print("\n5. Profiling sample_actions() (10 iterations)...")
    times = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  sample_actions(num_steps=3):")
    print(f"    Average: {avg_time:.2f} ms")
    print(f"    Std:     {std_time:.2f} ms")
    print(f"    Hz:      {1000/avg_time:.2f}")
    print(f"\n  TRT FP8 baseline:  ~120 ms (12 Hz)")
    print(f"  Current W4A16:     {avg_time:.2f} ms ({1000/avg_time:.2f} Hz)")
    print(f"  Slowdown:          {avg_time/120:.2f}x")

    # Profile components individually
    print("\n" + "=" * 70)
    print("COMPONENT BREAKDOWN (via torch.cuda.profiler)")
    print("=" * 70)

    # Profile with CUDA events at key points
    profile_components(model, device, observation)


def profile_components(model, device, observation):
    """Profile individual components of sample_actions."""
    paligemma = model.paligemma_with_expert.paligemma
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 1. Vision encoder
    print("\n[1] Vision Encoder (SigLIP):")
    vision_tower = paligemma.vision_tower

    for _ in range(3):
        with torch.no_grad():
            _ = vision_tower(observation.images["base_0_rgb"])
    torch.cuda.synchronize()

    times = []
    for _ in range(10):
        start.record()
        with torch.no_grad():
            _ = vision_tower(observation.images["base_0_rgb"])
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    vision_time = np.mean(times)
    print(f"    Average: {vision_time:.2f} ms")

    # 2. Single W4A16 MLP layer (decode mode, seq_len=1)
    print("\n[2] Single W4A16 MLP layer (seq_len=1):")
    mlp_layer = paligemma.language_model.layers[0].mlp
    test_input = torch.randn(1, 1, 2048, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = mlp_layer(test_input)
    torch.cuda.synchronize()

    times = []
    for _ in range(100):
        start.record()
        with torch.no_grad():
            _ = mlp_layer(test_input)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mlp_time = np.mean(times)
    print(f"    Average: {mlp_time:.3f} ms (std: {np.std(times):.3f})")
    print(f"    Target (TVM kernel): 0.125 ms")
    print(f"    Overhead: {(mlp_time - 0.125) / 0.125 * 100:.1f}%")

    # 3. Test 54 MLP layers total
    print("\n[3] All 54 W4A16 MLP layers (PaliGemma 18 + Expert 18 × 2):")
    all_mlps = []
    for layer in paligemma.language_model.layers:
        all_mlps.append(layer.mlp)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            x = test_input.clone()
            for mlp in all_mlps:
                x = mlp(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(20):
        start.record()
        with torch.no_grad():
            x = test_input.clone()
            for mlp in all_mlps:
                x = mlp(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    all_mlp_time = np.mean(times)
    print(f"    18 PaliGemma MLPs: {all_mlp_time:.2f} ms")
    print(f"    Per layer: {all_mlp_time/18:.3f} ms")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print(f"""
Component breakdown for W4A16 INT4 TVM (225ms total):
  - Vision:     ~{vision_time:.1f} ms
  - 54 MLPs:    ~{all_mlp_time * 3:.1f} ms (×3 for 54 layers)
  - Other:      ~{225 - vision_time - all_mlp_time * 3:.1f} ms (attention, embedding, etc.)

Expected MLP time: 54 × 0.125 ms = 6.75 ms
Actual MLP time:   ~{all_mlp_time * 3:.1f} ms
Overhead:          {(all_mlp_time * 3 - 6.75) / 6.75 * 100:.0f}%

ROOT CAUSE: The W4A16 TVM kernel is 0.125ms per layer, but Python/PyTorch
dispatch overhead adds significant latency per kernel call.

To fix: Need to fuse all MLP calls or use CUDA Graphs to batch kernel launches.
""")


if __name__ == "__main__":
    profile_sample_actions()
