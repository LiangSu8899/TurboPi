#!/usr/bin/env python3
"""
Analyze sample_actions() breakdown.

Find where 171ms "Other" time is spent.
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import torch
import numpy as np


def analyze_model_structure():
    """Analyze model structure and W4A16 replacement."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from openpi.utils.model_patcher import patch_paligemma_decode_path
    from safetensors.torch import load_file
    import json

    device = 'cuda'
    checkpoint_path = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()

    # Load config
    with open(checkpoint_path / "config.json") as f:
        model_config = json.load(f)

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=model_config.get("action_dim", 32),
        action_horizon=model_config.get("action_horizon", 50),
        max_token_len=model_config.get("tokenizer_max_length", 200),
        pi05=True,
        dtype="bfloat16",
    )

    model = PI0Pytorch(pi0_config)
    weights_path = checkpoint_path / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("=" * 70)
    print("MODEL STRUCTURE ANALYSIS")
    print("=" * 70)

    # Count layers before patching
    paligemma = model.paligemma_with_expert.paligemma
    expert = model.paligemma_with_expert.action_expert

    print(f"\n1. PaliGemma LM layers: {len(paligemma.language_model.layers)}")
    print(f"   Each layer MLP has:")
    mlp = paligemma.language_model.layers[0].mlp
    print(f"     - gate_proj: {mlp.gate_proj}")
    print(f"     - up_proj:   {mlp.up_proj}")
    print(f"     - down_proj: {mlp.down_proj}")

    print(f"\n2. Action Expert layers: {len(expert.layers)}")
    expert_mlp = expert.layers[0].mlp
    print(f"   Each layer MLP has:")
    print(f"     - gate_proj: {expert_mlp.gate_proj}")
    print(f"     - up_proj:   {expert_mlp.up_proj}")
    print(f"     - down_proj: {expert_mlp.down_proj}")

    # Apply W4A16 and count
    print("\n3. Applying W4A16 quantization...")
    stats = patch_paligemma_decode_path(model, verbose=False)
    print(f"   Replaced: {stats['replaced']} layers")

    # Check what was replaced
    from openpi.modules.w4a16_linear import W4A16Linear

    paligemma_w4a16 = 0
    expert_w4a16 = 0

    for layer in paligemma.language_model.layers:
        if isinstance(layer.mlp.gate_proj, W4A16Linear):
            paligemma_w4a16 += 3  # gate, up, down

    for layer in expert.layers:
        if isinstance(layer.mlp.gate_proj, W4A16Linear):
            expert_w4a16 += 3

    print(f"\n4. W4A16 distribution:")
    print(f"   PaliGemma MLP: {paligemma_w4a16} Linear layers")
    print(f"   Action Expert MLP: {expert_w4a16} Linear layers")
    print(f"   Total: {paligemma_w4a16 + expert_w4a16} Linear layers")


def profile_sample_actions_breakdown():
    """Profile sample_actions with fine-grained timing."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
    from openpi.utils.model_patcher import patch_paligemma_decode_path
    from safetensors.torch import load_file
    import json

    device = 'cuda'
    checkpoint_path = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()

    # Load model
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

    model = PI0Pytorch(pi0_config)
    state_dict = load_file(checkpoint_path / "model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Apply W4A16
    stats = patch_paligemma_decode_path(model, verbose=False)

    # Create observation
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

    print("\n" + "=" * 70)
    print("SAMPLE_ACTIONS BREAKDOWN (num_steps=3)")
    print("=" * 70)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
    torch.cuda.synchronize()

    # Profile with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Test different num_steps
    for num_steps in [1, 3, 5, 10]:
        times = []
        for _ in range(5):
            start.record()
            with torch.no_grad():
                _ = model.sample_actions(device, observation, num_steps=num_steps, use_kv_cache=True)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        avg = np.mean(times)
        print(f"  num_steps={num_steps}: {avg:.2f} ms ({1000/avg:.2f} Hz)")

    # Calculate per-step time
    print("\n  Analysis:")
    t1 = 0
    t3 = 0
    t5 = 0
    for _ in range(5):
        start.record()
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=1, use_kv_cache=True)
        end.record()
        torch.cuda.synchronize()
        t1 += start.elapsed_time(end)

        start.record()
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        end.record()
        torch.cuda.synchronize()
        t3 += start.elapsed_time(end)

        start.record()
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=5, use_kv_cache=True)
        end.record()
        torch.cuda.synchronize()
        t5 += start.elapsed_time(end)

    t1 /= 5
    t3 /= 5
    t5 /= 5

    # Base time (vision + prefix + 1 denoise)
    base_time = t1
    # Per-step denoise time
    step_time = (t3 - t1) / 2  # 3 steps - 1 step = 2 extra steps
    step_time2 = (t5 - t3) / 2  # 5 steps - 3 steps = 2 extra steps

    print(f"    Base time (1 step):     {base_time:.2f} ms")
    print(f"    Per denoise step:       {step_time:.2f} ms (from 1->3 steps)")
    print(f"    Per denoise step:       {step_time2:.2f} ms (from 3->5 steps)")
    print(f"    Vision+Prefix estimate: {base_time - step_time:.2f} ms")

    # What takes time in each denoise step?
    print("\n  Denoise step breakdown (estimated):")
    print(f"    PaliGemma 18 layers:  ~{18 * 0.814:.1f} ms (MLP only)")
    print(f"    Expert 18 layers:     ~{18 * 0.814:.1f} ms (MLP only)")
    print(f"    Attention overhead:   ~{step_time - 2*18*0.814:.1f} ms")


if __name__ == "__main__":
    analyze_model_structure()
    profile_sample_actions_breakdown()
