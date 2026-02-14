#!/usr/bin/env python3
"""Debug TRT loop step-by-step comparison."""

import sys
import os
import torch
import torch.nn.functional as F
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.backends.cudnn.enabled = False

CHECKPOINT_DIR = os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero")


def load_original_model():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from safetensors.torch import load_file

    checkpoint_dir = Path(CHECKPOINT_DIR)
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
    else:
        model_config = {}

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
    weights_path = checkpoint_dir / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device="cuda")
    model.eval()

    return model


def compare_tensors(name, a, b, threshold=0.99):
    cos_sim = F.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0)
    ).item()
    max_diff = (a.float() - b.float()).abs().max().item()
    status = "✅" if cos_sim > threshold else "❌"
    print(f"  {status} {name}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}")
    return cos_sim > threshold


def main():
    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
    from denoise_torch_trt_static import StaticDenoiseStep, load_weights_from_checkpoint

    device = "cuda"
    batch_size = 1
    action_horizon = 50
    action_dim = 32
    prefix_len = 968
    num_steps = 10
    dt = -1.0 / num_steps

    print("=" * 70)
    print("Debug: TRT Loop Step-by-Step Comparison")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    orig_model = load_original_model()

    trt_step = StaticDenoiseStep(
        batch_size=batch_size,
        action_horizon=action_horizon,
        action_dim=action_dim,
        prefix_len=prefix_len,
    ).to(device).half()
    load_weights_from_checkpoint(trt_step, CHECKPOINT_DIR, device)
    trt_step.eval()

    # Create identical inputs
    torch.manual_seed(42)
    initial_noise = torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=torch.bfloat16)

    torch.manual_seed(123)
    prefix_kv_cache = []
    for _ in range(18):
        k = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        prefix_kv_cache.append((k, v))

    valid_tokens = 560
    prefix_pad_masks = torch.zeros(batch_size, prefix_len, dtype=torch.bool, device=device)
    prefix_pad_masks[:, :valid_tokens] = True

    state = torch.randn(batch_size, action_dim, device=device, dtype=torch.bfloat16)

    # Prepare TRT common inputs
    cached_keys = torch.stack([kv[0] for kv in prefix_kv_cache], dim=0)
    cached_values = torch.stack([kv[1] for kv in prefix_kv_cache], dim=0)

    # No attention mask needed - original model doesn't use mask in denoise_step_with_cache

    suffix_pad_masks = torch.ones(batch_size, action_horizon, device=device, dtype=torch.bool)
    prefix_offsets = torch.sum(prefix_pad_masks.long(), dim=-1, keepdim=True)
    suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1

    # Initialize
    x_orig = initial_noise.clone()
    x_trt = initial_noise.clone().half()

    print("\n" + "=" * 70)
    print("Running both loops step-by-step...")
    print("=" * 70)

    model_dtype = orig_model.action_in_proj.weight.dtype

    for step in range(num_steps):
        timestep_val = 1.0 + step * dt
        timestep = torch.tensor([timestep_val], device=device, dtype=torch.float32)

        print(f"\n--- Step {step} (timestep={timestep_val:.2f}) ---")

        # Check x_t before step
        compare_tensors(f"x_t (input)", x_orig, x_trt)

        # Compute adarms_cond for this step
        time_emb = create_sinusoidal_pos_embedding(
            timestep, orig_model.action_in_proj.out_features,
            min_period=4e-3, max_period=4.0,
            device=torch.device(device)
        ).to(model_dtype)

        with torch.no_grad():
            x = orig_model.time_mlp_in(time_emb)
            x = F.silu(x)
            x = orig_model.time_mlp_out(x)
            adarms_cond = F.silu(x)

        # Original step
        with torch.no_grad():
            v_orig = orig_model.denoise_step_with_cache(
                state, prefix_kv_cache, prefix_pad_masks, x_orig, timestep
            )

        # TRT step (no mask - matching original model)
        with torch.no_grad():
            v_trt = trt_step(
                x_trt,
                suffix_position_ids,
                adarms_cond.half(),
                cached_keys.half(),
                cached_values.half(),
            )

        # Compare velocity outputs
        compare_tensors(f"v_t (velocity)", v_orig, v_trt)

        # Update
        x_orig = x_orig + dt * v_orig
        x_trt = x_trt + dt * v_trt

        # Check x_t after step
        compare_tensors(f"x_t (output)", x_orig, x_trt)

        print(f"  Original x_t mean: {x_orig.float().mean().item():.6f}")
        print(f"  TRT x_t mean:      {x_trt.float().mean().item():.6f}")

    print("\n" + "=" * 70)
    print("Final comparison")
    print("=" * 70)
    compare_tensors("Final output", x_orig, x_trt)


if __name__ == "__main__":
    main()
