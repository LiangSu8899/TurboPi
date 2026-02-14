#!/usr/bin/env python3
"""Test TRT 10-step denoise loop vs original model.

This verifies that the fixed TRT implementation produces similar outputs
to the original model over the full 10-step denoising process.
"""

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
    """Load the original PI0Pytorch model."""
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


def main():
    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
    from denoise_torch_trt_static import StaticDenoiseLoop, load_weights_from_checkpoint

    device = "cuda"
    batch_size = 1
    action_horizon = 50
    action_dim = 32
    prefix_len = 968
    num_steps = 10

    print("=" * 70)
    print("TRT 10-Step Denoise Loop Test")
    print("=" * 70)

    # Load original model
    print("\nLoading original model...")
    orig_model = load_original_model()

    # Create and load TRT module
    print("Creating TRT module...")
    trt_loop = StaticDenoiseLoop(
        batch_size=batch_size,
        action_horizon=action_horizon,
        action_dim=action_dim,
        prefix_len=prefix_len,
        num_steps=num_steps,
    ).to(device).half()
    load_weights_from_checkpoint(trt_loop, CHECKPOINT_DIR, device)
    trt_loop.eval()

    # Create identical inputs
    torch.manual_seed(42)
    initial_noise = torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=torch.bfloat16)

    # Create KV cache
    print("Creating KV cache...")
    torch.manual_seed(123)
    prefix_kv_cache = []
    for _ in range(18):
        k = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        prefix_kv_cache.append((k, v))

    # Masks
    valid_tokens = 560
    prefix_pad_masks = torch.zeros(batch_size, prefix_len, dtype=torch.bool, device=device)
    prefix_pad_masks[:, :valid_tokens] = True

    state = torch.randn(batch_size, action_dim, device=device, dtype=torch.bfloat16)

    # ============================================================
    # Run original model (10 steps)
    # ============================================================
    print("\n" + "=" * 70)
    print("Running Original Model (10 steps)...")
    print("=" * 70)

    dt = -1.0 / num_steps
    x_orig = initial_noise.clone()
    orig_step_outputs = []

    with torch.no_grad():
        for step in range(num_steps):
            timestep_val = 1.0 + step * dt
            timestep = torch.tensor([timestep_val], device=device, dtype=torch.float32)

            v_t = orig_model.denoise_step_with_cache(
                state, prefix_kv_cache, prefix_pad_masks, x_orig, timestep
            )
            x_orig = x_orig + dt * v_t
            orig_step_outputs.append(x_orig.clone())

            if step % 3 == 0 or step == num_steps - 1:
                print(f"  Step {step}: mean={x_orig.float().mean().item():.6f}, std={x_orig.float().std().item():.6f}")

    orig_final = x_orig.clone()

    # ============================================================
    # Run TRT model (10 steps)
    # ============================================================
    print("\n" + "=" * 70)
    print("Running TRT Model (10 steps)...")
    print("=" * 70)

    # Prepare TRT inputs
    # Pre-compute all adarms_conds
    model_dtype = orig_model.action_in_proj.weight.dtype
    adarms_conds = []
    for i in range(num_steps):
        timestep_val = 1.0 + i * dt
        timestep = torch.tensor([timestep_val], device=device, dtype=torch.float32)

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

        adarms_conds.append(adarms_cond)

    static_adarms_conds = torch.stack(adarms_conds, dim=0).half()

    # Position IDs
    prefix_offset = torch.sum(prefix_pad_masks.long(), dim=-1, keepdim=True)
    suffix_position_ids = prefix_offset + torch.arange(action_horizon, device=device, dtype=torch.long)

    # Stack KV cache for TRT format
    cached_keys = torch.stack([kv[0] for kv in prefix_kv_cache], dim=0)
    cached_values = torch.stack([kv[1] for kv in prefix_kv_cache], dim=0)

    # Note: No attention mask needed - matching original model (uses SDPA without mask)

    # Run TRT loop
    with torch.no_grad():
        trt_final = trt_loop(
            initial_noise.half(),
            suffix_position_ids,
            static_adarms_conds,
            cached_keys.half(),
            cached_values.half(),
        )

    print(f"  Final: mean={trt_final.float().mean().item():.6f}, std={trt_final.float().std().item():.6f}")

    # ============================================================
    # Compare results
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Original vs TRT")
    print("=" * 70)

    orig_flat = orig_final.float().flatten()
    trt_flat = trt_final.float().flatten()

    cos_sim = F.cosine_similarity(orig_flat.unsqueeze(0), trt_flat.unsqueeze(0)).item()
    max_diff = (orig_final.float() - trt_final.float()).abs().max().item()
    mean_diff = (orig_final.float() - trt_final.float()).abs().mean().item()

    print(f"\n  Original final mean: {orig_final.float().mean().item():.6f}")
    print(f"  TRT final mean:      {trt_final.float().mean().item():.6f}")
    print(f"\n  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max diff:          {max_diff:.6f}")
    print(f"  Mean diff:         {mean_diff:.6f}")

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    if cos_sim > 0.99:
        print(f"  ✅ EXCELLENT: cos_sim={cos_sim:.6f} > 0.99")
        print("     TRT model should work correctly!")
    elif cos_sim > 0.95:
        print(f"  ⚠️ GOOD: cos_sim={cos_sim:.6f} > 0.95")
        print("     TRT model should mostly work, minor accuracy loss expected")
    elif cos_sim > 0.8:
        print(f"  ⚠️ WARNING: cos_sim={cos_sim:.6f} > 0.8")
        print("     Some accuracy issues may occur")
    else:
        print(f"  ❌ FAILED: cos_sim={cos_sim:.6f} <= 0.8")
        print("     TRT model has significant issues")

    # Print first few values for debugging
    print(f"\n  First 5 values (Original): {orig_final[0, 0, :5].float().tolist()}")
    print(f"  First 5 values (TRT):      {trt_final[0, 0, :5].float().tolist()}")

    return 0 if cos_sim > 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
