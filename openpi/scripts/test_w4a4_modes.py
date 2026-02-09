#!/usr/bin/env python3
"""
对比 W4A4 的 Simulation 模式 vs CUTLASS 模式
"""

import torch
import torch.nn.functional as F
import sys
import json
import pathlib

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from safetensors.torch import load_file
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
from openpi.models_pytorch.nvfp4_mlp import replace_paligemma_mlp_with_nvfp4


def create_observation(device, dtype):
    return Observation(
        images={
            "base_0_rgb": torch.randn(1, 3, 224, 224, device=device, dtype=dtype),
            "left_wrist_0_rgb": torch.randn(1, 3, 224, 224, device=device, dtype=dtype),
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, device=device, dtype=dtype),
        },
        image_masks={
            "base_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(1, device=device, dtype=torch.bool),
        },
        state=torch.randn(1, 32, device=device, dtype=dtype),
        tokenized_prompt=torch.zeros(1, 200, device=device, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(1, 200, device=device, dtype=torch.bool),
    )


def main():
    print("=" * 70)
    print("W4A4 模式对比: Simulation vs CUTLASS")
    print("=" * 70)

    checkpoint_dir = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()
    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Load config
    with open(checkpoint_dir / "config.json") as f:
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

    state_dict = load_file(checkpoint_dir / "model.safetensors")
    observation = create_observation(device, dtype)

    # BF16 Baseline
    print("\n[1] BF16 Baseline...")
    model_bf16 = PI0Pytorch(pi0_config)
    model_bf16.load_state_dict(state_dict, strict=False)
    model_bf16 = model_bf16.to(device='cuda')
    model_bf16.eval()

    torch.manual_seed(42)
    with torch.no_grad():
        actions_bf16 = model_bf16.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
    print(f"  BF16: {actions_bf16.shape}")
    del model_bf16
    torch.cuda.empty_cache()

    # W4A4 Simulation
    print("\n[2] W4A4 Simulation Mode...")
    model_sim = PI0Pytorch(pi0_config)
    model_sim.load_state_dict(state_dict, strict=False)
    model_sim = model_sim.to(device='cuda')
    model_sim.eval()

    replaced = replace_paligemma_mlp_with_nvfp4(model_sim, use_cutlass=False)

    torch.manual_seed(42)
    with torch.no_grad():
        actions_sim = model_sim.sample_actions(device, observation, num_steps=3, use_kv_cache=True)

    has_nan_sim = torch.isnan(actions_sim).any().item()
    if has_nan_sim:
        print("  Simulation: NaN!")
        cos_sim_sim = float('nan')
    else:
        cos_sim_sim = F.cosine_similarity(
            actions_sim.flatten().float().unsqueeze(0),
            actions_bf16.flatten().float().unsqueeze(0)
        ).item()
        print(f"  Simulation: cos_sim = {cos_sim_sim:.6f}")

    del model_sim
    torch.cuda.empty_cache()

    # W4A4 CUTLASS
    print("\n[3] W4A4 CUTLASS Mode...")
    model_cutlass = PI0Pytorch(pi0_config)
    model_cutlass.load_state_dict(state_dict, strict=False)
    model_cutlass = model_cutlass.to(device='cuda')
    model_cutlass.eval()

    replaced = replace_paligemma_mlp_with_nvfp4(model_cutlass, use_cutlass=True)

    # Prepare CUTLASS weights
    paligemma_lm = model_cutlass.paligemma_with_expert.paligemma.language_model
    for layer in paligemma_lm.layers:
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(mlp, proj_name)
                if hasattr(proj, 'prepare_for_cutlass'):
                    proj.use_cutlass = True
                    proj.prepare_for_cutlass()

    torch.manual_seed(42)
    with torch.no_grad():
        try:
            actions_cutlass = model_cutlass.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
            has_nan_cutlass = torch.isnan(actions_cutlass).any().item()
        except Exception as e:
            print(f"  CUTLASS error: {e}")
            has_nan_cutlass = True
            actions_cutlass = None

    if has_nan_cutlass:
        print("  CUTLASS: NaN!")
        cos_sim_cutlass = float('nan')
    else:
        cos_sim_cutlass = F.cosine_similarity(
            actions_cutlass.flatten().float().unsqueeze(0),
            actions_bf16.flatten().float().unsqueeze(0)
        ).item()
        print(f"  CUTLASS: cos_sim = {cos_sim_cutlass:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n{'Mode':<20} {'Cosine Sim':<15} {'Status':<10}")
    print("-" * 50)
    print(f"{'BF16 (baseline)':<20} {'1.0000':<15} {'OK':<10}")
    print(f"{'W4A4 Simulation':<20} {cos_sim_sim if not has_nan_sim else 'NaN':<15} {'NaN' if has_nan_sim else 'OK':<10}")
    print(f"{'W4A4 CUTLASS':<20} {cos_sim_cutlass if not has_nan_cutlass else 'NaN':<15} {'NaN' if has_nan_cutlass else 'OK':<10}")
    print("-" * 50)

    if not has_nan_sim and not has_nan_cutlass:
        print(f"\nSimulation vs CUTLASS 差异: {abs(cos_sim_sim - cos_sim_cutlass):.6f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
