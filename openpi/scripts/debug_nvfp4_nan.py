#!/usr/bin/env python3
"""
Debug NVFP4 NaN 问题
"""

import torch
import torch.nn.functional as F
import sys
import json
import pathlib

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from safetensors.torch import load_file
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
from openpi.models_pytorch.nvfp4_mlp import replace_paligemma_mlp_with_nvfp4, NVFP4Linear


def check_tensor(name, tensor):
    """检查 tensor 是否有 NaN/Inf"""
    if tensor is None:
        return
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        print(f"  ⚠️  {name}: NaN={has_nan}, Inf={has_inf}, shape={tensor.shape}")
    else:
        print(f"  ✓ {name}: OK, range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}]")


def main():
    print("=" * 70)
    print("Debug NVFP4 NaN Issue")
    print("=" * 70)

    checkpoint_dir = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()
    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Load model
    print("\n[1] Loading model...")
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
    model = model.to(device='cuda')
    model.eval()

    # Check original weights
    print("\n[2] Checking original MLP weights...")
    paligemma_lm = model.paligemma_with_expert.paligemma.language_model
    layer0_mlp = paligemma_lm.layers[0].mlp
    check_tensor("gate_proj.weight", layer0_mlp.gate_proj.weight)
    check_tensor("up_proj.weight", layer0_mlp.up_proj.weight)
    check_tensor("down_proj.weight", layer0_mlp.down_proj.weight)

    # Replace with NVFP4
    print("\n[3] Replacing MLP with NVFP4...")
    replaced = replace_paligemma_mlp_with_nvfp4(model, use_cutlass=True)
    print(f"  Replaced {replaced} layers")

    # Prepare CUTLASS
    print("\n[4] Preparing CUTLASS weights...")
    for layer in paligemma_lm.layers:
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, proj_name):
                    proj = getattr(mlp, proj_name)
                    if hasattr(proj, 'prepare_for_cutlass'):
                        proj.use_cutlass = True
                        proj.prepare_for_cutlass()

    # Check NVFP4 weights
    print("\n[5] Checking NVFP4 layer 0 weights...")
    layer0_mlp = paligemma_lm.layers[0].mlp
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(layer0_mlp, proj_name)
        if isinstance(proj, NVFP4Linear):
            print(f"\n  {proj_name}:")
            check_tensor("    weight", proj.weight)
            check_tensor("    weight_q", proj.weight_q)
            check_tensor("    weight_scales", proj.weight_scales)
            check_tensor("    weight_packed", proj.weight_packed)
            check_tensor("    weight_scales_cutlass", proj.weight_scales_cutlass)

    # Test single layer forward
    print("\n[6] Testing single NVFP4 layer forward...")
    gate_proj = layer0_mlp.gate_proj
    test_input = torch.randn(1, 2048, device=device, dtype=torch.float32)
    check_tensor("test_input", test_input)

    with torch.no_grad():
        output = gate_proj(test_input)
    check_tensor("output", output)

    # Test with simulation mode
    print("\n[7] Testing with simulation mode (no CUTLASS)...")
    gate_proj.use_cutlass = False
    with torch.no_grad():
        output_sim = gate_proj(test_input)
    check_tensor("output_sim", output_sim)

    # Compare
    if not torch.isnan(output).any() and not torch.isnan(output_sim).any():
        cos_sim = F.cosine_similarity(
            output.flatten().float().unsqueeze(0),
            output_sim.flatten().float().unsqueeze(0)
        ).item()
        print(f"\n  CUTLASS vs Simulation: cos_sim={cos_sim:.6f}")

    # Test full model forward
    print("\n[8] Testing full model forward (BF16 baseline)...")
    # Temporarily disable NVFP4
    for layer in paligemma_lm.layers:
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(mlp, proj_name)
                if isinstance(proj, NVFP4Linear):
                    proj.use_cutlass = False

    observation = Observation(
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

    with torch.no_grad():
        actions_sim = model.sample_actions(device, observation, num_steps=1, use_kv_cache=True)
    check_tensor("actions (simulation mode)", actions_sim)

    # Test with CUTLASS
    print("\n[9] Testing full model forward (CUTLASS)...")
    for layer in paligemma_lm.layers:
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(mlp, proj_name)
                if isinstance(proj, NVFP4Linear):
                    proj.use_cutlass = True

    with torch.no_grad():
        actions_cutlass = model.sample_actions(device, observation, num_steps=1, use_kv_cache=True)
    check_tensor("actions (CUTLASS mode)", actions_cutlass)

    # Compare
    if not torch.isnan(actions_sim).any() and not torch.isnan(actions_cutlass).any():
        cos_sim = F.cosine_similarity(
            actions_sim.flatten().float().unsqueeze(0),
            actions_cutlass.flatten().float().unsqueeze(0)
        ).item()
        print(f"\n  CUTLASS vs Simulation: cos_sim={cos_sim:.6f}")
    else:
        print("\n  ❌ Cannot compare - one or both outputs contain NaN")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
