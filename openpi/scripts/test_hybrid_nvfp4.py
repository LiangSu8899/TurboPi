#!/usr/bin/env python3
"""
测试混合精度方案：只对前 N 层使用 NVFP4，后面层保持 BF16

找出最佳的层数阈值，在精度和速度之间取得平衡。
"""

import torch
import torch.nn.functional as F
import sys
import json
import pathlib

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from safetensors.torch import load_file
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
from openpi.models_pytorch.nvfp4_mlp import NVFP4Linear, NVFP4MLP, BLOCK_SIZE


def replace_partial_mlp_with_nvfp4(model, num_layers_to_replace=10, use_cutlass=True):
    """
    只替换前 num_layers_to_replace 层的 MLP 为 NVFP4
    后面的层保持 BF16
    """
    replaced_count = 0

    paligemma_lm = model.paligemma_with_expert.paligemma.language_model

    for layer_idx, layer in enumerate(paligemma_lm.layers):
        if layer_idx >= num_layers_to_replace:
            # 保持 BF16
            continue

        if hasattr(layer, 'mlp'):
            original_mlp = layer.mlp
            # 创建 NVFP4 MLP
            nvfp4_mlp = NVFP4MLP.from_gemma_mlp(original_mlp, BLOCK_SIZE, use_cutlass=use_cutlass)

            # Prepare CUTLASS weights
            if use_cutlass:
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    proj = getattr(nvfp4_mlp, proj_name)
                    if hasattr(proj, 'prepare_for_cutlass'):
                        proj.use_cutlass = True
                        proj.prepare_for_cutlass()

            # Replace
            layer.mlp = nvfp4_mlp
            replaced_count += 1

    return replaced_count


def test_hybrid_precision():
    print("=" * 70)
    print("测试混合精度方案")
    print("=" * 70)

    checkpoint_dir = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()
    device = torch.device('cuda')
    dtype = torch.bfloat16

    # First load BF16 baseline
    print("\n[1] Loading BF16 baseline...")
    config_path = checkpoint_dir / "config.json"
    with open(config_path) as f:
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

    model_bf16 = PI0Pytorch(pi0_config)
    state_dict = load_file(checkpoint_dir / "model.safetensors")
    model_bf16.load_state_dict(state_dict, strict=False)
    model_bf16 = model_bf16.to(device='cuda')
    model_bf16.eval()

    # Get BF16 baseline actions
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

    torch.manual_seed(42)
    with torch.no_grad():
        actions_bf16 = model_bf16.sample_actions(device, observation, num_steps=3, use_kv_cache=True)

    print(f"  BF16 actions: {actions_bf16.shape}")

    # Test different numbers of NVFP4 layers
    print("\n[2] Testing different NVFP4 layer counts...")
    print("-" * 70)
    print(f"{'Layers':>8s} | {'Cosine':>8s} | {'MAE':>8s} | {'Max Err':>8s} | Status")
    print("-" * 70)

    results = []

    for num_layers in [0, 4, 8, 10, 12, 14, 16, 18]:
        # Reload model
        model = PI0Pytorch(pi0_config)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device='cuda')
        model.eval()

        # Replace partial layers
        if num_layers > 0:
            replaced = replace_partial_mlp_with_nvfp4(model, num_layers, use_cutlass=True)
        else:
            replaced = 0

        # Get actions
        torch.manual_seed(42)
        with torch.no_grad():
            try:
                actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
                has_nan = torch.isnan(actions).any().item()
            except Exception as e:
                actions = None
                has_nan = True

        if actions is not None and not has_nan:
            cos_sim = F.cosine_similarity(
                actions.flatten().float().unsqueeze(0),
                actions_bf16.flatten().float().unsqueeze(0)
            ).item()
            mae = (actions - actions_bf16).abs().mean().item()
            max_err = (actions - actions_bf16).abs().max().item()
            status = "✓" if cos_sim > 0.8 else ("⚠" if cos_sim > 0.5 else "✗")
        else:
            cos_sim = float('nan')
            mae = float('nan')
            max_err = float('nan')
            status = "NaN"

        results.append({
            'num_layers': num_layers,
            'cos_sim': cos_sim,
            'mae': mae,
            'max_err': max_err,
        })

        print(f"{num_layers:>8d} | {cos_sim:>8.4f} | {mae:>8.4f} | {max_err:>8.4f} | {status}")

        # Clean up
        del model
        torch.cuda.empty_cache()

    print("-" * 70)

    # Find best configuration
    valid_results = [r for r in results if not (r['cos_sim'] != r['cos_sim'])]  # Filter NaN
    if valid_results:
        best = max(valid_results, key=lambda x: x['cos_sim'])
        print(f"\n最佳配置: {best['num_layers']} 层 NVFP4 (cos_sim={best['cos_sim']:.4f})")

        # Recommendation
        if best['cos_sim'] > 0.9:
            print("  推荐：可以使用此配置")
        elif best['cos_sim'] > 0.8:
            print("  可用：精度有轻微损失，建议实际测试")
        else:
            print("  不推荐：精度损失过大")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_hybrid_precision()
