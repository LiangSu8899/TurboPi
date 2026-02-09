#!/usr/bin/env python3
"""
检查 MLP 激活值范围，看是否会触发 FP8 Scale Overflow
"""

import torch
import sys
import json
import pathlib

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from safetensors.torch import load_file
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation

# FP8 Scale 限制
FP8_MAX = 448.0
NVFP4_MAX = 6.0
CLAMP_THRESHOLD = FP8_MAX * NVFP4_MAX  # 2688


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
    print("检查 MLP 激活值范围 (FP8 Scale Overflow 检测)")
    print("=" * 70)
    print(f"Clamp 阈值: ±{CLAMP_THRESHOLD}")

    checkpoint_dir = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()
    device = torch.device('cuda')
    dtype = torch.bfloat16

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

    model = PI0Pytorch(pi0_config)
    state_dict = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    # 记录 MLP 层的输入范围
    activation_stats = {}

    def make_hook(name):
        def hook(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
                if isinstance(x, torch.Tensor):
                    min_val = x.min().item()
                    max_val = x.max().item()
                    abs_max = max(abs(min_val), abs(max_val))
                    exceeds = abs_max > CLAMP_THRESHOLD
                    if name not in activation_stats:
                        activation_stats[name] = {
                            'min': min_val,
                            'max': max_val,
                            'abs_max': abs_max,
                            'exceeds_clamp': exceeds,
                        }
                    else:
                        # Update with max values across all calls
                        activation_stats[name]['min'] = min(activation_stats[name]['min'], min_val)
                        activation_stats[name]['max'] = max(activation_stats[name]['max'], max_val)
                        activation_stats[name]['abs_max'] = max(activation_stats[name]['abs_max'], abs_max)
                        activation_stats[name]['exceeds_clamp'] = activation_stats[name]['abs_max'] > CLAMP_THRESHOLD
        return hook

    # 注册 hooks
    hooks = []
    paligemma_lm = model.paligemma_with_expert.paligemma.language_model
    for layer_idx, layer in enumerate(paligemma_lm.layers):
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(mlp, proj_name)
                name = f"layer{layer_idx}.{proj_name}"
                h = proj.register_forward_pre_hook(make_hook(name))
                hooks.append(h)

    # 运行推理
    observation = create_observation(device, dtype)
    torch.manual_seed(42)  # 固定种子

    with torch.no_grad():
        actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)

    # 移除 hooks
    for h in hooks:
        h.remove()

    # 分析结果
    print("\n" + "-" * 70)
    print(f"{'Layer':<25} {'Min':<12} {'Max':<12} {'|Max|':<12} {'Exceeds?':<10}")
    print("-" * 70)

    overflow_count = 0
    max_activation = 0

    for name in sorted(activation_stats.keys()):
        stats = activation_stats[name]
        status = "YES!" if stats['exceeds_clamp'] else "No"
        print(f"{name:<25} {stats['min']:<12.2f} {stats['max']:<12.2f} {stats['abs_max']:<12.2f} {status:<10}")

        if stats['exceeds_clamp']:
            overflow_count += 1
        max_activation = max(max_activation, stats['abs_max'])

    print("-" * 70)
    print(f"\n总层数: {len(activation_stats)}")
    print(f"超过阈值的层数: {overflow_count}")
    print(f"最大激活值: {max_activation:.2f}")
    print(f"Clamp 阈值: {CLAMP_THRESHOLD}")

    if overflow_count > 0:
        print(f"\n警告: {overflow_count} 层的激活值超过 FP8 Scale 限制!")
        print("Clamp 会导致精度损失。")
    else:
        print("\n所有激活值都在安全范围内，不需要 Clamp。")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
