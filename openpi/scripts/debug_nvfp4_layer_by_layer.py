#!/usr/bin/env python3
"""
逐层调试 NVFP4 NaN 问题

找出是哪一层导致 NaN
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


def check_tensor_brief(name, tensor):
    """简短检查 tensor"""
    if tensor is None:
        return "None"
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan:
        return f"NaN! shape={tensor.shape}"
    elif has_inf:
        return f"Inf! shape={tensor.shape}"
    else:
        return f"OK [{tensor.min().item():.2f}, {tensor.max().item():.2f}]"


def main():
    print("=" * 70)
    print("逐层调试 NVFP4 NaN")
    print("=" * 70)

    checkpoint_dir = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()
    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Load model
    print("\n[1] Loading and preparing model...")
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

    model = PI0Pytorch(pi0_config)
    state_dict = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    # Replace with NVFP4 and prepare
    replaced = replace_paligemma_mlp_with_nvfp4(model, use_cutlass=True)
    print(f"  Replaced {replaced} layers")

    paligemma_lm = model.paligemma_with_expert.paligemma.language_model
    for layer in paligemma_lm.layers:
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(mlp, proj_name)
                if hasattr(proj, 'prepare_for_cutlass'):
                    proj.use_cutlass = True
                    proj.prepare_for_cutlass()

    # Test different input sizes
    print("\n[2] Testing different input sizes...")
    print("-" * 50)

    test_sizes = [
        (1, 2048),      # 标准 hidden size
        (1, 8192),      # intermediate size
        (10, 2048),     # 小 batch
        (100, 2048),    # 中 batch
        (256, 2048),    # 大 batch
        (512, 2048),    # 更大 batch
        (1, 768),       # 可能的 token embedding size
        (200, 2048),    # 接近 max_token_len
    ]

    gate_proj = paligemma_lm.layers[0].mlp.gate_proj

    for batch, hidden in test_sizes:
        if hidden != gate_proj.in_features:
            continue  # Skip mismatched sizes

        x = torch.randn(batch, hidden, device=device, dtype=torch.float32)
        with torch.no_grad():
            try:
                out = gate_proj(x)
                status = check_tensor_brief(f"[{batch}x{hidden}]", out)
            except Exception as e:
                status = f"ERROR: {str(e)[:50]}"
        print(f"  [{batch:4d} x {hidden:4d}] -> {status}")

    # Test each layer individually
    print("\n[3] Testing each layer with batch=256...")
    print("-" * 50)

    x = torch.randn(256, 2048, device=device, dtype=torch.float32)

    for layer_idx, layer in enumerate(paligemma_lm.layers):
        mlp = layer.mlp
        results = []

        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(mlp, proj_name)
            if isinstance(proj, NVFP4Linear):
                # Use correct input size for down_proj
                if proj_name == 'down_proj':
                    test_x = torch.randn(256, proj.in_features, device=device, dtype=torch.float32)
                else:
                    test_x = x

                with torch.no_grad():
                    try:
                        out = proj(test_x)
                        status = "✓" if not torch.isnan(out).any() else "NaN!"
                    except Exception as e:
                        status = "ERR"
                results.append(f"{proj_name[0]}:{status}")

        print(f"  Layer {layer_idx:2d}: {', '.join(results)}")

    # Test with hooks to find exact failure point
    print("\n[4] Testing full forward with hooks...")
    print("-" * 50)

    nan_found = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    nan_found.append(f"{name}: NaN detected")
            elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor) and torch.isnan(o).any():
                        nan_found.append(f"{name}[{i}]: NaN detected")
        return hook

    # Register hooks
    hooks = []
    for layer_idx, layer in enumerate(paligemma_lm.layers):
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(mlp, proj_name)
            h = proj.register_forward_hook(make_hook(f"layer{layer_idx}.{proj_name}"))
            hooks.append(h)

    # Run forward
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

    try:
        with torch.no_grad():
            actions = model.sample_actions(device, observation, num_steps=1, use_kv_cache=True)
        result = check_tensor_brief("actions", actions)
    except Exception as e:
        result = f"ERROR: {str(e)[:100]}"

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"  Final output: {result}")

    if nan_found:
        print(f"\n  Found {len(nan_found)} NaN occurrences:")
        for loc in nan_found[:20]:  # Show first 20
            print(f"    - {loc}")
        if len(nan_found) > 20:
            print(f"    ... and {len(nan_found) - 20} more")
    else:
        print("  No NaN detected in MLP layers (issue may be elsewhere)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
