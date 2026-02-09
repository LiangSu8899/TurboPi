#!/usr/bin/env python3
"""
调试 NVFP4 层的实际输入范围

找出为什么 layer16/17 在完整推理时产生 NaN
"""

import torch
import sys
import json
import pathlib

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from safetensors.torch import load_file
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
from openpi.models_pytorch.nvfp4_mlp import replace_paligemma_mlp_with_nvfp4, NVFP4Linear


def main():
    print("=" * 70)
    print("调试 NVFP4 层的实际输入范围")
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

    # Record input/output of problematic layers
    print("\n[2] Recording inputs to layer 15-17...")
    print("-" * 70)

    input_records = {}
    output_records = {}

    def make_pre_hook(name):
        def hook(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
                if isinstance(x, torch.Tensor):
                    has_nan = torch.isnan(x).any().item()
                    has_inf = torch.isinf(x).any().item()
                    input_records[name] = {
                        'shape': x.shape,
                        'dtype': str(x.dtype),
                        'min': x.min().item() if not has_nan else 'NaN',
                        'max': x.max().item() if not has_nan else 'NaN',
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                    }
        return hook

    def make_post_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                output_records[name] = {
                    'shape': output.shape,
                    'dtype': str(output.dtype),
                    'min': output.min().item() if not has_nan else 'NaN',
                    'max': output.max().item() if not has_nan else 'NaN',
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                }
        return hook

    # Register hooks on layers 14-17
    hooks = []
    for layer_idx in range(14, 18):
        layer = paligemma_lm.layers[layer_idx]
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(mlp, proj_name)
            name = f"layer{layer_idx}.{proj_name}"
            h1 = proj.register_forward_pre_hook(make_pre_hook(name))
            h2 = proj.register_forward_hook(make_post_hook(name))
            hooks.append(h1)
            hooks.append(h2)

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

    with torch.no_grad():
        try:
            actions = model.sample_actions(device, observation, num_steps=1, use_kv_cache=True)
        except Exception as e:
            print(f"  Forward failed: {e}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print results
    print("\n[3] Input/Output analysis:")
    print("-" * 70)

    for layer_idx in range(14, 18):
        print(f"\nLayer {layer_idx}:")
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            name = f"layer{layer_idx}.{proj_name}"

            if name in input_records:
                inp = input_records[name]
                status = "NaN!" if inp['has_nan'] else ("Inf!" if inp['has_inf'] else "OK")
                print(f"  {proj_name:10s} INPUT:  {str(inp['shape']):20s} [{inp['min']:.2f}, {inp['max']:.2f}] {status}")

            if name in output_records:
                out = output_records[name]
                status = "NaN!" if out['has_nan'] else ("Inf!" if out['has_inf'] else "OK")
                print(f"  {proj_name:10s} OUTPUT: {str(out['shape']):20s} [{out['min']:.2f}, {out['max']:.2f}] {status}")

    # Test with the actual problematic input size
    print("\n" + "=" * 70)
    print("[4] Testing layer 16 down_proj with actual input shape...")
    print("-" * 70)

    if 'layer16.down_proj' in input_records:
        inp = input_records['layer16.down_proj']
        shape = inp['shape']
        print(f"  Actual shape: {shape}")

        # Create test input with same shape
        layer16_down = paligemma_lm.layers[16].mlp.down_proj

        # Test with normal input
        test_input = torch.randn(shape, device=device, dtype=torch.float32)
        with torch.no_grad():
            out = layer16_down(test_input)
        has_nan = torch.isnan(out).any().item()
        print(f"  Random input: {'NaN!' if has_nan else 'OK'}")

        # Test with larger range input (simulating accumulated errors)
        test_input_large = torch.randn(shape, device=device, dtype=torch.float32) * 100
        with torch.no_grad():
            out_large = layer16_down(test_input_large)
        has_nan_large = torch.isnan(out_large).any().item()
        print(f"  Large input (x100): {'NaN!' if has_nan_large else 'OK'}")

        # Test the actual range seen
        if not inp['has_nan']:
            actual_min = inp['min']
            actual_max = inp['max']
            test_input_actual = torch.rand(shape, device=device, dtype=torch.float32) * (actual_max - actual_min) + actual_min
            with torch.no_grad():
                out_actual = layer16_down(test_input_actual)
            has_nan_actual = torch.isnan(out_actual).any().item()
            print(f"  Actual range [{actual_min:.2f}, {actual_max:.2f}]: {'NaN!' if has_nan_actual else 'OK'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
