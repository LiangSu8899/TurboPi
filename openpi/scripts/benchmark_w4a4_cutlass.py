#!/usr/bin/env python3
"""
Benchmark W4A4 CUTLASS 速度
"""

import torch
import time
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
    print("W4A4 CUTLASS 速度测试")
    print("=" * 70)

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

    state_dict = load_file(checkpoint_dir / "model.safetensors")
    observation = create_observation(device, dtype)

    num_warmup = 3
    num_runs = 5

    # BF16 Baseline
    print("\n[1] BF16 Baseline...")
    model_bf16 = PI0Pytorch(pi0_config)
    model_bf16.load_state_dict(state_dict, strict=False)
    model_bf16 = model_bf16.to(device='cuda')
    model_bf16.eval()

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model_bf16.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model_bf16.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()
    bf16_time = (time.perf_counter() - start) / num_runs
    print(f"  BF16: {bf16_time*1000:.1f} ms ({1/bf16_time:.2f} Hz)")

    del model_bf16
    torch.cuda.empty_cache()

    # W4A4 CUTLASS
    print("\n[2] W4A4 CUTLASS...")
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

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model_cutlass.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model_cutlass.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()
    cutlass_time = (time.perf_counter() - start) / num_runs
    print(f"  CUTLASS: {cutlass_time*1000:.1f} ms ({1/cutlass_time:.2f} Hz)")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Time (ms)':<12} {'Hz':<10} {'Speedup':<10}")
    print("-" * 55)
    print(f"{'BF16':<20} {bf16_time*1000:<12.1f} {1/bf16_time:<10.2f} {'1.0x':<10}")
    print(f"{'W4A4 CUTLASS':<20} {cutlass_time*1000:<12.1f} {1/cutlass_time:<10.2f} {bf16_time/cutlass_time:.2f}x")
    print("-" * 55)

    # Check if speedup is worth it
    speedup = bf16_time / cutlass_time
    if speedup > 1.5:
        print(f"\nW4A4 CUTLASS 有效! {speedup:.2f}x 加速")
    elif speedup > 1.0:
        print(f"\nW4A4 CUTLASS 轻微加速: {speedup:.2f}x")
    else:
        print(f"\nW4A4 CUTLASS 没有加速 ({speedup:.2f}x)，在线量化开销太大")
        print("需要实现预量化激活或使用 W4A16")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
