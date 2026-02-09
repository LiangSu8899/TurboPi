#!/usr/bin/env python3
"""
对比 NVFP4 vs BF16 模型输出

验证 NVFP4 模型是否能产生与 BF16 模型足够接近的 action 输出。
如果 action 差异太大，机器人会"抖动"或失败。
"""

import torch
import torch.nn.functional as F
import sys
import time
import json
import pathlib

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from safetensors.torch import load_file
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
from openpi.models_pytorch.nvfp4_mlp import replace_paligemma_mlp_with_nvfp4


def load_bf16_model(checkpoint_dir: str):
    """加载 BF16 基准模型"""
    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()

    config_path = checkpoint_path / "config.json"
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
    weights_path = checkpoint_path / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    return model, pi0_config


def load_nvfp4_model(checkpoint_dir: str):
    """加载 NVFP4 模型"""
    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()

    config_path = checkpoint_path / "config.json"
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
    weights_path = checkpoint_path / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    # Replace MLP with NVFP4
    print("Replacing MLP layers with NVFP4...")
    replaced = replace_paligemma_mlp_with_nvfp4(model, use_cutlass=True)
    print(f"Replaced {replaced} MLP layers")

    # Prepare CUTLASS weights
    print("Preparing CUTLASS weights...")
    paligemma_lm = model.paligemma_with_expert.paligemma.language_model
    for layer in paligemma_lm.layers:
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, proj_name):
                    proj = getattr(mlp, proj_name)
                    if hasattr(proj, 'prepare_for_cutlass'):
                        proj.use_cutlass = True
                        proj.prepare_for_cutlass()

    return model, pi0_config


def create_test_observation(device, dtype, seed=42):
    """创建测试 observation"""
    torch.manual_seed(seed)
    batch_size = 1

    return Observation(
        images={
            "base_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype),
            "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype),
            "right_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224, device=device, dtype=dtype),
        },
        image_masks={
            "base_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(batch_size, device=device, dtype=torch.bool),
        },
        state=torch.randn(batch_size, 32, device=device, dtype=dtype),
        tokenized_prompt=torch.zeros(batch_size, 200, device=device, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(batch_size, 200, device=device, dtype=torch.bool),
    )


def compare_actions(actions_bf16, actions_nvfp4):
    """比较两个 action 序列"""
    # Flatten for comparison
    a1 = actions_bf16.flatten().float()
    a2 = actions_nvfp4.flatten().float()

    # Cosine similarity
    cos_sim = F.cosine_similarity(a1.unsqueeze(0), a2.unsqueeze(0)).item()

    # L2 distance
    l2_dist = torch.norm(a1 - a2).item()

    # Max absolute error
    max_err = (a1 - a2).abs().max().item()

    # Mean absolute error
    mae = (a1 - a2).abs().mean().item()

    # Per-step comparison (actions are [B, T, D])
    T = actions_bf16.shape[1]
    step_cos_sims = []
    for t in range(T):
        s1 = actions_bf16[0, t].flatten().float()
        s2 = actions_nvfp4[0, t].flatten().float()
        step_cos = F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item()
        step_cos_sims.append(step_cos)

    return {
        'cosine_similarity': cos_sim,
        'l2_distance': l2_dist,
        'max_absolute_error': max_err,
        'mean_absolute_error': mae,
        'step_cosine_sims': step_cos_sims,
    }


def main():
    print("=" * 70)
    print("NVFP4 vs BF16 Action Output Comparison")
    print("=" * 70)

    checkpoint_dir = "~/.cache/openpi/checkpoints/pi05_libero"
    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Load BF16 model
    print("\n[1/4] Loading BF16 model...")
    model_bf16, config = load_bf16_model(checkpoint_dir)
    print("  BF16 model loaded")

    # Load NVFP4 model
    print("\n[2/4] Loading NVFP4 model...")
    model_nvfp4, _ = load_nvfp4_model(checkpoint_dir)
    print("  NVFP4 model loaded")

    # Run comparison with multiple seeds
    print("\n[3/4] Comparing action outputs...")
    print("-" * 50)

    all_results = []
    num_tests = 5

    for seed in range(num_tests):
        observation = create_test_observation(device, dtype, seed=seed)

        # BF16 inference
        with torch.no_grad():
            actions_bf16 = model_bf16.sample_actions(
                device, observation, num_steps=3, use_kv_cache=True
            )

        # NVFP4 inference (with same observation)
        observation = create_test_observation(device, dtype, seed=seed)
        with torch.no_grad():
            actions_nvfp4 = model_nvfp4.sample_actions(
                device, observation, num_steps=3, use_kv_cache=True
            )

        # Compare
        result = compare_actions(actions_bf16, actions_nvfp4)
        all_results.append(result)

        print(f"  Test {seed+1}: cos_sim={result['cosine_similarity']:.4f}, "
              f"MAE={result['mean_absolute_error']:.4f}, "
              f"Max={result['max_absolute_error']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("[4/4] Summary")
    print("=" * 70)

    avg_cos = sum(r['cosine_similarity'] for r in all_results) / len(all_results)
    avg_mae = sum(r['mean_absolute_error'] for r in all_results) / len(all_results)
    avg_max = sum(r['max_absolute_error'] for r in all_results) / len(all_results)

    print(f"\n  Average Cosine Similarity: {avg_cos:.4f}")
    print(f"  Average MAE:               {avg_mae:.4f}")
    print(f"  Average Max Error:         {avg_max:.4f}")

    # Per-step analysis (from last test)
    print("\n  Per-step cosine similarity (last test):")
    for t, cos in enumerate(all_results[-1]['step_cosine_sims'][:10]):
        print(f"    Step {t}: {cos:.4f}")
    if len(all_results[-1]['step_cosine_sims']) > 10:
        print(f"    ... ({len(all_results[-1]['step_cosine_sims'])} total steps)")

    # Verdict
    print("\n" + "-" * 50)
    if avg_cos > 0.95:
        print("  ✅ NVFP4 actions are highly consistent with BF16 (>0.95)")
        print("     机器人行为应该与 BF16 几乎一致")
    elif avg_cos > 0.90:
        print("  ⚠️  NVFP4 actions moderately consistent (0.90-0.95)")
        print("     可能有轻微抖动，建议实际测试")
    elif avg_cos > 0.80:
        print("  ⚠️  NVFP4 actions weakly consistent (0.80-0.90)")
        print("     可能有明显差异，需要谨慎")
    else:
        print("  ❌ NVFP4 actions significantly different (<0.80)")
        print("     不建议使用，需要进一步调试")

    print("=" * 70)


if __name__ == "__main__":
    main()
