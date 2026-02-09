#!/usr/bin/env python3
"""
比较不同量化方案的精度和速度

测试三种方案:
1. W4A4 (NVFP4) - 权重4位, 激活4位
2. W4A8 - 权重4位, 激活8位 (FP8)
3. W4A16 - 权重4位, 激活16位 (BF16)

对比 BF16 baseline 的:
- 精度 (Cosine Similarity, MAE)
- 速度 (推理时间)
- 内存占用
"""

import torch
import torch.nn.functional as F
import sys
import json
import pathlib
import time

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from safetensors.torch import load_file
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation


def create_observation(device, dtype):
    """创建测试 Observation。"""
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


def load_base_model():
    """加载基础 BF16 模型。"""
    checkpoint_dir = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()
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

    return model, state_dict, pi0_config


def get_bf16_baseline(model, observation, device, num_runs=3):
    """获取 BF16 baseline 结果。"""
    torch.manual_seed(42)
    with torch.no_grad():
        actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)

    # Timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_runs

    return actions, elapsed


def test_w4a16(state_dict, pi0_config, observation, device, num_runs=3):
    """测试 W4A16 量化。"""
    from openpi.models_pytorch.w4a16_mlp import replace_paligemma_mlp_with_w4a16

    model = PI0Pytorch(pi0_config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    # Replace with W4A16
    replaced = replace_paligemma_mlp_with_w4a16(model, cache_dequantized=True)

    # Warmup
    torch.manual_seed(42)
    with torch.no_grad():
        actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)

    # Check for NaN
    has_nan = torch.isnan(actions).any().item()
    if has_nan:
        return None, None, True

    # Timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_runs

    return actions, elapsed, False


def test_w4a8(state_dict, pi0_config, observation, device, num_runs=3):
    """测试 W4A8 量化。"""
    from openpi.models_pytorch.w4a8_mlp import replace_paligemma_mlp_with_w4a8

    model = PI0Pytorch(pi0_config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    # Replace with W4A8
    replaced = replace_paligemma_mlp_with_w4a8(model, cache_dequantized=True)

    # Warmup
    torch.manual_seed(42)
    with torch.no_grad():
        actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)

    # Check for NaN
    has_nan = torch.isnan(actions).any().item()
    if has_nan:
        return None, None, True

    # Timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_runs

    return actions, elapsed, False


def test_w4a4_sim(state_dict, pi0_config, observation, device, num_runs=3):
    """测试 W4A4 (NVFP4) 量化 (模拟模式, 不使用 CUTLASS)。"""
    from openpi.models_pytorch.nvfp4_mlp import replace_paligemma_mlp_with_nvfp4

    model = PI0Pytorch(pi0_config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    # Replace with NVFP4 (simulation mode)
    replaced = replace_paligemma_mlp_with_nvfp4(model, use_cutlass=False)

    # Warmup
    torch.manual_seed(42)
    with torch.no_grad():
        actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)

    # Check for NaN
    has_nan = torch.isnan(actions).any().item()
    if has_nan:
        return None, None, True

    # Timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_runs

    return actions, elapsed, False


def compute_metrics(actions, baseline):
    """计算精度指标。"""
    if actions is None or baseline is None:
        return {'cos_sim': float('nan'), 'mae': float('nan'), 'max_err': float('nan')}

    cos_sim = F.cosine_similarity(
        actions.flatten().float().unsqueeze(0),
        baseline.flatten().float().unsqueeze(0)
    ).item()

    mae = (actions - baseline).abs().mean().item()
    max_err = (actions - baseline).abs().max().item()

    return {'cos_sim': cos_sim, 'mae': mae, 'max_err': max_err}


def main():
    print("=" * 70)
    print("量化方案对比测试")
    print("=" * 70)

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Load base model
    print("\n[1] Loading base model...")
    model, state_dict, pi0_config = load_base_model()

    # Create observation
    observation = create_observation(device, dtype)

    # Test BF16 baseline
    print("\n[2] Testing BF16 baseline...")
    torch.cuda.empty_cache()
    actions_bf16, time_bf16 = get_bf16_baseline(model, observation, device)
    print(f"  BF16 time: {time_bf16*1000:.1f} ms")
    print(f"  Actions shape: {actions_bf16.shape}")

    del model
    torch.cuda.empty_cache()

    results = {}

    # Test W4A16
    print("\n[3] Testing W4A16...")
    torch.cuda.empty_cache()
    actions_w4a16, time_w4a16, has_nan = test_w4a16(state_dict, pi0_config, observation, device)
    if has_nan:
        print("  W4A16: NaN detected!")
        results['W4A16'] = {'status': 'NaN', 'time': None, 'metrics': None}
    else:
        metrics = compute_metrics(actions_w4a16, actions_bf16)
        results['W4A16'] = {'status': 'OK', 'time': time_w4a16, 'metrics': metrics}
        print(f"  Time: {time_w4a16*1000:.1f} ms")
        print(f"  Cosine sim: {metrics['cos_sim']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
    torch.cuda.empty_cache()

    # Test W4A8
    print("\n[4] Testing W4A8...")
    torch.cuda.empty_cache()
    actions_w4a8, time_w4a8, has_nan = test_w4a8(state_dict, pi0_config, observation, device)
    if has_nan:
        print("  W4A8: NaN detected!")
        results['W4A8'] = {'status': 'NaN', 'time': None, 'metrics': None}
    else:
        metrics = compute_metrics(actions_w4a8, actions_bf16)
        results['W4A8'] = {'status': 'OK', 'time': time_w4a8, 'metrics': metrics}
        print(f"  Time: {time_w4a8*1000:.1f} ms")
        print(f"  Cosine sim: {metrics['cos_sim']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
    torch.cuda.empty_cache()

    # Test W4A4 (simulation mode)
    print("\n[5] Testing W4A4 (NVFP4 sim)...")
    torch.cuda.empty_cache()
    actions_w4a4, time_w4a4, has_nan = test_w4a4_sim(state_dict, pi0_config, observation, device)
    if has_nan:
        print("  W4A4: NaN detected!")
        results['W4A4'] = {'status': 'NaN', 'time': None, 'metrics': None}
    else:
        metrics = compute_metrics(actions_w4a4, actions_bf16)
        results['W4A4'] = {'status': 'OK', 'time': time_w4a4, 'metrics': metrics}
        print(f"  Time: {time_w4a4*1000:.1f} ms")
        print(f"  Cosine sim: {metrics['cos_sim']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\n{'Method':<15} {'Status':<8} {'Time (ms)':<12} {'Cos Sim':<12} {'MAE':<12} {'Speedup':<8}")
    print("-" * 75)

    print(f"{'BF16':<15} {'OK':<8} {time_bf16*1000:<12.1f} {'1.0000':<12} {'0.0000':<12} {'1.0x':<8}")

    for method, data in results.items():
        if data['status'] == 'NaN':
            print(f"{method:<15} {'NaN':<8} {'-':<12} {'-':<12} {'-':<12} {'-':<8}")
        else:
            speedup = time_bf16 / data['time'] if data['time'] else 0
            m = data['metrics']
            print(f"{method:<15} {'OK':<8} {data['time']*1000:<12.1f} {m['cos_sim']:<12.6f} {m['mae']:<12.6f} {speedup:<.2f}x")

    print("-" * 75)

    # Recommendation
    print("\n推荐:")
    best_method = None
    best_score = 0

    for method, data in results.items():
        if data['status'] == 'OK' and data['metrics']['cos_sim'] > 0.95:
            score = data['metrics']['cos_sim'] * (time_bf16 / data['time'])
            if score > best_score:
                best_score = score
                best_method = method

    if best_method:
        print(f"  最佳方案: {best_method}")
        print(f"  精度: {results[best_method]['metrics']['cos_sim']:.4f}")
        print(f"  加速: {time_bf16/results[best_method]['time']:.2f}x")
    else:
        print("  没有满足精度要求的量化方案")
        print("  建议使用 BF16 或 FP8 (非量化方案)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
