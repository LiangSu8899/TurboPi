#!/usr/bin/env python3
"""
NVFP4 Packed End-to-End Test

测试 NVFP4 Packed kernel 的:
1. 精度: 与 BF16 baseline 对比 cosine similarity
2. 延迟: MLP 层单独计时
3. 端到端: 整体推理延迟

Author: Claude Code
Date: 2026-02-10
"""

import sys
import os
import time
import argparse
import json
import pathlib

# Setup paths
for path in [
    os.path.join(os.path.dirname(__file__), "..", "src"),
    os.path.join(os.path.dirname(__file__), "..", "nvfp4_packed_plugin", "python"),
]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.nn.functional as F

# Disable cuDNN for Jetson compatibility
torch.backends.cudnn.enabled = False


def load_model(checkpoint_dir):
    """Load PI0 model."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from safetensors.torch import load_file

    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()

    # Load config
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
    else:
        model_config = {}

    max_token_len = model_config.get("tokenizer_max_length", 200)

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=model_config.get("action_dim", 32),
        action_horizon=model_config.get("action_horizon", 50),
        max_token_len=max_token_len,
        pi05=True,
        dtype="bfloat16",
    )

    model = PI0Pytorch(pi0_config)

    # Load weights
    weights_path = checkpoint_path / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    return model, pi0_config


def get_mlp_layers(model):
    """Get all MLP layers from model."""
    mlp_layers = []
    try:
        lm = model.paligemma_with_expert.paligemma.language_model
        for layer_idx, layer in enumerate(lm.layers):
            if hasattr(layer, 'mlp') and layer.mlp is not None:
                mlp_layers.append((layer_idx, layer.mlp))
    except AttributeError:
        pass
    return mlp_layers


def benchmark_mlp_layer(mlp, input_tensor, warmup=50, runs=200):
    """Benchmark a single MLP layer."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = mlp(input_tensor)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = mlp(input_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / runs * 1000  # ms


def test_accuracy(model_bf16, model_quant, device):
    """Test accuracy between BF16 and quantized model."""
    print("\n" + "=" * 60)
    print("Accuracy Test: BF16 vs NVFP4 Packed")
    print("=" * 60)

    # Get MLP layers
    mlp_bf16 = get_mlp_layers(model_bf16)
    mlp_quant = get_mlp_layers(model_quant)

    if not mlp_bf16 or not mlp_quant:
        print("ERROR: Could not find MLP layers")
        return

    # Test each layer
    hidden_size = 2048
    batch_size = 1

    cos_sims = []
    max_diffs = []

    for (idx, mlp_b), (_, mlp_q) in zip(mlp_bf16[:5], mlp_quant[:5]):  # Test first 5 layers
        # Create random input
        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            out_bf16 = mlp_b(x)
            out_quant = mlp_q(x)

        # Compute metrics
        cos_sim = F.cosine_similarity(
            out_bf16.flatten().float().unsqueeze(0),
            out_quant.flatten().float().unsqueeze(0)
        ).item()

        max_diff = (out_bf16 - out_quant).abs().max().item()

        cos_sims.append(cos_sim)
        max_diffs.append(max_diff)

        print(f"Layer {idx}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}")

    print(f"\nAverage cosine similarity: {np.mean(cos_sims):.6f}")
    print(f"Average max diff: {np.mean(max_diffs):.6f}")

    return np.mean(cos_sims)


def test_mlp_latency(model, name, device):
    """Test MLP layer latency."""
    print(f"\n" + "=" * 60)
    print(f"MLP Latency Test: {name}")
    print("=" * 60)

    mlp_layers = get_mlp_layers(model)
    if not mlp_layers:
        print("ERROR: Could not find MLP layers")
        return

    hidden_size = 2048
    batch_size = 1
    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    latencies = []
    for idx, mlp in mlp_layers[:5]:  # Test first 5 layers
        latency = benchmark_mlp_layer(mlp, x)
        latencies.append(latency)
        print(f"Layer {idx}: {latency:.4f} ms")

    avg_latency = np.mean(latencies)
    print(f"\nAverage MLP latency: {avg_latency:.4f} ms")
    print(f"18 layers total: {avg_latency * 18:.2f} ms")

    return avg_latency


def test_end_to_end_latency(model, device, warmup=10, runs=50):
    """Test end-to-end inference latency."""
    print("\n" + "=" * 60)
    print("End-to-End Latency Test")
    print("=" * 60)

    from openpi.models_pytorch.pi0_pytorch import Observation

    # Create dummy input
    batch_size = 1
    img_size = 224

    observation = Observation(
        images={
            "base_0_rgb": torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=torch.bfloat16),
            "left_wrist_0_rgb": torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=torch.bfloat16),
            "right_wrist_0_rgb": torch.zeros(batch_size, 3, img_size, img_size, device=device, dtype=torch.bfloat16),
        },
        image_masks={
            "base_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(batch_size, device=device, dtype=torch.bool),
        },
        state=torch.randn(batch_size, 32, device=device, dtype=torch.bfloat16),
        tokenized_prompt=torch.randint(0, 1000, (batch_size, 200), device=device, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(batch_size, 200, device=device, dtype=torch.bool),
    )

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({runs} iterations)...")
    latencies = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    hz = 1000 / avg_latency

    print(f"\nResults:")
    print(f"  Average latency: {avg_latency:.2f} ms (std: {std_latency:.2f})")
    print(f"  Throughput: {hz:.2f} Hz")

    return avg_latency, hz


def main():
    parser = argparse.ArgumentParser(description="NVFP4 Packed E2E Test")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="~/.cache/openpi/checkpoints/pi05_libero",
                        help="Model checkpoint directory")
    parser.add_argument("--test", type=str, nargs='+',
                        default=['accuracy', 'mlp_latency', 'e2e'],
                        choices=['accuracy', 'mlp_latency', 'e2e'],
                        help="Tests to run")
    args = parser.parse_args()

    print("=" * 60)
    print("NVFP4 Packed End-to-End Test")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Tests: {args.test}")

    device = torch.device('cuda')

    # =========================================================================
    # Test 1: Accuracy
    # =========================================================================
    if 'accuracy' in args.test:
        print("\n\n>>> Loading BF16 model...")
        model_bf16, _ = load_model(args.checkpoint_dir)

        print("\n>>> Loading NVFP4 Packed model...")
        model_quant, _ = load_model(args.checkpoint_dir)

        # Apply NVFP4 Packed quantization
        try:
            from nvfp4_packed import replace_mlp_with_nvfp4_packed
            replaced = replace_mlp_with_nvfp4_packed(model_quant)
            print(f"Replaced {replaced} MLP layers with NVFP4 Packed")
        except ImportError:
            # Fallback to existing NVFP4 implementation
            from openpi.models_pytorch.nvfp4_mlp import replace_paligemma_mlp_with_nvfp4
            replaced = replace_paligemma_mlp_with_nvfp4(model_quant, use_cutlass=False)
            print(f"Replaced {replaced} MLP layers with NVFP4 (simulation)")

        cos_sim = test_accuracy(model_bf16, model_quant, device)

        del model_bf16
        torch.cuda.empty_cache()

        if cos_sim and cos_sim > 0.99:
            print("\n[PASS] Accuracy test passed (cos_sim > 0.99)")
        else:
            print("\n[WARN] Accuracy may need improvement")

    # =========================================================================
    # Test 2: MLP Latency
    # =========================================================================
    if 'mlp_latency' in args.test:
        if 'accuracy' not in args.test:
            print("\n>>> Loading model for MLP latency test...")
            model_quant, _ = load_model(args.checkpoint_dir)

            try:
                from nvfp4_packed import replace_mlp_with_nvfp4_packed
                replaced = replace_mlp_with_nvfp4_packed(model_quant)
            except ImportError:
                from openpi.models_pytorch.nvfp4_mlp import replace_paligemma_mlp_with_nvfp4
                replaced = replace_paligemma_mlp_with_nvfp4(model_quant, use_cutlass=False)

        # Also test BF16 baseline for comparison
        print("\n>>> Testing BF16 MLP latency...")
        model_bf16_mlp, _ = load_model(args.checkpoint_dir)
        bf16_latency = test_mlp_latency(model_bf16_mlp, "BF16", device)
        del model_bf16_mlp
        torch.cuda.empty_cache()

        print("\n>>> Testing NVFP4 Packed MLP latency...")
        nvfp4_latency = test_mlp_latency(model_quant, "NVFP4 Packed", device)

        if bf16_latency and nvfp4_latency:
            speedup = bf16_latency / nvfp4_latency
            print(f"\n>>> MLP Speedup: {speedup:.2f}x")

    # =========================================================================
    # Test 3: End-to-End Latency
    # =========================================================================
    if 'e2e' in args.test:
        # BF16 baseline
        print("\n\n>>> Testing BF16 end-to-end latency...")
        if 'mlp_latency' not in args.test or 'accuracy' not in args.test:
            model_bf16_e2e, _ = load_model(args.checkpoint_dir)
        else:
            model_bf16_e2e, _ = load_model(args.checkpoint_dir)

        bf16_latency, bf16_hz = test_end_to_end_latency(model_bf16_e2e, device)
        del model_bf16_e2e
        torch.cuda.empty_cache()

        # NVFP4 Packed
        print("\n\n>>> Testing NVFP4 Packed end-to-end latency...")
        model_nvfp4, _ = load_model(args.checkpoint_dir)
        try:
            from nvfp4_packed import replace_mlp_with_nvfp4_packed
            replaced = replace_mlp_with_nvfp4_packed(model_nvfp4)
        except ImportError:
            from openpi.models_pytorch.nvfp4_mlp import replace_paligemma_mlp_with_nvfp4
            replaced = replace_paligemma_mlp_with_nvfp4(model_nvfp4, use_cutlass=False)

        nvfp4_latency, nvfp4_hz = test_end_to_end_latency(model_nvfp4, device)

        # Summary
        print("\n" + "=" * 60)
        print("End-to-End Summary")
        print("=" * 60)
        print(f"{'Method':<20} {'Latency (ms)':<15} {'Hz':<10}")
        print("-" * 45)
        print(f"{'BF16':<20} {bf16_latency:<15.2f} {bf16_hz:<10.2f}")
        print(f"{'NVFP4 Packed':<20} {nvfp4_latency:<15.2f} {nvfp4_hz:<10.2f}")
        print("-" * 45)
        speedup = bf16_latency / nvfp4_latency if nvfp4_latency > 0 else 0
        print(f"Speedup: {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
