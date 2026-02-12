#!/usr/bin/env python3
"""Debug TRT FP8 Accuracy Issue.

The TRT FP8 prefill produces 5142% error vs BF16.
This script investigates:
1. KV cache output comparison
2. Layer-by-layer output comparison
3. Weight loading verification

Author: Claude Code
Date: 2026-02-12
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import json


def load_model():
    """Load model WITHOUT any quantization."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from safetensors.torch import load_file

    device = 'cuda'
    checkpoint_path = pathlib.Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()

    with open(checkpoint_path / "config.json") as f:
        model_config = json.load(f)

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

    print("Loading model (unpatched BF16)...")
    model = PI0Pytorch(pi0_config)
    state_dict = load_file(checkpoint_path / "model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, pi0_config, device, max_token_len, str(checkpoint_path)


def create_test_observation(device, max_token_len):
    """Create test observation."""
    from openpi.models_pytorch.pi0_pytorch import Observation

    # Use fixed seed for reproducibility
    torch.manual_seed(42)

    return Observation(
        images={
            "base_0_rgb": torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device=device),
            "left_wrist_0_rgb": torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device=device),
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, dtype=torch.bfloat16, device=device),
        },
        image_masks={
            "base_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
        },
        state=torch.randn(1, 32, dtype=torch.bfloat16, device=device),
        tokenized_prompt=torch.randint(0, 1000, (1, max_token_len), dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.ones(1, max_token_len, dtype=torch.bool, device=device),
    )


def debug_kv_cache_comparison():
    """Compare KV cache outputs between BF16 and TRT FP8."""

    print("=" * 70)
    print("DEBUG: TRT FP8 vs BF16 KV Cache Comparison")
    print("=" * 70)

    # Load model
    model, config, device, max_token_len, checkpoint_dir = load_model()
    observation = create_test_observation(device, max_token_len)

    # Preprocess
    import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
    obs_proc = _preprocessing.preprocess_observation_pytorch(observation, train=False)
    images = list(obs_proc.images.values())
    img_masks = list(obs_proc.image_masks.values())
    lang_tokens = obs_proc.tokenized_prompt
    lang_masks = obs_proc.tokenized_prompt_mask

    # =========================================================================
    # Get BF16 KV Cache from original model
    # =========================================================================
    print("\n[1] Computing BF16 KV Cache from original model...")

    with torch.no_grad():
        prefix_embs_bf16, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        bf16_kv_cache = model.compute_prefix_kv_cache(
            prefix_embs_bf16, prefix_pad_masks, prefix_att_masks
        )

    print(f"    prefix_embs shape: {prefix_embs_bf16.shape}")
    print(f"    BF16 KV Cache: {len(bf16_kv_cache)} layers")
    print(f"    Layer 0 K shape: {bf16_kv_cache[0][0].shape}")
    print(f"    Layer 0 V shape: {bf16_kv_cache[0][1].shape}")

    # =========================================================================
    # Get TRT FP8 KV Cache
    # =========================================================================
    print("\n[2] Computing TRT FP8 KV Cache...")

    # Load TRT FP8 Engine (without compiling TRT to save time - just use FP16 fallback)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "torch_trt_fp8_kv_cache",
        pathlib.Path(__file__).parent.parent / "src" / "openpi" / "inference" / "torch_trt_fp8_kv_cache.py"
    )
    trt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trt_module)
    TorchTRTFP8KVCacheEngine = trt_module.TorchTRTFP8KVCacheEngine

    # Don't compile TRT - just use FP16 fallback to check accuracy
    trt_engine = TorchTRTFP8KVCacheEngine(
        checkpoint_dir=checkpoint_dir,
        device=device,
        compile_trt=False,  # Use FP16 fallback first
    )

    # Run TRT FP8 engine
    trt_kv_cache = trt_engine.infer_list(prefix_embs_bf16, None, None)

    print(f"    TRT KV Cache: {len(trt_kv_cache)} layers")
    print(f"    Layer 0 K shape: {trt_kv_cache[0][0].shape}")
    print(f"    Layer 0 V shape: {trt_kv_cache[0][1].shape}")

    # =========================================================================
    # Compare KV Caches
    # =========================================================================
    print("\n[3] Comparing KV Caches...")

    for i in range(min(5, len(bf16_kv_cache))):
        bf16_k, bf16_v = bf16_kv_cache[i]
        trt_k, trt_v = trt_kv_cache[i]

        # Handle shape mismatch
        if bf16_k.shape != trt_k.shape:
            print(f"\n  Layer {i} SHAPE MISMATCH:")
            print(f"    BF16 K: {bf16_k.shape}, TRT K: {trt_k.shape}")
            print(f"    BF16 V: {bf16_v.shape}, TRT V: {trt_v.shape}")
            continue

        k_diff = (bf16_k - trt_k).abs()
        v_diff = (bf16_v - trt_v).abs()

        print(f"\n  Layer {i}:")
        print(f"    K max diff: {k_diff.max().item():.6f}, mean diff: {k_diff.mean().item():.6f}")
        print(f"    V max diff: {v_diff.max().item():.6f}, mean diff: {v_diff.mean().item():.6f}")

        # Check relative error
        k_rel_err = (k_diff / (bf16_k.abs() + 1e-8)).mean() * 100
        v_rel_err = (v_diff / (bf16_v.abs() + 1e-8)).mean() * 100
        print(f"    K rel error: {k_rel_err.item():.2f}%, V rel error: {v_rel_err.item():.2f}%")

    # =========================================================================
    # Compare denoise outputs
    # =========================================================================
    print("\n[4] Comparing Denoise Outputs...")

    state = observation.state
    torch.manual_seed(42)
    noise = model.sample_noise(
        (1, model.config.action_horizon, model.config.action_dim), device
    ).to(torch.bfloat16)

    def run_denoise(kv_cache, cache_name):
        actions = noise.clone()
        for step in range(3):
            timestep = torch.tensor(
                [step / 3],
                dtype=torch.float32,
                device=device
            ).expand(1)
            with torch.no_grad():
                actions = model.denoise_step_with_cache(
                    state, kv_cache, prefix_pad_masks, actions, timestep
                )
        return actions

    bf16_actions = run_denoise(bf16_kv_cache, "BF16")
    trt_actions = run_denoise(trt_kv_cache, "TRT")

    print(f"\n  BF16 Actions:")
    print(f"    Shape: {bf16_actions.shape}")
    print(f"    Range: [{bf16_actions.min().item():.4f}, {bf16_actions.max().item():.4f}]")
    print(f"    Mean: {bf16_actions.mean().item():.4f}")
    print(f"    Std: {bf16_actions.std().item():.4f}")

    print(f"\n  TRT Actions (FP16 fallback):")
    print(f"    Shape: {trt_actions.shape}")
    print(f"    Range: [{trt_actions.min().item():.4f}, {trt_actions.max().item():.4f}]")
    print(f"    Mean: {trt_actions.mean().item():.4f}")
    print(f"    Std: {trt_actions.std().item():.4f}")

    diff = (bf16_actions - trt_actions).abs()
    print(f"\n  Difference:")
    print(f"    Max abs diff: {diff.max().item():.6f}")
    print(f"    Mean abs diff: {diff.mean().item():.6f}")
    rel_err = (diff / (bf16_actions.abs() + 1e-8)).mean() * 100
    print(f"    Relative error: {rel_err.item():.2f}%")

    # =========================================================================
    # Weight comparison
    # =========================================================================
    print("\n[5] Comparing Weights...")

    # Get weights from original model
    orig_layer0 = model.paligemma_with_expert.paligemma.model.language_model.layers[0]

    # Get weights from TRT engine
    trt_layer0 = trt_engine.model.layers[0]

    # Compare MLP weights
    orig_gate = orig_layer0.mlp.gate_proj.weight.data
    trt_gate = trt_layer0.gate_proj.weight.data

    gate_diff = (orig_gate - trt_gate).abs()
    print(f"\n  Layer 0 MLP gate_proj weight:")
    print(f"    Original shape: {orig_gate.shape}, TRT shape: {trt_gate.shape}")
    print(f"    Max diff: {gate_diff.max().item():.8f}")
    print(f"    Mean diff: {gate_diff.mean().item():.8f}")

    # Compare attention weights
    orig_qkv = orig_layer0.self_attn.qkv_proj.weight.data
    trt_q = trt_layer0.self_attn.q_proj.weight.data
    trt_k = trt_layer0.self_attn.k_proj.weight.data
    trt_v = trt_layer0.self_attn.v_proj.weight.data

    print(f"\n  Layer 0 Attention weights:")
    print(f"    Original QKV shape: {orig_qkv.shape}")
    print(f"    TRT Q shape: {trt_q.shape}, K shape: {trt_k.shape}, V shape: {trt_v.shape}")

    # The original model might use fused QKV proj
    # Split original QKV to compare
    # QKV = [Q (2048x2048), K (256x2048), V (256x2048)]
    # Total = 2048 + 256 + 256 = 2560 rows for each layer

    q_size = trt_q.shape[0]  # 2048
    k_size = trt_k.shape[0]  # 256
    v_size = trt_v.shape[0]  # 256

    if orig_qkv.shape[0] == q_size + k_size + v_size:
        orig_q = orig_qkv[:q_size]
        orig_k = orig_qkv[q_size:q_size+k_size]
        orig_v = orig_qkv[q_size+k_size:]

        q_diff = (orig_q - trt_q).abs()
        k_diff = (orig_k - trt_k).abs()
        v_diff = (orig_v - trt_v).abs()

        print(f"    Q weight max diff: {q_diff.max().item():.8f}")
        print(f"    K weight max diff: {k_diff.max().item():.8f}")
        print(f"    V weight max diff: {v_diff.max().item():.8f}")
    else:
        print(f"    [WARN] QKV size mismatch")

    return {
        'bf16_actions': bf16_actions,
        'trt_actions': trt_actions,
        'rel_err': rel_err.item(),
    }


if __name__ == "__main__":
    results = debug_kv_cache_comparison()
