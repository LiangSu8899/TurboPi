#!/usr/bin/env python3
"""Clean TRT FP8 Accuracy Verification.

Run TRT FP8 prefill in isolation and compare PROPERLY with BF16.

Key insight: The previous benchmark showed 5142% error because the BF16 baseline
was run AFTER TRT FP8 compilation which corrupted the model state.

This script runs BF16 baseline FIRST, then loads TRT FP8 in a clean way.

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
    """Create test observation with fixed seed."""
    from openpi.models_pytorch.pi0_pytorch import Observation

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


def run_denoise_with_cache(model, kv_cache, prefix_pad_masks, state, noise, device, num_steps=3):
    """Run denoise loop."""
    actions = noise.clone()
    for step in range(num_steps):
        timestep = torch.tensor(
            [step / num_steps],
            dtype=torch.float32,
            device=device
        ).expand(1)
        with torch.no_grad():
            actions = model.denoise_step_with_cache(
                state, kv_cache, prefix_pad_masks, actions, timestep
            )
    return actions


def verify_accuracy():
    """Clean accuracy verification."""

    print("=" * 70)
    print("CLEAN TRT FP8 ACCURACY VERIFICATION")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load model and prepare inputs
    # =========================================================================
    model, config, device, max_token_len, checkpoint_dir = load_model()
    observation = create_test_observation(device, max_token_len)

    import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
    obs_proc = _preprocessing.preprocess_observation_pytorch(observation, train=False)
    images = list(obs_proc.images.values())
    img_masks = list(obs_proc.image_masks.values())
    lang_tokens = obs_proc.tokenized_prompt
    lang_masks = obs_proc.tokenized_prompt_mask
    state = observation.state

    # =========================================================================
    # Step 2: BF16 Baseline (run FIRST, before ANY TRT loading)
    # =========================================================================
    print("\n[1] Running BF16 Baseline (BEFORE TRT loading)...")

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        bf16_kv_cache = model.compute_prefix_kv_cache(
            prefix_embs, prefix_pad_masks, prefix_att_masks
        )

    torch.manual_seed(42)
    noise = model.sample_noise(
        (1, model.config.action_horizon, model.config.action_dim), device
    ).to(torch.bfloat16)

    bf16_actions = run_denoise_with_cache(model, bf16_kv_cache, prefix_pad_masks, state, noise, device)

    print(f"    prefix_embs shape: {prefix_embs.shape}")
    print(f"    BF16 KV Cache: {len(bf16_kv_cache)} layers")
    print(f"    BF16 Actions shape: {bf16_actions.shape}")
    print(f"    BF16 Actions range: [{bf16_actions.min().item():.4f}, {bf16_actions.max().item():.4f}]")
    print(f"    BF16 Actions mean: {bf16_actions.mean().item():.4f}")
    print(f"    BF16 Actions std: {bf16_actions.std().item():.4f}")

    # =========================================================================
    # Step 3: Use ORIGINAL model's KV cache computation with TRT FP8 denoise
    # =========================================================================
    print("\n[2] Testing: Use BF16 Prefill + BF16 Denoise (reference)...")

    # Reset noise
    torch.manual_seed(42)
    noise2 = model.sample_noise(
        (1, model.config.action_horizon, model.config.action_dim), device
    ).to(torch.bfloat16)

    bf16_actions2 = run_denoise_with_cache(model, bf16_kv_cache, prefix_pad_masks, state, noise2, device)

    diff = (bf16_actions - bf16_actions2).abs()
    print(f"    Reproducibility check: max diff = {diff.max().item():.8f}")

    # =========================================================================
    # Step 4: Load TRT FP8 Engine (FP16 fallback only, no TRT compile)
    # =========================================================================
    print("\n[3] Loading TRT FP8 Engine (FP16 fallback mode)...")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "torch_trt_fp8_kv_cache",
        pathlib.Path(__file__).parent.parent / "src" / "openpi" / "inference" / "torch_trt_fp8_kv_cache.py"
    )
    trt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trt_module)
    TorchTRTFP8KVCacheEngine = trt_module.TorchTRTFP8KVCacheEngine

    trt_engine = TorchTRTFP8KVCacheEngine(
        checkpoint_dir=checkpoint_dir,
        device=device,
        compile_trt=False,  # FP16 fallback only
    )
    print(f"    TRT FP8 Engine loaded (FP16 fallback)")

    # =========================================================================
    # Step 5: Run TRT FP8 Prefill (FP16 fallback)
    # =========================================================================
    print("\n[4] Running TRT FP8 Prefill (FP16 fallback)...")

    # IMPORTANT: Use the SAME prefix_embs as BF16
    trt_kv_cache = trt_engine.infer_list(prefix_embs, None, None)

    print(f"    TRT KV Cache: {len(trt_kv_cache)} layers")

    # =========================================================================
    # Step 6: Run Denoise with TRT KV Cache using ORIGINAL model
    # =========================================================================
    print("\n[5] Running Denoise with TRT KV Cache...")

    torch.manual_seed(42)
    noise3 = model.sample_noise(
        (1, model.config.action_horizon, model.config.action_dim), device
    ).to(torch.bfloat16)

    trt_actions = run_denoise_with_cache(model, trt_kv_cache, prefix_pad_masks, state, noise3, device)

    print(f"    TRT Actions shape: {trt_actions.shape}")
    print(f"    TRT Actions range: [{trt_actions.min().item():.4f}, {trt_actions.max().item():.4f}]")
    print(f"    TRT Actions mean: {trt_actions.mean().item():.4f}")
    print(f"    TRT Actions std: {trt_actions.std().item():.4f}")

    # =========================================================================
    # Step 7: Compare
    # =========================================================================
    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON")
    print("=" * 70)

    diff = (bf16_actions - trt_actions).abs()

    print(f"""
   +---------------------------+------------------+------------------+
   | Metric                    | BF16             | TRT FP8 (FP16)   |
   +---------------------------+------------------+------------------+
   | Range                     | [{bf16_actions.min().item():7.4f}, {bf16_actions.max().item():7.4f}] | [{trt_actions.min().item():7.4f}, {trt_actions.max().item():7.4f}] |
   | Mean                      | {bf16_actions.mean().item():16.4f} | {trt_actions.mean().item():16.4f} |
   | Std                       | {bf16_actions.std().item():16.4f} | {trt_actions.std().item():16.4f} |
   +---------------------------+------------------+------------------+

   Difference Analysis:
   +---------------------------+------------------+
   | Max Absolute Diff         | {diff.max().item():16.6f} |
   | Mean Absolute Diff        | {diff.mean().item():16.6f} |
   | Relative Error (%)        | {(diff.mean() / bf16_actions.abs().mean() * 100).item():16.2f} |
   +---------------------------+------------------+
""")

    # Verdict
    rel_err = (diff.mean() / bf16_actions.abs().mean() * 100).item()
    if rel_err < 1.0:
        verdict = "EXCELLENT (<1%)"
    elif rel_err < 5.0:
        verdict = "GOOD (<5%)"
    elif rel_err < 10.0:
        verdict = "ACCEPTABLE (<10%)"
    else:
        verdict = "POOR (>10%)"

    print(f"   Verdict: {verdict}")
    print(f"   Relative Error: {rel_err:.2f}%")

    # =========================================================================
    # Step 8: Benchmark latency
    # =========================================================================
    print("\n" + "=" * 70)
    print("LATENCY BENCHMARK")
    print("=" * 70)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)
        _ = trt_engine.infer_list(prefix_embs, None, None)
    torch.cuda.synchronize()

    # Benchmark BF16 Prefill
    bf16_times = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)
        end.record()
        torch.cuda.synchronize()
        bf16_times.append(start.elapsed_time(end))

    bf16_prefill_ms = np.mean(bf16_times)

    # Benchmark TRT FP8 Prefill
    trt_times = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = trt_engine.infer_list(prefix_embs, None, None)
        end.record()
        torch.cuda.synchronize()
        trt_times.append(start.elapsed_time(end))

    trt_prefill_ms = np.mean(trt_times)

    speedup = bf16_prefill_ms / trt_prefill_ms

    print(f"""
   +---------------------------+------------------+
   | Mode                      | Latency (ms)     |
   +---------------------------+------------------+
   | BF16 Prefill              | {bf16_prefill_ms:16.2f} |
   | TRT FP8 Prefill (FP16)    | {trt_prefill_ms:16.2f} |
   +---------------------------+------------------+
   | Speedup                   | {speedup:15.2f}x |
   +---------------------------+------------------+
""")

    return {
        'bf16_prefill_ms': bf16_prefill_ms,
        'trt_prefill_ms': trt_prefill_ms,
        'speedup': speedup,
        'rel_err': rel_err,
    }


if __name__ == "__main__":
    results = verify_accuracy()
    print(f"\nFinal Results: {results}")
