#!/usr/bin/env python3
"""Detailed Full Pipeline Benchmark with Accurate Component Timing.

Separates timing for:
1. Vision (TRT FP16)
2. Embed Prefix (BF16) - embedding images, language tokens, state
3. Prefill (TRT FP8) - only the transformer forward pass
4. Denoise (BF16 Expert) - action denoising loop

Author: Claude Code
Date: 2026-02-12
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class DetailedPipelineResult:
    """Detailed pipeline benchmark results."""
    mode: str
    vision_ms: float
    embed_prefix_ms: float
    prefill_ms: float
    denoise_ms: float
    total_ms: float
    hz: float


def load_model_unpatched():
    """Load model WITHOUT any quantization (pure BF16)."""
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


def run_denoise_bf16_expert(model, prefix_kv_cache, prefix_pad_masks, state, noise, num_steps=3):
    """Run denoise with BF16 Expert (seq=50)."""
    actions = noise.clone()

    for step in range(num_steps):
        timestep = torch.tensor(
            [step / num_steps],
            dtype=torch.float32,
            device=state.device
        ).expand(actions.shape[0])

        with torch.no_grad():
            actions = model.denoise_step_with_cache(
                state, prefix_kv_cache, prefix_pad_masks, actions, timestep
            )

    return actions


def benchmark_detailed_pipeline(num_iters=30):
    """Benchmark with detailed component separation."""

    print("=" * 70)
    print("DETAILED PIPELINE BENCHMARK")
    print("Accurate component timing separation")
    print("=" * 70)

    # Load unpatched model
    model, config, device, max_token_len, checkpoint_dir = load_model_unpatched()
    observation = create_test_observation(device, max_token_len)

    # Preprocess observation
    import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
    obs_proc = _preprocessing.preprocess_observation_pytorch(observation, train=False)
    images = list(obs_proc.images.values())
    img_masks = list(obs_proc.image_masks.values())
    lang_tokens = obs_proc.tokenized_prompt
    lang_masks = obs_proc.tokenized_prompt_mask
    state = observation.state

    # =========================================================================
    # BASELINE: Full BF16 (MUST run BEFORE TRT FP8 compilation!)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BASELINE: Full BF16 (No Optimization)")
    print("=" * 70)

    for _ in range(3):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
    torch.cuda.synchronize()

    baseline_times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        end.record()
        torch.cuda.synchronize()
        baseline_times.append(start.elapsed_time(end))

    baseline_ms = np.mean(baseline_times)
    print(f"\n  Baseline: {baseline_ms:.1f} ms ({1000/baseline_ms:.1f} Hz)")

    # =========================================================================
    # Load TRT Vision
    # =========================================================================
    print("\n[1] Loading TRT Vision...")
    trt_vision = None
    try:
        from openpi.modules.vision_trt import VisionEncoderWrapper, get_default_engine_path
        engine_path = get_default_engine_path()
        if engine_path:
            trt_vision = VisionEncoderWrapper(model, engine_path, device, use_trt=True)
            print(f"    TRT Vision loaded: {engine_path}")
    except Exception as e:
        print(f"    TRT Vision failed: {e}")

    # =========================================================================
    # Load TRT FP8 Prefill Engine (AFTER baseline!)
    # =========================================================================
    print("\n[2] Loading TRT FP8 Prefill Engine...")
    print("    (Compiling 18 layers - takes ~6 minutes first time)")

    trt_prefill = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "torch_trt_fp8_kv_cache",
            pathlib.Path(__file__).parent.parent / "src" / "openpi" / "inference" / "torch_trt_fp8_kv_cache.py"
        )
        trt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trt_module)
        TorchTRTFP8KVCacheEngine = trt_module.TorchTRTFP8KVCacheEngine

        trt_prefill = TorchTRTFP8KVCacheEngine(
            checkpoint_dir=checkpoint_dir,
            device=device,
            compile_trt=True,
        )
        print(f"    TRT FP8 Prefill loaded: {trt_prefill._trt_compiled_count}/18 layers")
    except Exception as e:
        print(f"    TRT FP8 Prefill failed: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # DETAILED BENCHMARK
    # =========================================================================
    print("\n" + "=" * 70)
    print("DETAILED PIPELINE: TRT Vision + Embed + TRT FP8 Prefill + BF16 Denoise")
    print("=" * 70)

    if trt_prefill is None:
        print("\n  [SKIP] TRT FP8 Prefill not available")
        return {'baseline_ms': baseline_ms}

    # Warmup
    print("\n  [WARMUP]...")

    if trt_vision:
        for img in images:
            _ = trt_vision.forward(img)

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

    prefix_kv_cache = trt_prefill.infer_list(prefix_embs, None, None)
    torch.cuda.synchronize()

    noise = model.sample_noise(
        (1, model.config.action_horizon, model.config.action_dim), device
    ).to(torch.bfloat16)

    for _ in range(5):
        _ = run_denoise_bf16_expert(model, prefix_kv_cache, prefix_pad_masks, state, noise)
    torch.cuda.synchronize()

    # =========================================================================
    # Benchmark with 4 timing points
    # =========================================================================
    print("\n  [BENCHMARK] Running detailed pipeline...")

    vision_times, embed_times, prefill_times, denoise_times, total_times = [], [], [], [], []

    for _ in range(num_iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t4 = torch.cuda.Event(enable_timing=True)

        t0.record()

        # === VISION (TRT) ===
        if trt_vision:
            image_embeddings = [trt_vision.forward(img) for img in images]
        else:
            image_embeddings = [model.paligemma_with_expert.embed_image(img) for img in images]
        t1.record()

        # === EMBED PREFIX (BF16) ===
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
        t2.record()

        # === TRT FP8 PREFILL ONLY ===
        prefix_kv_cache = trt_prefill.infer_list(prefix_embs, None, None)
        t3.record()

        # === DENOISE (BF16 Expert) ===
        actions = run_denoise_bf16_expert(model, prefix_kv_cache, prefix_pad_masks, state, noise)
        t4.record()

        torch.cuda.synchronize()

        vision_times.append(t0.elapsed_time(t1))
        embed_times.append(t1.elapsed_time(t2))
        prefill_times.append(t2.elapsed_time(t3))
        denoise_times.append(t3.elapsed_time(t4))
        total_times.append(t0.elapsed_time(t4))

    result = DetailedPipelineResult(
        mode="Detailed Pipeline",
        vision_ms=np.mean(vision_times),
        embed_prefix_ms=np.mean(embed_times),
        prefill_ms=np.mean(prefill_times),
        denoise_ms=np.mean(denoise_times),
        total_ms=np.mean(total_times),
        hz=1000/np.mean(total_times),
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("DETAILED PERFORMANCE SUMMARY")
    print("=" * 70)

    speedup = baseline_ms / result.total_ms
    bf16_prefill_ms = 122.0  # Reference from relay test (includes embed)

    print(f"""
   +-----------------------------------------------+----------+--------+
   | Mode                                          | Time(ms) | Hz     |
   +-----------------------------------------------+----------+--------+
   | Baseline BF16                                 | {baseline_ms:7.1f}  | {1000/baseline_ms:5.1f}  |
   | Optimized Pipeline                            | {result.total_ms:7.1f}  | {result.hz:5.1f}  |
   +-----------------------------------------------+----------+--------+
   | Speedup                                       |   {speedup:.2f}x  |        |
   +-----------------------------------------------+----------+--------+

   DETAILED Component Breakdown:
   +---------------------------+----------+----------+
   | Component                 | Time(ms) | % Total  |
   +---------------------------+----------+----------+
   | Vision (TRT)              | {result.vision_ms:7.1f}  | {result.vision_ms/result.total_ms*100:5.1f}%  |
   | Embed Prefix (BF16)       | {result.embed_prefix_ms:7.1f}  | {result.embed_prefix_ms/result.total_ms*100:5.1f}%  |
   | Prefill (TRT FP8 ONLY)    | {result.prefill_ms:7.1f}  | {result.prefill_ms/result.total_ms*100:5.1f}%  |
   | Denoise (BF16 Expert)     | {result.denoise_ms:7.1f}  | {result.denoise_ms/result.total_ms*100:5.1f}%  |
   +---------------------------+----------+----------+
   | Total                     | {result.total_ms:7.1f}  | 100.0%  |
   +---------------------------+----------+----------+

   Analysis:
   - Vision (TRT):         {result.vision_ms:.1f}ms - Already optimized
   - Embed Prefix (BF16):  {result.embed_prefix_ms:.1f}ms - Embedding overhead
   - Prefill (TRT FP8):    {result.prefill_ms:.1f}ms - Standalone was 46.67ms
   - Denoise (BF16):       {result.denoise_ms:.1f}ms - Expert stays BF16 (seq=50)

   Previous "Prefill" measurement included embed_prefix:
   - Old measurement: 82.8ms = embed_prefix + prefill
   - Now separated: embed={result.embed_prefix_ms:.1f}ms + prefill={result.prefill_ms:.1f}ms
""")

    # =========================================================================
    # Accuracy check
    # =========================================================================
    print("\n" + "=" * 70)
    print("ACCURACY CHECK")
    print("=" * 70)
    print(f"  Action shape: {actions.shape}")
    print(f"  Action range: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
    print(f"  Action mean:  {actions.mean().item():.4f}")
    print(f"  Action std:   {actions.std().item():.4f}")

    # =========================================================================
    # Run BF16 Baseline for Accuracy Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON: TRT FP8 vs BF16")
    print("=" * 70)

    # Get BF16 actions for comparison
    with torch.no_grad():
        bf16_actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)

    # Compare
    if bf16_actions.shape == actions.shape:
        diff = (actions - bf16_actions).abs()
        print(f"  BF16 Action range: [{bf16_actions.min().item():.4f}, {bf16_actions.max().item():.4f}]")
        print(f"  BF16 Action mean:  {bf16_actions.mean().item():.4f}")
        print(f"  BF16 Action std:   {bf16_actions.std().item():.4f}")
        print(f"\n  Difference Analysis:")
        print(f"    Max abs diff:    {diff.max().item():.6f}")
        print(f"    Mean abs diff:   {diff.mean().item():.6f}")
        print(f"    Relative error:  {(diff.mean() / bf16_actions.abs().mean() * 100).item():.4f}%")
    else:
        print(f"  [WARN] Shape mismatch: TRT={actions.shape}, BF16={bf16_actions.shape}")

    return {
        'baseline_ms': baseline_ms,
        'result': result,
        'speedup': speedup,
        'trt_actions': actions,
        'bf16_actions': bf16_actions if 'bf16_actions' in dir() else None,
    }


if __name__ == "__main__":
    results = benchmark_detailed_pipeline()
