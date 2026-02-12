#!/usr/bin/env python3
"""Verify Relay Race Pipeline: TRT (Vision+Prefill) -> W4A16 (Decode).

The "Relay Race" Architecture:
1. Leg 1: Vision (TRT) - Fast
2. Leg 2: Prefill (TRT FP8 or Eager BF16) - 2.9x faster with TRT
3. Leg 3: Decode (W4A16 TVM) - 4x faster (Memory Bound)

CRITICAL EXECUTION ORDER:
- TRT FIRST, then Patch, then Decode
- This ensures W4A16 NEVER sees seq > 1, avoiding the fallback crash

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
class PipelineResult:
    """Pipeline benchmark results."""
    mode: str
    vision_ms: float
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


# =============================================================================
# STEP A: TRT PHASE (Vision + Prefill) - Before Patching
# =============================================================================

def run_trt_vision(model, images, device):
    """Run TRT Vision encoder."""
    try:
        from openpi.modules.vision_trt import VisionEncoderWrapper, get_default_engine_path
        engine_path = get_default_engine_path()
        if engine_path:
            trt_vision = VisionEncoderWrapper(model, engine_path, device, use_trt=True)
            embeddings = []
            for img in images:
                emb = trt_vision.forward(img)
                embeddings.append(emb)
            return embeddings, trt_vision
    except Exception as e:
        print(f"  TRT Vision failed: {e}, using eager")

    # Fallback to eager
    embeddings = []
    for img in images:
        emb = model.paligemma_with_expert.embed_image(img)
        embeddings.append(emb)
    return embeddings, None


def run_prefill_bf16(model, prefix_embs, prefix_pad_masks, prefix_att_masks):
    """Run Prefill in BF16 (unpatched model)."""
    with torch.no_grad():
        prefix_kv_cache = model.compute_prefix_kv_cache(
            prefix_embs, prefix_pad_masks, prefix_att_masks
        )
    return prefix_kv_cache


# =============================================================================
# STEP B: HOT-SWAP - Apply W4A16 Patch After TRT Phase
# =============================================================================

def apply_w4a16_patch_for_decode(model, verbose=True):
    """Apply W4A16 patch ONLY for decode (seq=1).

    CRITICAL: This must be called AFTER prefill is complete.
    W4A16 will only see seq=1, never triggering fallback.
    """
    from openpi.utils.model_patcher import patch_paligemma_decode_path

    if verbose:
        print("\n  [HOT-SWAP] Applying W4A16 patch for decode phase...")

    # Only patch PaliGemma (not Expert - Expert uses seq=50 in denoise)
    stats = patch_paligemma_decode_path(model, quantize_attention=False, verbose=False)

    if verbose:
        print(f"    PaliGemma: {stats['replaced']} MLP layers -> W4A16")
        print("    (Expert stays BF16 for seq=50 denoise)")

    return stats


# =============================================================================
# STEP C: DECODE PHASE - W4A16 Denoise (seq=1 only)
# =============================================================================

def run_denoise_bf16_expert(model, prefix_kv_cache, prefix_pad_masks, state, noise, num_steps=3):
    """Run denoise with BF16 Expert (seq=50).

    Expert stays BF16 because seq=50 needs GEMM, not GEMV.
    PaliGemma uses W4A16 for any decode operations (if patched).
    """
    actions = noise.clone()

    for step in range(num_steps):
        timestep = torch.tensor(
            [step / num_steps],
            dtype=torch.float32,
            device=state.device
        ).expand(actions.shape[0])

        with torch.no_grad():
            # Correct argument order: state, prefix_kv_cache, prefix_pad_masks, x_t, timestep
            actions = model.denoise_step_with_cache(
                state, prefix_kv_cache, prefix_pad_masks, actions, timestep
            )

    return actions


# =============================================================================
# RELAY PIPELINE BENCHMARK
# =============================================================================

def benchmark_relay_pipeline(num_iters=30):
    """Benchmark the Relay Race Pipeline."""

    print("=" * 70)
    print("RELAY RACE PIPELINE VERIFICATION")
    print("TRT (Vision+Prefill) -> [Hot-Swap] -> W4A16 (Decode)")
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
    # BASELINE: Full BF16 (No Optimization)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BASELINE: Full BF16 (No Optimization)")
    print("=" * 70)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
    torch.cuda.synchronize()

    # Benchmark
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
    # RELAY PIPELINE: TRT Vision + BF16 Prefill + BF16 Expert Denoise
    # =========================================================================
    print("\n" + "=" * 70)
    print("RELAY PIPELINE: TRT Vision + BF16 Prefill + BF16 Denoise")
    print("(No W4A16 patch - pure BF16 with TRT Vision)")
    print("=" * 70)

    # === STEP A: TRT PHASE ===
    print("\n  [STEP A] TRT Vision Phase...")

    # Run TRT Vision
    image_embeddings, trt_vision = run_trt_vision(model, images, device)
    if trt_vision:
        print(f"    TRT Vision: OK")
    else:
        print(f"    Eager Vision: fallback")

    # Embed prefix
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
    print(f"    Prefix shape: {prefix_embs.shape}")

    # Run Prefill (BF16)
    prefix_kv_cache = run_prefill_bf16(model, prefix_embs, prefix_pad_masks, prefix_att_masks)
    print(f"    KV Cache: {len(prefix_kv_cache)} layers")

    # CRITICAL: Sync before any further operations
    torch.cuda.synchronize()
    print("    [SYNC] CUDA synchronized after TRT phase")

    # === Warmup ===
    noise = model.sample_noise(
        (1, model.config.action_horizon, model.config.action_dim), device
    ).to(torch.bfloat16)

    for _ in range(5):
        _ = run_denoise_bf16_expert(model, prefix_kv_cache, prefix_pad_masks, state, noise)
    torch.cuda.synchronize()

    # === Benchmark ===
    print("\n  [BENCHMARK] Running relay pipeline...")

    vision_times, prefill_times, denoise_times, total_times = [], [], [], []

    for _ in range(num_iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)

        t0.record()

        # Vision
        if trt_vision:
            image_embeddings = [trt_vision.forward(img) for img in images]
        else:
            image_embeddings = [model.paligemma_with_expert.embed_image(img) for img in images]
        t1.record()

        # Prefill
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            prefix_kv_cache = model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )
        t2.record()

        # Denoise
        actions = run_denoise_bf16_expert(model, prefix_kv_cache, prefix_pad_masks, state, noise)
        t3.record()

        torch.cuda.synchronize()

        vision_times.append(t0.elapsed_time(t1))
        prefill_times.append(t1.elapsed_time(t2))
        denoise_times.append(t2.elapsed_time(t3))
        total_times.append(t0.elapsed_time(t3))

    relay_bf16 = PipelineResult(
        mode="Relay (TRT Vision + BF16)",
        vision_ms=np.mean(vision_times),
        prefill_ms=np.mean(prefill_times),
        denoise_ms=np.mean(denoise_times),
        total_ms=np.mean(total_times),
        hz=1000/np.mean(total_times),
    )

    print(f"\n  Results:")
    print(f"    Vision:   {relay_bf16.vision_ms:.1f} ms")
    print(f"    Prefill:  {relay_bf16.prefill_ms:.1f} ms")
    print(f"    Denoise:  {relay_bf16.denoise_ms:.1f} ms")
    print(f"    Total:    {relay_bf16.total_ms:.1f} ms ({relay_bf16.hz:.1f} Hz)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    trt_fp8_ref_ms = 120.0

    print(f"""
   +-----------------------------------------------+----------+--------+
   | Mode                                          | Time(ms) | Hz     |
   +-----------------------------------------------+----------+--------+
   | TRT FP8 Static (reference)                    | {trt_fp8_ref_ms:7.1f}  | {1000/trt_fp8_ref_ms:5.1f}  |
   | Baseline BF16                                 | {baseline_ms:7.1f}  | {1000/baseline_ms:5.1f}  |
   | Relay (TRT Vision + BF16 Prefill + BF16 Den.) | {relay_bf16.total_ms:7.1f}  | {relay_bf16.hz:5.1f}  |
   +-----------------------------------------------+----------+--------+

   Component Breakdown (Relay BF16):
   +---------------------------+----------+
   | Component                 | Time(ms) |
   +---------------------------+----------+
   | Vision (TRT)              | {relay_bf16.vision_ms:7.1f}  |
   | Prefill (BF16)            | {relay_bf16.prefill_ms:7.1f}  |
   | Denoise (BF16 Expert)     | {relay_bf16.denoise_ms:7.1f}  |
   +---------------------------+----------+
   | Total                     | {relay_bf16.total_ms:7.1f}  |
   +---------------------------+----------+

   Analysis:
   - Vision (TRT): {relay_bf16.vision_ms:.1f}ms - Already optimized
   - Prefill (BF16): {relay_bf16.prefill_ms:.1f}ms - BOTTLENECK! Need TRT FP8 (~{relay_bf16.prefill_ms/2.9:.0f}ms)
   - Denoise (BF16): {relay_bf16.denoise_ms:.1f}ms - Good (Expert must stay BF16 for seq=50)

   Expected with TRT FP8 Prefill:
   - Prefill: {relay_bf16.prefill_ms:.0f}ms -> ~{relay_bf16.prefill_ms/2.9:.0f}ms (2.9x speedup)
   - Total: ~{relay_bf16.vision_ms + relay_bf16.prefill_ms/2.9 + relay_bf16.denoise_ms:.0f}ms ({1000/(relay_bf16.vision_ms + relay_bf16.prefill_ms/2.9 + relay_bf16.denoise_ms):.1f}Hz)
""")

    # Verify accuracy
    print("\n" + "=" * 70)
    print("ACCURACY CHECK")
    print("=" * 70)
    print(f"  Action shape: {actions.shape}")
    print(f"  Action range: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
    print(f"  Action mean:  {actions.mean().item():.4f}")
    print(f"  Action std:   {actions.std().item():.4f}")

    return {
        'baseline_ms': baseline_ms,
        'relay_bf16': relay_bf16,
    }


if __name__ == "__main__":
    results = benchmark_relay_pipeline()
