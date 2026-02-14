#!/usr/bin/env python3
"""Debug TRT vs Original step-by-step to find divergence."""

import sys
import os
import torch
import torch.nn.functional as F
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.backends.cudnn.enabled = False

CHECKPOINT_DIR = os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero")


def load_original_model():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from safetensors.torch import load_file

    checkpoint_dir = Path(CHECKPOINT_DIR)
    config_path = checkpoint_dir / "config.json"
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
    weights_path = checkpoint_dir / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device="cuda")
    model.eval()

    return model


def compare_tensors(name, a, b, threshold=0.99):
    """Compare two tensors and print diagnostics."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False

    cos_sim = F.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0)
    ).item()

    max_diff = (a.float() - b.float()).abs().max().item()

    status = "✅" if cos_sim > threshold else "❌"
    print(f"  {status} {name}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}, "
          f"a_mean={a.float().mean().item():.6f}, b_mean={b.float().mean().item():.6f}")

    return cos_sim > threshold


def main():
    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
    from transformers.models.gemma import modeling_gemma
    from denoise_torch_trt_static import StaticDenoiseStep, load_weights_from_checkpoint

    device = "cuda"
    batch_size = 1
    action_horizon = 50
    action_dim = 32
    prefix_len = 968

    print("=" * 70)
    print("Debug: TRT vs Original Step-by-Step Comparison")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    orig_model = load_original_model()

    trt_step = StaticDenoiseStep(
        batch_size=batch_size,
        action_horizon=action_horizon,
        action_dim=action_dim,
        prefix_len=prefix_len,
    ).to(device).half()
    load_weights_from_checkpoint(trt_step, CHECKPOINT_DIR, device)
    trt_step.eval()

    # Create identical inputs
    torch.manual_seed(42)
    x_t = torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=torch.bfloat16)

    torch.manual_seed(123)
    prefix_kv_cache = []
    for _ in range(18):
        k = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        prefix_kv_cache.append((k, v))

    valid_tokens = 560
    prefix_pad_masks = torch.zeros(batch_size, prefix_len, dtype=torch.bool, device=device)
    prefix_pad_masks[:, :valid_tokens] = True

    state = torch.randn(batch_size, action_dim, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([1.0], device=device, dtype=torch.float32)

    # ============================================================
    # Step 1: Check action_in_proj
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: action_in_proj")
    print("=" * 70)

    model_dtype = orig_model.action_in_proj.weight.dtype
    orig_action_emb = orig_model.action_in_proj(x_t.to(model_dtype))

    trt_action_emb = trt_step.action_in_proj(x_t.half())

    print(f"\n  Input x_t dtype: {x_t.dtype}")
    print(f"  Original weight dtype: {orig_model.action_in_proj.weight.dtype}")
    print(f"  TRT weight dtype: {trt_step.action_in_proj.weight.dtype}")

    compare_tensors("action_in_proj output", orig_action_emb, trt_action_emb)

    # ============================================================
    # Step 2: Check adarms_cond computation
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: adarms_cond (time embedding)")
    print("=" * 70)

    time_emb = create_sinusoidal_pos_embedding(
        timestep, orig_model.action_in_proj.out_features,
        min_period=4e-3, max_period=4.0,
        device=torch.device(device)
    ).to(model_dtype)

    with torch.no_grad():
        x = orig_model.time_mlp_in(time_emb)
        x = F.silu(x)
        x = orig_model.time_mlp_out(x)
        adarms_cond = F.silu(x)

    print(f"\n  adarms_cond shape: {adarms_cond.shape}")
    print(f"  adarms_cond dtype: {adarms_cond.dtype}")
    print(f"  adarms_cond mean: {adarms_cond.float().mean().item():.6f}")

    # ============================================================
    # Step 3: Check RoPE
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3: RoPE embeddings")
    print("=" * 70)

    suffix_pad_masks = torch.ones(batch_size, action_horizon, device=device, dtype=torch.bool)
    prefix_offsets = torch.sum(prefix_pad_masks.long(), dim=-1, keepdim=True)
    suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1

    # Original RoPE
    paligemma_lm = orig_model.paligemma_with_expert.paligemma.language_model
    dummy = torch.zeros(batch_size, action_horizon, 256, device=device, dtype=orig_action_emb.dtype)
    orig_cos, orig_sin = paligemma_lm.rotary_emb(dummy, suffix_position_ids)

    # TRT RoPE
    trt_cos, trt_sin = trt_step.rotary_emb(trt_action_emb, suffix_position_ids)

    print(f"\n  suffix_position_ids range: {suffix_position_ids.min().item()} - {suffix_position_ids.max().item()}")
    compare_tensors("RoPE cos", orig_cos, trt_cos)
    compare_tensors("RoPE sin", orig_sin, trt_sin)

    # ============================================================
    # Step 4: Check Layer 0 input_layernorm
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 4: Layer 0 input_layernorm")
    print("=" * 70)

    gemma_expert = orig_model.paligemma_with_expert.gemma_expert.model
    orig_layer0 = gemma_expert.layers[0]
    trt_layer0 = trt_step.layers[0]

    suffix_embs = orig_action_emb.to(torch.bfloat16)

    with torch.no_grad():
        orig_normed, orig_gate = orig_layer0.input_layernorm(suffix_embs, cond=adarms_cond)

    with torch.no_grad():
        trt_normed, trt_gate = trt_layer0.input_layernorm(trt_action_emb, cond=adarms_cond.half())

    compare_tensors("input_layernorm output", orig_normed, trt_normed)
    compare_tensors("input_layernorm gate", orig_gate, trt_gate)

    # ============================================================
    # Step 5: Check Q, K, V projections
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 5: Q, K, V projections (Layer 0)")
    print("=" * 70)

    input_shape = orig_normed.shape[:-1]
    hidden_shape = (*input_shape, -1, orig_layer0.self_attn.head_dim)

    with torch.no_grad():
        orig_q = orig_layer0.self_attn.q_proj(orig_normed).view(hidden_shape).transpose(1, 2)
        orig_k = orig_layer0.self_attn.k_proj(orig_normed).view(hidden_shape).transpose(1, 2)
        orig_v = orig_layer0.self_attn.v_proj(orig_normed).view(hidden_shape).transpose(1, 2)

    with torch.no_grad():
        trt_q = trt_layer0.self_attn.q_proj(trt_normed)
        trt_q = trt_q.view(batch_size, action_horizon, 8, 256).transpose(1, 2)
        trt_k = trt_layer0.self_attn.k_proj(trt_normed)
        trt_k = trt_k.view(batch_size, action_horizon, 1, 256).transpose(1, 2)
        trt_v = trt_layer0.self_attn.v_proj(trt_normed)
        trt_v = trt_v.view(batch_size, action_horizon, 1, 256).transpose(1, 2)

    compare_tensors("Q projection", orig_q, trt_q)
    compare_tensors("K projection", orig_k, trt_k)
    compare_tensors("V projection", orig_v, trt_v)

    # ============================================================
    # Step 6: Check full single step output
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 6: Full single step output")
    print("=" * 70)

    with torch.no_grad():
        orig_output = orig_model.denoise_step_with_cache(
            state, prefix_kv_cache, prefix_pad_masks, x_t, timestep
        )

    # Prepare TRT inputs (no mask - matching original model)
    cached_keys = torch.stack([kv[0] for kv in prefix_kv_cache], dim=0)
    cached_values = torch.stack([kv[1] for kv in prefix_kv_cache], dim=0)

    with torch.no_grad():
        trt_output = trt_step(
            x_t.half(),
            suffix_position_ids,
            adarms_cond.half(),
            cached_keys.half(),
            cached_values.half(),
        )

    compare_tensors("Full step output", orig_output, trt_output)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
