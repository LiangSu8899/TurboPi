#!/usr/bin/env python3
"""Step-by-step comparison of full inference between JAX and PyTorch."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

print("=" * 60)
print("Full Inference Step-by-Step Comparison")
print("=" * 60)

np.random.seed(42)

# Create test inputs
batch_size = 1
image_np = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
image2_np = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
image3_np = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
state_np = np.random.randn(batch_size, 32).astype(np.float32) * 0.1
token_ids_np = np.zeros((batch_size, 64), dtype=np.int32)
token_ids_np[:, :5] = [100, 200, 300, 400, 500]
token_mask_np = np.zeros((batch_size, 64), dtype=bool)
token_mask_np[:, :5] = True
noise_np = np.random.randn(batch_size, 50, 32).astype(np.float32)

# ============== JAX ==============
print("\n" + "=" * 60)
print("JAX Model")
print("=" * 60)

import jax
import jax.numpy as jnp
from openpi.training import config as _config
from openpi.models import model as _model

train_config = _config.get_config("pi05_libero")
jax_params = _model.restore_params("/openpi_cache/openpi-assets/checkpoints/pi05_libero/params", dtype=jnp.float32)
jax_model = train_config.model.load(jax_params)

# Create JAX observation
jax_observation = _model.Observation(
    images={
        "base_0_rgb": jnp.array(image_np),
        "left_wrist_0_rgb": jnp.array(image2_np),
        "right_wrist_0_rgb": jnp.array(image3_np),
    },
    image_masks={
        "base_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
        "left_wrist_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
        "right_wrist_0_rgb": jnp.zeros(batch_size, dtype=jnp.bool_),
    },
    state=jnp.array(state_np),
    tokenized_prompt=jnp.array(token_ids_np),
    tokenized_prompt_mask=jnp.array(token_mask_np),
)

# Step 1: Preprocess observation
print("\n--- Step 1: Preprocess observation ---")
jax_obs_processed = _model.preprocess_observation(None, jax_observation, train=False)
print(f"  State shape: {jax_obs_processed.state.shape}")
for key in jax_obs_processed.images:
    img = jax_obs_processed.images[key]
    print(f"  Image {key}: shape={np.array(img).shape}, range=[{float(np.array(img).min()):.4f}, {float(np.array(img).max()):.4f}]")

# Step 2: Embed prefix
print("\n--- Step 2: Embed prefix ---")
jax_prefix_tokens, jax_prefix_mask, jax_prefix_ar_mask = jax_model.embed_prefix(jax_obs_processed)
jax_prefix_tokens_np = np.asarray(jax_prefix_tokens, dtype=np.float32)
print(f"  Prefix tokens shape: {jax_prefix_tokens_np.shape}")
print(f"  Prefix tokens range: [{float(jax_prefix_tokens_np.min()):.4f}, {float(jax_prefix_tokens_np.max()):.4f}]")
print(f"  Prefix tokens mean: {float(jax_prefix_tokens_np.mean()):.6f}")
print(f"  Prefix mask: {np.array(jax_prefix_mask)}")

# Step 3: Forward prefix through LLM to get KV cache
print("\n--- Step 3: Forward prefix through LLM ---")
from openpi.models.pi0 import make_attn_mask
jax_prefix_attn_mask = make_attn_mask(jax_prefix_mask, jax_prefix_ar_mask)
jax_positions = jnp.cumsum(jax_prefix_mask, axis=1) - 1
_, jax_kv_cache = jax_model.PaliGemma.llm([jax_prefix_tokens, None], mask=jax_prefix_attn_mask, positions=jax_positions)
print(f"  KV cache type: {type(jax_kv_cache)}")
print(f"  KV cache len: {len(jax_kv_cache)}")
# Check the structure
if jax_kv_cache:
    print(f"  First element type: {type(jax_kv_cache[0])}")
    if isinstance(jax_kv_cache[0], (list, tuple)):
        print(f"  First element len: {len(jax_kv_cache[0])}")
        if len(jax_kv_cache[0]) >= 2:
            k = jax_kv_cache[0][0]
            v = jax_kv_cache[0][1]
            print(f"  First layer K shape: {np.array(k).shape}")
            print(f"  First layer V shape: {np.array(v).shape}")
            print(f"  First layer K mean: {float(np.asarray(k, dtype=np.float32).mean()):.6f}")

# Step 4: Embed suffix for first denoising step
print("\n--- Step 4: Embed suffix (t=1.0) ---")
jax_noise = jnp.array(noise_np)
jax_time = jnp.array(1.0)
jax_suffix_tokens, jax_suffix_mask, jax_suffix_ar_mask, jax_adarms_cond = jax_model.embed_suffix(
    jax_obs_processed, jax_noise, jnp.broadcast_to(jax_time, batch_size)
)
jax_suffix_tokens_np = np.asarray(jax_suffix_tokens, dtype=np.float32)
print(f"  Suffix tokens shape: {jax_suffix_tokens_np.shape}")
print(f"  Suffix tokens range: [{float(jax_suffix_tokens_np.min()):.4f}, {float(jax_suffix_tokens_np.max()):.4f}]")
print(f"  Suffix tokens mean: {float(jax_suffix_tokens_np.mean()):.6f}")
print(f"  AdaRMS cond shape: {np.array(jax_adarms_cond).shape if jax_adarms_cond is not None else 'None'}")
if jax_adarms_cond is not None:
    jax_adarms_np = np.asarray(jax_adarms_cond, dtype=np.float32)
    print(f"  AdaRMS cond range: [{float(jax_adarms_np.min()):.4f}, {float(jax_adarms_np.max()):.4f}]")
    print(f"  AdaRMS cond first 10: {jax_adarms_np[0, :10]}")

# Step 5: Forward suffix through LLM
print("\n--- Step 5: Forward suffix through LLM ---")
import einops
jax_suffix_attn_mask = make_attn_mask(jax_suffix_mask, jax_suffix_ar_mask)
jax_prefix_attn_mask_for_suffix = einops.repeat(jax_prefix_mask, "b p -> b s p", s=jax_suffix_tokens.shape[1])
jax_full_attn_mask = jnp.concatenate([jax_prefix_attn_mask_for_suffix, jax_suffix_attn_mask], axis=-1)
jax_suffix_positions = jnp.sum(jax_prefix_mask, axis=-1)[:, None] + jnp.cumsum(jax_suffix_mask, axis=-1) - 1

(_, jax_suffix_out), _ = jax_model.PaliGemma.llm(
    [None, jax_suffix_tokens],
    mask=jax_full_attn_mask,
    positions=jax_suffix_positions,
    kv_cache=jax_kv_cache,
    adarms_cond=[None, jax_adarms_cond],
)
jax_suffix_out_np = np.asarray(jax_suffix_out, dtype=np.float32)
print(f"  Suffix output shape: {jax_suffix_out_np.shape}")
print(f"  Suffix output range: [{float(jax_suffix_out_np.min()):.4f}, {float(jax_suffix_out_np.max()):.4f}]")
print(f"  Suffix output mean: {float(jax_suffix_out_np.mean()):.6f}")

# Step 6: Action output projection
print("\n--- Step 6: Action output projection ---")
jax_action_input = jax_suffix_out[:, -jax_model.action_horizon:]
jax_v_t = jax_model.action_out_proj(jax_action_input)
jax_v_t_np = np.asarray(jax_v_t, dtype=np.float32)
print(f"  v_t shape: {jax_v_t_np.shape}")
print(f"  v_t range: [{float(jax_v_t_np.min()):.4f}, {float(jax_v_t_np.max()):.4f}]")
print(f"  v_t mean: {float(jax_v_t_np.mean()):.6f}")
print(f"  v_t first action: {jax_v_t_np[0, 0, :7]}")

# ============== PyTorch ==============
print("\n" + "=" * 60)
print("PyTorch Model")
print("=" * 60)

import torch
import safetensors.torch
from openpi.models_pytorch import pi0_pytorch

pytorch_model = pi0_pytorch.PI0Pytorch(config=train_config.model)
safetensors.torch.load_model(pytorch_model, "/openpi_cache/checkpoints/pi05_libero/model.safetensors", strict=False)

# Fix weight tying
paligemma = pytorch_model.paligemma_with_expert.paligemma
embed_tokens = paligemma.model.language_model.embed_tokens.weight
lm_head = paligemma.lm_head.weight
if embed_tokens.shape == lm_head.shape:
    with torch.no_grad():
        embed_tokens.copy_(lm_head)
    print("Fixed weight tying")

pytorch_model.eval()

# Create PyTorch observation (NCHW format)
pt_observation = _model.Observation(
    images={
        "base_0_rgb": torch.from_numpy(image_np).permute(0, 3, 1, 2),
        "left_wrist_0_rgb": torch.from_numpy(image2_np).permute(0, 3, 1, 2),
        "right_wrist_0_rgb": torch.from_numpy(image3_np).permute(0, 3, 1, 2),
    },
    image_masks={
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool),
    },
    state=torch.from_numpy(state_np),
    tokenized_prompt=torch.from_numpy(token_ids_np.astype(np.int64)),
    tokenized_prompt_mask=torch.from_numpy(token_mask_np),
)

# Step 1: Preprocess observation
print("\n--- Step 1: Preprocess observation ---")
with torch.no_grad():
    pt_images, pt_img_masks, pt_lang_tokens, pt_lang_masks, pt_state = pytorch_model._preprocess_observation(pt_observation, train=False)
print(f"  State shape: {pt_state.shape}")
for i, (img, mask) in enumerate(zip(pt_images, pt_img_masks)):
    print(f"  Image {i}: shape={img.shape}, range=[{img.min().item():.4f}, {img.max().item():.4f}], mask={mask}")

# Step 2: Embed prefix
print("\n--- Step 2: Embed prefix ---")
with torch.no_grad():
    pt_prefix_embs, pt_prefix_pad_masks, pt_prefix_att_masks = pytorch_model.embed_prefix(
        pt_images, pt_img_masks, pt_lang_tokens, pt_lang_masks
    )
    pt_prefix_embs_np = pt_prefix_embs.float().numpy()
print(f"  Prefix tokens shape: {pt_prefix_embs_np.shape}")
print(f"  Prefix tokens range: [{pt_prefix_embs_np.min():.4f}, {pt_prefix_embs_np.max():.4f}]")
print(f"  Prefix tokens mean: {pt_prefix_embs_np.mean():.6f}")
print(f"  Prefix pad mask: {pt_prefix_pad_masks.numpy()}")

# Step 3: Forward prefix through LLM to get KV cache
print("\n--- Step 3: Forward prefix through LLM ---")
with torch.no_grad():
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    pt_prefix_att_2d_masks = make_att_2d_masks(pt_prefix_pad_masks, pt_prefix_att_masks)
    pt_prefix_att_2d_masks_4d = pytorch_model._prepare_attention_masks_4d(pt_prefix_att_2d_masks)
    pt_prefix_position_ids = torch.cumsum(pt_prefix_pad_masks, dim=1) - 1

    pytorch_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, pt_past_key_values = pytorch_model.paligemma_with_expert.forward(
        attention_mask=pt_prefix_att_2d_masks_4d,
        position_ids=pt_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[pt_prefix_embs, None],
        use_cache=True,
    )
print(f"  KV cache layers: {len(pt_past_key_values)}")
if pt_past_key_values:
    k = pt_past_key_values[0][0]  # key_states
    v = pt_past_key_values[0][1]  # value_states
    print(f"  First layer K shape: {k.shape}")
    print(f"  First layer V shape: {v.shape}")
    print(f"  First layer K mean: {k.float().mean().item():.6f}")

# Step 4: Embed suffix for first denoising step
print("\n--- Step 4: Embed suffix (t=1.0) ---")
with torch.no_grad():
    pt_noise = torch.from_numpy(noise_np)
    pt_time = torch.tensor([1.0], dtype=torch.float32)
    pt_suffix_embs, pt_suffix_pad_masks, pt_suffix_att_masks, pt_adarms_cond = pytorch_model.embed_suffix(
        pt_state, pt_noise, pt_time
    )
    pt_suffix_embs_np = pt_suffix_embs.float().numpy()
print(f"  Suffix tokens shape: {pt_suffix_embs_np.shape}")
print(f"  Suffix tokens range: [{pt_suffix_embs_np.min():.4f}, {pt_suffix_embs_np.max():.4f}]")
print(f"  Suffix tokens mean: {pt_suffix_embs_np.mean():.6f}")
print(f"  AdaRMS cond shape: {pt_adarms_cond.shape if pt_adarms_cond is not None else 'None'}")
if pt_adarms_cond is not None:
    pt_adarms_np = pt_adarms_cond.float().numpy()
    print(f"  AdaRMS cond range: [{pt_adarms_np.min():.4f}, {pt_adarms_np.max():.4f}]")
    print(f"  AdaRMS cond first 10: {pt_adarms_np[0, :10]}")

# Step 5: Forward suffix through LLM
print("\n--- Step 5: Forward suffix through LLM ---")
with torch.no_grad():
    # Convert suffix_embs to bfloat16 if needed
    if pytorch_model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        pt_suffix_embs_bf16 = pt_suffix_embs.to(dtype=torch.bfloat16)
    else:
        pt_suffix_embs_bf16 = pt_suffix_embs

    suffix_len = pt_suffix_pad_masks.shape[1]
    prefix_len = pt_prefix_pad_masks.shape[1]
    pt_prefix_pad_2d_masks = pt_prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    pt_suffix_att_2d_masks = make_att_2d_masks(pt_suffix_pad_masks, pt_suffix_att_masks)
    pt_full_att_2d_masks = torch.cat([pt_prefix_pad_2d_masks, pt_suffix_att_2d_masks], dim=2)
    pt_full_att_2d_masks_4d = pytorch_model._prepare_attention_masks_4d(pt_full_att_2d_masks)

    pt_prefix_offsets = torch.sum(pt_prefix_pad_masks, dim=-1)[:, None]
    pt_suffix_position_ids = pt_prefix_offsets + torch.cumsum(pt_suffix_pad_masks, dim=1) - 1

    pytorch_model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    pt_outputs_embeds, _ = pytorch_model.paligemma_with_expert.forward(
        attention_mask=pt_full_att_2d_masks_4d,
        position_ids=pt_suffix_position_ids,
        past_key_values=pt_past_key_values,
        inputs_embeds=[None, pt_suffix_embs_bf16],
        use_cache=False,
        adarms_cond=[None, pt_adarms_cond],
    )
    pt_suffix_out = pt_outputs_embeds[1]
    pt_suffix_out_np = pt_suffix_out.float().numpy()
print(f"  Suffix output shape: {pt_suffix_out_np.shape}")
print(f"  Suffix output range: [{pt_suffix_out_np.min():.4f}, {pt_suffix_out_np.max():.4f}]")
print(f"  Suffix output mean: {pt_suffix_out_np.mean():.6f}")

# Step 6: Action output projection
print("\n--- Step 6: Action output projection ---")
with torch.no_grad():
    pt_action_input = pt_suffix_out[:, -pytorch_model.config.action_horizon:]
    pt_action_input_f32 = pt_action_input.to(dtype=torch.float32)
    pt_v_t = pytorch_model.action_out_proj(pt_action_input_f32)
    pt_v_t_np = pt_v_t.numpy()
print(f"  v_t shape: {pt_v_t_np.shape}")
print(f"  v_t range: [{pt_v_t_np.min():.4f}, {pt_v_t_np.max():.4f}]")
print(f"  v_t mean: {pt_v_t_np.mean():.6f}")
print(f"  v_t first action: {pt_v_t_np[0, 0, :7]}")

# ============== Comparison ==============
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)

print("\n| Step | JAX Mean | PyTorch Mean | Diff |")
print("|------|----------|--------------|------|")
print(f"| Prefix embs | {float(jax_prefix_tokens_np.mean()):.6f} | {pt_prefix_embs_np.mean():.6f} | {abs(float(jax_prefix_tokens_np.mean()) - pt_prefix_embs_np.mean()):.6f} |")
print(f"| Suffix embs | {float(jax_suffix_tokens_np.mean()):.6f} | {pt_suffix_embs_np.mean():.6f} | {abs(float(jax_suffix_tokens_np.mean()) - pt_suffix_embs_np.mean()):.6f} |")
print(f"| Suffix out | {float(jax_suffix_out_np.mean()):.6f} | {pt_suffix_out_np.mean():.6f} | {abs(float(jax_suffix_out_np.mean()) - pt_suffix_out_np.mean()):.6f} |")
print(f"| v_t | {float(jax_v_t_np.mean()):.6f} | {pt_v_t_np.mean():.6f} | {abs(float(jax_v_t_np.mean()) - pt_v_t_np.mean()):.6f} |")

print("\n| Step | JAX Range | PyTorch Range |")
print("|------|-----------|---------------|")
print(f"| v_t | [{float(jax_v_t_np.min()):.4f}, {float(jax_v_t_np.max()):.4f}] | [{pt_v_t_np.min():.4f}, {pt_v_t_np.max():.4f}] |")

print("\nFirst action comparison:")
print(f"  JAX:     {jax_v_t_np[0, 0, :7]}")
print(f"  PyTorch: {pt_v_t_np[0, 0, :7]}")
print(f"  Diff:    {jax_v_t_np[0, 0, :7] - pt_v_t_np[0, 0, :7]}")

print("\n" + "=" * 60)
