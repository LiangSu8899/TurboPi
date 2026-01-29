#!/usr/bin/env python3
"""Step-by-step comparison of JAX and PyTorch models."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

print("=" * 60)
print("Step-by-Step JAX vs PyTorch Comparison")
print("=" * 60)

np.random.seed(42)
batch_size = 1

# Create identical inputs
base_image_np = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
left_wrist_np = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
right_wrist_np = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
state_np = np.random.randn(batch_size, 32).astype(np.float32) * 0.1
tokens_np = np.zeros((batch_size, 64), dtype=np.int32)
tokens_np[:, :5] = [100, 200, 300, 400, 500]
token_mask_np = np.zeros((batch_size, 64), dtype=bool)
token_mask_np[:, :5] = True
noise_np = np.random.randn(batch_size, 50, 32).astype(np.float32)

# ========== Load JAX Model ==========
print("\n[JAX] Loading model...")
import jax
import jax.numpy as jnp
from openpi.training import config as _config
from openpi.models import model as _model

train_config = _config.get_config("pi05_libero")
jax_params = _model.restore_params(
    "/openpi_cache/openpi-assets/checkpoints/pi05_libero/params",
    dtype=jnp.float32
)
jax_model = train_config.model.load(jax_params)
print("[JAX] Model loaded")

# ========== Load PyTorch Model ==========
print("\n[PyTorch] Loading model...")
import torch
import safetensors.torch
from openpi.models_pytorch import pi0_pytorch

pytorch_model = pi0_pytorch.PI0Pytorch(config=train_config.model)
safetensors.torch.load_model(pytorch_model, "/openpi_cache/checkpoints/pi05_libero/model.safetensors", strict=False)

# Fix weight tying
paligemma = pytorch_model.paligemma_with_expert.paligemma
with torch.no_grad():
    paligemma.model.language_model.embed_tokens.weight.copy_(paligemma.lm_head.weight)

pytorch_model.eval()
print("[PyTorch] Model loaded")

# ========== Compare time MLP ==========
print("\n" + "=" * 40)
print("Step 1: Time Embedding + MLP")
print("=" * 40)

# JAX time embedding
from openpi.models import pi0 as _pi0
time_val = 1.0
jax_time = jnp.array([time_val])
jax_time_emb = _pi0.posemb_sincos(jax_time, 1024, min_period=4e-3, max_period=4.0)
print(f"[JAX] time_emb: shape={jax_time_emb.shape}, range=[{jax_time_emb.min():.4f}, {jax_time_emb.max():.4f}]")

# Apply JAX time MLP (for pi05)
jax_time_mlp_out = jax_model.time_mlp_in(jax_time_emb)
jax_time_mlp_out = jax.nn.swish(jax_time_mlp_out)
jax_time_mlp_out = jax_model.time_mlp_out(jax_time_mlp_out)
jax_time_mlp_out = jax.nn.swish(jax_time_mlp_out)
print(f"[JAX] time_emb after MLP: range=[{jax_time_mlp_out.min():.4f}, {jax_time_mlp_out.max():.4f}]")
print(f"[JAX] time_emb mean={jax_time_mlp_out.mean():.6f}, std={jax_time_mlp_out.std():.6f}")

# PyTorch time embedding
pt_time = torch.tensor([time_val], dtype=torch.float32)
pt_time_emb = pi0_pytorch.create_sinusoidal_pos_embedding(
    pt_time, 1024, min_period=4e-3, max_period=4.0, device=pt_time.device
)
pt_time_emb = pt_time_emb.to(torch.float32)  # Convert to float32
print(f"[PyTorch] time_emb: shape={pt_time_emb.shape}, range=[{pt_time_emb.min():.4f}, {pt_time_emb.max():.4f}]")

# Apply PyTorch time MLP
with torch.no_grad():
    pt_time_mlp_out = pytorch_model.time_mlp_in(pt_time_emb)
    pt_time_mlp_out = torch.nn.functional.silu(pt_time_mlp_out)
    pt_time_mlp_out = pytorch_model.time_mlp_out(pt_time_mlp_out)
    pt_time_mlp_out = torch.nn.functional.silu(pt_time_mlp_out)
print(f"[PyTorch] time_emb after MLP: range=[{pt_time_mlp_out.min():.4f}, {pt_time_mlp_out.max():.4f}]")
print(f"[PyTorch] time_emb mean={pt_time_mlp_out.mean():.6f}, std={pt_time_mlp_out.std():.6f}")

# Compare time embeddings
jax_time_emb_np = np.array(jax_time_emb)
pt_time_emb_np = pt_time_emb.numpy()
time_emb_diff = np.abs(jax_time_emb_np - pt_time_emb_np).max()
print(f"\nTime embedding diff: {time_emb_diff:.6f}")

jax_time_mlp_np = np.array(jax_time_mlp_out)
pt_time_mlp_np = pt_time_mlp_out.numpy()
time_mlp_diff = np.abs(jax_time_mlp_np - pt_time_mlp_np).max()
print(f"Time MLP output diff: {time_mlp_diff:.6f}")

# ========== Compare action projection ==========
print("\n" + "=" * 40)
print("Step 2: Action Projection")
print("=" * 40)

jax_action = jnp.array(noise_np)
pt_action = torch.from_numpy(noise_np)

jax_action_emb = jax_model.action_in_proj(jax_action)
with torch.no_grad():
    pt_action_emb = pytorch_model.action_in_proj(pt_action)

print(f"[JAX] action_emb: shape={jax_action_emb.shape}, range=[{jax_action_emb.min():.4f}, {jax_action_emb.max():.4f}]")
print(f"[PyTorch] action_emb: shape={pt_action_emb.shape}, range=[{pt_action_emb.min():.4f}, {pt_action_emb.max():.4f}]")

jax_action_emb_np = np.array(jax_action_emb)
pt_action_emb_np = pt_action_emb.numpy()
action_emb_diff = np.abs(jax_action_emb_np - pt_action_emb_np).max()
print(f"Action embedding diff: {action_emb_diff:.6f}")

# ========== Compare action_out_proj ==========
print("\n" + "=" * 40)
print("Step 3: Action Output Projection")
print("=" * 40)

# Create dummy transformer output
dummy_transformer_out = np.random.randn(batch_size, 50, 1024).astype(np.float32)
jax_out = jnp.array(dummy_transformer_out)
pt_out = torch.from_numpy(dummy_transformer_out)

jax_final = jax_model.action_out_proj(jax_out)
with torch.no_grad():
    pt_final = pytorch_model.action_out_proj(pt_out)

print(f"[JAX] final output: shape={jax_final.shape}, range=[{jax_final.min():.4f}, {jax_final.max():.4f}]")
print(f"[PyTorch] final output: shape={pt_final.shape}, range=[{pt_final.min():.4f}, {pt_final.max():.4f}]")

jax_final_np = np.array(jax_final)
pt_final_np = pt_final.numpy()
final_diff = np.abs(jax_final_np - pt_final_np).max()
print(f"Final output diff: {final_diff:.6f}")

# ========== Summary ==========
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Time embedding diff: {time_emb_diff:.6f}")
print(f"Time MLP output diff: {time_mlp_diff:.6f}")
print(f"Action embedding diff: {action_emb_diff:.6f}")
print(f"Action out_proj diff: {final_diff:.6f}")

if time_mlp_diff > 0.01 or action_emb_diff > 0.01 or final_diff > 0.01:
    print("\nWARNING: Significant differences found!")
    if time_mlp_diff > 0.01:
        print("  - Time MLP outputs differ significantly")
    if action_emb_diff > 0.01:
        print("  - Action embeddings differ significantly")
    if final_diff > 0.01:
        print("  - Final projections differ significantly")
else:
    print("\nAll projection layers match closely!")

print("\n" + "=" * 60)
