#!/usr/bin/env python3
"""Compare JAX and PyTorch model outputs on CPU to identify differences."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU

import numpy as np

print("=" * 60)
print("JAX vs PyTorch Comparison (CPU)")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic test inputs
batch_size = 1
# Images should be float32 in HWC format for JAX Observation
# Use correct key names for LIBERO: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
base_image_np = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)  # HWC format
left_wrist_image_np = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
right_wrist_image_np = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)  # Empty for LIBERO
state_np = np.random.randn(batch_size, 32).astype(np.float32) * 0.1
token_ids_np = np.zeros((batch_size, 64), dtype=np.int32)
token_ids_np[:, :5] = [100, 200, 300, 400, 500]
token_mask_np = np.zeros((batch_size, 64), dtype=bool)
token_mask_np[:, :5] = True

# Create deterministic noise
noise_np = np.random.randn(batch_size, 50, 32).astype(np.float32)

print(f"\nTest inputs:")
print(f"  Base image: shape={base_image_np.shape}")
print(f"  State: shape={state_np.shape}, range=[{state_np.min():.4f}, {state_np.max():.4f}]")
print(f"  Noise: shape={noise_np.shape}")

# ============== JAX Model ==============
print("\n" + "=" * 60)
print("Loading JAX Model")
print("=" * 60)

import jax
import jax.numpy as jnp
print(f"JAX devices: {jax.devices()}")

from openpi.training import config as _config
from openpi.models import model as _model
from openpi.models import pi0 as _pi0

# Load JAX checkpoint
jax_checkpoint_dir = "/openpi_cache/openpi-assets/checkpoints/pi05_libero"
train_config = _config.get_config("pi05_libero")

print(f"\nLoading JAX params from {jax_checkpoint_dir}...")
jax_params = _model.restore_params(f"{jax_checkpoint_dir}/params", dtype=jnp.float32)
print("JAX params loaded!")

# Create JAX model
print("Creating JAX model...")
jax_model = train_config.model.load(jax_params)
print("JAX model created!")

# Create observation for JAX (images in HWC format)
jax_observation = _model.Observation(
    images={
        "base_0_rgb": jnp.array(base_image_np),  # (B, H, W, C)
        "left_wrist_0_rgb": jnp.array(left_wrist_image_np),
        "right_wrist_0_rgb": jnp.array(right_wrist_image_np),
    },
    image_masks={
        "base_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
        "left_wrist_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
        "right_wrist_0_rgb": jnp.zeros(batch_size, dtype=jnp.bool_),  # Empty camera
    },
    state=jnp.array(state_np),
    tokenized_prompt=jnp.array(token_ids_np),
    tokenized_prompt_mask=jnp.array(token_mask_np),
)

# Run JAX inference
print("\nRunning JAX inference...")
jax_noise = jnp.array(noise_np)
jax_rng = jax.random.key(0)
jax_actions = jax_model.sample_actions(jax_rng, jax_observation, num_steps=10, noise=jax_noise)
jax_actions_np = np.array(jax_actions)

print(f"JAX actions:")
print(f"  shape: {jax_actions_np.shape}")
print(f"  range: [{jax_actions_np.min():.6f}, {jax_actions_np.max():.6f}]")
print(f"  mean: {jax_actions_np.mean():.6f}, std: {jax_actions_np.std():.6f}")
print(f"  first action: {jax_actions_np[0, 0, :7]}")

# ============== PyTorch Model ==============
print("\n" + "=" * 60)
print("Loading PyTorch Model")
print("=" * 60)

import torch
import safetensors.torch
print(f"PyTorch device: cpu")

from openpi.models_pytorch import pi0_pytorch

# Create PyTorch model
print("Creating PyTorch model...")
pytorch_model = pi0_pytorch.PI0Pytorch(config=train_config.model)

# Load PyTorch weights
pytorch_checkpoint_path = "/openpi_cache/checkpoints/pi05_libero/model.safetensors"
print(f"Loading PyTorch weights from {pytorch_checkpoint_path}...")
safetensors.torch.load_model(pytorch_model, pytorch_checkpoint_path, strict=False)

# Fix weight tying
paligemma = pytorch_model.paligemma_with_expert.paligemma
embed_tokens = paligemma.model.language_model.embed_tokens.weight
lm_head = paligemma.lm_head.weight
if embed_tokens.shape == lm_head.shape:
    with torch.no_grad():
        embed_tokens.copy_(lm_head)
    print("Fixed weight tying")

pytorch_model.eval()
print("PyTorch model loaded!")

# Create observation for PyTorch (NCHW format for PyTorch)
# PyTorch preprocessing expects [B, C, H, W] format
base_image_pt = torch.from_numpy(base_image_np).permute(0, 3, 1, 2)  # HWC -> CHW
left_wrist_image_pt = torch.from_numpy(left_wrist_image_np).permute(0, 3, 1, 2)
right_wrist_image_pt = torch.from_numpy(right_wrist_image_np).permute(0, 3, 1, 2)

pytorch_observation = _model.Observation(
    images={
        "base_0_rgb": base_image_pt,  # (B, C, H, W)
        "left_wrist_0_rgb": left_wrist_image_pt,
        "right_wrist_0_rgb": right_wrist_image_pt,
    },
    image_masks={
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool),  # Empty camera
    },
    state=torch.from_numpy(state_np),
    tokenized_prompt=torch.from_numpy(token_ids_np.astype(np.int64)),
    tokenized_prompt_mask=torch.from_numpy(token_mask_np),
)

# Run PyTorch inference
print("\nRunning PyTorch inference...")
pytorch_noise = torch.from_numpy(noise_np)
with torch.no_grad():
    pytorch_actions = pytorch_model.sample_actions("cpu", pytorch_observation, num_steps=10, noise=pytorch_noise)
pytorch_actions_np = pytorch_actions.numpy()

print(f"PyTorch actions:")
print(f"  shape: {pytorch_actions_np.shape}")
print(f"  range: [{pytorch_actions_np.min():.6f}, {pytorch_actions_np.max():.6f}]")
print(f"  mean: {pytorch_actions_np.mean():.6f}, std: {pytorch_actions_np.std():.6f}")
print(f"  first action: {pytorch_actions_np[0, 0, :7]}")

# ============== Comparison ==============
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)

diff = jax_actions_np - pytorch_actions_np
print(f"\nDifference (JAX - PyTorch):")
print(f"  max abs diff: {np.abs(diff).max():.6f}")
print(f"  mean abs diff: {np.abs(diff).mean():.6f}")
print(f"  std diff: {diff.std():.6f}")

# Check if they're close
if np.abs(diff).max() < 0.01:
    print("\n✓ JAX and PyTorch outputs are VERY CLOSE (max diff < 0.01)")
elif np.abs(diff).max() < 0.1:
    print("\n⚠ JAX and PyTorch outputs are SOMEWHAT CLOSE (max diff < 0.1)")
else:
    print("\n✗ JAX and PyTorch outputs are DIFFERENT (max diff >= 0.1)")
    print("  This may indicate a bug in the PyTorch implementation!")

# Detailed comparison
print("\nDetailed comparison of first action:")
print(f"  JAX:     {jax_actions_np[0, 0, :7]}")
print(f"  PyTorch: {pytorch_actions_np[0, 0, :7]}")
print(f"  Diff:    {diff[0, 0, :7]}")

print("\n" + "=" * 60)
