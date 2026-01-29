#!/usr/bin/env python3
"""Compare SigLIP vision encoder outputs between JAX and PyTorch."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

print("=" * 60)
print("SigLIP Vision Encoder Comparison")
print("=" * 60)

# Set random seed
np.random.seed(42)

# Create test image (HWC format, float32, range [0, 1])
batch_size = 1
image_np = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)

print(f"\nTest image: shape={image_np.shape}, range=[{image_np.min():.4f}, {image_np.max():.4f}]")

# ============== JAX SigLIP ==============
print("\n" + "=" * 60)
print("Loading JAX Model")
print("=" * 60)

import jax
import jax.numpy as jnp
print(f"JAX devices: {jax.devices()}")

from openpi.training import config as _config
from openpi.models import model as _model

jax_checkpoint_dir = "/openpi_cache/openpi-assets/checkpoints/pi05_libero"
train_config = _config.get_config("pi05_libero")

print(f"\nLoading JAX params...")
jax_params = _model.restore_params(f"{jax_checkpoint_dir}/params", dtype=jnp.float32)
print("JAX params loaded!")

print("Creating JAX model...")
jax_model = train_config.model.load(jax_params)
print("JAX model created!")

# Get JAX image tokens
print("\nRunning JAX SigLIP...")
jax_image = jnp.array(image_np)  # (B, H, W, C)
print(f"  JAX image input shape: {jax_image.shape}")

# Access SigLIP through PaliGemma
jax_paligemma = jax_model.PaliGemma
jax_img_tokens, _ = jax_paligemma.img(jax_image, train=False)
jax_img_tokens_np = np.asarray(jax_img_tokens, dtype=np.float32)

print(f"  JAX image tokens shape: {jax_img_tokens_np.shape}")
print(f"  JAX image tokens range: [{float(jax_img_tokens_np.min()):.6f}, {float(jax_img_tokens_np.max()):.6f}]")
print(f"  JAX image tokens mean: {float(jax_img_tokens_np.mean()):.6f}")
print(f"  JAX image tokens std: {float(jax_img_tokens_np.std()):.6f}")
print(f"  JAX first 10 values: {jax_img_tokens_np[0, 0, :10]}")

# ============== PyTorch SigLIP ==============
print("\n" + "=" * 60)
print("Loading PyTorch Model")
print("=" * 60)

import torch
import safetensors.torch

from openpi.models_pytorch import pi0_pytorch

print("Creating PyTorch model...")
pytorch_model = pi0_pytorch.PI0Pytorch(config=train_config.model)

pytorch_checkpoint_path = "/openpi_cache/checkpoints/pi05_libero/model.safetensors"
print(f"Loading PyTorch weights...")
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

# Get PyTorch image tokens
print("\nRunning PyTorch SigLIP...")
# PyTorch expects NCHW format
pt_image = torch.from_numpy(image_np).permute(0, 3, 1, 2)  # (B, C, H, W)
print(f"  PyTorch image input shape: {pt_image.shape}")

# Access SigLIP through PaliGemma
with torch.no_grad():
    # Use embed_image method
    pt_img_tokens = pytorch_model.paligemma_with_expert.embed_image(pt_image)
    # Convert to float32 for numpy compatibility
    pt_img_tokens_np = pt_img_tokens.float().numpy()

print(f"  PyTorch image tokens shape: {pt_img_tokens_np.shape}")
print(f"  PyTorch image tokens range: [{pt_img_tokens_np.min():.6f}, {pt_img_tokens_np.max():.6f}]")
print(f"  PyTorch image tokens mean: {pt_img_tokens_np.mean():.6f}")
print(f"  PyTorch image tokens std: {pt_img_tokens_np.std():.6f}")
print(f"  PyTorch first 10 values: {pt_img_tokens_np[0, 0, :10]}")

# ============== Comparison ==============
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)

# Check shapes
if jax_img_tokens_np.shape != pt_img_tokens_np.shape:
    print(f"\n⚠ Shape mismatch!")
    print(f"  JAX: {jax_img_tokens_np.shape}")
    print(f"  PyTorch: {pt_img_tokens_np.shape}")
else:
    diff = jax_img_tokens_np - pt_img_tokens_np
    print(f"\nImage tokens difference:")
    print(f"  max abs diff: {np.abs(diff).max():.6f}")
    print(f"  mean abs diff: {np.abs(diff).mean():.6f}")
    print(f"  std diff: {diff.std():.6f}")

    if np.abs(diff).max() < 0.01:
        print("\n✓ SigLIP outputs are VERY CLOSE (max diff < 0.01)")
    elif np.abs(diff).max() < 0.1:
        print("\n⚠ SigLIP outputs are SOMEWHAT CLOSE (max diff < 0.1)")
    else:
        print("\n✗ SigLIP outputs are DIFFERENT (max diff >= 0.1)")
        print("  The vision encoder is the source of divergence!")

# Compare first few positions in detail
print("\nDetailed comparison (first position, first 10 dims):")
print(f"  JAX:     {jax_img_tokens_np[0, 0, :10]}")
print(f"  PyTorch: {pt_img_tokens_np[0, 0, :10]}")
if jax_img_tokens_np.shape == pt_img_tokens_np.shape:
    print(f"  Diff:    {(jax_img_tokens_np - pt_img_tokens_np)[0, 0, :10]}")

print("\n" + "=" * 60)
