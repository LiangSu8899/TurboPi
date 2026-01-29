#!/usr/bin/env python3
"""Direct comparison of JAX and PyTorch model outputs."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import numpy as np
import torch
import os

print("=" * 60)
print("Direct JAX vs PyTorch Model Comparison")
print("=" * 60)

# Set random seed
np.random.seed(42)

# Create deterministic inputs
batch_size = 1
image_np = np.random.randint(0, 255, (batch_size, 224, 224, 3), dtype=np.uint8)  # HWC format
wrist_image_np = np.random.randint(0, 255, (batch_size, 224, 224, 3), dtype=np.uint8)
state_np = np.random.randn(batch_size, 8).astype(np.float32) * 0.1
prompt = "pick up the object"

print(f"Input shapes:")
print(f"  Image: {image_np.shape}")
print(f"  State: {state_np.shape}")
print(f"  Prompt: {prompt}")

# ============== Load PyTorch Model ==============
print("\n" + "=" * 60)
print("Loading PyTorch Model")
print("=" * 60)

from openpi.training import config as _config
from openpi.policies import policy_config

train_config = _config.get_config("pi05_libero")
checkpoint_dir = "/openpi_cache/checkpoints/pi05_libero"

pytorch_policy = policy_config.create_trained_policy(
    train_config,
    checkpoint_dir,
    pytorch_device="cuda"
)
print("PyTorch policy loaded")

# Prepare input dict
element = {
    "observation/image": image_np[0],  # Remove batch dim for policy.infer
    "observation/wrist_image": wrist_image_np[0],
    "observation/state": state_np[0],
    "prompt": prompt,
}

# Run inference
print("\nRunning PyTorch inference...")
with torch.no_grad():
    pytorch_result = pytorch_policy.infer(element)

pytorch_actions = pytorch_result["actions"]
print(f"PyTorch actions shape: {pytorch_actions.shape}")
print(f"PyTorch actions range: [{pytorch_actions.min():.4f}, {pytorch_actions.max():.4f}]")
print(f"PyTorch first action (first 7 dims): {pytorch_actions[0, :7]}")

# ============== Try to Load JAX Model ==============
print("\n" + "=" * 60)
print("Attempting to Load JAX Model")
print("=" * 60)

try:
    import jax
    import jax.numpy as jnp

    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")

    # Check if we have GPU
    if not jax.devices('gpu'):
        print("WARNING: No JAX GPU found, using CPU")

    # Load JAX checkpoint
    from openpi.models import model as _model

    # Try loading JAX model
    jax_checkpoint_path = checkpoint_dir + "/params"
    if os.path.exists(jax_checkpoint_path):
        print(f"Loading JAX params from: {jax_checkpoint_path}")

        # Create JAX policy
        jax_policy = policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            pytorch_device=None  # Force JAX mode
        )

        # Run inference
        print("\nRunning JAX inference...")
        jax_result = jax_policy.infer(element)

        jax_actions = jax_result["actions"]
        print(f"JAX actions shape: {jax_actions.shape}")
        print(f"JAX actions range: [{jax_actions.min():.4f}, {jax_actions.max():.4f}]")
        print(f"JAX first action (first 7 dims): {jax_actions[0, :7]}")

        # Compare
        print("\n" + "=" * 60)
        print("Comparison")
        print("=" * 60)

        diff = np.abs(pytorch_actions - jax_actions)
        print(f"Absolute difference range: [{diff.min():.6f}, {diff.max():.6f}]")
        print(f"Mean absolute difference: {diff.mean():.6f}")

        if diff.max() < 0.01:
            print("RESULT: Models match within tolerance!")
        else:
            print("RESULT: Significant difference detected!")
            print("\nPer-dimension comparison (first action):")
            for i in range(min(7, pytorch_actions.shape[1])):
                pt_val = pytorch_actions[0, i]
                jax_val = jax_actions[0, i]
                print(f"  Dim {i}: PyTorch={pt_val:.4f}, JAX={jax_val:.4f}, diff={abs(pt_val-jax_val):.4f}")
    else:
        print(f"JAX params not found at: {jax_checkpoint_path}")
        print("Checking what files exist...")
        for f in os.listdir(checkpoint_dir):
            print(f"  {f}")

except ImportError as e:
    print(f"JAX import error: {e}")
except Exception as e:
    print(f"Error loading JAX model: {e}")
    import traceback
    traceback.print_exc()

# ============== Additional PyTorch Analysis ==============
print("\n" + "=" * 60)
print("Additional PyTorch Model Analysis")
print("=" * 60)

# Get the underlying model
pytorch_model = pytorch_policy._model
print(f"Model type: {type(pytorch_model)}")
print(f"Model config: action_horizon={pytorch_model.config.action_horizon}, action_dim={pytorch_model.config.action_dim}")

# Check denoising steps
print(f"\nDefault num_steps for sampling: {pytorch_policy._sample_kwargs}")

# Try with more denoising steps
print("\nTrying with different num_steps...")
for num_steps in [5, 10, 20, 50]:
    with torch.no_grad():
        result = pytorch_policy.infer(element, noise=None)
    print(f"  num_steps={num_steps}: action range [{result['actions'].min():.4f}, {result['actions'].max():.4f}]")
