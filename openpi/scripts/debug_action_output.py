#!/usr/bin/env python3
"""Quick diagnostic script to debug action output from PyTorch model."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import numpy as np
import torch

# First check norm stats
import json
norm_stats_path = "/openpi_cache/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
with open(norm_stats_path) as f:
    norm_stats = json.load(f)["norm_stats"]

print("=" * 60)
print("Norm Stats for Actions")
print("=" * 60)
print(f"q01: {norm_stats['actions']['q01']}")
print(f"q99: {norm_stats['actions']['q99']}")
print(f"mean: {norm_stats['actions']['mean']}")
print(f"std: {norm_stats['actions']['std']}")

# Calculate expected raw output range (model outputs normalized values)
q01 = np.array(norm_stats['actions']['q01'])
q99 = np.array(norm_stats['actions']['q99'])
print("\n" + "=" * 60)
print("Expected Action Ranges After Unnormalization")
print("=" * 60)
print(f"Min (q01): {q01}")
print(f"Max (q99): {q99}")
print(f"Range: {q99 - q01}")

# Test unnormalize function
def unnormalize_quantile(x, q01, q99):
    """Quantile unnormalization: [-1, 1] -> [q01, q99]"""
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

# Test with boundary values
print("\n" + "=" * 60)
print("Test Unnormalization")
print("=" * 60)
print(f"If model outputs -1.0 -> {unnormalize_quantile(-1.0, q01, q99)}")
print(f"If model outputs  0.0 -> {unnormalize_quantile(0.0, q01, q99)}")
print(f"If model outputs  1.0 -> {unnormalize_quantile(1.0, q01, q99)}")

# Now load and test the model
print("\n" + "=" * 60)
print("Loading PyTorch Model")
print("=" * 60)

from openpi.training import config as _config
from openpi.policies import policy_config

# Get the config
train_config = _config.get_config("pi05_libero")

# Create policy
checkpoint_dir = "/openpi_cache/checkpoints/pi05_libero"
policy = policy_config.create_trained_policy(
    train_config,
    checkpoint_dir,
    pytorch_device="cuda"
)

print("Model loaded successfully")

# Create dummy input
print("\n" + "=" * 60)
print("Testing with dummy input")
print("=" * 60)

# Create a simple test observation
dummy_obs = {
    "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(8, dtype=np.float32),  # 8-dim state
    "prompt": "pick up the object",
}

# Run inference
with torch.no_grad():
    result = policy.infer(dummy_obs)

actions = result["actions"]
print(f"Actions shape: {actions.shape}")
print(f"Actions dtype: {actions.dtype}")
print(f"Actions range: [{actions.min():.4f}, {actions.max():.4f}]")
print(f"Actions mean: {actions.mean():.4f}")
print(f"Actions std: {actions.std():.4f}")
print(f"\nFirst action: {actions[0]}")
print(f"Last action: {actions[-1]}")

# Check if actions are in reasonable range
print("\n" + "=" * 60)
print("Action Analysis")
print("=" * 60)
for i, dim_name in enumerate(['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']):
    dim_actions = actions[:, i]
    expected_min, expected_max = q01[i], q99[i]
    print(f"Dim {i} ({dim_name}): range [{dim_actions.min():.4f}, {dim_actions.max():.4f}], expected [{expected_min:.4f}, {expected_max:.4f}]")

# Check timing
timing = result.get("policy_timing", {})
print(f"\nInference time: {timing.get('infer_ms', 'N/A')} ms")
