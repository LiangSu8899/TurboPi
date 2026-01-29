#!/usr/bin/env python3
"""Test different action_horizon settings."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import numpy as np
import torch
import os

print("=" * 60)
print("Testing Action Horizon Configuration")
print("=" * 60)

# Set random seed
np.random.seed(42)

# Create test inputs
batch_size = 1
image_np = np.random.randint(0, 255, (batch_size, 224, 224, 3), dtype=np.uint8)
wrist_image_np = np.random.randint(0, 255, (batch_size, 224, 224, 3), dtype=np.uint8)
state_np = np.random.randn(batch_size, 8).astype(np.float32) * 0.1
prompt = "pick up the object"

# Test with action_horizon=10 (OpenPi default)
print("\n" + "=" * 60)
print("Test 1: action_horizon=10 (OpenPi default)")
print("=" * 60)

from openpi.training import config as _config
from openpi.policies import policy_config

train_config = _config.get_config("pi05_libero")
print(f"Config action_horizon: {train_config.model.action_horizon}")

checkpoint_dir = "/openpi_cache/checkpoints/pi05_libero"
policy = policy_config.create_trained_policy(
    train_config,
    checkpoint_dir,
    pytorch_device="cuda"
)

element = {
    "observation/image": image_np[0],
    "observation/wrist_image": wrist_image_np[0],
    "observation/state": state_np[0],
    "prompt": prompt,
}

with torch.no_grad():
    result = policy.infer(element)

print(f"Actions shape: {result['actions'].shape}")
print(f"Actions range: [{result['actions'].min():.4f}, {result['actions'].max():.4f}]")
print(f"First action (first 7 dims): {result['actions'][0, :7]}")

# Test with action_horizon=50 (LeRobot/PyTorch checkpoint default)
print("\n" + "=" * 60)
print("Test 2: action_horizon=50 (LeRobot default)")
print("=" * 60)

# Modify the model config to use action_horizon=50
from openpi.models import pi0_config
modified_model_config = pi0_config.Pi0Config(
    pi05=True,
    action_horizon=50,  # Changed from 10 to 50
    discrete_state_input=False
)

# Create a modified train config
from dataclasses import replace
modified_train_config = replace(train_config, model=modified_model_config)
print(f"Modified config action_horizon: {modified_train_config.model.action_horizon}")

try:
    policy50 = policy_config.create_trained_policy(
        modified_train_config,
        checkpoint_dir,
        pytorch_device="cuda"
    )

    with torch.no_grad():
        result50 = policy50.infer(element)

    print(f"Actions shape: {result50['actions'].shape}")
    print(f"Actions range: [{result50['actions'].min():.4f}, {result50['actions'].max():.4f}]")
    print(f"First action (first 7 dims): {result50['actions'][0, :7]}")

    # Compare
    print("\n" + "=" * 60)
    print("Comparison (first 10 actions)")
    print("=" * 60)
    print(f"action_horizon=10: action[0] = {result['actions'][0, :7]}")
    print(f"action_horizon=50: action[0] = {result50['actions'][0, :7]}")

    # Check if they're different
    if result['actions'].shape != result50['actions'].shape:
        print(f"\n*** DIFFERENT SHAPES! {result['actions'].shape} vs {result50['actions'].shape} ***")

    # Check first 10 actions if possible
    min_len = min(result['actions'].shape[0], result50['actions'].shape[0])
    diff = np.abs(result['actions'][:min_len] - result50['actions'][:min_len])
    print(f"\nDifference (first {min_len} actions): max={diff.max():.4f}, mean={diff.mean():.4f}")

except Exception as e:
    print(f"Error with action_horizon=50: {e}")
    import traceback
    traceback.print_exc()

# Check the actual weights dimensions
print("\n" + "=" * 60)
print("Checking Model Weights Dimensions")
print("=" * 60)

model = policy._model
print(f"action_in_proj: input={model.action_in_proj.in_features}, output={model.action_in_proj.out_features}")
print(f"action_out_proj: input={model.action_out_proj.in_features}, output={model.action_out_proj.out_features}")
print(f"time_mlp_in: input={model.time_mlp_in.in_features}, output={model.time_mlp_in.out_features}")

# The key question: does the model architecture depend on action_horizon?
# In pi0_pytorch.py, action_horizon affects:
# 1. The shape of noisy_actions input (batch, action_horizon, action_dim)
# 2. The number of action tokens extracted from suffix_out

print("\n" + "=" * 60)
print("Checking Checkpoint Config")
print("=" * 60)
checkpoint_config_path = os.path.join(checkpoint_dir, "config.json")
if os.path.exists(checkpoint_config_path):
    import json
    with open(checkpoint_config_path) as f:
        ckpt_config = json.load(f)
    print(f"Checkpoint config: {json.dumps(ckpt_config, indent=2)}")
else:
    print("No config.json found in checkpoint directory")

# Check the other PyTorch checkpoint
alt_checkpoint_dir = "/openpi_cache/pytorch_checkpoints/pi05_libero"
alt_config_path = os.path.join(alt_checkpoint_dir, "config.json")
if os.path.exists(alt_config_path):
    print(f"\nAlternate checkpoint config ({alt_checkpoint_dir}):")
    with open(alt_config_path) as f:
        alt_config = json.load(f)
    print(json.dumps(alt_config, indent=2))
