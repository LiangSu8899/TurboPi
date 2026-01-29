#!/usr/bin/env python3
"""Debug normalization/unnormalization to verify action pipeline."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import numpy as np
import json
import torch

print("=" * 60)
print("FULL PIPELINE DEBUG")
print("=" * 60)

# First, check the actual policy configuration
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi import transforms

train_config = _config.get_config("pi05_libero")
print(f"\n1. Model Config:")
print(f"   action_horizon: {train_config.model.action_horizon}")
print(f"   model_type: {train_config.model.model_type}")

# Check data config
checkpoint_dir = "/openpi_cache/checkpoints/pi05_libero"
data_config = train_config.data.create(
    "/app/assets/pi05_libero",
    train_config.model
)

print(f"\n2. Data Config (CRITICAL):")
print(f"   use_quantile_norm: {data_config.use_quantile_norm}")
if data_config.norm_stats:
    action_stats = data_config.norm_stats.get('actions')
    if action_stats:
        print(f"   action q01: {action_stats.q01}")
        print(f"   action q99: {action_stats.q99}")

# Create policy and inspect transforms
print("\n3. Creating Policy...")
policy = policy_config.create_trained_policy(
    train_config,
    checkpoint_dir,
    pytorch_device="cuda"
)

print(f"\n4. Policy Transform Info:")
print(f"   Input transform: {type(policy._input_transform).__name__}")
print(f"   Output transform: {type(policy._output_transform).__name__}")

# Test inference
print("\n5. Test Inference...")
np.random.seed(42)
element = {
    "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/state": np.array([0.0, 0.1, 0.8, 3.0, 0.0, 0.0, 0.03, -0.03], dtype=np.float32),
    "prompt": "pick up the black bowl",
}

with torch.no_grad():
    result = policy.infer(element)

actions = result['actions']
print(f"\n6. Final Actions (after all transforms):")
print(f"   Shape: {actions.shape}")
print(f"   Range: [{actions.min():.4f}, {actions.max():.4f}]")
print(f"   First 3 actions:")
for i in range(min(3, len(actions))):
    print(f"     {i}: {actions[i]}")

# Check if actions are in reasonable range for OSC_POSE
print(f"\n7. OSC_POSE Compatibility Check:")
print(f"   OSC_POSE expects input in [-1, 1]")
print(f"   OSC_POSE scales position to ±0.05m, rotation to ±0.5 rad")
osc_pos = actions[:, :3]
osc_rot = actions[:, 3:6]
print(f"   Position actions range: [{osc_pos.min():.4f}, {osc_pos.max():.4f}]")
print(f"   Rotation actions range: [{osc_rot.min():.4f}, {osc_rot.max():.4f}]")
if np.abs(actions[:, :6]).max() > 1.0:
    print(f"   WARNING: Actions exceed [-1, 1] range!")
    print(f"   Expected displacement: {actions[0, :3] * 0.05} m (scaled)")
else:
    print(f"   ✓ Actions within valid range")
    print(f"   Expected displacement: {actions[0, :3] * 0.05} m (scaled)")

print("\n" + "=" * 60)

# Load norm stats
norm_stats_path = "/openpi_cache/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
with open(norm_stats_path) as f:
    norm_stats = json.load(f)["norm_stats"]

action_q01 = np.array(norm_stats['actions']['q01'])
action_q99 = np.array(norm_stats['actions']['q99'])
state_q01 = np.array(norm_stats['state']['q01'])
state_q99 = np.array(norm_stats['state']['q99'])

print("\nAction normalization stats (first 7 dims):")
print(f"  q01: {action_q01}")
print(f"  q99: {action_q99}")
print(f"  Range: {action_q99 - action_q01}")

print("\nState normalization stats:")
print(f"  q01: {state_q01}")
print(f"  q99: {state_q99}")

# Test normalization functions
def normalize_quantile(x, q01, q99):
    """Normalize: [q01, q99] -> [-1, 1]"""
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

def unnormalize_quantile(x, q01, q99):
    """Unnormalize: [-1, 1] -> [q01, q99]"""
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

print("\n" + "=" * 60)
print("Testing Action Unnormalization")
print("=" * 60)

# Test with boundary values
print("\nIf model outputs -1.0 (normalized min):")
unnorm_minus1 = unnormalize_quantile(-1.0, action_q01, action_q99)
print(f"  Unnormalized: {unnorm_minus1}")

print("\nIf model outputs 0.0 (normalized mid):")
unnorm_zero = unnormalize_quantile(0.0, action_q01, action_q99)
print(f"  Unnormalized: {unnorm_zero}")

print("\nIf model outputs 1.0 (normalized max):")
unnorm_plus1 = unnormalize_quantile(1.0, action_q01, action_q99)
print(f"  Unnormalized: {unnorm_plus1}")

# Test with actual model output (from debug_libero_real_obs.py)
print("\n" + "=" * 60)
print("Testing with Actual Model Output")
print("=" * 60)

# These are the actual unnormalized actions from the debug script
# First action: [-0.16, -0.037, 0.029, 0.0065, 0.056, 0.0071, 0.197]
actual_unnorm_action = np.array([-0.16093423, -0.03688405, 0.02873519, 0.0064965, 0.05590397, 0.00713836, 0.19674311])

print(f"Actual unnormalized action: {actual_unnorm_action}")

# Reverse: what was the normalized action?
reversed_norm = normalize_quantile(actual_unnorm_action, action_q01, action_q99)
print(f"Reversed to normalized: {reversed_norm}")
print(f"  (Expected range: [-1, 1])")

# Check if these normalized values are reasonable
print("\nNormalized action analysis:")
for i, (norm_val, unnorm_val) in enumerate(zip(reversed_norm, actual_unnorm_action)):
    dim_range = action_q99[i] - action_q01[i]
    pct_of_range = (unnorm_val - action_q01[i]) / dim_range * 100
    print(f"  Dim {i}: norm={norm_val:.3f}, unnorm={unnorm_val:.4f}, {pct_of_range:.1f}% of range [{action_q01[i]:.3f}, {action_q99[i]:.3f}]")

print("\n" + "=" * 60)
print("What's Expected from Training Data")
print("=" * 60)

# If training demos have these action ranges, what does a "typical" action look like?
print("\nAction ranges in training data:")
print("  x: ±0.84m (delta position)")
print("  y: ±0.83m")
print("  z: ±0.94m")
print("  rx: ±0.13 rad")
print("  ry: ±0.18 rad")
print("  rz: ±0.25 rad")
print("  gripper: ±1.0")

print("\nNOTE: These are the 1st-99th percentile ranges, meaning 98% of training")
print("actions fall within these ranges. If model outputs are near 0, the robot")
print("will barely move.")

print("\n" + "=" * 60)
print("Position Change Calculation")
print("=" * 60)

# From debug output: first action x = -0.16, position change = -0.001297m
# That's a ratio of about 0.008
# This suggests the OSC controller might be scaling actions differently

print("\nFrom debug output:")
print("  Action x = -0.16 m (delta command)")
print("  Position change x = -0.001297 m")
print("  Ratio = 0.008 (about 1:125 scaling)")

print("\nThis massive scaling suggests either:")
print("  1. OSC_POSE controller scales actions internally")
print("  2. The action representation is different (velocity vs position)")
print("  3. Control frequency affects the scaling (20Hz)")

print("\nIf OSC interprets actions as velocity (m/s), then with 20Hz control:")
print("  -0.16 m/s * 0.05s = -0.008m per step")
print("  Actual: -0.001297m per step")
print("  Still 6x smaller than expected!")
