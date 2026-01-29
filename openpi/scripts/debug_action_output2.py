#!/usr/bin/env python3
"""Quick diagnostic script to debug action output from PyTorch model - RAW output check."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import numpy as np
import torch

print("=" * 60)
print("Loading Model Directly (No Policy Wrapper)")
print("=" * 60)

from openpi.training import config as _config
from openpi.models_pytorch import pi0_pytorch
import os

# Get the config
train_config = _config.get_config("pi05_libero")

# Load model directly
checkpoint_dir = "/openpi_cache/checkpoints/pi05_libero"
weight_path = os.path.join(checkpoint_dir, "model.safetensors")

model = train_config.model.load_pytorch(train_config, weight_path)
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
model = model.to("cuda")
model.eval()

print(f"Model loaded successfully")
print(f"Model config: action_horizon={model.config.action_horizon}, action_dim={model.config.action_dim}")

# Create simple observation
from openpi.models import model as _model

# Create dummy inputs matching model's expected format - (B, C, H, W) for PyTorch
batch_size = 1
image = torch.randint(0, 255, (batch_size, 3, 224, 224), dtype=torch.uint8, device="cuda")
wrist_image = torch.randint(0, 255, (batch_size, 3, 224, 224), dtype=torch.uint8, device="cuda")

# Normalize state to roughly match expected range (normalized)
state = torch.zeros(batch_size, 8, dtype=torch.bfloat16, device="cuda")

# Token IDs for "pick up the object" - simple prompt
token_ids = torch.zeros(batch_size, 64, dtype=torch.int32, device="cuda")  # padded
token_ids[:, :5] = torch.tensor([100, 200, 300, 400, 500])  # dummy tokens

# Use the proper image keys
images = {
    "base_0_rgb": (image.to(torch.bfloat16) / 127.5 - 1.0),  # Scale to [-1, 1]
    "left_wrist_0_rgb": (wrist_image.to(torch.bfloat16) / 127.5 - 1.0),
    "right_wrist_0_rgb": (wrist_image.to(torch.bfloat16) / 127.5 - 1.0),  # Use same as wrist for now
}
image_masks = {k: torch.ones(batch_size, dtype=torch.bool, device="cuda") for k in images.keys()}

obs = _model.Observation(
    images=images,
    image_masks=image_masks,
    state=state,
    tokenized_prompt=token_ids,
    tokenized_prompt_mask=torch.ones(batch_size, 64, dtype=torch.bool, device="cuda"),
)

print("\n" + "=" * 60)
print("Running Model.sample_actions()")
print("=" * 60)

with torch.no_grad():
    # Call sample_actions which internally handles denoising
    raw_actions = model.sample_actions("cuda", obs)

print(f"Raw actions shape: {raw_actions.shape}")
print(f"Raw actions dtype: {raw_actions.dtype}")
raw_np = raw_actions.cpu().float().numpy()
print(f"Raw actions range: [{raw_np.min():.4f}, {raw_np.max():.4f}]")
print(f"Raw actions mean: {raw_np.mean():.4f}")
print(f"Raw actions std: {raw_np.std():.4f}")
print(f"\nFirst action: {raw_np[0, 0]}")
print(f"Last action: {raw_np[0, -1]}")

# Per-dimension analysis
print("\n" + "=" * 60)
print("Per-Dimension Analysis of RAW Model Output")
print("=" * 60)
print("(Expected range should be approximately [-1, 1] for normalized output)")
for i, dim_name in enumerate(['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']):
    dim_actions = raw_np[0, :, i]
    print(f"Dim {i} ({dim_name}): range [{dim_actions.min():.4f}, {dim_actions.max():.4f}], mean={dim_actions.mean():.4f}")

# Now test unnormalization manually
print("\n" + "=" * 60)
print("Testing Manual Unnormalization")
print("=" * 60)
import json
norm_stats_path = "/openpi_cache/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
with open(norm_stats_path) as f:
    norm_stats = json.load(f)["norm_stats"]

q01 = np.array(norm_stats['actions']['q01'])
q99 = np.array(norm_stats['actions']['q99'])

def unnormalize_quantile(x, q01, q99):
    """Quantile unnormalization: [-1, 1] -> [q01, q99]
    Only applies to the first len(q01) dimensions."""
    dim = q01.shape[-1]
    if dim < x.shape[-1]:
        return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

# Check raw first 7 dimensions (actual LIBERO actions)
raw_first_7 = raw_np[0, :, :7]
print(f"\nRaw first 7 dims (actual actions) range: [{raw_first_7.min():.4f}, {raw_first_7.max():.4f}]")
print("NOTE: If model is trained correctly, this should be approximately [-1, 1]")

unnormalized = unnormalize_quantile(raw_np[0], q01, q99)
print(f"\nAfter unnormalization shape: {unnormalized.shape}")
print(f"After unnormalization (first 7 dims) range: [{unnormalized[:, :7].min():.4f}, {unnormalized[:, :7].max():.4f}]")
print(f"\nFirst action unnormalized (first 7 dims): {unnormalized[0, :7]}")
print(f"Last action unnormalized (first 7 dims): {unnormalized[-1, :7]}")

print("\n" + "=" * 60)
print("Per-Dimension After Unnormalization")
print("=" * 60)
for i, dim_name in enumerate(['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']):
    dim_actions = unnormalized[:, i]
    expected_min, expected_max = q01[i], q99[i]
    print(f"Dim {i} ({dim_name}): range [{dim_actions.min():.4f}, {dim_actions.max():.4f}], expected [{expected_min:.4f}, {expected_max:.4f}]")
