#!/usr/bin/env python3
"""Check PyTorch weight loading from LeRobot checkpoint."""

import sys
sys.path.insert(0, "/app/src")

import os
import safetensors.torch
import torch
from collections import OrderedDict

print("=" * 60)
print("Checking PyTorch Weight Loading")
print("=" * 60)

checkpoint_dir = "/openpi_cache/checkpoints/pi05_libero"
weight_path = os.path.join(checkpoint_dir, "model.safetensors")

# Load weights from safetensors
print("\n1. Loading safetensors weights...")
checkpoint_weights = safetensors.torch.load_file(weight_path)
print(f"   Total weights in checkpoint: {len(checkpoint_weights)}")

# Sample keys
print("\n2. Sample checkpoint keys:")
for i, key in enumerate(list(checkpoint_weights.keys())[:20]):
    shape = checkpoint_weights[key].shape
    print(f"   {key}: {shape}")

# Create model
print("\n3. Creating model...")
from openpi.training import config as _config
from openpi.models_pytorch import pi0_pytorch

train_config = _config.get_config("pi05_libero")
model = pi0_pytorch.PI0Pytorch(config=train_config.model)

# Get model state dict
model_state_dict = model.state_dict()
print(f"   Total params in model: {len(model_state_dict)}")

# Sample model keys
print("\n4. Sample model keys:")
for i, key in enumerate(list(model_state_dict.keys())[:20]):
    shape = model_state_dict[key].shape
    print(f"   {key}: {shape}")

# Check for mismatches
print("\n5. Checking for mismatches...")
checkpoint_keys = set(checkpoint_weights.keys())
model_keys = set(model_state_dict.keys())

missing_in_checkpoint = model_keys - checkpoint_keys
extra_in_checkpoint = checkpoint_keys - model_keys

if missing_in_checkpoint:
    print(f"\n   Missing in checkpoint ({len(missing_in_checkpoint)}):")
    for key in list(missing_in_checkpoint)[:20]:
        print(f"     {key}")
    if len(missing_in_checkpoint) > 20:
        print(f"     ... and {len(missing_in_checkpoint) - 20} more")

if extra_in_checkpoint:
    print(f"\n   Extra in checkpoint ({len(extra_in_checkpoint)}):")
    for key in list(extra_in_checkpoint)[:20]:
        print(f"     {key}")
    if len(extra_in_checkpoint) > 20:
        print(f"     ... and {len(extra_in_checkpoint) - 20} more")

# Check shape mismatches
print("\n6. Checking shape mismatches...")
shape_mismatches = []
for key in checkpoint_keys & model_keys:
    ckpt_shape = checkpoint_weights[key].shape
    model_shape = model_state_dict[key].shape
    if ckpt_shape != model_shape:
        shape_mismatches.append((key, ckpt_shape, model_shape))

if shape_mismatches:
    print(f"   Shape mismatches ({len(shape_mismatches)}):")
    for key, ckpt_shape, model_shape in shape_mismatches[:20]:
        print(f"     {key}: checkpoint {ckpt_shape} vs model {model_shape}")
else:
    print("   No shape mismatches found!")

# Actually load and check
print("\n7. Loading weights with strict=False...")
safetensors.torch.load_model(model, weight_path, strict=False)
print("   Weights loaded successfully!")

# Verify a few weights are non-zero
print("\n8. Verifying weights are loaded correctly...")
test_params = [
    "paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight",
    "action_in_proj.weight",
    "action_out_proj.weight",
    "time_mlp_in.weight",
]
for param_name in test_params:
    if param_name in model_state_dict:
        param = model.state_dict()[param_name]
        print(f"   {param_name}: mean={param.mean():.6f}, std={param.std():.6f}, zeros={torch.sum(param == 0).item()}/{param.numel()}")
