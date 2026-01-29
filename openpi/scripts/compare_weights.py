#!/usr/bin/env python3
"""Compare JAX and PyTorch checkpoint weights."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

print("=" * 60)
print("Weight Comparison: JAX vs PyTorch")
print("=" * 60)

# Load JAX weights
print("\n1. Loading JAX weights...")
import jax.numpy as jnp
from openpi.models import model as _model

jax_params = _model.restore_params(
    "/openpi_cache/openpi-assets/checkpoints/pi05_libero/params",
    dtype=jnp.float32
)

# Flatten JAX params
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

jax_flat = flatten_dict(jax_params)
print(f"   JAX has {len(jax_flat)} weights")

# Load PyTorch weights
print("\n2. Loading PyTorch weights...")
import safetensors.torch
pytorch_weights = safetensors.torch.load_file(
    "/openpi_cache/checkpoints/pi05_libero/model.safetensors"
)
print(f"   PyTorch has {len(pytorch_weights)} weights")

# Show sample keys
print("\n3. Sample JAX keys:")
for i, key in enumerate(list(jax_flat.keys())[:15]):
    print(f"   {key}: shape={np.array(jax_flat[key]).shape}")

print("\n4. Sample PyTorch keys:")
for i, key in enumerate(list(pytorch_weights.keys())[:15]):
    print(f"   {key}: shape={pytorch_weights[key].shape}")

# Compare specific weights
print("\n5. Detailed weight comparison:")

# Action input projection
jax_key = None
for k in jax_flat.keys():
    if 'action_in_proj' in k and 'kernel' in k:
        jax_key = k
        break

if jax_key:
    jax_w = np.array(jax_flat[jax_key])
    pt_w = pytorch_weights["action_in_proj.weight"].numpy()

    print(f"\n   action_in_proj:")
    print(f"     JAX key: {jax_key}")
    print(f"     JAX shape: {jax_w.shape}, PT shape: {pt_w.shape}")
    print(f"     JAX range: [{jax_w.min():.6f}, {jax_w.max():.6f}]")
    print(f"     PT range: [{pt_w.min():.6f}, {pt_w.max():.6f}]")

    # JAX Linear kernel is (in, out), PyTorch is (out, in)
    if jax_w.shape == pt_w.T.shape:
        diff = jax_w - pt_w.T
        print(f"     Max diff (JAX vs PT.T): {np.abs(diff).max():.6f}")
    elif jax_w.shape == pt_w.shape:
        diff = jax_w - pt_w
        print(f"     Max diff: {np.abs(diff).max():.6f}")
    else:
        print(f"     Shapes incompatible!")

# Action output projection
jax_key = None
for k in jax_flat.keys():
    if 'action_out_proj' in k and 'kernel' in k:
        jax_key = k
        break

if jax_key:
    jax_w = np.array(jax_flat[jax_key])
    pt_w = pytorch_weights["action_out_proj.weight"].numpy()

    print(f"\n   action_out_proj:")
    print(f"     JAX key: {jax_key}")
    print(f"     JAX shape: {jax_w.shape}, PT shape: {pt_w.shape}")
    print(f"     JAX range: [{jax_w.min():.6f}, {jax_w.max():.6f}]")
    print(f"     PT range: [{pt_w.min():.6f}, {pt_w.max():.6f}]")

    if jax_w.shape == pt_w.T.shape:
        diff = jax_w - pt_w.T
        print(f"     Max diff (JAX vs PT.T): {np.abs(diff).max():.6f}")

# Time MLP
for layer_name in ["time_mlp_in", "time_mlp_out"]:
    jax_key = None
    for k in jax_flat.keys():
        if layer_name in k and 'kernel' in k:
            jax_key = k
            break

    if jax_key and f"{layer_name}.weight" in pytorch_weights:
        jax_w = np.array(jax_flat[jax_key])
        pt_w = pytorch_weights[f"{layer_name}.weight"].numpy()

        print(f"\n   {layer_name}:")
        print(f"     JAX shape: {jax_w.shape}, PT shape: {pt_w.shape}")
        print(f"     JAX range: [{jax_w.min():.6f}, {jax_w.max():.6f}]")
        print(f"     PT range: [{pt_w.min():.6f}, {pt_w.max():.6f}]")

        if jax_w.shape == pt_w.T.shape:
            diff = jax_w - pt_w.T
            print(f"     Max diff (JAX vs PT.T): {np.abs(diff).max():.6f}")

print("\n" + "=" * 60)
