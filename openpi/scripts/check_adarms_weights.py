#!/usr/bin/env python3
"""Check adaRMS weights loading."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import torch
from safetensors.torch import load_file

print("=" * 60)
print("Checking adaRMS Weights")
print("=" * 60)

# Load checkpoint directly
checkpoint_path = "/openpi_cache/checkpoints/pi05_libero/model.safetensors"
checkpoint = load_file(checkpoint_path)

# Check for adaRMS related weights
adarms_keys = [k for k in checkpoint.keys() if 'norm' in k.lower() and 'dense' in k.lower()]
print(f"\nFound {len(adarms_keys)} adaRMS dense layer keys:")
for k in adarms_keys[:10]:
    tensor = checkpoint[k]
    print(f"  {k}: shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"    range=[{tensor.min():.6f}, {tensor.max():.6f}], mean={tensor.mean():.6f}")

# Now load model and compare
from openpi.training import config as _config
import os

train_config = _config.get_config("pi05_libero")
weight_path = os.path.join("/openpi_cache/checkpoints/pi05_libero", "model.safetensors")

print("\n" + "=" * 60)
print("Loading Model")
print("=" * 60)

model = train_config.model.load_pytorch(train_config, weight_path)
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
model = model.to("cuda")
model.eval()

# Check model's adaRMS layers
print("\nChecking model's adaRMS layers:")
for name, module in model.named_modules():
    if hasattr(module, 'dense') and module.dense is not None:
        if hasattr(module, '_use_adarms') and module._use_adarms:
            print(f"\n{name}:")
            print(f"  dense.weight: shape={module.dense.weight.shape}, dtype={module.dense.weight.dtype}")
            print(f"    range=[{module.dense.weight.min():.6f}, {module.dense.weight.max():.6f}]")
            print(f"  dense.bias: shape={module.dense.bias.shape}")
            print(f"    range=[{module.dense.bias.min():.6f}, {module.dense.bias.max():.6f}]")
            if hasattr(module, 'weight') and module.weight is not None:
                print(f"  weight buffer: shape={module.weight.shape}, value={module.weight[:5].tolist()}")
            break  # Just show first one

# Test a forward pass through the norm
print("\n" + "=" * 60)
print("Testing adaRMS Forward Pass")
print("=" * 60)

# Get a specific layer
layer = model.paligemma_with_expert.gemma_expert.model.layers[0]
norm = layer.input_layernorm

print(f"Layer input_layernorm type: {type(norm)}")
print(f"  _use_adarms: {norm._use_adarms}")
print(f"  cond_dim: {norm.cond_dim}")

# Create test inputs
batch_size = 1
hidden_dim = norm.dim
cond_dim = norm.cond_dim

hidden_states = torch.randn(batch_size, 10, hidden_dim, device="cuda", dtype=torch.bfloat16)
cond = torch.randn(batch_size, cond_dim, device="cuda", dtype=torch.bfloat16)

print(f"\nInput shapes:")
print(f"  hidden_states: {hidden_states.shape}, dtype={hidden_states.dtype}")
print(f"  cond: {cond.shape}, dtype={cond.dtype}")

# Forward pass
output, gate = norm(hidden_states, cond=cond)
print(f"\nOutput:")
print(f"  output shape: {output.shape}")
print(f"  output range: [{output.min():.4f}, {output.max():.4f}]")
print(f"  gate shape: {gate.shape}")
print(f"  gate range: [{gate.min():.4f}, {gate.max():.4f}]")

# Check if gate values are reasonable
print(f"\nGate analysis:")
print(f"  Gate should modulate the residual connection")
print(f"  If gate ≈ 0, residual dominates (no learned contribution)")
print(f"  If gate ≈ 1, learned contribution dominates")
print(f"  Current gate mean: {gate.mean():.4f}")
