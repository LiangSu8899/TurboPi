#!/usr/bin/env python3
"""Debug individual layer outputs."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import torch
print("=" * 60)
print("Debugging Layer Outputs")
print("=" * 60)

from openpi.training import config as _config
import os

# Load model
train_config = _config.get_config("pi05_libero")
weight_path = os.path.join("/openpi_cache/checkpoints/pi05_libero", "model.safetensors")
model = train_config.model.load_pytorch(train_config, weight_path)
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
model = model.to("cuda")
model.eval()

print("Model loaded")

# Test RMSNorm directly
print("\n=== Testing RMSNorm ===")
layer = model.paligemma_with_expert.gemma_expert.model.layers[0]
norm = layer.input_layernorm

# Create test input
batch = 1
seq_len = 10
hidden_dim = 1024

# Normal input (like after embedding)
x = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
print(f"Input: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}], std={x.std():.4f}")

# Test without conditioning
out_no_cond, gate_no_cond = norm(x, cond=None)
print(f"Output (no cond): range=[{out_no_cond.min():.4f}, {out_no_cond.max():.4f}], std={out_no_cond.std():.4f}")

# Test with conditioning
cond = torch.randn(batch, hidden_dim, device="cuda", dtype=torch.bfloat16)
print(f"Condition: shape={cond.shape}, range=[{cond.min():.4f}, {cond.max():.4f}], std={cond.std():.4f}")

out_with_cond, gate = norm(x, cond=cond)
print(f"Output (with cond): range=[{out_with_cond.min():.4f}, {out_with_cond.max():.4f}], std={out_with_cond.std():.4f}")
print(f"Gate: range=[{gate.min():.4f}, {gate.max():.4f}], mean={gate.mean():.4f}")

# Check the dense layer output scale
print("\n=== Checking adaRMS Dense Layer ===")
print(f"Dense weight: shape={norm.dense.weight.shape}, range=[{norm.dense.weight.min():.4f}, {norm.dense.weight.max():.4f}]")
print(f"Dense bias: range=[{norm.dense.bias.min():.4f}, {norm.dense.bias.max():.4f}]")

# Compute modulation manually
cond_f32 = cond.float()
modulation = norm.dense(cond_f32)
print(f"Modulation: shape={modulation.shape}, range=[{modulation.min():.4f}, {modulation.max():.4f}]")

scale, shift, gate_raw = torch.chunk(modulation, 3, dim=-1)
print(f"Scale: range=[{scale.min():.4f}, {scale.max():.4f}], mean={scale.mean():.4f}")
print(f"Shift: range=[{shift.min():.4f}, {shift.max():.4f}], mean={shift.mean():.4f}")
print(f"Gate raw: range=[{gate_raw.min():.4f}, {gate_raw.max():.4f}], mean={gate_raw.mean():.4f}")
print(f"Gate sigmoid: range=[{gate_raw.sigmoid().min():.4f}, {gate_raw.sigmoid().max():.4f}], mean={gate_raw.sigmoid().mean():.4f}")

# Now trace through a full layer
print("\n=== Full Decoder Layer ===")
print(f"Layer type: {type(layer)}")

# Simple forward through attention and MLP
hidden_states = x.clone()
print(f"Hidden input: range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}], std={hidden_states.std():.4f}")

# Input layernorm
normed, gate1 = layer.input_layernorm(hidden_states, cond=cond)
print(f"After input_layernorm: range=[{normed.min():.4f}, {normed.max():.4f}], std={normed.std():.4f}")
print(f"Gate1: mean={gate1.mean():.4f}")

# Self attention (simplified - just q/k/v proj)
q = layer.self_attn.q_proj(normed)
print(f"Q projection: range=[{q.min():.4f}, {q.max():.4f}], std={q.std():.4f}")

# Post attention layernorm (pretend we have attention output)
fake_attn_out = torch.randn_like(hidden_states)
normed2, gate2 = layer.post_attention_layernorm(fake_attn_out, cond=cond)
print(f"After post_attention_layernorm: range=[{normed2.min():.4f}, {normed2.max():.4f}], std={normed2.std():.4f}")

# MLP
mlp_out = layer.mlp(normed2.to(layer.mlp.gate_proj.weight.dtype))
print(f"MLP output: range=[{mlp_out.min():.4f}, {mlp_out.max():.4f}], std={mlp_out.std():.4f}")

# Check if final norm is applied
print("\n=== Final Model Norm ===")
final_norm = model.paligemma_with_expert.gemma_expert.model.norm
test_hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16) * 60  # Simulate large values
print(f"Test input (scaled up): range=[{test_hidden.min():.4f}, {test_hidden.max():.4f}], std={test_hidden.std():.4f}")

normed_final, gate_final = final_norm(test_hidden, cond=cond)
print(f"After final norm: range=[{normed_final.min():.4f}, {normed_final.max():.4f}], std={normed_final.std():.4f}")
