#!/usr/bin/env python3
"""Debug config sharing in attention layers."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config

config = Pi0Config(
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
    dtype="bfloat16",
)

model = PI0Pytorch(config)

print("=" * 60)
print("Config Object Sharing Debug")
print("=" * 60)

# Check Gemma Expert
expert = model.paligemma_with_expert.gemma_expert.model
model_config = expert.config
print(f"\nExpert model config id: {id(model_config)}")
print(f"  _attn_implementation: {model_config._attn_implementation}")

print(f"\nChecking first 3 layers of Gemma Expert:")
for i, layer in enumerate(expert.layers[:3]):
    layer_config = layer.self_attn.config
    print(f"  Layer {i} config id: {id(layer_config)}, _attn: {layer_config._attn_implementation}")
    print(f"    Same as model config: {id(layer_config) == id(model_config)}")

# Set the config
print(f"\nSetting model_config._attn_implementation = 'sdpa'")
model_config._attn_implementation = "sdpa"

print(f"\nAfter setting:")
print(f"  Model config _attn: {model_config._attn_implementation}")
for i, layer in enumerate(expert.layers[:3]):
    layer_config = layer.self_attn.config
    print(f"  Layer {i} _attn: {layer_config._attn_implementation}")

# Check PaliGemma
vlm = model.paligemma_with_expert.paligemma.language_model
vlm_config = vlm.config
print(f"\n\nVLM model config id: {id(vlm_config)}")
print(f"  _attn_implementation: {vlm_config._attn_implementation}")

print(f"\nChecking first 3 layers of VLM:")
for i, layer in enumerate(vlm.layers[:3]):
    layer_config = layer.self_attn.config
    print(f"  Layer {i} config id: {id(layer_config)}, _attn: {layer_config._attn_implementation}")
    print(f"    Same as model config: {id(layer_config) == id(vlm_config)}")
