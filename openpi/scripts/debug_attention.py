#!/usr/bin/env python3
"""Debug script to check attention implementation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

print("=" * 60)
print("Attention Implementation Debug")
print("=" * 60)

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config

config = Pi0Config(
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
    action_dim=32,
    action_horizon=50,
    max_token_len=200,
    max_state_dim=32,
    pi05=True,
    dtype="bfloat16",
)

model = PI0Pytorch(config)

# Check VLM attention
print("\nPaliGemma Language Model:")
vlm_config = model.paligemma_with_expert.paligemma.language_model.config
print(f"  _attn_implementation: {getattr(vlm_config, '_attn_implementation', 'NOT SET')}")

# Check vision encoder
print("\nVision Encoder:")
try:
    vision_config = model.paligemma_with_expert.paligemma.vision_tower.vision_model.config
    print(f"  _attn_implementation: {getattr(vision_config, '_attn_implementation', 'NOT SET')}")
except Exception as e:
    print(f"  Error: {e}")

# Check Gemma Expert
print("\nGemma Expert:")
expert_config = model.paligemma_with_expert.gemma_expert.config
print(f"  _attn_implementation: {getattr(expert_config, '_attn_implementation', 'NOT SET')}")

# Check what ALL_ATTENTION_FUNCTIONS contains
print("\n" + "=" * 60)
print("Available Attention Functions:")
print("=" * 60)
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    for name in ALL_ATTENTION_FUNCTIONS:
        print(f"  - {name}")
except ImportError:
    print("  ALL_ATTENTION_FUNCTIONS not available")

# Check if SDPA is available
print("\n" + "=" * 60)
print("PyTorch Backend Info:")
print("=" * 60)
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  cuDNN version: {torch.backends.cudnn.version()}")
print(f"  Flash Attention available: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"  Memory-efficient attention available: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"  Math SDPA enabled: {torch.backends.cuda.math_sdp_enabled()}")
