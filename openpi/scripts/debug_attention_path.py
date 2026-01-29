#!/usr/bin/env python3
"""Debug which attention path is being used during inference."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

print("=" * 60)
print("Debugging Attention Path During Inference")
print("=" * 60)

# Import model first to trigger patching
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation

# Now patch the attention functions
_attention_calls = {}

def make_tracker(name, original_fn):
    def tracked_fn(*args, **kwargs):
        _attention_calls[name] = _attention_calls.get(name, 0) + 1
        return original_fn(*args, **kwargs)
    return tracked_fn

# Patch the sdpa function
from transformers.integrations import sdpa_attention
original_sdpa = sdpa_attention.sdpa_attention_forward
sdpa_attention.sdpa_attention_forward = make_tracker("sdpa", original_sdpa)

# Patch eager attention in gemma
from transformers.models.gemma import modeling_gemma
if hasattr(modeling_gemma, 'eager_attention_forward'):
    original_eager = modeling_gemma.eager_attention_forward
    modeling_gemma.eager_attention_forward = make_tracker("eager", original_eager)

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

device = torch.device("cuda")
dtype = torch.bfloat16

model = PI0Pytorch(config)
model = model.to(device=device, dtype=dtype)
model.eval()

batch_size = 1
observation = Observation(
    images={
        "base_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype),
        "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype),
        "right_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224, device=device, dtype=dtype),
    },
    image_masks={
        "base_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
        "right_wrist_0_rgb": torch.zeros(batch_size, device=device, dtype=torch.bool),
    },
    state=torch.randn(batch_size, 32, device=device, dtype=dtype),
    tokenized_prompt=torch.zeros(batch_size, 200, device=device, dtype=torch.long),
    tokenized_prompt_mask=torch.ones(batch_size, 200, device=device, dtype=torch.bool),
)

print("\nRunning inference with 1 denoising step...")
_attention_calls.clear()

with torch.no_grad():
    actions = model.sample_actions(device, observation, num_steps=1)

print("\nAttention function calls:")
print("-" * 40)
for name, count in sorted(_attention_calls.items()):
    print(f"  {name}: {count} calls")

if not _attention_calls:
    print("  (no calls recorded - functions may be inlined or different)")

print("\nConfig check:")
print(f"  VLM _attn_implementation: {model.paligemma_with_expert.paligemma.language_model.config._attn_implementation}")
print(f"  Expert _attn_implementation: {model.paligemma_with_expert.gemma_expert.config._attn_implementation}")
