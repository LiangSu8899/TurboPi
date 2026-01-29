#!/usr/bin/env python3
"""Debug model internals to find the root cause of 0% success rate."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import numpy as np
import torch
import safetensors.torch

print("=" * 60)
print("MODEL INTERNALS DEBUG")
print("=" * 60)

# Load checkpoint weights
checkpoint_path = "/openpi_cache/checkpoints/pi05_libero/model.safetensors"
checkpoint_weights = safetensors.torch.load_file(checkpoint_path)

print(f"\n1. Checkpoint has {len(checkpoint_weights)} weights")

# Sample some key weights
print("\n2. Key weight statistics from checkpoint:")
key_weights = [
    "action_in_proj.weight",
    "action_out_proj.weight",
    "time_mlp_in.weight",
    "time_mlp_out.weight",
    "paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight",
    "paligemma_with_expert.paligemma.lm_head.weight",
    "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
]

for key in key_weights:
    if key in checkpoint_weights:
        w = checkpoint_weights[key]
        print(f"  {key}")
        print(f"    shape: {w.shape}, dtype: {w.dtype}")
        print(f"    range: [{w.min():.6f}, {w.max():.6f}]")
        print(f"    mean: {w.mean():.6f}, std: {w.std():.6f}")
    else:
        print(f"  {key}: NOT FOUND")

# Create model and load weights
print("\n3. Creating PyTorch model...")
from openpi.training import config as _config
from openpi.models_pytorch import pi0_pytorch

train_config = _config.get_config("pi05_libero")
model = pi0_pytorch.PI0Pytorch(config=train_config.model)
print(f"   Model created, action_horizon: {train_config.model.action_horizon}")

# Load weights
print("\n4. Loading weights...")
safetensors.torch.load_model(model, checkpoint_path, strict=False)

# Fix weight tying
paligemma = model.paligemma_with_expert.paligemma
embed_tokens = paligemma.model.language_model.embed_tokens.weight
lm_head = paligemma.lm_head.weight
if embed_tokens.shape == lm_head.shape:
    with torch.no_grad():
        embed_tokens.copy_(lm_head)
    print("   Fixed weight tying: copied lm_head.weight to embed_tokens.weight")

# Check weights after loading
print("\n5. Weights after loading into model:")
model_weights = {
    "action_in_proj": model.action_in_proj.weight,
    "action_out_proj": model.action_out_proj.weight,
    "time_mlp_in": model.time_mlp_in.weight,
    "time_mlp_out": model.time_mlp_out.weight,
}
for name, w in model_weights.items():
    print(f"  {name}.weight")
    print(f"    shape: {w.shape}, dtype: {w.dtype}")
    print(f"    range: [{w.min():.6f}, {w.max():.6f}]")
    print(f"    mean: {w.mean():.6f}, std: {w.std():.6f}")

# Check biases if they exist
print("\n6. Checking biases (should be None for Linear without bias):")
print(f"  action_in_proj.bias: {model.action_in_proj.bias}")
print(f"  action_out_proj.bias: {model.action_out_proj.bias}")
print(f"  time_mlp_in.bias: {model.time_mlp_in.bias}")
print(f"  time_mlp_out.bias: {model.time_mlp_out.bias}")

# Run a test inference
print("\n7. Test inference on synthetic data...")
model.eval()
model = model.to("cuda")
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

# Create synthetic observation
from openpi.models import model as _model
batch_size = 1
images = {
    "primary": torch.randint(0, 255, (batch_size, 3, 224, 224), dtype=torch.uint8, device="cuda"),
    "secondary": torch.randint(0, 255, (batch_size, 3, 224, 224), dtype=torch.uint8, device="cuda"),
}
image_masks = {
    "primary": torch.ones(batch_size, dtype=torch.bool, device="cuda"),
    "secondary": torch.ones(batch_size, dtype=torch.bool, device="cuda"),
}
state = torch.randn(batch_size, 32, dtype=torch.float32, device="cuda") * 0.1
tokenized_prompt = torch.zeros(batch_size, 64, dtype=torch.int64, device="cuda")
tokenized_prompt[:, :5] = torch.tensor([100, 200, 300, 400, 500])
tokenized_prompt_mask = torch.zeros(batch_size, 64, dtype=torch.bool, device="cuda")
tokenized_prompt_mask[:, :5] = True

observation = _model.Observation(
    images=images,
    image_masks=image_masks,
    state=state,
    tokenized_prompt=tokenized_prompt,
    tokenized_prompt_mask=tokenized_prompt_mask,
)

print(f"  Input state range: [{state.min():.4f}, {state.max():.4f}]")

with torch.no_grad():
    actions = model.sample_actions("cuda", observation, num_steps=10)

print(f"\n8. Raw model output:")
print(f"  actions shape: {actions.shape}")
print(f"  actions dtype: {actions.dtype}")
print(f"  actions range: [{actions.min():.6f}, {actions.max():.6f}]")
print(f"  actions mean: {actions.mean():.6f}, std: {actions.std():.6f}")
print(f"  First action: {actions[0, 0, :7].cpu().numpy()}")
print(f"  Last action: {actions[0, -1, :7].cpu().numpy()}")

# Check denoise step intermediate outputs
print("\n9. Analyzing denoise step...")
images_list, img_masks_list, lang_tokens, lang_masks, state_proc = model._preprocess_observation(observation, train=False)

prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images_list, img_masks_list, lang_tokens, lang_masks)
print(f"  prefix_embs shape: {prefix_embs.shape}")
print(f"  prefix_embs range: [{prefix_embs.min():.4f}, {prefix_embs.max():.4f}]")

# Test time embedding
time = torch.tensor([1.0], dtype=torch.float32, device="cuda")
time_emb = pi0_pytorch.create_sinusoidal_pos_embedding(
    time, model.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=time.device
)
print(f"\n  time_emb (t=1.0) shape: {time_emb.shape}")
print(f"  time_emb range: [{time_emb.min():.4f}, {time_emb.max():.4f}]")

# Time MLP output
time_emb_after_mlp = model.time_mlp_in(time_emb)
time_emb_after_mlp = torch.nn.functional.silu(time_emb_after_mlp)
time_emb_after_mlp = model.time_mlp_out(time_emb_after_mlp)
time_emb_after_mlp = torch.nn.functional.silu(time_emb_after_mlp)
print(f"  time_emb after MLP range: [{time_emb_after_mlp.min():.4f}, {time_emb_after_mlp.max():.4f}]")

# Test action embedding
noisy_action = torch.randn(batch_size, 50, 32, dtype=torch.bfloat16, device="cuda")
action_emb = model.action_in_proj(noisy_action)
print(f"\n  action_emb shape: {action_emb.shape}")
print(f"  action_emb range: [{action_emb.min():.4f}, {action_emb.max():.4f}]")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
