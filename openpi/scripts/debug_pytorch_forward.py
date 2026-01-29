#!/usr/bin/env python3
"""Debug PyTorch model forward pass step by step."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import safetensors.torch

print("=" * 60)
print("PyTorch Forward Pass Debug")
print("=" * 60)

np.random.seed(42)
batch_size = 1

# Load model
from openpi.training import config as _config
from openpi.models_pytorch import pi0_pytorch
from openpi.models import model as _model

train_config = _config.get_config("pi05_libero")
model = pi0_pytorch.PI0Pytorch(config=train_config.model)
safetensors.torch.load_model(model, "/openpi_cache/checkpoints/pi05_libero/model.safetensors", strict=False)

# Fix weight tying
paligemma = model.paligemma_with_expert.paligemma
with torch.no_grad():
    paligemma.model.language_model.embed_tokens.weight.copy_(paligemma.lm_head.weight)

model.eval()
print("Model loaded")
print(f"action_horizon: {model.config.action_horizon}")
print(f"action_dim: {model.config.action_dim}")

# Create test inputs
base_image = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)  # NCHW
left_wrist = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)
right_wrist = torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32)
state = torch.randn(batch_size, 32, dtype=torch.float32) * 0.1
tokens = torch.zeros(batch_size, 64, dtype=torch.int64)
tokens[:, :5] = torch.tensor([100, 200, 300, 400, 500])
token_mask = torch.zeros(batch_size, 64, dtype=torch.bool)
token_mask[:, :5] = True

observation = _model.Observation(
    images={
        "base_0_rgb": base_image,
        "left_wrist_0_rgb": left_wrist,
        "right_wrist_0_rgb": right_wrist,
    },
    image_masks={
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool),
    },
    state=state,
    tokenized_prompt=tokens,
    tokenized_prompt_mask=token_mask,
)

# Step 1: Preprocess observation
print("\n1. Preprocessing observation...")
from openpi.models_pytorch import preprocessing_pytorch
proc_obs = preprocessing_pytorch.preprocess_observation_pytorch(observation, train=False)
print(f"   Preprocessed images keys: {list(proc_obs.images.keys())}")
for key, img in proc_obs.images.items():
    print(f"   {key}: shape={img.shape}, range=[{img.min():.4f}, {img.max():.4f}]")

# Step 2: Embed prefix (images + language)
print("\n2. Embedding prefix...")
images = list(proc_obs.images.values())
img_masks = list(proc_obs.image_masks.values())
lang_tokens = proc_obs.tokenized_prompt
lang_masks = proc_obs.tokenized_prompt_mask

prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
print(f"   prefix_embs: shape={prefix_embs.shape}, range=[{prefix_embs.min():.4f}, {prefix_embs.max():.4f}]")
print(f"   prefix_pad_masks: shape={prefix_pad_masks.shape}")

# Step 3: Test time embedding
print("\n3. Testing time embedding...")
time = torch.tensor([1.0], dtype=torch.float32)
time_emb = pi0_pytorch.create_sinusoidal_pos_embedding(
    time, model.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=time.device
)
print(f"   time_emb (t=1.0): shape={time_emb.shape}, range=[{time_emb.min():.4f}, {time_emb.max():.4f}]")

# Step 4: Test time MLP
print("\n4. Testing time MLP...")
time_emb_mlp = model.time_mlp_in(time_emb)
time_emb_mlp = torch.nn.functional.silu(time_emb_mlp)
time_emb_mlp = model.time_mlp_out(time_emb_mlp)
time_emb_mlp = torch.nn.functional.silu(time_emb_mlp)
print(f"   time_emb after MLP: shape={time_emb_mlp.shape}, range=[{time_emb_mlp.min():.4f}, {time_emb_mlp.max():.4f}]")
print(f"   time_emb after MLP mean: {time_emb_mlp.mean():.4f}, std: {time_emb_mlp.std():.4f}")

# Step 5: Test action embedding
print("\n5. Testing action embedding...")
noisy_action = torch.randn(batch_size, 50, 32, dtype=torch.float32)
action_emb = model.action_in_proj(noisy_action)
print(f"   action_emb: shape={action_emb.shape}, range=[{action_emb.min():.4f}, {action_emb.max():.4f}]")

# Step 6: Embed suffix
print("\n6. Embedding suffix...")
suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(proc_obs.state, noisy_action, time)
print(f"   suffix_embs: shape={suffix_embs.shape}, range=[{suffix_embs.min():.4f}, {suffix_embs.max():.4f}]")
print(f"   adarms_cond: shape={adarms_cond.shape if adarms_cond is not None else None}")
if adarms_cond is not None:
    print(f"   adarms_cond range: [{adarms_cond.min():.4f}, {adarms_cond.max():.4f}]")

# Step 7: Run full sample_actions
print("\n7. Running sample_actions...")
noise = torch.randn(batch_size, 50, 32, dtype=torch.float32)
with torch.no_grad():
    actions = model.sample_actions("cpu", observation, num_steps=10, noise=noise)

print(f"   Final actions: shape={actions.shape}")
print(f"   Actions range: [{actions.min():.4f}, {actions.max():.4f}]")
print(f"   Actions mean: {actions.mean():.4f}, std: {actions.std():.4f}")
print(f"   First action: {actions[0, 0, :7].numpy()}")

# Step 8: Check denoise step outputs
print("\n8. Debugging denoise step...")
# Get KV cache
prefix_att_2d_masks = pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

_, past_key_values = model.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)

# Single denoise step
x_t = noise.clone()
time_step = torch.tensor([1.0], dtype=torch.float32)
v_t = model.denoise_step(proc_obs.state, prefix_pad_masks, past_key_values, x_t, time_step)
print(f"   v_t (t=1.0): shape={v_t.shape}, range=[{v_t.min():.4f}, {v_t.max():.4f}]")
print(f"   v_t mean: {v_t.mean():.4f}, std: {v_t.std():.4f}")

# Check action_out_proj
print("\n9. Checking action_out_proj...")
print(f"   action_out_proj weight: shape={model.action_out_proj.weight.shape}")
print(f"   action_out_proj weight range: [{model.action_out_proj.weight.min():.4f}, {model.action_out_proj.weight.max():.4f}]")
if model.action_out_proj.bias is not None:
    print(f"   action_out_proj bias range: [{model.action_out_proj.bias.min():.4f}, {model.action_out_proj.bias.max():.4f}]")

print("\n" + "=" * 60)
