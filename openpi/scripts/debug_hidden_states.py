#!/usr/bin/env python3
"""Debug transformer hidden states."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import torch
import numpy as np

print("=" * 60)
print("Debugging Transformer Hidden States")
print("=" * 60)

from openpi.training import config as _config
from openpi.models import model as _model
import os

# Load model
train_config = _config.get_config("pi05_libero")
weight_path = os.path.join("/openpi_cache/checkpoints/pi05_libero", "model.safetensors")
model = train_config.model.load_pytorch(train_config, weight_path)
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
model = model.to("cuda")
model.eval()

# Create simple inputs
batch_size = 1
image = torch.randint(0, 255, (batch_size, 3, 224, 224), dtype=torch.uint8, device="cuda")
wrist_image = torch.randint(0, 255, (batch_size, 3, 224, 224), dtype=torch.uint8, device="cuda")
state = torch.zeros(batch_size, 8, dtype=torch.bfloat16, device="cuda")
token_ids = torch.zeros(batch_size, 64, dtype=torch.int32, device="cuda")
token_ids[:, :5] = torch.tensor([100, 200, 300, 400, 500])

images = {
    "base_0_rgb": (image.to(torch.bfloat16) / 127.5 - 1.0),
    "left_wrist_0_rgb": (wrist_image.to(torch.bfloat16) / 127.5 - 1.0),
    "right_wrist_0_rgb": (wrist_image.to(torch.bfloat16) / 127.5 - 1.0),
}
image_masks = {k: torch.ones(batch_size, dtype=torch.bool, device="cuda") for k in images.keys()}

obs = _model.Observation(
    images=images,
    image_masks=image_masks,
    state=state,
    tokenized_prompt=token_ids,
    tokenized_prompt_mask=torch.ones(batch_size, 64, dtype=torch.bool, device="cuda"),
)

# Hook to capture intermediate outputs
hidden_states_list = []
def hook_fn(module, input, output):
    if isinstance(output, tuple):
        out = output[0]
    else:
        out = output
    hidden_states_list.append(out.detach().cpu().float())

# Register hook on final layer of gemma expert
model.paligemma_with_expert.gemma_expert.model.layers[-1].register_forward_hook(hook_fn)

# Run inference
print("\nRunning inference...")
with torch.no_grad():
    actions = model.sample_actions("cuda", obs, num_steps=10)

print(f"\nFinal actions shape: {actions.shape}")
print(f"Final actions range: [{actions.min():.4f}, {actions.max():.4f}]")

# Check captured hidden states
print(f"\nCaptured {len(hidden_states_list)} hidden states outputs")
for i, hs in enumerate(hidden_states_list[:3]):  # Show first 3
    print(f"  Step {i}: shape={hs.shape}, range=[{hs.min():.4f}, {hs.max():.4f}], std={hs.std():.4f}")

# Let's also check the action_out_proj input directly
# by running one step manually
print("\n" + "=" * 60)
print("Manual Denoising Step Analysis")
print("=" * 60)

# Preprocess observation
images_proc, img_masks, lang_tokens, lang_masks, state_proc = model._preprocess_observation(obs, train=False)

# Embed prefix
prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images_proc, img_masks, lang_tokens, lang_masks)
prefix_att_2d_masks = model.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

# Get KV cache
prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

_, past_key_values = model.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)

# Initialize noise
noise = model.sample_noise((batch_size, model.config.action_horizon, model.config.action_dim), "cuda")
print(f"Initial noise range: [{noise.min():.4f}, {noise.max():.4f}], std={noise.std():.4f}")

# One denoising step
time = torch.tensor(1.0, dtype=torch.float32, device="cuda")
x_t = noise

suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state_proc, x_t, time.expand(batch_size))
print(f"\nSuffix embeddings:")
print(f"  shape: {suffix_embs.shape}")
print(f"  range: [{suffix_embs.min():.4f}, {suffix_embs.max():.4f}]")
print(f"  std: {suffix_embs.std():.4f}")

print(f"\nadaRMS conditioning:")
print(f"  shape: {adarms_cond.shape}")
print(f"  range: [{adarms_cond.min():.4f}, {adarms_cond.max():.4f}]")
print(f"  std: {adarms_cond.std():.4f}")

# Run through transformer
if suffix_embs.dtype != torch.bfloat16:
    suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

suffix_len = suffix_pad_masks.shape[1]
prefix_len = prefix_pad_masks.shape[1]
prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
suffix_att_2d_masks = model.make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=-1)
full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks)
suffix_position_ids = prefix_pad_masks.sum(dim=-1).unsqueeze(-1) + torch.cumsum(suffix_pad_masks, dim=-1) - 1

outputs_embeds, _ = model.paligemma_with_expert.forward(
    attention_mask=full_att_2d_masks_4d,
    position_ids=suffix_position_ids,
    past_key_values=past_key_values,
    inputs_embeds=[None, suffix_embs],
    use_cache=False,
    adarms_cond=[None, adarms_cond],
)

suffix_out = outputs_embeds[1]
print(f"\nTransformer output (suffix_out):")
print(f"  shape: {suffix_out.shape}")
print(f"  range: [{suffix_out.min():.4f}, {suffix_out.max():.4f}]")
print(f"  std: {suffix_out.std():.4f}")

# Get action tokens only
action_out = suffix_out[:, -model.config.action_horizon:]
print(f"\nAction hidden states (last {model.config.action_horizon} tokens):")
print(f"  shape: {action_out.shape}")
print(f"  range: [{action_out.min():.4f}, {action_out.max():.4f}]")
print(f"  std: {action_out.std():.4f}")

# Project to actions
v_t = model.action_out_proj(action_out.float())
print(f"\nProjected velocity (v_t):")
print(f"  shape: {v_t.shape}")
print(f"  range: [{v_t.min():.4f}, {v_t.max():.4f}]")
print(f"  std: {v_t.std():.4f}")
print(f"  first 7 dims of first action: {v_t[0, 0, :7].tolist()}")

# Apply Euler step
dt = -1.0 / 10
x_new = x_t + dt * v_t
print(f"\nAfter one Euler step (dt={dt}):")
print(f"  x_new range: [{x_new.min():.4f}, {x_new.max():.4f}]")
print(f"  x_new std: {x_new.std():.4f}")
