#!/usr/bin/env python3
"""Compare JAX and PyTorch model outputs with identical inputs."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import numpy as np
import torch
import jax
import jax.numpy as jnp
import os

print("=" * 60)
print("Comparing JAX vs PyTorch Model Outputs")
print("=" * 60)

from openpi.training import config as _config
from openpi.models import model as _model

# Set random seed for reproducibility
np.random.seed(42)

# Create deterministic test inputs
batch_size = 1
image_np = np.random.randint(0, 255, (batch_size, 3, 224, 224), dtype=np.uint8)
wrist_image_np = np.random.randint(0, 255, (batch_size, 3, 224, 224), dtype=np.uint8)
state_np = np.random.randn(batch_size, 8).astype(np.float32) * 0.1  # Small random state
token_ids_np = np.zeros((batch_size, 64), dtype=np.int32)
token_ids_np[:, :5] = [100, 200, 300, 400, 500]  # dummy tokens

# Create deterministic noise (action_horizon=50)
noise_np = np.random.randn(batch_size, 50, 32).astype(np.float32)

print("\nTest inputs created:")
print(f"  Image: shape={image_np.shape}, range=[{image_np.min()}, {image_np.max()}]")
print(f"  State: shape={state_np.shape}, range=[{state_np.min():.4f}, {state_np.max():.4f}]")
print(f"  Noise: shape={noise_np.shape}, range=[{noise_np.min():.4f}, {noise_np.max():.4f}]")

# ============== PyTorch Model ==============
print("\n" + "=" * 60)
print("Loading PyTorch Model")
print("=" * 60)

train_config = _config.get_config("pi05_libero")
checkpoint_dir = "/openpi_cache/checkpoints/pi05_libero"
weight_path = os.path.join(checkpoint_dir, "model.safetensors")

pytorch_model = train_config.model.load_pytorch(train_config, weight_path)
pytorch_model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
pytorch_model = pytorch_model.to("cuda")
pytorch_model.eval()

# Prepare PyTorch inputs
def prepare_pytorch_obs():
    images = {
        "base_0_rgb": (torch.from_numpy(image_np).to(torch.bfloat16).cuda() / 127.5 - 1.0),
        "left_wrist_0_rgb": (torch.from_numpy(wrist_image_np).to(torch.bfloat16).cuda() / 127.5 - 1.0),
        "right_wrist_0_rgb": (torch.from_numpy(wrist_image_np).to(torch.bfloat16).cuda() / 127.5 - 1.0),
    }
    image_masks = {k: torch.ones(batch_size, dtype=torch.bool, device="cuda") for k in images.keys()}

    obs = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=torch.from_numpy(state_np).to(torch.bfloat16).cuda(),
        tokenized_prompt=torch.from_numpy(token_ids_np).to(torch.int32).cuda(),
        tokenized_prompt_mask=torch.ones(batch_size, 64, dtype=torch.bool, device="cuda"),
    )
    return obs

pytorch_obs = prepare_pytorch_obs()
pytorch_noise = torch.from_numpy(noise_np).to(torch.float32).cuda()

print("Running PyTorch inference...")
with torch.no_grad():
    pytorch_actions = pytorch_model.sample_actions("cuda", pytorch_obs, noise=pytorch_noise, num_steps=10)

pytorch_actions_np = pytorch_actions.cpu().float().numpy()
print(f"PyTorch actions shape: {pytorch_actions_np.shape}")
print(f"PyTorch actions range: [{pytorch_actions_np.min():.4f}, {pytorch_actions_np.max():.4f}]")
print(f"PyTorch actions mean: {pytorch_actions_np.mean():.4f}")
print(f"PyTorch actions std: {pytorch_actions_np.std():.4f}")
print(f"\nFirst action (first 7 dims): {pytorch_actions_np[0, 0, :7]}")
print(f"Last action (first 7 dims): {pytorch_actions_np[0, -1, :7]}")

# Per-dimension analysis
print("\n" + "-" * 40)
print("Per-dimension analysis (PyTorch):")
for i, dim_name in enumerate(['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']):
    dim_actions = pytorch_actions_np[0, :, i]
    print(f"  Dim {i} ({dim_name}): range [{dim_actions.min():.4f}, {dim_actions.max():.4f}], mean={dim_actions.mean():.4f}")

# ============== Check intermediate outputs ==============
print("\n" + "=" * 60)
print("Debugging Intermediate Outputs")
print("=" * 60)

# Check one denoising step manually
with torch.no_grad():
    # Preprocess observation
    images, img_masks, lang_tokens, lang_masks, state = pytorch_model._preprocess_observation(pytorch_obs, train=False)

    # Embed prefix
    prefix_embs, prefix_pad_masks, prefix_att_masks = pytorch_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    print(f"\nPrefix embeddings: shape={prefix_embs.shape}, range=[{prefix_embs.min():.4f}, {prefix_embs.max():.4f}]")
    print(f"Prefix pad masks: shape={prefix_pad_masks.shape}")

    # Prepare attention masks
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = pytorch_model._prepare_attention_masks_4d(prefix_att_2d_masks)

    # Get KV cache
    pytorch_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, past_key_values = pytorch_model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    print(f"KV cache created: {len(past_key_values)} layers")

    # Test one denoising step at t=1.0
    x_t = pytorch_noise
    time = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    # Embed suffix
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = pytorch_model.embed_suffix(state, x_t, time.expand(batch_size))

    print(f"\nSuffix embeddings (t=1.0): shape={suffix_embs.shape}")
    print(f"  range=[{suffix_embs.min():.4f}, {suffix_embs.max():.4f}]")
    print(f"  std={suffix_embs.std():.4f}")

    print(f"\nadaRMS conditioning: shape={adarms_cond.shape}")
    print(f"  range=[{adarms_cond.min():.4f}, {adarms_cond.max():.4f}]")
    print(f"  std={adarms_cond.std():.4f}")

    # Run denoise step
    v_t = pytorch_model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(batch_size))

    print(f"\nVelocity v_t (t=1.0): shape={v_t.shape}")
    print(f"  range=[{v_t.min():.4f}, {v_t.max():.4f}]")
    print(f"  std={v_t.std():.4f}")
    print(f"  first 7 dims: {v_t[0, 0, :7].cpu().numpy()}")

    # Apply Euler step
    dt = -1.0 / 10
    x_new = x_t + dt * v_t
    print(f"\nAfter Euler step (dt={dt}): x_new range=[{x_new.min():.4f}, {x_new.max():.4f}]")

# ============== Check weight statistics ==============
print("\n" + "=" * 60)
print("Weight Statistics")
print("=" * 60)

action_in_proj = pytorch_model.action_in_proj
action_out_proj = pytorch_model.action_out_proj

print(f"action_in_proj.weight: shape={action_in_proj.weight.shape}, range=[{action_in_proj.weight.min():.4f}, {action_in_proj.weight.max():.4f}]")
print(f"action_out_proj.weight: shape={action_out_proj.weight.shape}, range=[{action_out_proj.weight.min():.4f}, {action_out_proj.weight.max():.4f}]")

# Check time MLP
time_mlp_in = pytorch_model.time_mlp_in
time_mlp_out = pytorch_model.time_mlp_out
print(f"\ntime_mlp_in.weight: shape={time_mlp_in.weight.shape}, range=[{time_mlp_in.weight.min():.4f}, {time_mlp_in.weight.max():.4f}]")
print(f"time_mlp_out.weight: shape={time_mlp_out.weight.shape}, range=[{time_mlp_out.weight.min():.4f}, {time_mlp_out.weight.max():.4f}]")
