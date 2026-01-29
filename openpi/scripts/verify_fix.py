#!/usr/bin/env python3
"""Quick verification that the PyTorch model produces non-zero outputs after the fix."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import numpy as np
import torch

print("=" * 60)
print("PyTorch Model Verification (with KV cache fix)")
print("=" * 60)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Create test inputs
batch_size = 1
image_np = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
image2_np = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
image3_np = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
state_np = np.random.randn(batch_size, 32).astype(np.float32) * 0.1
token_ids_np = np.zeros((batch_size, 64), dtype=np.int32)
token_ids_np[:, :5] = [100, 200, 300, 400, 500]
token_mask_np = np.zeros((batch_size, 64), dtype=bool)
token_mask_np[:, :5] = True
noise_np = np.random.randn(batch_size, 50, 32).astype(np.float32)

print(f"\nTest inputs created")

# Load PyTorch model
print("\nLoading PyTorch model...")
import safetensors.torch
from openpi.training import config as _config
from openpi.models import model as _model
from openpi.models_pytorch import pi0_pytorch

train_config = _config.get_config("pi05_libero")
pytorch_model = pi0_pytorch.PI0Pytorch(config=train_config.model)
safetensors.torch.load_model(pytorch_model, "/openpi_cache/checkpoints/pi05_libero/model.safetensors", strict=False)

# Fix weight tying
paligemma = pytorch_model.paligemma_with_expert.paligemma
embed_tokens = paligemma.model.language_model.embed_tokens.weight
lm_head = paligemma.lm_head.weight
if embed_tokens.shape == lm_head.shape:
    with torch.no_grad():
        embed_tokens.copy_(lm_head)
    print("Fixed weight tying")

pytorch_model.to(device)
pytorch_model.eval()
print("PyTorch model loaded!")

# Create observation
pt_observation = _model.Observation(
    images={
        "base_0_rgb": torch.from_numpy(image_np).permute(0, 3, 1, 2).to(device),
        "left_wrist_0_rgb": torch.from_numpy(image2_np).permute(0, 3, 1, 2).to(device),
        "right_wrist_0_rgb": torch.from_numpy(image3_np).permute(0, 3, 1, 2).to(device),
    },
    image_masks={
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool, device=device),
    },
    state=torch.from_numpy(state_np).to(device),
    tokenized_prompt=torch.from_numpy(token_ids_np.astype(np.int64)).to(device),
    tokenized_prompt_mask=torch.from_numpy(token_mask_np).to(device),
)

# Run inference
print("\nRunning PyTorch inference (with fix)...")
pt_noise = torch.from_numpy(noise_np).to(device)
with torch.no_grad():
    pt_actions = pytorch_model.sample_actions(device, pt_observation, num_steps=10, noise=pt_noise)
    pt_actions_np = pt_actions.cpu().numpy()

print(f"\nPyTorch actions:")
print(f"  shape: {pt_actions_np.shape}")
print(f"  range: [{pt_actions_np.min():.6f}, {pt_actions_np.max():.6f}]")
print(f"  mean: {pt_actions_np.mean():.6f}, std: {pt_actions_np.std():.6f}")
print(f"  first action: {pt_actions_np[0, 0, :7]}")

# Check if outputs are non-trivial
if abs(pt_actions_np.mean()) < 0.001 and pt_actions_np.std() < 0.01:
    print("\n✗ WARNING: Outputs are near-zero! Fix may not be working correctly.")
else:
    print("\n✓ Outputs look reasonable (non-zero mean/std)")

# Compare with expected JAX output (from previous run)
# JAX first action was approximately: [-0.2507225, 0.23688425, -0.4281018, ...]
expected_range = (-0.5, 0.5)  # Rough expected range based on JAX output
if expected_range[0] < pt_actions_np.mean() < expected_range[1]:
    print(f"✓ Mean is within expected range {expected_range}")
else:
    print(f"⚠ Mean is outside expected range {expected_range}")

print("\n" + "=" * 60)
