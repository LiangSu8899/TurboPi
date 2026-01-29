#!/usr/bin/env python3
"""Debug with real LIBERO observation."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")
sys.path.insert(0, "/app/third_party/libero")

import collections
import logging
import math
import pathlib
import numpy as np
import torch

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi.training import config as _config
from openpi.policies import policy_config

logging.basicConfig(level=logging.INFO)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

print("=" * 60)
print("Loading Policy")
print("=" * 60)

train_config = _config.get_config("pi05_libero")
checkpoint_dir = "/openpi_cache/checkpoints/pi05_libero"
policy = policy_config.create_trained_policy(
    train_config,
    checkpoint_dir,
    pytorch_device="cuda"
)
print("Policy loaded successfully")

# Load norm stats for reference
import json
norm_stats_path = f"{checkpoint_dir}/assets/physical-intelligence/libero/norm_stats.json"
with open(norm_stats_path) as f:
    norm_stats = json.load(f)["norm_stats"]
q01 = np.array(norm_stats['actions']['q01'])
q99 = np.array(norm_stats['actions']['q99'])
print(f"\nAction norm stats:")
print(f"  q01: {q01}")
print(f"  q99: {q99}")

print("\n" + "=" * 60)
print("Setting up LIBERO Environment")
print("=" * 60)

np.random.seed(7)

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)  # First task
initial_states = task_suite.get_task_init_states(0)
env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, 7)

print(f"Task: {task_description}")

# Reset environment and wait for objects to settle
env.reset()
obs = env.set_init_state(initial_states[0])

# Wait for objects to settle
for _ in range(10):
    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

print("\n" + "=" * 60)
print("Observation Analysis")
print("=" * 60)

# Get observation
agentview_img = obs["agentview_image"]
wrist_img = obs["robot0_eye_in_hand_image"]

print(f"agentview_image: shape={agentview_img.shape}, dtype={agentview_img.dtype}, range=[{agentview_img.min()}, {agentview_img.max()}]")
print(f"wrist_image: shape={wrist_img.shape}, dtype={wrist_img.dtype}, range=[{wrist_img.min()}, {wrist_img.max()}]")

# Preprocess images (rotate 180 degrees)
img = np.ascontiguousarray(agentview_img[::-1, ::-1])
wrist_img_proc = np.ascontiguousarray(wrist_img[::-1, ::-1])

# Resize with padding
img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
wrist_img_proc = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img_proc, 224, 224))

print(f"\nAfter preprocessing:")
print(f"  img: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
print(f"  wrist_img: shape={wrist_img_proc.shape}, dtype={wrist_img_proc.dtype}")

# Get robot state
eef_pos = obs["robot0_eef_pos"]
eef_quat = obs["robot0_eef_quat"]
gripper_qpos = obs["robot0_gripper_qpos"]
state = np.concatenate((eef_pos, _quat2axisangle(eef_quat), gripper_qpos))

print(f"\nRobot state: shape={state.shape}")
print(f"  EEF position: {eef_pos}")
print(f"  EEF quaternion: {eef_quat}")
print(f"  EEF axis-angle: {_quat2axisangle(obs['robot0_eef_quat'])}")
print(f"  Gripper qpos: {gripper_qpos}")

print("\n" + "=" * 60)
print("Running Inference")
print("=" * 60)

element = {
    "observation/image": img,
    "observation/wrist_image": wrist_img_proc,
    "observation/state": state,
    "prompt": str(task_description),
}

print(f"Prompt: {task_description}")

with torch.no_grad():
    result = policy.infer(element)

actions = result["actions"]
print(f"\nActions shape: {actions.shape}")
print(f"Actions dtype: {actions.dtype}")
print(f"Actions range: [{actions.min():.4f}, {actions.max():.4f}]")
print(f"Actions mean: {actions.mean():.4f}")
print(f"Actions std: {actions.std():.4f}")

print("\n" + "=" * 60)
print("Per-Dimension Action Analysis")
print("=" * 60)

dim_names = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
print("\nFirst action (should be the first step of the trajectory):")
for i, dim_name in enumerate(dim_names):
    val = actions[0, i]
    expected_min, expected_max = q01[i], q99[i]
    print(f"  Dim {i} ({dim_name}): {val:.4f} (expected range: [{expected_min:.4f}, {expected_max:.4f}])")

print("\nAction trajectory (first 7 dims):")
for step in range(min(5, actions.shape[0])):
    print(f"  Step {step}: {actions[step, :7]}")

# Check if actions are within expected range
print("\n" + "=" * 60)
print("Range Check")
print("=" * 60)
in_range = True
for i, dim_name in enumerate(dim_names):
    dim_actions = actions[:, i]
    expected_min, expected_max = q01[i], q99[i]
    actual_min, actual_max = dim_actions.min(), dim_actions.max()
    coverage = (actual_max - actual_min) / (expected_max - expected_min + 1e-6) * 100
    status = "OK" if coverage > 10 else "LOW"
    print(f"  Dim {i} ({dim_name}): range [{actual_min:.4f}, {actual_max:.4f}] vs expected [{expected_min:.4f}, {expected_max:.4f}] - coverage {coverage:.1f}% {status}")
    if coverage < 5:
        in_range = False

if not in_range:
    print("\nWARNING: Action range is much smaller than expected. The model may not be working correctly.")
else:
    print("\nAction ranges look reasonable.")

# Test applying first action to environment
print("\n" + "=" * 60)
print("Testing Action in Environment")
print("=" * 60)

first_action = actions[0, :7].tolist()
print(f"First action: {first_action}")

# Get current EEF position
current_pos = obs["robot0_eef_pos"].copy()
print(f"Current EEF position: {current_pos}")

# Apply action
obs_after, reward, done, info = env.step(first_action)
new_pos = obs_after["robot0_eef_pos"]
print(f"New EEF position: {new_pos}")
print(f"Position change: {new_pos - current_pos}")
print(f"Position change magnitude: {np.linalg.norm(new_pos - current_pos):.6f} m")

# Apply more actions
for i in range(1, min(5, actions.shape[0])):
    action = actions[i, :7].tolist()
    prev_pos = obs_after["robot0_eef_pos"].copy()
    obs_after, _, _, _ = env.step(action)
    new_pos = obs_after["robot0_eef_pos"]
    print(f"Step {i+1}: pos change = {np.linalg.norm(new_pos - prev_pos):.6f} m")

env.close()
print("\nEnvironment closed.")
