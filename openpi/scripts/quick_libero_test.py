#!/usr/bin/env python3
"""Quick LIBERO test to diagnose action output issues."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")
sys.path.insert(0, "/app/third_party/libero")

import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import pathlib

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

print("=" * 60)
print("Quick LIBERO Test")
print("=" * 60)

# Load policy
train_config = _config.get_config("pi05_libero")
checkpoint_dir = "/openpi_cache/checkpoints/pi05_libero"
policy = policy_config.create_trained_policy(
    train_config,
    checkpoint_dir,
    pytorch_device="cuda"
)
print("Policy loaded successfully")

# Initialize LIBERO environment
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
initial_states = task_suite.get_task_init_states(0)

task_description = task.language
task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
env.seed(7)

print(f"\nTask: {task_description}")

# Reset and get initial state
env.reset()
obs = env.set_init_state(initial_states[0])

# Wait for stabilization
for _ in range(10):
    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

# Process observation
def process_obs(obs):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))

    state = np.concatenate((
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }

def _quat2axisangle(quat):
    import math
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

# Get one action prediction
element = process_obs(obs)

print("\n" + "=" * 60)
print("Input Observation")
print("=" * 60)
print(f"Image shape: {element['observation/image'].shape}")
print(f"Wrist image shape: {element['observation/wrist_image'].shape}")
print(f"State shape: {element['observation/state'].shape}")
print(f"State values: {element['observation/state']}")
print(f"Prompt: {element['prompt']}")

# Run policy inference
result = policy.infer(element)

print("\n" + "=" * 60)
print("Policy Output")
print("=" * 60)
actions = result["actions"]
print(f"Actions shape: {actions.shape}")
print(f"Actions dtype: {actions.dtype}")
print(f"Actions range: [{actions.min():.4f}, {actions.max():.4f}]")
print(f"\nFirst action: {actions[0]}")
print(f"Last action: {actions[-1]}")

# Per-dimension analysis
print("\n" + "=" * 60)
print("Per-Dimension Action Analysis")
print("=" * 60)
dim_names = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
expected_ranges = [
    (-0.75, 0.94),  # x
    (-0.80, 0.86),  # y
    (-0.94, 0.94),  # z
    (-0.12, 0.14),  # rx
    (-0.17, 0.18),  # ry
    (-0.19, 0.31),  # rz
    (-1.0, 1.0),    # gripper
]

for i, (dim_name, (exp_min, exp_max)) in enumerate(zip(dim_names, expected_ranges)):
    dim_actions = actions[:, i]
    in_range = (dim_actions.min() >= exp_min - 0.1) and (dim_actions.max() <= exp_max + 0.1)
    status = "OK" if in_range else "OUT OF RANGE"
    print(f"Dim {i} ({dim_name}): [{dim_actions.min():.4f}, {dim_actions.max():.4f}], expected ~[{exp_min:.2f}, {exp_max:.2f}] {status}")

# Execute a few steps and observe
print("\n" + "=" * 60)
print("Executing 5 steps")
print("=" * 60)
initial_pos = obs["robot0_eef_pos"].copy()

for step in range(5):
    action = actions[step]
    print(f"Step {step}: action = {action}")
    obs, reward, done, info = env.step(action.tolist())
    current_pos = obs["robot0_eef_pos"]
    print(f"  Robot EEF position: {current_pos}")
    print(f"  Delta from start: {current_pos - initial_pos}")
    if done:
        print("Task completed!")
        break

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
