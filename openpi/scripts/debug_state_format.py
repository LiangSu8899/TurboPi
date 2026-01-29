#!/usr/bin/env python3
"""Debug state format for LIBERO."""

import sys
sys.path.insert(0, "/app/third_party/libero")

import numpy as np
import pathlib
import math

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def _quat2axisangle(quat):
    """Convert quaternion to axis-angle."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

print("=" * 60)
print("State Format Debug")
print("=" * 60)

# Initialize environment
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
initial_states = task_suite.get_task_init_states(0)

task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
env = OffScreenRenderEnv(**env_args)
env.seed(7)

# Reset and get obs
env.reset()
obs = env.set_init_state(initial_states[0])

# Wait for objects to settle
DUMMY_ACTION = [0.0] * 6 + [-1.0]
for _ in range(10):
    obs, _, _, _ = env.step(DUMMY_ACTION)

print("\nObservation Keys:")
for key in sorted(obs.keys()):
    val = obs[key]
    if isinstance(val, np.ndarray):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"  {key}: {type(val)}")

print("\nRelevant State Components:")
print(f"  robot0_eef_pos: {obs['robot0_eef_pos']} (shape: {obs['robot0_eef_pos'].shape})")
print(f"  robot0_eef_quat: {obs['robot0_eef_quat']} (shape: {obs['robot0_eef_quat'].shape})")
print(f"  robot0_gripper_qpos: {obs['robot0_gripper_qpos']} (shape: {obs['robot0_gripper_qpos'].shape})")

# Construct state as in main.py
axis_angle = _quat2axisangle(obs["robot0_eef_quat"])
state = np.concatenate((
    obs["robot0_eef_pos"],
    axis_angle,
    obs["robot0_gripper_qpos"],
))
print(f"\nConstructed State:")
print(f"  Shape: {state.shape}")
print(f"  Values: {state}")

# Check against norm_stats
import json
norm_stats_path = "/openpi_cache/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
with open(norm_stats_path) as f:
    norm_stats = json.load(f)["norm_stats"]

state_q01 = np.array(norm_stats['state']['q01'])
state_q99 = np.array(norm_stats['state']['q99'])

print(f"\nState vs Norm Stats:")
print(f"  State shape: {state.shape}, Norm stats shape: {state_q01.shape}")
for i in range(len(state)):
    in_range = state_q01[i] <= state[i] <= state_q99[i]
    status = "✓" if in_range else "✗"
    print(f"  Dim {i}: value={state[i]:.4f}, range=[{state_q01[i]:.4f}, {state_q99[i]:.4f}] {status}")

env.close()
