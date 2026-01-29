#!/usr/bin/env python3
"""Test robosuite OSC_POSE controller behavior directly."""

import sys
sys.path.insert(0, "/app/third_party/libero")

import numpy as np
import pathlib

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

print("=" * 60)
print("Testing robosuite OSC_POSE Controller")
print("=" * 60)

# Initialize environment
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
initial_states = task_suite.get_task_init_states(0)

task_description = task.language
task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
env = OffScreenRenderEnv(**env_args)
env.seed(7)

print(f"Task: {task_description}")

# Reset and initialize
env.reset()
obs = env.set_init_state(initial_states[0])

# Wait for objects to settle
DUMMY_ACTION = [0.0] * 6 + [-1.0]
for _ in range(10):
    obs, _, _, _ = env.step(DUMMY_ACTION)

print("\nInitial robot state:")
print(f"  EEF position: {obs['robot0_eef_pos']}")

# Get the robot controller config
robot = env.env.robots[0]
print(f"\nRobot type: {type(robot)}")
print(f"Controller type: {type(robot.controller)}")

# Check controller config
controller = robot.controller
print(f"\nController config:")
print(f"  control_freq: {controller.control_freq if hasattr(controller, 'control_freq') else 'N/A'}")
print(f"  input_max: {controller.input_max if hasattr(controller, 'input_max') else 'N/A'}")
print(f"  input_min: {controller.input_min if hasattr(controller, 'input_min') else 'N/A'}")
print(f"  output_max: {controller.output_max if hasattr(controller, 'output_max') else 'N/A'}")
print(f"  output_min: {controller.output_min if hasattr(controller, 'output_min') else 'N/A'}")

# Test different action magnitudes
print("\n" + "=" * 60)
print("Testing Different Action Magnitudes")
print("=" * 60)

# Reset to initial state
obs = env.set_init_state(initial_states[0])
for _ in range(10):
    obs, _, _, _ = env.step(DUMMY_ACTION)

test_actions = [
    ("small", [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    ("medium", [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    ("large", [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    ("max", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    ("negative", [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
]

for name, action in test_actions:
    # Reset to initial state
    obs = env.set_init_state(initial_states[0])
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    initial_pos = obs['robot0_eef_pos'].copy()

    # Apply action for 10 steps
    for _ in range(10):
        obs, _, _, _ = env.step(action)

    final_pos = obs['robot0_eef_pos']
    pos_change = final_pos - initial_pos

    print(f"\n{name} action (x={action[0]}):")
    print(f"  Position change after 10 steps: {pos_change}")
    print(f"  Total x displacement: {pos_change[0]:.6f} m")
    print(f"  Per-step x displacement: {pos_change[0]/10:.6f} m")

# Test with actual model output magnitude
print("\n" + "=" * 60)
print("Testing with Model-Like Actions")
print("=" * 60)

# From our debug: first action x = -0.16
model_like_actions = [
    ("model_first", [-0.16, -0.04, 0.03, 0.007, 0.06, 0.007, 0.2]),
    ("model_scaled_10x", [-1.6, -0.4, 0.3, 0.07, 0.6, 0.07, 1.0]),
]

for name, action in model_like_actions:
    # Reset to initial state
    obs = env.set_init_state(initial_states[0])
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    initial_pos = obs['robot0_eef_pos'].copy()

    # Apply action for 5 steps (like replan_steps)
    for _ in range(5):
        # Clip to valid range
        clipped_action = np.clip(action, -1.0, 1.0)
        obs, _, _, _ = env.step(clipped_action)

    final_pos = obs['robot0_eef_pos']
    pos_change = final_pos - initial_pos

    print(f"\n{name}:")
    print(f"  Original action: {action[:7]}")
    print(f"  Clipped action: {np.clip(action, -1.0, 1.0)[:7]}")
    print(f"  Position change after 5 steps: {pos_change}")
    print(f"  Total displacement: {np.linalg.norm(pos_change):.6f} m")

env.close()
print("\nEnvironment closed.")
