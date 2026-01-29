#!/usr/bin/env python3
"""LIBERO evaluation using PyTorch policy directly."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")
sys.path.insert(0, "/app/third_party/libero")

import collections
import logging
import math
import pathlib
import argparse

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import tqdm

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


def eval_libero(args):
    np.random.seed(args.seed)

    # Load policy
    logging.info("Loading PyTorch policy...")
    train_config = _config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        pytorch_device="cuda"
    )
    logging.info("Policy loaded successfully")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}, {num_tasks_in_suite} tasks")

    # Get max steps for this suite
    max_steps_dict = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_dict.get(args.task_suite_name, 300)

    # Evaluation loop
    total_episodes, total_successes = 0, 0
    task_range = range(args.task_start, min(args.task_end, num_tasks_in_suite))

    for task_id in tqdm.tqdm(task_range, desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials), desc=f"Task {task_id}", leave=False):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])

            t = 0
            done = False
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                if not action_plan:
                    # Get new action chunk
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate((
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )),
                        "prompt": str(task_description),
                    }

                    result = policy.infer(element)
                    action_chunk = result["actions"]
                    action_plan.extend(action_chunk[:args.replan_steps])

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())

                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            task_episodes += 1
            total_episodes += 1

            if total_episodes % 5 == 0:
                logging.info(f"Progress: {total_episodes} episodes, {total_successes} successes ({100*total_successes/total_episodes:.1f}%)")

        print(f"\n>>> Task {task_id} ({task_description}): {task_successes}/{task_episodes} ({100*task_successes/task_episodes:.1f}%)", flush=True)

        # Early stop if success rate is 0 after first task
        if args.early_stop and task_id == args.task_start and task_successes == 0:
            print("\n>>> WARNING: 0% success rate on first task - stopping early", flush=True)
            break

    print("\n" + "=" * 60, flush=True)
    print(f">>> Final Results: {total_successes}/{total_episodes} ({100*total_successes/total_episodes:.1f}%)", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--checkpoint_dir", default="/openpi_cache/checkpoints/pi05_libero")
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--task_start", type=int, default=0)
    parser.add_argument("--task_end", type=int, default=10)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--early_stop", action="store_true", help="Stop early if first task has 0% success")
    args = parser.parse_args()

    eval_libero(args)
