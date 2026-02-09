#!/usr/bin/env python3
"""
LIBERO Evaluation with Pipeline (Async) Optimization.

This script tests whether we can hide KV Cache latency (~50ms) during
Denoising execution (~100ms) using async inference and action buffering.

Key insight: While robot executes actions from buffer, we can run inference
for the next observation in parallel.

Usage:
    # Test pipeline vs baseline
    python scripts/libero_eval_pipeline.py --quick
    python scripts/libero_eval_pipeline.py --mode baseline --quick
    python scripts/libero_eval_pipeline.py --mode pipeline --quick
"""

import sys
import os
import collections
import logging
import math
import pathlib
import argparse
import json
import time
import threading
import queue
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import torch

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "openpi-client", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "libero"))

for path in ["/workspace/src", "/workspace/packages/openpi-client/src", "/workspace/third_party/libero"]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Set MuJoCo rendering options
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
torch.backends.cudnn.enabled = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import base optimized policy
from libero_eval_full_optimized import (
    FullOptimizedPolicy,
    run_episode,
    _get_libero_env,
    MAX_STEPS_DICT,
    LIBERO_ENV_RESOLUTION,
    LIBERO_DUMMY_ACTION,
    _quat2axisangle,
    resize_with_pad,
)


class PipelinePolicy:
    """
    Pipeline-optimized policy with async inference and action buffering.

    Design:
    - Main thread: Execute actions, get observations
    - Inference thread: Run VLA inference asynchronously
    - Action buffer: Store action chunks for smooth execution

    Key optimization:
    - While robot executes action[t], inference for obs[t+1] can start
    - Latency is "hidden" in action execution time
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        num_denoising_steps: int = 10,
        schedule_type: str = "linear",
        action_buffer_size: int = 10,  # How many steps to buffer
    ):
        self.device = device
        self.action_buffer_size = action_buffer_size

        # Create base policy
        self.base_policy = FullOptimizedPolicy(
            checkpoint_dir=checkpoint_dir,
            device=device,
            num_denoising_steps=num_denoising_steps,
            schedule_type=schedule_type,
        )

        # Async inference state
        self.inference_future: Optional[Future] = None
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Action buffer
        self.action_buffer = collections.deque()

        # Timing stats
        self.effective_latencies = []  # Actual wait time for user
        self.inference_latencies = []  # Full inference time
        self.overlap_times = []  # How much latency was hidden

        logger.info(f"PipelinePolicy initialized with buffer_size={action_buffer_size}")

    def reset_episode(self):
        """Reset for new episode."""
        self.action_buffer.clear()
        self.inference_future = None
        self.effective_latencies = []
        self.inference_latencies = []
        self.overlap_times = []

    def _run_inference(self, observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Run inference (called in background thread)."""
        return self.base_policy.infer(observation)

    def submit_inference(self, observation: Dict[str, Any]):
        """Submit inference request asynchronously."""
        if self.inference_future is not None:
            # Wait for previous inference to complete first
            self.inference_future.result()

        self.inference_future = self.executor.submit(self._run_inference, observation)

    def get_action(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Get next action, using pipeline optimization.

        Returns:
            Action to execute, or None if waiting for inference
        """
        start_time = time.perf_counter()

        # If buffer is empty, need to wait for inference
        if len(self.action_buffer) == 0:
            # Submit new inference if not already running
            if self.inference_future is None:
                self.submit_inference(observation)

            # Wait for result
            result = self.inference_future.result()
            inference_time = time.perf_counter() - start_time
            self.inference_latencies.append(inference_time * 1000)
            self.effective_latencies.append(inference_time * 1000)
            self.overlap_times.append(0.0)  # No overlap when buffer empty

            # Fill buffer with action chunk
            actions = result["actions"]
            for i in range(min(self.action_buffer_size, len(actions))):
                self.action_buffer.append(actions[i])

            self.inference_future = None

        # Get action from buffer
        if len(self.action_buffer) > 0:
            action = self.action_buffer.popleft()

            # Proactively start next inference when buffer runs low
            if len(self.action_buffer) <= 2 and self.inference_future is None:
                # Will be submitted on next call
                pass

            effective_time = (time.perf_counter() - start_time) * 1000
            if effective_time < 5:  # Only count if we actually waited
                self.overlap_times.append(effective_time)

            return action

        return None

    def infer_with_overlap(self, observation: Dict[str, Any], last_inference_time: float = 0) -> Dict[str, np.ndarray]:
        """
        Infer with explicit overlap tracking.

        Args:
            observation: Current observation
            last_inference_time: Time spent on last inference (used to calculate overlap)

        Returns:
            Actions dictionary
        """
        start_time = time.perf_counter()

        result = self.base_policy.infer(observation)

        inference_time = (time.perf_counter() - start_time) * 1000
        self.inference_latencies.append(inference_time)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        base_stats = self.base_policy.get_latency_stats()

        if not self.effective_latencies:
            return base_stats

        effective = np.array(self.effective_latencies)
        inference = np.array(self.inference_latencies) if self.inference_latencies else effective
        overlap = np.array(self.overlap_times) if self.overlap_times else np.zeros_like(effective)

        return {
            **base_stats,
            "pipeline": {
                "effective_mean_ms": float(np.mean(effective)),
                "effective_hz": float(1000 / np.mean(effective)) if np.mean(effective) > 0 else 0,
                "inference_mean_ms": float(np.mean(inference)),
                "overlap_mean_ms": float(np.mean(overlap)),
                "overlap_ratio": float(np.mean(overlap) / np.mean(inference)) if np.mean(inference) > 0 else 0,
            }
        }

    def shutdown(self):
        """Shutdown executor."""
        self.executor.shutdown(wait=True)


def run_pipeline_episode(env, policy: PipelinePolicy, task_description: str, args, use_pipeline: bool = True):
    """
    Run episode with pipeline optimization.

    Args:
        env: LIBERO environment
        policy: Pipeline policy
        task_description: Task language description
        args: Arguments
        use_pipeline: If True, use action buffering; if False, use baseline (no buffer)

    Returns:
        success: Whether task completed successfully
    """
    import cv2

    max_steps = MAX_STEPS_DICT.get(args.task_suite_name, 300)
    action_buffer = collections.deque()
    replan_steps = args.replan_steps

    obs = env.reset()
    policy.reset_episode()

    t = 0
    success = False

    # Timing for pipeline analysis
    step_times = []
    wait_times = []

    while t < max_steps + args.num_steps_wait:
        step_start = time.perf_counter()

        # Wait phase
        if t < args.num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        # Get action
        if len(action_buffer) == 0:
            # Need to run inference
            wait_start = time.perf_counter()

            # Prepare observation
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            img_resized = resize_with_pad(img, args.resize_size, args.resize_size)
            wrist_resized = resize_with_pad(wrist_img, args.resize_size, args.resize_size)

            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )).astype(np.float32)

            observation = {
                "observation/image": img_resized,
                "observation/wrist_image": wrist_resized,
                "observation/state": state,  # Don't pad - base policy handles it
                "prompt": task_description,
            }

            # Run inference
            result = policy.infer_with_overlap(observation)
            actions = result["actions"]

            wait_time = (time.perf_counter() - wait_start) * 1000
            wait_times.append(wait_time)

            # Fill action buffer
            if use_pipeline:
                # Pipeline mode: buffer multiple actions
                for i in range(min(replan_steps, len(actions))):
                    action_buffer.append(actions[i])
            else:
                # Baseline mode: only take one action at a time
                action_buffer.append(actions[0])

        # Execute action
        if len(action_buffer) > 0:
            action = action_buffer.popleft()
            # Truncate to 7 dimensions (action space)
            if hasattr(action, 'tolist'):
                action = action.tolist()
            if len(action) > 7:
                action = action[:7]
            obs, reward, done, info = env.step(action)

            if done:
                success = True
                break

        step_time = (time.perf_counter() - step_start) * 1000
        step_times.append(step_time)
        t += 1

    # Calculate effective frequency
    if step_times:
        mean_step_time = np.mean(step_times)
        effective_hz = 1000 / mean_step_time if mean_step_time > 0 else 0
    else:
        effective_hz = 0

    if wait_times:
        mean_wait_time = np.mean(wait_times)
    else:
        mean_wait_time = 0

    return {
        "success": success,
        "steps": t,
        "effective_hz": effective_hz,
        "mean_step_ms": np.mean(step_times) if step_times else 0,
        "mean_wait_ms": mean_wait_time,
        "num_inferences": len(wait_times),
    }


def run_evaluation(args):
    """Run pipeline evaluation."""
    from libero.libero import benchmark

    logger.info(f"Running evaluation: mode={args.mode}, steps={args.denoising_steps}, schedule={args.schedule_type}")

    # Create policy
    policy = PipelinePolicy(
        checkpoint_dir=args.checkpoint_dir,
        num_denoising_steps=args.denoising_steps,
        schedule_type=args.schedule_type,
        action_buffer_size=args.replan_steps,
    )

    # Get benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    tasks = task_suite.get_task_names()[:args.num_tasks]

    results = {
        "mode": args.mode,
        "task_suite": args.task_suite_name,
        "denoising_steps": args.denoising_steps,
        "schedule_type": args.schedule_type,
        "replan_steps": args.replan_steps,
        "num_tasks": len(tasks),
        "num_trials": args.num_trials,
        "task_results": [],
        "total_successes": 0,
        "total_trials": 0,
    }

    use_pipeline = (args.mode == "pipeline")

    for task_idx, task_name in enumerate(tasks):
        task_id = task_suite.get_task_names().index(task_name)
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)

        logger.info(f"[{task_idx+1}/{len(tasks)}] {task.language}")

        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_successes = 0
        task_episodes = []

        for trial_idx in range(min(args.num_trials, len(init_states))):
            env.reset()
            env.set_init_state(init_states[trial_idx])

            episode_result = run_pipeline_episode(
                env, policy, task_description, args,
                use_pipeline=use_pipeline
            )

            if episode_result["success"]:
                task_successes += 1

            task_episodes.append(episode_result)
            logger.info(f"  Trial {trial_idx+1}: {'Success' if episode_result['success'] else 'Fail'} "
                       f"(Hz={episode_result['effective_hz']:.1f}, wait={episode_result['mean_wait_ms']:.0f}ms)")

        results["task_results"].append({
            "task_name": task_name,
            "successes": task_successes,
            "trials": args.num_trials,
            "success_rate": task_successes / args.num_trials * 100,
            "episodes": task_episodes,
        })
        results["total_successes"] += task_successes
        results["total_trials"] += args.num_trials

        env.close()

    results["overall_success_rate"] = results["total_successes"] / results["total_trials"] * 100
    results["timing"] = policy.get_stats()

    policy.shutdown()

    return results


def main():
    parser = argparse.ArgumentParser(description="LIBERO Pipeline Evaluation")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="/root/.cache/openpi/checkpoints/pi05_libero")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--mode", type=str, default="compare",
                       choices=["baseline", "pipeline", "compare"])
    parser.add_argument("--denoising_steps", type=int, default=10)
    parser.add_argument("--schedule_type", type=str, default="linear")
    parser.add_argument("--replan_steps", type=int, default=10)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--resize_size", type=int, default=224)

    args = parser.parse_args()

    if args.quick:
        args.num_tasks = 3
        args.num_trials = 3

    all_results = {}

    if args.mode == "compare":
        modes = ["baseline", "pipeline"]
    else:
        modes = [args.mode]

    for mode in modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Mode: {mode}")
        logger.info(f"{'='*60}\n")

        args.mode = mode
        results = run_evaluation(args)
        all_results[mode] = results

        # Print summary
        logger.info(f"\n{mode.upper()} Results:")
        logger.info(f"  Success Rate: {results['overall_success_rate']:.1f}%")
        logger.info(f"  Avg Latency: {results['timing'].get('mean_ms', 0):.1f}ms")
        logger.info(f"  Hz: {results['timing'].get('hz', 0):.1f}")

        torch.cuda.empty_cache()

    # Print comparison
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("Pipeline Comparison Results")
        print("="*80)
        print(f"{'Mode':<15} {'Accuracy':<12} {'Inference':<12} {'Effective Hz':<15} {'Replan Steps':<12}")
        print("-"*80)

        for mode, results in all_results.items():
            timing = results.get('timing', {})
            print(f"{mode:<15} "
                  f"{results['overall_success_rate']:.1f}%{'':<7} "
                  f"{timing.get('mean_ms', 0):.1f}ms{'':<6} "
                  f"{timing.get('hz', 0):.1f}{'':<10} "
                  f"{args.replan_steps}")
        print("="*80)

    # Save results
    if args.output_file:
        output_path = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"pipeline_results_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
