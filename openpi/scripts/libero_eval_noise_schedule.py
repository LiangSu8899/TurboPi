#!/usr/bin/env python3
"""
LIBERO Evaluation with Noise Schedule Comparison.

Compares different noise schedules (linear, cosine, quadratic, sigmoid)
for flow matching denoising while keeping 10 steps.

Usage:
    python scripts/libero_eval_noise_schedule.py --quick
    python scripts/libero_eval_noise_schedule.py --schedule cosine --quick
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
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
SEQ_LEN = 968

MAX_STEPS_DICT = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class NoiseScheduleDenoiseLoop:
    """Denoising loop with configurable noise schedule."""

    def __init__(self, model, num_steps: int = 10, schedule_type: str = "linear"):
        self.model = model
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.device = next(model.parameters()).device

        # Pre-compute timesteps for this schedule
        from openpi.models_pytorch.pi0_pytorch import get_noise_schedule
        self.timesteps = get_noise_schedule(num_steps, schedule_type, self.device)

        logger.info(f"NoiseScheduleDenoiseLoop: {num_steps} steps, schedule={schedule_type}")
        logger.info(f"  Timesteps: {[f'{t:.3f}' for t in self.timesteps.tolist()[:5]]}...{[f'{t:.3f}' for t in self.timesteps.tolist()[-3:]]}")

    def infer(self, observation, noise=None):
        """Run denoising with configured schedule."""
        return self.model.sample_actions(
            device=self.device,
            observation=observation,
            noise=noise,
            num_steps=self.num_steps,
            use_kv_cache=True,
            schedule_type=self.schedule_type,
        )


class NoiseSchedulePolicy:
    """Policy with noise schedule support."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        num_denoising_steps: int = 10,
        schedule_type: str = "linear",
    ):
        self.device = device
        self.num_denoising_steps = num_denoising_steps
        self.schedule_type = schedule_type
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser()

        self.latency_records = []

        self._load_model()
        self._load_tokenizer()
        self._load_norm_stats()

        logger.info(f"Policy initialized: {num_denoising_steps} steps, schedule={schedule_type}")

    def _load_model(self):
        """Load PyTorch model."""
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
        from safetensors.torch import load_file

        config_path = self.checkpoint_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
        else:
            model_config = {}

        pi0_config = Pi0Config(
            paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
            action_dim=model_config.get("action_dim", 32),
            action_horizon=model_config.get("action_horizon", 50),
            max_token_len=model_config.get("tokenizer_max_length", 200),
            pi05=True,
            dtype="bfloat16",
        )

        self.model = PI0Pytorch(pi0_config)

        weights_path = self.checkpoint_dir / "model.safetensors"
        state_dict = load_file(weights_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device=self.device)
        self.model.eval()

        self.action_dim = pi0_config.action_dim
        self.action_horizon = pi0_config.action_horizon
        self.max_token_len = pi0_config.max_token_len
        self.max_state_dim = 32

        logger.info(f"Loaded model from {self.checkpoint_dir}")

    def _load_tokenizer(self):
        """Load SentencePiece tokenizer."""
        import sentencepiece as spm

        tokenizer_paths = [
            self.checkpoint_dir / "tokenizer.model",
            pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model",
            pathlib.Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model"),
        ]

        for path in tokenizer_paths:
            if path.exists():
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.Load(str(path))
                logger.info(f"Loaded tokenizer from {path}")
                return

        raise FileNotFoundError("Tokenizer not found")

    def _load_norm_stats(self):
        """Load normalization statistics."""
        norm_stats_paths = [
            self.checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            self.checkpoint_dir / "norm_stats.json",
        ]

        for path in norm_stats_paths:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                self.norm_stats = data.get("norm_stats", data)
                logger.info(f"Loaded norm stats from {path}")
                return

        logger.warning("No norm_stats.json found")
        self.norm_stats = None

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return state

        stats = self.norm_stats.get("state", {})
        q01 = stats.get("q01")
        q99 = stats.get("q99")

        if q01 is not None and q99 is not None:
            q01 = np.array(q01, dtype=np.float32)[:len(state)]
            q99 = np.array(q99, dtype=np.float32)[:len(state)]
            return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return state

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return actions

        stats = self.norm_stats.get("actions", {})
        q01 = stats.get("q01")
        q99 = stats.get("q99")

        if q01 is not None and q99 is not None:
            q01 = np.array(q01, dtype=np.float32)
            q99 = np.array(q99, dtype=np.float32)
            action_dim = min(actions.shape[-1], len(q01))

            unnormalized = actions.copy()
            unnormalized[..., :action_dim] = (actions[..., :action_dim] + 1.0) / 2.0 * (q99[:action_dim] - q01[:action_dim] + 1e-6) + q01[:action_dim]
            return unnormalized
        return actions

    def _preprocess(self, observation: Dict[str, Any]):
        """Preprocess observation."""
        from openpi.models_pytorch.pi0_pytorch import Observation

        img = observation.get("observation/image")
        wrist_img = observation.get("observation/wrist_image")
        state = observation.get("observation/state")
        prompt = observation.get("prompt", "")

        # Convert images
        if img is not None:
            img_tensor = torch.from_numpy(img).float() / 127.5 - 1.0
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            img_tensor = torch.zeros(1, 3, 224, 224) - 1.0

        if wrist_img is not None:
            wrist_tensor = torch.from_numpy(wrist_img).float() / 127.5 - 1.0
            if wrist_tensor.ndim == 3:
                wrist_tensor = wrist_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            wrist_tensor = torch.zeros(1, 3, 224, 224) - 1.0

        images = {
            "base_0_rgb": img_tensor.to(self.device, dtype=torch.bfloat16),
            "left_wrist_0_rgb": wrist_tensor.to(self.device, dtype=torch.bfloat16),
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, device=self.device, dtype=torch.bfloat16) - 1.0,
        }
        image_masks = {
            "base_0_rgb": torch.ones(1, device=self.device, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, device=self.device, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(1, device=self.device, dtype=torch.bool),
        }

        # Process state
        if state is not None:
            state_np = np.asarray(state, dtype=np.float32)
            state_np = self._normalize_state(state_np)
            state_tensor = torch.from_numpy(state_np).float()
            if state_tensor.shape[-1] < self.max_state_dim:
                padding = torch.zeros(self.max_state_dim - state_tensor.shape[-1])
                state_tensor = torch.cat([state_tensor, padding])
            state_tensor = state_tensor.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        else:
            state_tensor = torch.zeros(1, self.max_state_dim, device=self.device, dtype=torch.bfloat16)

        # Tokenize prompt
        token_ids = self.tokenizer.Encode(prompt, add_bos=True)
        if len(token_ids) > self.max_token_len:
            token_ids = token_ids[:self.max_token_len]
        padding_len = self.max_token_len - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * padding_len
        token_ids = token_ids + [0] * padding_len

        tokenized_prompt = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        tokenized_prompt_mask = torch.tensor([attention_mask], dtype=torch.bool, device=self.device)

        return Observation(
            images=images,
            image_masks=image_masks,
            state=state_tensor,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=None,
            token_loss_mask=None,
        )

    @torch.no_grad()
    def infer(self, observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Run inference with configured schedule."""
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        obs = self._preprocess(observation)

        # Run with configured schedule
        actions = self.model.sample_actions(
            device=self.device,
            observation=obs,
            num_steps=self.num_denoising_steps,
            use_kv_cache=True,
            schedule_type=self.schedule_type,
        )

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.latency_records.append(latency_ms)

        # Post-process
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics."""
        if not self.latency_records:
            return {}

        latencies = np.array(self.latency_records)
        return {
            "count": len(latencies),
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "hz": float(1000 / np.mean(latencies)),
        }


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle."""
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    """Create LIBERO environment."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def resize_with_pad(img, target_h, target_w):
    """Resize image with padding."""
    try:
        from openpi_client import image_tools
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, target_h, target_w)
        )
    except ImportError:
        import cv2
        return cv2.resize(img, (target_w, target_h))


def run_episode(env, policy, task_description, args):
    """Run a single episode."""
    action_plan = collections.deque()
    max_steps = MAX_STEPS_DICT.get(args.task_suite_name, 300)

    t = 0
    obs = env.reset()

    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if len(action_plan) == 0:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            img = resize_with_pad(img, args.resize_size, args.resize_size)
            wrist_img = resize_with_pad(wrist_img, args.resize_size, args.resize_size)

            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )).astype(np.float32)

            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": str(task_description),
            }

            result = policy.infer(element)
            action_chunk = result["actions"]
            action_plan.extend(action_chunk[:args.replan_steps])

        action = action_plan.popleft()
        if hasattr(action, 'tolist'):
            action = action.tolist()
        if len(action) > 7:
            action = action[:7]

        obs, reward, done, info = env.step(action)

        if done:
            return True

        t += 1

    return False


def eval_libero(args):
    """Main evaluation function."""
    np.random.seed(args.seed)

    from libero.libero import benchmark
    import tqdm

    # Create policy with specified schedule
    logger.info(f"Creating Policy: {args.denoising_steps} steps, schedule={args.schedule}")
    policy = NoiseSchedulePolicy(
        checkpoint_dir=args.checkpoint_dir,
        num_denoising_steps=args.denoising_steps,
        schedule_type=args.schedule,
    )

    # Initialize LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    logger.info(f"Task suite: {args.task_suite_name}, {num_tasks_in_suite} tasks")

    # Determine task range
    if args.quick:
        task_start = 0
        task_end = min(3, num_tasks_in_suite)
        num_trials = 3
        logger.info(f"Quick mode: {task_end} tasks, {num_trials} trials each")
    else:
        task_start = args.task_start
        task_end = min(args.task_end, num_tasks_in_suite)
        num_trials = args.num_trials

    task_range = range(task_start, task_end)

    # Evaluation loop
    total_episodes = 0
    total_successes = 0
    task_results = []

    for task_id in tqdm.tqdm(task_range, desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes = 0
        task_successes = 0

        for episode_idx in tqdm.tqdm(range(num_trials), desc=f"Task {task_id}", leave=False):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])

            success = run_episode(env, policy, task_description, args)

            if success:
                task_successes += 1
                total_successes += 1

            task_episodes += 1
            total_episodes += 1

        task_result = {
            "task_id": task_id,
            "description": str(task_description),
            "successes": task_successes,
            "episodes": task_episodes,
            "success_rate": 100 * task_successes / task_episodes if task_episodes > 0 else 0,
        }
        task_results.append(task_result)

        print(f"\n>>> Task {task_id}: {task_successes}/{task_episodes} ({task_result['success_rate']:.1f}%)")
        env.close()

    # Get latency stats
    latency_stats = policy.get_latency_stats()

    # Final results
    print("\n" + "=" * 70)
    print(f"NOISE SCHEDULE EVALUATION: {args.schedule.upper()}")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Task Suite: {args.task_suite_name}")
    print(f"  Denoising Steps: {args.denoising_steps}")
    print(f"  Noise Schedule: {args.schedule}")
    print(f"  Tasks: {task_end - task_start}, Trials per task: {num_trials}")

    print(f"\nAccuracy:")
    print(f"  Total: {total_successes}/{total_episodes} ({100*total_successes/total_episodes:.1f}%)")

    print(f"\nLatency ({latency_stats.get('count', 0)} inferences):")
    print(f"  Mean: {latency_stats.get('mean_ms', 0):.2f} ms")
    print(f"  Std:  {latency_stats.get('std_ms', 0):.2f} ms")
    print(f"  P50:  {latency_stats.get('p50_ms', 0):.2f} ms")
    print(f"  P95:  {latency_stats.get('p95_ms', 0):.2f} ms")
    print(f"  Hz:   {latency_stats.get('hz', 0):.1f}")

    print("\n" + "=" * 70)

    # Save results
    if args.output_file:
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "task_suite": args.task_suite_name,
                "denoising_steps": args.denoising_steps,
                "schedule_type": args.schedule,
                "num_trials": num_trials,
                "seed": args.seed,
            },
            "accuracy": {
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "success_rate": 100 * total_successes / total_episodes,
            },
            "latency": latency_stats,
            "tasks": task_results,
        }
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")

    return {
        "schedule": args.schedule,
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate": 100 * total_successes / total_episodes,
        "latency": latency_stats,
    }


def compare_schedules(args):
    """Compare all noise schedules."""
    schedules = ["linear", "cosine", "quadratic", "sigmoid"]
    results = {}

    for schedule in schedules:
        print(f"\n{'='*70}")
        print(f"Testing schedule: {schedule.upper()}")
        print('='*70)

        args.schedule = schedule
        result = eval_libero(args)
        results[schedule] = result

    # Print comparison
    print("\n" + "=" * 70)
    print("NOISE SCHEDULE COMPARISON")
    print("=" * 70)
    print(f"\n{'Schedule':<12} | {'Accuracy':<12} | {'Mean Latency':<15} | {'Hz':<10}")
    print("-" * 55)

    for schedule in schedules:
        r = results[schedule]
        acc = f"{r['success_rate']:.1f}%"
        lat = f"{r['latency'].get('mean_ms', 0):.2f} ms"
        hz = f"{r['latency'].get('hz', 0):.1f}"
        print(f"{schedule:<12} | {acc:<12} | {lat:<15} | {hz:<10}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIBERO evaluation with Noise Schedule comparison")

    parser.add_argument("--task_suite_name", default="libero_spatial",
                       choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--task_start", type=int, default=0)
    parser.add_argument("--task_end", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=10)

    parser.add_argument("--checkpoint_dir",
                       default=os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero"))
    parser.add_argument("--denoising_steps", type=int, default=10)
    parser.add_argument("--schedule", type=str, default="linear",
                       choices=["linear", "cosine", "quadratic", "sigmoid", "all"])

    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 tasks, 3 trials")
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    if args.schedule == "all":
        compare_schedules(args)
    else:
        eval_libero(args)
