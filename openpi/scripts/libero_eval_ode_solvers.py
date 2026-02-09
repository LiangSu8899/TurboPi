#!/usr/bin/env python3
"""
LIBERO Evaluation for ODE Solver Comparison.

Tests different ODE solvers on LIBERO tasks using pure PyTorch path.
This script compares Euler vs higher-order methods on actual task success rate.

Usage:
    # Quick test with different solvers
    python scripts/libero_eval_ode_solvers.py --solver euler --quick
    python scripts/libero_eval_ode_solvers.py --solver dpm_solver_2 --quick

    # Compare all solvers
    python scripts/libero_eval_ode_solvers.py --solver all --quick
"""

import sys
import os
import logging
import argparse
import json
import time
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import cv2

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEPS_DICT = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
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


def resize_with_pad(img, target_h, target_w):
    """Resize image with padding to preserve aspect ratio."""
    try:
        from openpi_client import image_tools
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, target_h, target_w)
        )
    except ImportError:
        return cv2.resize(img, (target_w, target_h))


class LiberoPolicy:
    """Pure PyTorch policy for LIBERO evaluation with ODE solver support."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda",
                 num_denoising_steps: int = 10, solver_type: str = "euler"):
        self.device = device
        self.num_denoising_steps = num_denoising_steps
        self.solver_type = solver_type

        # Load model
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
        from safetensors.torch import load_file

        self.checkpoint_dir = Path(checkpoint_dir)
        config_path = self.checkpoint_dir / "config.json"

        with open(config_path) as f:
            model_config = json.load(f)

        self.pi0_config = Pi0Config(
            paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
            action_dim=model_config.get("action_dim", 32),
            action_horizon=model_config.get("action_horizon", 50),
            max_token_len=model_config.get("tokenizer_max_length", 200),
            pi05=True,
            dtype="bfloat16",
        )

        self.model = PI0Pytorch(self.pi0_config)
        weights_path = self.checkpoint_dir / "model.safetensors"
        state_dict = load_file(weights_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device=device)
        self.model.eval()

        # Tokenizer (SentencePiece)
        import sentencepiece as spm
        tokenizer_paths = [
            self.checkpoint_dir / "tokenizer.model",
            Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model",
            Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model"),
        ]
        self.tokenizer = None
        for path in tokenizer_paths:
            if path.exists():
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.Load(str(path))
                logger.info(f"Loaded tokenizer from {path}")
                break
        if self.tokenizer is None:
            raise RuntimeError("Could not find tokenizer.model")

        # Image processor
        from transformers import SiglipImageProcessor
        self.image_processor = SiglipImageProcessor(
            size={"height": 224, "width": 224},
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )

        # Load normalization statistics
        self._load_norm_stats()

        # Latency tracking
        self.latencies = []

        logger.info(f"Policy initialized with solver_type={solver_type}")

    def _load_norm_stats(self):
        """Load normalization statistics for action unnormalization."""
        norm_stats_paths = [
            self.checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            self.checkpoint_dir / "norm_stats.json",
            Path.home() / ".cache/openpi/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json",
        ]

        for path in norm_stats_paths:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                self.norm_stats = data.get("norm_stats", data)
                logger.info(f"Loaded norm stats from {path}")
                return

        logger.warning("No norm_stats.json found - actions will not be unnormalized")
        self.norm_stats = None

    def _unnormalize_actions(self, actions):
        """Unnormalize actions from [-1, 1] to original scale."""
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

    def process_observation(self, obs, instruction):
        """Convert LIBERO observation to model input."""
        from openpi.models_pytorch.pi0_pytorch import Observation

        def process_image(img):
            processed = self.image_processor(images=img, return_tensors="pt")
            return processed["pixel_values"].squeeze(0).to(self.device, dtype=torch.bfloat16)

        images = {}
        image_masks = {}

        # Process available images (LIBERO uses different key names and images need flipping + resizing)
        if "agentview_image" in obs:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            img = resize_with_pad(img, 224, 224)
            images["base_0_rgb"] = process_image(img)
            image_masks["base_0_rgb"] = torch.ones(1, device=self.device, dtype=torch.bool)

        if "robot0_eye_in_hand_image" in obs:
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            wrist_img = resize_with_pad(wrist_img, 224, 224)
            images["left_wrist_0_rgb"] = process_image(wrist_img)
            image_masks["left_wrist_0_rgb"] = torch.ones(1, device=self.device, dtype=torch.bool)

        # Pad to 3 images
        dummy_image = torch.zeros(3, 224, 224, device=self.device, dtype=torch.bfloat16) - 1.0
        for name in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            if name not in images:
                images[name] = dummy_image
                image_masks[name] = torch.zeros(1, device=self.device, dtype=torch.bool)

        # Stack images
        stacked_images = torch.stack([images[k] for k in sorted(images.keys())], dim=0).unsqueeze(0)
        stacked_masks = torch.stack([image_masks[k] for k in sorted(image_masks.keys())], dim=1)

        # Process state (use axis-angle instead of quaternion for rotation)
        state = obs.get("robot0_eef_pos", np.zeros(3)).tolist()
        quat = obs.get("robot0_eef_quat", np.zeros(4))
        state += _quat2axisangle(quat).tolist()  # Convert quaternion to axis-angle
        state += obs.get("robot0_gripper_qpos", np.zeros(2)).tolist()
        state = state[:self.pi0_config.max_state_dim]
        state += [0.0] * (self.pi0_config.max_state_dim - len(state))
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.bfloat16).unsqueeze(0)

        # Tokenize instruction (SentencePiece)
        token_ids = self.tokenizer.Encode(instruction, add_bos=True)
        max_len = self.pi0_config.max_token_len
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        pad_len = max_len - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        token_ids = token_ids + [0] * pad_len

        tokenized_prompt = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        tokenized_prompt_mask = torch.tensor([attention_mask], device=self.device, dtype=torch.bool)

        return Observation(
            images={"base_0_rgb": stacked_images[:, 0],
                   "left_wrist_0_rgb": stacked_images[:, 1],
                   "right_wrist_0_rgb": stacked_images[:, 2]},
            image_masks={"base_0_rgb": stacked_masks[:, 0],
                        "left_wrist_0_rgb": stacked_masks[:, 1],
                        "right_wrist_0_rgb": stacked_masks[:, 2]},
            state=state_tensor,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

    def get_action(self, obs, instruction):
        """Get action from model."""
        observation = self.process_observation(obs, instruction)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            actions = self.model.sample_actions(
                self.device, observation,
                num_steps=self.num_denoising_steps,
                solver_type=self.solver_type,
            )

        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        self.latencies.append(latency)

        # Return first action (unnormalized)
        action = actions[0, 0].float().cpu().numpy()
        action = self._unnormalize_actions(action)
        return action


def run_libero_evaluation(
    policy,
    task_suite_name: str,
    task_start: int = 0,
    task_end: int = 3,
    num_trials: int = 3,
    seed: int = 7,
):
    """Run LIBERO evaluation."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from tqdm import tqdm

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    max_steps = MAX_STEPS_DICT.get(task_suite_name, 300)

    task_ids = list(range(task_start, min(task_end, task_suite.n_tasks)))

    results = []
    total_success = 0
    total_episodes = 0

    for task_id in tqdm(task_ids, desc="Tasks"):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        # Use correct path for bddl file
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        task_successes = 0

        # Get initial states for deterministic evaluation
        initial_states = task_suite.get_task_init_states(task_id)
        num_steps_wait = 10  # Warmup steps

        for trial in tqdm(range(num_trials), desc=f"Task {task_id}", leave=False):
            # Use initial states for deterministic evaluation
            env.reset()
            obs = env.set_init_state(initial_states[trial % len(initial_states)])

            success = False
            step = 0
            while step < max_steps + num_steps_wait:
                # Warmup period with dummy actions
                if step < num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    step += 1
                    continue

                action = policy.get_action(obs, task_description)

                # Pad action if needed
                if len(action) < 7:
                    action = np.concatenate([action, np.zeros(7 - len(action))])
                action = action[:7]

                obs, reward, done, info = env.step(action)

                if done:
                    success = True
                    break

                step += 1

            if success:
                task_successes += 1
                total_success += 1
            total_episodes += 1

        env.close()

        results.append({
            "task_id": task_id,
            "description": task_description,
            "successes": task_successes,
            "episodes": num_trials,
            "success_rate": task_successes / num_trials * 100,
        })

        print(f"\n>>> Task {task_id}: {task_successes}/{num_trials} ({task_successes/num_trials*100:.1f}%)")

    return {
        "total_success": total_success,
        "total_episodes": total_episodes,
        "success_rate": total_success / total_episodes * 100,
        "tasks": results,
        "latencies": policy.latencies,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default=str(Path.home() / ".cache/openpi/checkpoints/pi05_libero"))
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_start", type=int, default=0)
    parser.add_argument("--task_end", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--denoising_steps", type=int, default=10)
    parser.add_argument("--solver", type=str, default="euler",
                       choices=["euler", "midpoint", "heun", "dpm_solver_2", "rk4", "all"])
    parser.add_argument("--quick", action="store_true", help="Quick test (3 tasks, 3 trials)")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.quick:
        args.task_end = 3
        args.num_trials = 3

    solvers = ["euler", "midpoint", "heun", "dpm_solver_2"] if args.solver == "all" else [args.solver]

    all_results = {}

    for solver in solvers:
        print("\n" + "=" * 70)
        print(f"TESTING SOLVER: {solver}")
        print("=" * 70)

        policy = LiberoPolicy(
            args.checkpoint,
            num_denoising_steps=args.denoising_steps,
            solver_type=solver,
        )

        result = run_libero_evaluation(
            policy,
            args.task_suite_name,
            args.task_start,
            args.task_end,
            args.num_trials,
            args.seed,
        )

        result["solver"] = solver
        result["mean_latency_ms"] = np.mean(result["latencies"])
        result["std_latency_ms"] = np.std(result["latencies"])

        all_results[solver] = result

        print(f"\n{solver} Results:")
        print(f"  Success Rate: {result['success_rate']:.1f}%")
        print(f"  Latency: {result['mean_latency_ms']:.2f} ± {result['std_latency_ms']:.2f} ms")

    # Summary
    if len(solvers) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n{'Solver':<15} | {'Success Rate':<15} | {'Latency (ms)':<15}")
        print("-" * 50)

        for solver, result in all_results.items():
            print(f"{solver:<15} | {result['success_rate']:.1f}%           | {result['mean_latency_ms']:.1f}")

    # Save results
    output_file = args.output_file or f"ode_solver_libero_{args.solver}.json"
    with open(output_file, "w") as f:
        save_data = {
            solver: {
                "success_rate": r["success_rate"],
                "total_success": r["total_success"],
                "total_episodes": r["total_episodes"],
                "mean_latency_ms": r["mean_latency_ms"],
                "std_latency_ms": r["std_latency_ms"],
                "tasks": r["tasks"],
            }
            for solver, r in all_results.items()
        }
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
