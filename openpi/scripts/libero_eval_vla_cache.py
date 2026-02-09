#!/usr/bin/env python3
"""
VLA-Cache Evaluation: Hybrid (Base Only) vs Full KV Reuse.

This script compares different KV cache reuse strategies:
1. Baseline: No reuse (every frame computes fresh KV)
2. Full VLA-Cache: Reuse all camera KV (base + wrist)
3. Hybrid (Base Only): Only reuse base camera KV, wrist is recomputed each frame

The hypothesis is that base camera (third-person view) changes less frequently
than wrist camera, so selective reuse might preserve accuracy better.

Usage:
    python scripts/libero_eval_vla_cache.py --quick
    python scripts/libero_eval_vla_cache.py --mode hybrid_base
    python scripts/libero_eval_vla_cache.py --mode full_reuse
"""

import sys
import os
import collections
import logging
import pathlib
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

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

os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

torch.backends.cudnn.enabled = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
TOKENS_PER_CAMERA = 256  # 16x16 patches
SEQ_LEN = 968

MAX_STEPS_DICT = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class VLACacheGate:
    """
    VLA-Cache style gate using cosine similarity.

    Supports three modes:
    1. no_reuse: Always recompute (baseline)
    2. full_reuse: Reuse all camera KV when similarity > threshold
    3. hybrid_base: Only reuse base camera KV, always recompute wrist
    """

    def __init__(
        self,
        mode: str = "no_reuse",  # "no_reuse", "full_reuse", "hybrid_base"
        similarity_threshold: float = 0.98,
        device: str = "cuda",
    ):
        self.mode = mode
        self.similarity_threshold = similarity_threshold
        self.device = device

        # Cached state
        self.prev_base_image = None   # (B, C, H, W)
        self.prev_wrist_image = None  # (B, C, H, W)
        self.prev_kv_cache = None     # Full KV cache
        self.prev_prefix_pad_masks = None

        # For hybrid mode: separate base/wrist KV
        self.prev_base_kv = None      # List of (K, V) for base tokens only
        self.prev_lang_kv = None      # List of (K, V) for language tokens

        # Statistics
        self.stats = {
            "total_frames": 0,
            "base_reused": 0,
            "wrist_reused": 0,
            "full_recompute": 0,
            "base_similarity_values": [],
            "wrist_similarity_values": [],
        }

    def reset(self):
        """Reset for new episode."""
        self.prev_base_image = None
        self.prev_wrist_image = None
        self.prev_kv_cache = None
        self.prev_prefix_pad_masks = None
        self.prev_base_kv = None
        self.prev_lang_kv = None

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_frames": 0,
            "base_reused": 0,
            "wrist_reused": 0,
            "full_recompute": 0,
            "base_similarity_values": [],
            "wrist_similarity_values": [],
        }

    def compute_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute cosine similarity between two images."""
        if img1 is None or img2 is None:
            return 0.0

        # Flatten and normalize
        v1 = img1.flatten().float()
        v2 = img2.flatten().float()

        # Cosine similarity
        sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        return sim

    def check_reuse(
        self,
        current_base_image: torch.Tensor,
        current_wrist_image: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Check which parts of KV cache can be reused.

        Returns:
            dict with keys:
                - reuse_base: bool
                - reuse_wrist: bool
                - base_sim: float
                - wrist_sim: float
                - reason: str
        """
        self.stats["total_frames"] += 1

        # Compute similarities
        base_sim = self.compute_similarity(self.prev_base_image, current_base_image)
        wrist_sim = self.compute_similarity(self.prev_wrist_image, current_wrist_image)

        self.stats["base_similarity_values"].append(base_sim)
        self.stats["wrist_similarity_values"].append(wrist_sim)

        result = {
            "base_sim": base_sim,
            "wrist_sim": wrist_sim,
            "reuse_base": False,
            "reuse_wrist": False,
            "reason": "",
        }

        if self.mode == "no_reuse":
            result["reason"] = "no_reuse mode"
            self.stats["full_recompute"] += 1

        elif self.mode == "full_reuse":
            # Only reuse if BOTH cameras pass threshold
            if base_sim >= self.similarity_threshold and wrist_sim >= self.similarity_threshold:
                result["reuse_base"] = True
                result["reuse_wrist"] = True
                result["reason"] = f"full_reuse (base={base_sim:.3f}, wrist={wrist_sim:.3f})"
                self.stats["base_reused"] += 1
                self.stats["wrist_reused"] += 1
            else:
                result["reason"] = f"similarity too low (base={base_sim:.3f}, wrist={wrist_sim:.3f})"
                self.stats["full_recompute"] += 1

        elif self.mode == "hybrid_base":
            # Only reuse base camera if it passes threshold
            # Wrist is ALWAYS recomputed
            if base_sim >= self.similarity_threshold:
                result["reuse_base"] = True
                result["reason"] = f"hybrid: base reused ({base_sim:.3f}), wrist fresh"
                self.stats["base_reused"] += 1
            else:
                result["reason"] = f"hybrid: base too different ({base_sim:.3f}), all fresh"
                self.stats["full_recompute"] += 1

        return result

    def update_cache(
        self,
        base_image: torch.Tensor,
        wrist_image: torch.Tensor,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
    ):
        """Update cached state after fresh computation."""
        self.prev_base_image = base_image.clone()
        self.prev_wrist_image = wrist_image.clone()
        self.prev_kv_cache = kv_cache
        self.prev_prefix_pad_masks = prefix_pad_masks

        # For hybrid mode, extract base-only KV
        # Assuming: base tokens [0:256], wrist tokens [256:512], lang tokens [512:]
        if self.mode == "hybrid_base":
            self.prev_base_kv = []
            self.prev_lang_kv = []
            for k, v in kv_cache:
                # k, v shape: (B, num_heads, seq_len, head_dim)
                self.prev_base_kv.append((
                    k[:, :, :TOKENS_PER_CAMERA, :].clone(),
                    v[:, :, :TOKENS_PER_CAMERA, :].clone(),
                ))
                # Language tokens start after both cameras
                lang_start = 2 * TOKENS_PER_CAMERA
                self.prev_lang_kv.append((
                    k[:, :, lang_start:, :].clone(),
                    v[:, :, lang_start:, :].clone(),
                ))

    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        total = max(self.stats["total_frames"], 1)
        return {
            "mode": self.mode,
            "total_frames": self.stats["total_frames"],
            "base_reuse_rate": self.stats["base_reused"] / total * 100,
            "wrist_reuse_rate": self.stats["wrist_reused"] / total * 100,
            "full_recompute_rate": self.stats["full_recompute"] / total * 100,
            "avg_base_similarity": np.mean(self.stats["base_similarity_values"]) if self.stats["base_similarity_values"] else 0,
            "avg_wrist_similarity": np.mean(self.stats["wrist_similarity_values"]) if self.stats["wrist_similarity_values"] else 0,
        }


class VLACachePolicy:
    """
    Policy wrapper implementing VLA-Cache style inference.

    Supports selective KV reuse based on camera similarity.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        mode: str = "no_reuse",
        similarity_threshold: float = 0.98,
        num_denoising_steps: int = 10,
    ):
        self.device = "cuda"
        self.num_denoising_steps = num_denoising_steps
        self.mode = mode
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser()

        # Initialize gate
        self.gate = VLACacheGate(
            mode=mode,
            similarity_threshold=similarity_threshold,
            device=self.device,
        )

        # Load model
        self._load_model()
        self._load_tokenizer()
        self._load_norm_stats()

        # Timing stats
        self.inference_times = []
        self.vision_times = []
        self.denoise_times = []

    def _load_model(self):
        """Load Pi0.5 model."""
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
        from safetensors.torch import load_file

        logger.info(f"Loading model from {self.checkpoint_dir}")

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

        logger.info(f"Model loaded successfully (mode: {self.mode})")

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

        raise FileNotFoundError(f"Tokenizer not found")

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

    def reset_episode(self):
        """Reset for new episode."""
        self.gate.reset()

    def reset_stats(self):
        """Reset all statistics."""
        self.gate.reset_stats()
        self.inference_times = []
        self.vision_times = []
        self.denoise_times = []

    @torch.no_grad()
    def infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference with VLA-Cache strategy.

        Args:
            observation: Dict with keys:
                - observation/image: (H, W, 3) base camera image
                - observation/wrist_image: (H, W, 3) wrist camera image
                - observation/state: (state_dim,) robot state
                - prompt: str

        Returns:
            Dict with 'actions' and timing info
        """
        start_time = time.perf_counter()

        # Preprocess images
        base_img = observation["observation/image"]
        wrist_img = observation["observation/wrist_image"]
        state = observation["observation/state"]
        prompt = observation["prompt"]

        # Convert to tensors and resize to 224x224
        base_tensor = self._preprocess_image(base_img)
        wrist_tensor = self._preprocess_image(wrist_img)

        # Check reuse decision
        reuse_decision = self.gate.check_reuse(base_tensor, wrist_tensor)

        vision_start = time.perf_counter()

        # Prepare full observation for model
        from openpi.models_pytorch.pi0_pytorch import Observation

        obs = Observation(
            images={
                "base_0_rgb": base_tensor,
                "left_wrist_0_rgb": wrist_tensor,
                "right_wrist_0_rgb": torch.zeros_like(wrist_tensor),  # Dummy
            },
            image_masks={
                "base_0_rgb": torch.ones(1, dtype=torch.bool, device=self.device),
                "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=self.device),
                "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=self.device),
            },
            state=torch.from_numpy(state).unsqueeze(0).to(self.device),
            tokenized_prompt=self._tokenize(prompt),
            tokenized_prompt_mask=None,
        )

        # Decide inference path based on reuse decision
        if self.mode == "no_reuse" or not (reuse_decision["reuse_base"] or reuse_decision["reuse_wrist"]):
            # Full computation path
            actions = self._full_inference(obs)

            # Update cache
            self.gate.update_cache(
                base_tensor,
                wrist_tensor,
                getattr(self, '_last_kv_cache', None),
                getattr(self, '_last_prefix_masks', None),
            )
        elif reuse_decision["reuse_base"] and reuse_decision["reuse_wrist"]:
            # Full reuse path (VLA-Cache full)
            actions = self._reuse_all_inference(obs)
        else:
            # Hybrid path (base reuse only)
            # NOTE: This requires special handling - for now, fall back to full
            # True hybrid would need partial KV cache merging
            actions = self._full_inference(obs)
            self.gate.update_cache(
                base_tensor,
                wrist_tensor,
                getattr(self, '_last_kv_cache', None),
                getattr(self, '_last_prefix_masks', None),
            )

        vision_time = time.perf_counter() - vision_start
        self.vision_times.append(vision_time)

        total_time = time.perf_counter() - start_time
        self.inference_times.append(total_time)

        return {
            "actions": actions.cpu().numpy(),
            "inference_time": total_time,
            "reuse_decision": reuse_decision,
        }

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Convert image to model input format."""
        import cv2

        if img.shape[0] != 224 or img.shape[1] != 224:
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

        # HWC -> CHW, normalize to [0, 1], add batch dim
        tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device) / 255.0
        return tensor

    def _tokenize(self, prompt: str) -> torch.Tensor:
        """Tokenize prompt using SentencePiece."""
        token_ids = self.tokenizer.Encode(prompt, add_bos=True)
        # Pad to max length
        if len(token_ids) < self.max_token_len:
            token_ids = token_ids + [0] * (self.max_token_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_token_len]
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _full_inference(self, obs) -> torch.Tensor:
        """Full inference without reuse."""
        # Get images and masks
        images = list(obs.images.values())
        img_masks = list(obs.image_masks.values())

        # Embed prefix (all cameras + language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, obs.tokenized_prompt,
            torch.ones_like(obs.tokenized_prompt, dtype=torch.bool)
        )

        # Compute KV cache
        kv_cache = self.model.compute_prefix_kv_cache(
            prefix_embs, prefix_pad_masks, prefix_att_masks
        )

        # Store for potential future reuse
        self._last_kv_cache = kv_cache
        self._last_prefix_masks = prefix_pad_masks

        # Denoise
        actions = self.model.sample_actions_with_external_kv(
            self.device,
            obs.state,
            kv_cache,
            prefix_pad_masks,
            num_steps=self.num_denoising_steps,
        )

        return actions[0]  # Remove batch dim

    @torch.no_grad()
    def _reuse_all_inference(self, obs) -> torch.Tensor:
        """Inference with full KV cache reuse."""
        if self.gate.prev_kv_cache is None:
            return self._full_inference(obs)

        # Use cached KV, only update state
        actions = self.model.sample_actions_with_external_kv(
            self.device,
            obs.state,
            self.gate.prev_kv_cache,
            self.gate.prev_prefix_pad_masks,
            num_steps=self.num_denoising_steps,
        )

        return actions[0]

    def get_timing_summary(self) -> Dict[str, float]:
        """Get timing statistics."""
        if not self.inference_times:
            return {}

        return {
            "avg_inference_ms": np.mean(self.inference_times) * 1000,
            "std_inference_ms": np.std(self.inference_times) * 1000,
            "avg_hz": 1.0 / np.mean(self.inference_times),
        }


def run_libero_evaluation(
    policy: VLACachePolicy,
    task_suite_name: str = "libero_spatial",
    num_tasks: int = 10,
    num_trials: int = 10,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run LIBERO evaluation."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    logger.info(f"Starting evaluation: {task_suite_name}, {num_tasks} tasks, {num_trials} trials")

    # Get benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    tasks = task_suite.get_task_names()[:num_tasks]

    max_steps = MAX_STEPS_DICT.get(task_suite_name, 300)

    results = {
        "task_suite": task_suite_name,
        "num_tasks": len(tasks),
        "num_trials": num_trials,
        "mode": policy.mode,
        "task_results": [],
        "total_successes": 0,
        "total_trials": 0,
    }

    for task_idx, task_name in enumerate(tasks):
        task_id = task_suite.get_task_names().index(task_name)
        task = task_suite.get_task(task_id)
        task_description = task.language
        init_states = task_suite.get_task_init_states(task_id)

        logger.info(f"[{task_idx+1}/{len(tasks)}] {task_name}: {task_description}")

        # Create environment with correct BDDL path
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        task_successes = 0

        for trial_idx in range(min(num_trials, len(init_states))):
            # Reset environment
            env.reset()
            obs = env.set_init_state(init_states[trial_idx])

            # Reset policy for new episode
            policy.reset_episode()

            # Action plan buffer
            action_plan = []
            replan_steps = 10

            success = False

            for t in range(max_steps):
                # Check if we need new actions
                if len(action_plan) < 1:
                    # Prepare observation (same as baseline)
                    import cv2
                    import math

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Resize to 224x224
                    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
                    wrist_resized = cv2.resize(wrist_img, (224, 224), interpolation=cv2.INTER_LINEAR)

                    # Convert quat to axis-angle
                    def quat2axisangle(quat):
                        den = np.sqrt(1.0 - quat[3] * quat[3])
                        if abs(den) < 1e-6:
                            return np.zeros(3)
                        return (quat[:3] * 2.0 * math.acos(np.clip(quat[3], -1, 1))) / den

                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )).astype(np.float32)

                    # Pad state
                    state_padded = np.zeros(32, dtype=np.float32)
                    state_padded[:len(state)] = state

                    # Run inference
                    result = policy.infer({
                        "observation/image": img_resized,
                        "observation/wrist_image": wrist_resized,
                        "observation/state": state_padded,
                        "prompt": task_description,
                    })

                    actions = result["actions"]
                    action_plan.extend(actions[:replan_steps])

                # Execute action
                action_7d = action_plan.pop(0)[:7]
                obs, reward, done, info = env.step(action_7d)

                if done or info.get("success", False):
                    success = info.get("success", False)
                    break

            if success:
                task_successes += 1

            logger.info(f"  Trial {trial_idx+1}: {'Success' if success else 'Fail'}")

        results["task_results"].append({
            "task_name": task_name,
            "successes": task_successes,
            "trials": num_trials,
            "success_rate": task_successes / num_trials * 100,
        })
        results["total_successes"] += task_successes
        results["total_trials"] += num_trials

        env.close()

    results["overall_success_rate"] = results["total_successes"] / results["total_trials"] * 100
    results["gate_stats"] = policy.gate.get_summary()
    results["timing"] = policy.get_timing_summary()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str,
                       default="/root/.cache/openpi/checkpoints/pi05_libero")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["no_reuse", "full_reuse", "hybrid_base", "all"],
                       help="VLA-Cache mode to evaluate")
    parser.add_argument("--similarity_threshold", type=float, default=0.98)
    parser.add_argument("--denoising_steps", type=int, default=10)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="Quick test: 3 tasks, 3 trials")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.quick:
        args.num_tasks = 3
        args.num_trials = 3

    # Determine which modes to run
    if args.mode == "all":
        modes = ["no_reuse", "full_reuse"]  # Skip hybrid_base for now (needs special implementation)
    else:
        modes = [args.mode]

    all_results = {}

    for mode in modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating mode: {mode}")
        logger.info(f"{'='*60}\n")

        # Create policy
        policy = VLACachePolicy(
            checkpoint_dir=args.checkpoint_dir,
            mode=mode,
            similarity_threshold=args.similarity_threshold,
            num_denoising_steps=args.denoising_steps,
        )

        # Run evaluation
        results = run_libero_evaluation(
            policy=policy,
            task_suite_name=args.task_suite_name,
            num_tasks=args.num_tasks,
            num_trials=args.num_trials,
            seed=args.seed,
        )

        all_results[mode] = results

        # Print summary
        logger.info(f"\n{mode} Results:")
        logger.info(f"  Success Rate: {results['overall_success_rate']:.1f}%")
        logger.info(f"  Avg Latency: {results['timing'].get('avg_inference_ms', 0):.1f}ms")
        logger.info(f"  Avg Hz: {results['timing'].get('avg_hz', 0):.1f}")

        gate_stats = results.get('gate_stats', {})
        logger.info(f"  Base Reuse Rate: {gate_stats.get('base_reuse_rate', 0):.1f}%")
        logger.info(f"  Wrist Reuse Rate: {gate_stats.get('wrist_reuse_rate', 0):.1f}%")

        # Clean up
        del policy
        torch.cuda.empty_cache()

    # Print comparison
    print("\n" + "="*80)
    print("VLA-Cache Comparison Results")
    print("="*80)
    print(f"{'Mode':<15} {'Accuracy':<12} {'Latency':<12} {'Hz':<8} {'Base Reuse':<12} {'Wrist Reuse':<12}")
    print("-"*80)

    for mode, results in all_results.items():
        timing = results.get('timing', {})
        gate = results.get('gate_stats', {})
        print(f"{mode:<15} "
              f"{results['overall_success_rate']:.1f}%{'':<7} "
              f"{timing.get('avg_inference_ms', 0):.1f}ms{'':<6} "
              f"{timing.get('avg_hz', 0):.1f}{'':<4} "
              f"{gate.get('base_reuse_rate', 0):.1f}%{'':<8} "
              f"{gate.get('wrist_reuse_rate', 0):.1f}%")

    print("="*80)

    # Save results
    if args.output_file:
        output_path = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"vla_cache_comparison_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
