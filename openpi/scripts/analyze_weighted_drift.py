#!/usr/bin/env python3
"""
Step 2: Analyze Weighted Drift Distribution

This script analyzes the distribution of different drift metrics to find optimal thresholds
for the Attention-Weighted Drift gate.

Compares:
1. State Drift - Joint angle changes (fastest check)
2. Raw Pixel Diff - Unweighted pixel changes
3. Weighted Drift - Attention-weighted pixel changes

Key verification points:
1. State Drift should spike during key actions (grasp, place)
2. Weighted Drift should be more discriminative than Raw Diff
3. Hand-eye camera edge noise should be filtered

Usage:
    python scripts/analyze_weighted_drift.py --quick
    python scripts/analyze_weighted_drift.py --num_episodes 3 --output_dir drift_analysis
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
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Constants
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


class ProAttentionGateAnalyzer:
    """
    Analyzer for Pro Attention Gate metrics.
    Computes State Drift, Raw Diff, and Weighted Drift for each frame.
    """

    def __init__(self, drift_threshold=0.5, state_threshold=0.05, dilation_kernel=5):
        self.drift_threshold = drift_threshold
        self.state_threshold = state_threshold
        self.dilation_kernel = dilation_kernel

        self.prev_image_small = None   # (B, C, 28, 28)
        self.prev_attention = None     # (B, num_cameras, 16, 16)
        self.prev_state = None         # (B, state_dim)

    def reset(self):
        """Reset state for new episode."""
        self.prev_image_small = None
        self.prev_attention = None
        self.prev_state = None

    def compute_metrics(
        self,
        current_images: Dict[str, torch.Tensor],  # {name: (B, C, H, W)}
        current_state: torch.Tensor,  # (B, state_dim)
        current_attention: torch.Tensor,  # (B, num_cameras, 16, 16)
    ) -> Dict[str, float]:
        """
        Compute all drift metrics for current frame.

        Returns:
            Dict with keys:
            - state_drift: Max absolute joint change
            - raw_diff_base: Raw pixel diff for base camera
            - raw_diff_wrist: Raw pixel diff for wrist camera
            - weighted_drift_base: Attention-weighted drift for base camera
            - weighted_drift_wrist: Attention-weighted drift for wrist camera
            - total_weighted_drift: Sum of weighted drifts
        """
        # Stack images for processing
        camera_names = list(current_images.keys())
        images_stacked = torch.stack([current_images[name] for name in camera_names], dim=1)
        # images_stacked: (B, num_cameras, C, H, W)

        B, num_cameras, C, H, W = images_stacked.shape

        # Downsample to 28x28
        images_flat = images_stacked.view(B * num_cameras, C, H, W)
        images_small = F.avg_pool2d(images_flat, kernel_size=8, stride=8)  # (B*num_cameras, C, 28, 28)
        images_small = images_small.view(B, num_cameras, C, 28, 28)

        metrics = {}

        # ========== State Drift ==========
        if self.prev_state is not None:
            state_diff = torch.abs(current_state - self.prev_state)
            metrics["state_drift"] = state_diff.max().item()
        else:
            metrics["state_drift"] = 0.0

        # ========== First frame - no diff available ==========
        if self.prev_image_small is None:
            metrics["raw_diff_base"] = 0.0
            metrics["raw_diff_wrist"] = 0.0
            metrics["weighted_drift_base"] = 0.0
            metrics["weighted_drift_wrist"] = 0.0
            metrics["total_weighted_drift"] = 0.0

            # Update state for next frame
            self.prev_image_small = images_small.clone()
            self.prev_attention = current_attention.clone()
            self.prev_state = current_state.clone()

            return metrics

        # ========== Compute Pixel Diffs ==========
        pixel_diff = torch.abs(images_small - self.prev_image_small).mean(dim=2)  # (B, num_cameras, 28, 28)

        # Raw diff per camera (sum of pixel diffs)
        raw_diffs = pixel_diff.sum(dim=(2, 3))  # (B, num_cameras)
        metrics["raw_diff_base"] = raw_diffs[0, 0].item() if num_cameras > 0 else 0.0
        metrics["raw_diff_wrist"] = raw_diffs[0, 1].item() if num_cameras > 1 else 0.0

        # ========== Attention-Weighted Drift ==========
        if self.prev_attention is not None:
            # Resize attention to 28x28
            attn_resized = F.interpolate(
                self.prev_attention.view(B * num_cameras, 1, 16, 16),
                size=(28, 28),
                mode='bilinear',
                align_corners=False
            ).view(B, num_cameras, 28, 28)

            # Dilate attention
            attn_dilated = self._dilate_attention(attn_resized)

            # Weighted drift
            weighted_diff = (pixel_diff * attn_dilated).sum(dim=(2, 3))  # (B, num_cameras)

            metrics["weighted_drift_base"] = weighted_diff[0, 0].item() if num_cameras > 0 else 0.0
            metrics["weighted_drift_wrist"] = weighted_diff[0, 1].item() if num_cameras > 1 else 0.0
            metrics["total_weighted_drift"] = weighted_diff.sum().item()
        else:
            metrics["weighted_drift_base"] = 0.0
            metrics["weighted_drift_wrist"] = 0.0
            metrics["total_weighted_drift"] = 0.0

        # Update state for next frame
        self.prev_image_small = images_small.clone()
        self.prev_attention = current_attention.clone()
        self.prev_state = current_state.clone()

        return metrics

    def _dilate_attention(self, attention_map: torch.Tensor) -> torch.Tensor:
        """Dilate attention map using max pooling."""
        B, num_cameras, H, W = attention_map.shape
        attn_flat = attention_map.view(B * num_cameras, 1, H, W)

        padding = self.dilation_kernel // 2
        dilated = F.max_pool2d(
            attn_flat,
            kernel_size=self.dilation_kernel,
            stride=1,
            padding=padding
        )

        return dilated.view(B, num_cameras, H, W)


# Import visualization policy from Step 1
from visualize_attention_importance import (
    AttentionVisualizationPolicy,
    extract_patch_importance,
    _quat2axisangle,
    _get_libero_env,
    resize_with_pad,
)


def collect_trajectory_data(
    policy: AttentionVisualizationPolicy,
    env,
    task_description: str,
    max_steps: int = 200,
    replan_steps: int = 5,
    resize_size: int = 224,
) -> List[Dict]:
    """
    Collect trajectory data including images, states, actions, and attention maps.

    Returns:
        List of frame data dicts
    """
    analyzer = ProAttentionGateAnalyzer()
    trajectory_data = []

    action_plan = collections.deque()
    t = 0
    frame_idx = 0
    obs = env.reset()

    while t < max_steps + 10:
        if t < 10:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if len(action_plan) == 0:
            # Get images
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            img_resized = resize_with_pad(img, resize_size, resize_size)
            wrist_resized = resize_with_pad(wrist_img, resize_size, resize_size)

            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )).astype(np.float32)

            element = {
                "observation/image": img_resized,
                "observation/wrist_image": wrist_resized,
                "observation/state": state,
                "prompt": str(task_description),
            }

            # Get actions and attention
            actions, importance_maps = policy.infer_with_attention(element)
            action_plan.extend(actions[:replan_steps])

            # Prepare tensors for analyzer
            current_images = {
                "base": torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(policy.device) / 255.0,
                "wrist": torch.from_numpy(wrist_resized).float().permute(2, 0, 1).unsqueeze(0).to(policy.device) / 255.0,
            }
            current_state = torch.from_numpy(state).unsqueeze(0).to(policy.device)
            # Only use first 2 cameras (base and wrist), ignore dummy right_wrist
            current_attention = importance_maps[:, :2, :, :]  # (B, 2, 16, 16)

            # Compute metrics
            metrics = analyzer.compute_metrics(current_images, current_state, current_attention)

            frame_data = {
                "frame": frame_idx,
                "timestep": t,
                **metrics,
                "eef_pos": obs["robot0_eef_pos"].tolist(),
                "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
            }
            trajectory_data.append(frame_data)

            frame_idx += 1

        action = action_plan.popleft()
        if hasattr(action, 'tolist'):
            action = action.tolist()
        if len(action) > 7:
            action = action[:7]

        obs, reward, done, info = env.step(action)

        if done:
            logger.info(f"Episode completed successfully at frame {frame_idx}")
            break

        t += 1

    return trajectory_data


def plot_drift_comparison(
    trajectory_data: List[Dict],
    output_path: str,
):
    """Plot comparison of different drift metrics over time."""
    frames = [d["frame"] for d in trajectory_data]
    state_drifts = [d["state_drift"] for d in trajectory_data]
    raw_base = [d["raw_diff_base"] for d in trajectory_data]
    raw_wrist = [d["raw_diff_wrist"] for d in trajectory_data]
    weighted_base = [d["weighted_drift_base"] for d in trajectory_data]
    weighted_wrist = [d["weighted_drift_wrist"] for d in trajectory_data]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: State Drift
    axes[0].plot(frames, state_drifts, 'b-', linewidth=2, label='State Drift')
    axes[0].set_ylabel('State Drift')
    axes[0].set_title('State Drift (Joint Changes) - Layer 1 Guard')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Raw Pixel Diff
    axes[1].plot(frames, raw_base, 'g-', linewidth=2, label='Base Camera')
    axes[1].plot(frames, raw_wrist, 'r-', linewidth=2, label='Wrist Camera')
    axes[1].set_ylabel('Raw Pixel Diff')
    axes[1].set_title('Raw Pixel Difference (No Attention Weighting)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Weighted Drift
    axes[2].plot(frames, weighted_base, 'g-', linewidth=2, label='Base Camera')
    axes[2].plot(frames, weighted_wrist, 'r-', linewidth=2, label='Wrist Camera')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Weighted Drift')
    axes[2].set_title('Attention-Weighted Drift (With Dilation)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def analyze_threshold_candidates(
    all_trajectories: List[List[Dict]],
    output_dir: str,
):
    """Analyze different threshold values and their effects."""
    # Collect all metrics
    all_state_drifts = []
    all_weighted_drifts = []
    all_raw_diffs = []

    for traj in all_trajectories:
        for d in traj:
            all_state_drifts.append(d["state_drift"])
            all_weighted_drifts.append(d["total_weighted_drift"])
            all_raw_diffs.append(d["raw_diff_base"] + d["raw_diff_wrist"])

    # Convert to numpy
    state_drifts = np.array(all_state_drifts)
    weighted_drifts = np.array(all_weighted_drifts)
    raw_diffs = np.array(all_raw_diffs)

    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # State drift distribution
    axes[0].hist(state_drifts[state_drifts > 0], bins=50, color='blue', alpha=0.7)
    axes[0].set_xlabel('State Drift')
    axes[0].set_ylabel('Count')
    axes[0].set_title('State Drift Distribution')
    axes[0].axvline(np.percentile(state_drifts, 90), color='red', linestyle='--', label=f'P90: {np.percentile(state_drifts, 90):.4f}')
    axes[0].legend()

    # Weighted drift distribution
    axes[1].hist(weighted_drifts[weighted_drifts > 0], bins=50, color='green', alpha=0.7)
    axes[1].set_xlabel('Weighted Drift')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Weighted Drift Distribution')
    axes[1].axvline(np.percentile(weighted_drifts, 50), color='orange', linestyle='--', label=f'P50: {np.percentile(weighted_drifts, 50):.4f}')
    axes[1].axvline(np.percentile(weighted_drifts, 90), color='red', linestyle='--', label=f'P90: {np.percentile(weighted_drifts, 90):.4f}')
    axes[1].legend()

    # Raw diff distribution
    axes[2].hist(raw_diffs[raw_diffs > 0], bins=50, color='purple', alpha=0.7)
    axes[2].set_xlabel('Raw Diff')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Raw Pixel Diff Distribution')
    axes[2].axvline(np.percentile(raw_diffs, 90), color='red', linestyle='--', label=f'P90: {np.percentile(raw_diffs, 90):.4f}')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drift_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Compute threshold statistics
    stats = {
        "state_drift": {
            "min": float(state_drifts.min()),
            "max": float(state_drifts.max()),
            "mean": float(state_drifts.mean()),
            "std": float(state_drifts.std()),
            "p50": float(np.percentile(state_drifts, 50)),
            "p75": float(np.percentile(state_drifts, 75)),
            "p90": float(np.percentile(state_drifts, 90)),
            "p95": float(np.percentile(state_drifts, 95)),
        },
        "weighted_drift": {
            "min": float(weighted_drifts.min()),
            "max": float(weighted_drifts.max()),
            "mean": float(weighted_drifts.mean()),
            "std": float(weighted_drifts.std()),
            "p50": float(np.percentile(weighted_drifts, 50)),
            "p75": float(np.percentile(weighted_drifts, 75)),
            "p90": float(np.percentile(weighted_drifts, 90)),
            "p95": float(np.percentile(weighted_drifts, 95)),
        },
        "raw_diff": {
            "min": float(raw_diffs.min()),
            "max": float(raw_diffs.max()),
            "mean": float(raw_diffs.mean()),
            "std": float(raw_diffs.std()),
            "p50": float(np.percentile(raw_diffs, 50)),
            "p75": float(np.percentile(raw_diffs, 75)),
            "p90": float(np.percentile(raw_diffs, 90)),
            "p95": float(np.percentile(raw_diffs, 95)),
        },
    }

    # Recommend thresholds
    stats["recommended_thresholds"] = {
        "state_threshold": stats["state_drift"]["p90"],
        "drift_threshold": stats["weighted_drift"]["p75"],
        "comment": "state_threshold at P90 catches major joint movements. drift_threshold at P75 is balanced between reuse rate and safety."
    }

    return stats


def run_drift_analysis(args):
    """Main analysis function."""
    np.random.seed(args.seed)

    from libero.libero import benchmark

    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create policy
    logger.info("Loading model for drift analysis...")
    policy = AttentionVisualizationPolicy(
        checkpoint_dir=args.checkpoint_dir,
    )

    # Initialize LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    num_tasks = min(3, task_suite.n_tasks) if args.quick else min(args.num_tasks, task_suite.n_tasks)
    num_episodes = 1 if args.quick else args.num_episodes

    logger.info(f"Analyzing {num_tasks} tasks, {num_episodes} episodes each")
    logger.info(f"Output directory: {output_dir}")

    all_trajectories = []

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        logger.info(f"Task {task_id}: {task_description}")

        for episode_idx in range(num_episodes):
            logger.info(f"  Episode {episode_idx + 1}/{num_episodes}")

            env.reset()
            env.set_init_state(initial_states[episode_idx % len(initial_states)])

            trajectory_data = collect_trajectory_data(
                policy,
                env,
                task_description,
                max_steps=200,
                replan_steps=args.replan_steps,
                resize_size=args.resize_size,
            )

            all_trajectories.append(trajectory_data)

            # Save trajectory data
            traj_path = output_dir / f"trajectory_task{task_id}_ep{episode_idx}.json"
            with open(traj_path, 'w') as f:
                json.dump({
                    "task_id": task_id,
                    "task_description": str(task_description),
                    "episode": episode_idx,
                    "frames": trajectory_data,
                }, f, indent=2)

            # Plot drift comparison for this trajectory
            if len(trajectory_data) > 1:
                plot_path = output_dir / f"drift_task{task_id}_ep{episode_idx}.png"
                plot_drift_comparison(trajectory_data, str(plot_path))

        env.close()

    # Analyze threshold candidates
    logger.info("Analyzing threshold candidates...")
    stats = analyze_threshold_candidates(all_trajectories, str(output_dir))

    # Save analysis results
    stats_path = output_dir / "threshold_analysis.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("DRIFT ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nData collected: {len(all_trajectories)} trajectories")
    total_frames = sum(len(t) for t in all_trajectories)
    print(f"Total frames: {total_frames}")

    print("\n--- State Drift Statistics ---")
    sd = stats["state_drift"]
    print(f"  Mean: {sd['mean']:.6f} ± {sd['std']:.6f}")
    print(f"  P50: {sd['p50']:.6f}, P90: {sd['p90']:.6f}, P95: {sd['p95']:.6f}")

    print("\n--- Weighted Drift Statistics ---")
    wd = stats["weighted_drift"]
    print(f"  Mean: {wd['mean']:.6f} ± {wd['std']:.6f}")
    print(f"  P50: {wd['p50']:.6f}, P90: {wd['p90']:.6f}, P95: {wd['p95']:.6f}")

    print("\n--- Raw Diff Statistics ---")
    rd = stats["raw_diff"]
    print(f"  Mean: {rd['mean']:.6f} ± {rd['std']:.6f}")
    print(f"  P50: {rd['p50']:.6f}, P90: {rd['p90']:.6f}, P95: {rd['p95']:.6f}")

    print("\n--- Recommended Thresholds ---")
    rec = stats["recommended_thresholds"]
    print(f"  state_threshold: {rec['state_threshold']:.6f}")
    print(f"  drift_threshold: {rec['drift_threshold']:.6f}")
    print(f"  Note: {rec['comment']}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze weighted drift distribution")

    parser.add_argument("--task_suite_name", default="libero_spatial",
                       choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--checkpoint_dir",
                       default=os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero"))

    parser.add_argument("--num_tasks", type=int, default=3,
                       help="Number of tasks to analyze")
    parser.add_argument("--num_episodes", type=int, default=2,
                       help="Number of episodes per task")
    parser.add_argument("--output_dir", type=str, default="drift_analysis",
                       help="Output directory")

    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: 1 task, 1 episode")

    args = parser.parse_args()
    run_drift_analysis(args)
