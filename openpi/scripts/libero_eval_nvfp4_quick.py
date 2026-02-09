#!/usr/bin/env python3
"""
Quick LIBERO evaluation with NVFP4 MLP layers.

This script tests the NVFP4 CUTLASS GEMM kernel with corrected scale layout
on a small subset of LIBERO tasks.

Usage:
    python scripts/libero_eval_nvfp4_quick.py --num_tasks 2 --num_trials 3
"""

import sys
import os
import logging
import time
import argparse
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

# Disable cuDNN for Jetson compatibility
torch.backends.cudnn.enabled = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# LIBERO constants
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = 220

# Try to import LIBERO
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    logger.warning("LIBERO not available - will run in dummy mode")


def resize_with_pad(img, target_h, target_w):
    """Resize image with padding."""
    import cv2
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    result = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    result[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return result


def create_policy_with_nvfp4(checkpoint_dir: str, use_cutlass: bool = True):
    """Create policy with NVFP4 MLP layers."""
    import json
    import pathlib
    from safetensors.torch import load_file
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from openpi.models_pytorch.nvfp4_mlp import replace_paligemma_mlp_with_nvfp4

    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()
    logger.info(f"Loading PI0 model from {checkpoint_path}...")

    # Load config
    config_path = checkpoint_path / "config.json"
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

    model = PI0Pytorch(pi0_config)

    # Load weights
    weights_path = checkpoint_path / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    logger.info("Replacing MLP layers with NVFP4...")

    # Replace MLP layers
    replaced = replace_paligemma_mlp_with_nvfp4(model)
    logger.info(f"Replaced {replaced} MLP layers with NVFP4")

    # Prepare CUTLASS weights
    if use_cutlass:
        logger.info("Preparing NVFP4 weights for CUTLASS...")
        paligemma_lm = model.paligemma_with_expert.paligemma.language_model
        for layer_idx, layer in enumerate(paligemma_lm.layers):
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(mlp, proj_name):
                        proj = getattr(mlp, proj_name)
                        if hasattr(proj, 'prepare_for_cutlass'):
                            proj.use_cutlass = True
                            proj.prepare_for_cutlass()
        logger.info("CUTLASS preparation complete")

    return model, pi0_config


def run_episode(policy, env, task_description, max_steps=MAX_STEPS, resize_size=224, replan_steps=8):
    """Run a single episode."""
    import cv2

    obs = env.reset()

    action_plan = []
    t = 0
    success = False

    while t < max_steps:
        if len(action_plan) == 0:
            # Get observation
            img = obs['agentview_image']
            wrist_img = obs['robot0_eye_in_hand_image']
            state = obs['robot0_eef_pos'].tolist() + obs['robot0_eef_quat'].tolist() + [obs['robot0_gripper_qpos'][0]]

            # Resize
            img = resize_with_pad(img, resize_size, resize_size)
            wrist_img = resize_with_pad(wrist_img, resize_size, resize_size)

            # Prepare inputs
            inputs = {
                'observation/image': torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float().cuda() / 255.0,
                'observation/wrist_image': torch.from_numpy(wrist_img).unsqueeze(0).permute(0, 3, 1, 2).float().cuda() / 255.0,
                'observation/state': torch.tensor([state], dtype=torch.float32).cuda(),
                'prompt': [task_description],
            }

            # Inference
            with torch.no_grad():
                outputs = policy.sample_actions(inputs, num_steps=3)

            action_chunk = outputs['actions'][0].cpu().numpy()
            action_plan.extend(action_chunk[:replan_steps])

        # Execute action
        action = action_plan.pop(0)
        obs, reward, done, info = env.step(action)
        t += 1

        if done or info.get('success', False):
            success = info.get('success', False)
            break

    return success


def main():
    parser = argparse.ArgumentParser(description="Quick NVFP4 LIBERO evaluation")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="~/.cache/openpi/checkpoints/pi05_libero",
                       help="Model checkpoint directory")
    parser.add_argument("--num_tasks", type=int, default=2, help="Number of tasks to test")
    parser.add_argument("--num_trials", type=int, default=3, help="Trials per task")
    parser.add_argument("--use_cutlass", action="store_true", default=True, help="Use CUTLASS NVFP4")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dummy", action="store_true", help="Run dummy inference benchmark only")
    args = parser.parse_args()

    print("=" * 70)
    print("NVFP4 LIBERO Quick Evaluation")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Tasks: {args.num_tasks}")
    print(f"  Trials per task: {args.num_trials}")
    print(f"  CUTLASS: {args.use_cutlass}")
    print("=" * 70)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create policy
    policy, pi0_config = create_policy_with_nvfp4(args.checkpoint_dir, args.use_cutlass)

    if not LIBERO_AVAILABLE or args.dummy:
        # Dummy test - just run inference
        logger.info("Running dummy inference test...")
        from openpi.models_pytorch.pi0_pytorch import Observation

        device = torch.device('cuda')
        dtype = torch.bfloat16
        batch_size = 1

        # Create proper Observation object
        observation = Observation(
            images={
                "base_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype),
                "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype),
                "right_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224, device=device, dtype=dtype),
            },
            image_masks={
                "base_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
                "left_wrist_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
                "right_wrist_0_rgb": torch.zeros(batch_size, device=device, dtype=torch.bool),
            },
            state=torch.randn(batch_size, 32, device=device, dtype=dtype),
            tokenized_prompt=torch.zeros(batch_size, 200, device=device, dtype=torch.long),
            tokenized_prompt_mask=torch.ones(batch_size, 200, device=device, dtype=torch.bool),
        )

        # Warmup
        print("Warmup...")
        for _ in range(3):
            with torch.no_grad():
                _ = policy.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
            torch.cuda.synchronize()

        # Benchmark
        print("Benchmark...")
        torch.cuda.synchronize()
        start = time.time()
        num_iters = 5
        for _ in range(num_iters):
            with torch.no_grad():
                actions = policy.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
            torch.cuda.synchronize()
        elapsed = time.time() - start

        hz = num_iters / elapsed
        print(f"\nNVFP4 Inference: {hz:.2f} Hz ({elapsed/num_iters*1000:.1f} ms/iter)")
        print(f"Actions shape: {actions.shape}")
        return

    # Run LIBERO evaluation
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()

    results = []

    for task_idx in range(min(args.num_tasks, len(task_suite.tasks))):
        task = task_suite.tasks[task_idx]
        task_name = task.name
        task_description = task.language

        logger.info(f"\nTask {task_idx + 1}/{args.num_tasks}: {task_name}")
        logger.info(f"  Description: {task_description}")

        # Create environment
        env_args = {
            "bddl_file_name": task.bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(args.seed)

        task_successes = []
        for trial in range(args.num_trials):
            success = run_episode(policy, env, task_description)
            task_successes.append(success)
            logger.info(f"    Trial {trial + 1}: {'SUCCESS' if success else 'FAIL'}")

        env.close()

        success_rate = sum(task_successes) / len(task_successes)
        results.append({
            'task': task_name,
            'success_rate': success_rate,
            'successes': sum(task_successes),
            'trials': len(task_successes),
        })

        logger.info(f"  Success rate: {success_rate:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    total_success = sum(r['successes'] for r in results)
    total_trials = sum(r['trials'] for r in results)
    overall_rate = total_success / total_trials if total_trials > 0 else 0

    for r in results:
        print(f"  {r['task']}: {r['success_rate']:.1%} ({r['successes']}/{r['trials']})")

    print("-" * 70)
    print(f"  Overall: {overall_rate:.1%} ({total_success}/{total_trials})")
    print("=" * 70)


if __name__ == "__main__":
    main()
