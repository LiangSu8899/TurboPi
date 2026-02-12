#!/usr/bin/env python3
"""
LIBERO evaluation comparing different quantization methods.

Tests W4A16, W4A8, W4A4 (simulation), and BF16 baseline on LIBERO tasks.

Usage:
    python scripts/libero_eval_quantization_compare.py --num_tasks 2 --num_trials 5
"""

import sys
import os
import logging
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

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


def load_base_model(checkpoint_dir: str):
    """Load base PI0 model."""
    import json
    import pathlib
    from safetensors.torch import load_file
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config

    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()

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

    # Load weights
    weights_path = checkpoint_path / "model.safetensors"
    state_dict = load_file(weights_path)

    return state_dict, pi0_config


def create_bf16_model(state_dict, pi0_config):
    """Create BF16 baseline model."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    model = PI0Pytorch(pi0_config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    return model


def create_w4a16_model(state_dict, pi0_config):
    """Create W4A16 quantized model."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models_pytorch.w4a16_mlp import replace_paligemma_mlp_with_w4a16

    model = PI0Pytorch(pi0_config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    replaced = replace_paligemma_mlp_with_w4a16(model, cache_dequantized=True)
    logger.info(f"W4A16: Replaced {replaced} MLP layers")

    return model


def create_w4a8_model(state_dict, pi0_config):
    """Create W4A8 quantized model."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models_pytorch.w4a8_mlp import replace_paligemma_mlp_with_w4a8

    model = PI0Pytorch(pi0_config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    replaced = replace_paligemma_mlp_with_w4a8(model, cache_dequantized=True)
    logger.info(f"W4A8: Replaced {replaced} MLP layers")

    return model


def create_w4a4_model(state_dict, pi0_config):
    """Create W4A4 (NVFP4 simulation) quantized model."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models_pytorch.nvfp4_mlp import replace_paligemma_mlp_with_nvfp4

    model = PI0Pytorch(pi0_config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    # Use simulation mode (not CUTLASS) for accuracy test
    replaced = replace_paligemma_mlp_with_nvfp4(model, use_cutlass=False)
    logger.info(f"W4A4: Replaced {replaced} MLP layers (simulation mode)")

    return model


def benchmark_inference(model, device, num_warmup=3, num_runs=5):
    """Benchmark inference latency."""
    from openpi.models_pytorch.pi0_pytorch import Observation

    dtype = torch.bfloat16
    batch_size = 1

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
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / num_runs) * 1000
    hz = num_runs / elapsed

    return latency_ms, hz


def run_episode(model, env, task_description, device, max_steps=MAX_STEPS, resize_size=224, replan_steps=8):
    """Run a single episode."""
    import cv2
    from openpi.models_pytorch.pi0_pytorch import Observation

    obs = env.reset()

    action_plan = []
    t = 0
    success = False
    inference_times = []

    while t < max_steps:
        if len(action_plan) == 0:
            # Get observation
            img = obs['agentview_image']
            wrist_img = obs['robot0_eye_in_hand_image']
            state = np.concatenate([
                obs['robot0_eef_pos'],
                obs['robot0_eef_quat'],
                [obs['robot0_gripper_qpos'][0]]
            ])

            # Resize
            img = resize_with_pad(img, resize_size, resize_size)
            wrist_img = resize_with_pad(wrist_img, resize_size, resize_size)

            # Convert to tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            # Create observation
            observation = Observation(
                images={
                    "base_0_rgb": img_tensor.to(torch.bfloat16),
                    "left_wrist_0_rgb": wrist_tensor.to(torch.bfloat16),
                    "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, device=device, dtype=torch.bfloat16),
                },
                image_masks={
                    "base_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
                    "left_wrist_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
                    "right_wrist_0_rgb": torch.zeros(1, device=device, dtype=torch.bool),
                },
                state=torch.zeros(1, 32, device=device, dtype=torch.bfloat16),  # Padded state
                tokenized_prompt=torch.zeros(1, 200, device=device, dtype=torch.long),
                tokenized_prompt_mask=torch.ones(1, 200, device=device, dtype=torch.bool),
            )

            # Set state
            observation.state[:, :8] = state_tensor.to(torch.bfloat16)

            # Inference
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
            torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - start)

            action_chunk = actions[0].cpu().numpy()
            # Extract 7-DOF action (pos + quat + gripper)
            action_plan.extend(action_chunk[:replan_steps, :7])

        # Execute action
        action = action_plan.pop(0)
        obs, reward, done, info = env.step(action)
        t += 1

        if done or info.get('success', False):
            success = info.get('success', False)
            break

    avg_inference_ms = np.mean(inference_times) * 1000 if inference_times else 0
    return success, avg_inference_ms


def get_bddl_full_path(bddl_file, suite_name="libero_spatial"):
    """Get full path to BDDL file."""
    import libero.libero as libero_mod
    libero_root = os.path.dirname(libero_mod.__file__)
    full_path = os.path.join(libero_root, "bddl_files", suite_name, bddl_file)
    if os.path.exists(full_path):
        return full_path
    # Try without suite name
    full_path = os.path.join(libero_root, "bddl_files", bddl_file)
    if os.path.exists(full_path):
        return full_path
    return bddl_file  # Return original if not found


def evaluate_method(method_name, model, state_dict, pi0_config, tasks, num_trials, seed, device, suite_name="libero_spatial"):
    """Evaluate a single quantization method."""
    results = []

    for task_idx, task in enumerate(tasks):
        task_name = task.name
        task_description = task.language

        logger.info(f"\n[{method_name}] Task {task_idx + 1}/{len(tasks)}: {task_name}")
        logger.info(f"  Description: {task_description}")

        # Get full BDDL path
        bddl_path = get_bddl_full_path(task.bddl_file, suite_name)
        logger.info(f"  BDDL: {bddl_path}")

        # Create environment
        env_args = {
            "bddl_file_name": bddl_path,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        task_successes = []
        task_latencies = []

        for trial in range(num_trials):
            success, avg_latency = run_episode(model, env, task_description, device)
            task_successes.append(success)
            task_latencies.append(avg_latency)
            logger.info(f"    Trial {trial + 1}: {'SUCCESS' if success else 'FAIL'} ({avg_latency:.1f} ms)")

        env.close()

        success_rate = sum(task_successes) / len(task_successes)
        avg_latency = np.mean(task_latencies)

        results.append({
            'task': task_name,
            'success_rate': success_rate,
            'successes': sum(task_successes),
            'trials': len(task_successes),
            'avg_latency_ms': avg_latency,
        })

        logger.info(f"  Success rate: {success_rate:.1%}, Avg latency: {avg_latency:.1f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="LIBERO quantization comparison")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="~/.cache/openpi/checkpoints/pi05_libero",
                       help="Model checkpoint directory")
    parser.add_argument("--num_tasks", type=int, default=2, help="Number of tasks to test")
    parser.add_argument("--num_trials", type=int, default=5, help="Trials per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--methods", type=str, nargs='+',
                       default=['bf16', 'w4a16', 'w4a8'],
                       choices=['bf16', 'w4a16', 'w4a8', 'w4a4'],
                       help="Methods to test")
    parser.add_argument("--dummy", action="store_true", help="Run dummy inference benchmark only")
    args = parser.parse_args()

    print("=" * 70)
    print("LIBERO Quantization Methods Comparison")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Tasks: {args.num_tasks}")
    print(f"  Trials per task: {args.num_trials}")
    print(f"  Methods: {args.methods}")
    print("=" * 70)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda')

    # Load base model weights
    logger.info("Loading model weights...")
    state_dict, pi0_config = load_base_model(args.checkpoint_dir)

    # Method creators
    method_creators = {
        'bf16': create_bf16_model,
        'w4a16': create_w4a16_model,
        'w4a8': create_w4a8_model,
        'w4a4': create_w4a4_model,
    }

    all_results = {}

    if args.dummy or not LIBERO_AVAILABLE:
        # Dummy benchmark mode
        print("\n" + "=" * 70)
        print("Inference Latency Benchmark (Dummy Mode)")
        print("=" * 70)

        for method in args.methods:
            print(f"\n[{method.upper()}] Creating model...")
            torch.cuda.empty_cache()

            model = method_creators[method](state_dict, pi0_config)

            latency_ms, hz = benchmark_inference(model, device)
            print(f"  Latency: {latency_ms:.1f} ms ({hz:.2f} Hz)")

            all_results[method] = {
                'latency_ms': latency_ms,
                'hz': hz,
                'success_rate': None,  # N/A in dummy mode
            }

            del model
            torch.cuda.empty_cache()

        # Summary
        print("\n" + "=" * 70)
        print("Latency Summary")
        print("=" * 70)
        print(f"\n{'Method':<12} {'Latency (ms)':<15} {'Hz':<10}")
        print("-" * 40)
        for method, data in all_results.items():
            print(f"{method.upper():<12} {data['latency_ms']:<15.1f} {data['hz']:<10.2f}")
        print("-" * 40)

        return

    # LIBERO evaluation mode
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    tasks = task_suite.tasks[:args.num_tasks]

    for method in args.methods:
        print(f"\n{'='*70}")
        print(f"Testing: {method.upper()}")
        print("=" * 70)

        torch.cuda.empty_cache()
        model = method_creators[method](state_dict, pi0_config)

        # Benchmark latency first
        latency_ms, hz = benchmark_inference(model, device)
        print(f"Inference latency: {latency_ms:.1f} ms ({hz:.2f} Hz)")

        # Run LIBERO evaluation
        results = evaluate_method(method, model, state_dict, pi0_config, tasks, args.num_trials, args.seed, device, "libero_spatial")

        total_success = sum(r['successes'] for r in results)
        total_trials = sum(r['trials'] for r in results)
        overall_rate = total_success / total_trials if total_trials > 0 else 0
        avg_latency = np.mean([r['avg_latency_ms'] for r in results])

        all_results[method] = {
            'results': results,
            'success_rate': overall_rate,
            'total_success': total_success,
            'total_trials': total_trials,
            'latency_ms': latency_ms,
            'hz': hz,
            'avg_task_latency_ms': avg_latency,
        }

        del model
        torch.cuda.empty_cache()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<10} {'Success Rate':<15} {'Latency (ms)':<15} {'Hz':<10} {'Rec':<5}")
    print("-" * 60)

    best_method = None
    best_score = 0

    for method, data in all_results.items():
        success_str = f"{data['success_rate']:.1%}" if data['success_rate'] is not None else "N/A"
        print(f"{method.upper():<10} {success_str:<15} {data['latency_ms']:<15.1f} {data['hz']:<10.2f}")

        # Score = success_rate * hz (higher is better)
        if data['success_rate'] is not None:
            score = data['success_rate'] * data['hz']
            if score > best_score:
                best_score = score
                best_method = method

    print("-" * 60)

    if best_method:
        print(f"\nRecommendation: {best_method.upper()}")
        print(f"  Success Rate: {all_results[best_method]['success_rate']:.1%}")
        print(f"  Latency: {all_results[best_method]['latency_ms']:.1f} ms ({all_results[best_method]['hz']:.2f} Hz)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
