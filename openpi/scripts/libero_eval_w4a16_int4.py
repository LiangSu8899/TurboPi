#!/usr/bin/env python3
"""
LIBERO W4A16 INT4 TVM Benchmark
===============================

Evaluate W4A16 INT4 TVM kernel (128-bit vectorized) on LIBERO tasks.
Uses the new optimized kernel: 0.125ms per layer.

Usage:
    docker exec $ENV_VARS turbo_pi_eval python /workspace/scripts/libero_eval_w4a16_int4.py \
        --num_tasks 2 --num_trials 3

Author: Claude Code
Date: 2026-02-11
"""

import sys
import os
import collections
import logging
import math
import time
import argparse
import json
import pathlib
from datetime import datetime

# Setup paths
script_dir = pathlib.Path(__file__).parent
for path in [
    script_dir.parent / "src",
    script_dir.parent / "packages" / "openpi-client" / "src",
    script_dir.parent / "third_party" / "libero",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Set MuJoCo rendering options
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch

# Disable cuDNN for Jetson compatibility
torch.backends.cudnn.enabled = False

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from safetensors.torch import load_file
import sentencepiece as spm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def load_norm_stats(checkpoint_dir):
    """Load normalization stats from checkpoint."""
    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()
    norm_stats_path = checkpoint_path / "assets/physical-intelligence/libero/norm_stats.json"

    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            data = json.load(f)
        norm_stats = data.get("norm_stats", data)
        logger.info(f"Loaded normalization stats from {norm_stats_path}")
        return norm_stats
    else:
        logger.warning(f"Normalization stats not found at {norm_stats_path}")
        return None


def normalize_state(state, norm_stats):
    """Normalize state using mean and std."""
    if norm_stats is None:
        return state
    mean = np.array(norm_stats["state"]["mean"])
    std = np.array(norm_stats["state"]["std"])
    return (state - mean) / (std + 1e-8)


def unnormalize_actions(actions, norm_stats):
    """Unnormalize actions using mean and std."""
    if norm_stats is None:
        return actions
    mean = np.array(norm_stats["actions"]["mean"])
    std = np.array(norm_stats["actions"]["std"])
    return actions * std + mean


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle."""
    quat = np.array(quat, dtype=np.float64)
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
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def load_tokenizer(max_token_len=200):
    """Load SentencePiece tokenizer."""
    tokenizer_paths = [
        pathlib.Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model"),
        pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model",
    ]
    for path in tokenizer_paths:
        if path.exists():
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.Load(str(path))
            logger.info(f"Loaded tokenizer from {path}")
            return tokenizer, max_token_len
    raise FileNotFoundError(f"Tokenizer not found in {tokenizer_paths}")


def create_model_w4a16_int4(checkpoint_dir):
    """Create PI0 model with W4A16 INT4 TVM quantization (new optimized kernel)."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from openpi.utils.model_patcher import patch_paligemma_decode_path

    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()

    # Load config
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
    else:
        model_config = {}

    max_token_len = model_config.get("tokenizer_max_length", 200)

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=model_config.get("action_dim", 32),
        action_horizon=model_config.get("action_horizon", 50),
        max_token_len=max_token_len,
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

    # Load tokenizer
    tokenizer, _ = load_tokenizer(max_token_len)

    # Apply W4A16 INT4 TVM quantization to PaliGemma MLP layers
    logger.info("Applying W4A16 INT4 TVM quantization...")
    stats = patch_paligemma_decode_path(model, verbose=True)
    logger.info(f"W4A16 INT4: Replaced {stats['replaced']} layers, saved {stats['memory_saved_mb']:.1f} MB")

    return model, pi0_config, tokenizer


def prepare_observation(tokenizer, img, wrist_img, state, prompt, device, max_token_len=200):
    """Prepare observation for model inference with proper tokenization."""
    from openpi.models_pytorch.pi0_pytorch import Observation

    # Convert images to tensor: [H,W,C] uint8 -> [C,H,W] float [0,1] -> bfloat16
    def img_to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(device).to(torch.bfloat16)

    # State: [8] float64 -> bfloat16 tensor padded to [32]
    state_tensor = torch.zeros(1, 32, device=device, dtype=torch.bfloat16)
    state_tensor[0, :len(state)] = torch.tensor(state, dtype=torch.bfloat16)

    # Tokenize prompt using SentencePiece
    token_ids = tokenizer.Encode(prompt, add_bos=True)
    # Pad/truncate to max_token_len
    if len(token_ids) > max_token_len:
        token_ids = token_ids[:max_token_len]
    token_mask = [1] * len(token_ids)
    # Pad
    pad_len = max_token_len - len(token_ids)
    token_ids = token_ids + [0] * pad_len
    token_mask = token_mask + [0] * pad_len

    tokenized_prompt = torch.tensor([token_ids], device=device, dtype=torch.long)
    tokenized_prompt_mask = torch.tensor([token_mask], device=device, dtype=torch.bool)

    observation = Observation(
        images={
            "base_0_rgb": img_to_tensor(img),
            "left_wrist_0_rgb": img_to_tensor(wrist_img),
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, device=device, dtype=torch.bfloat16),
        },
        image_masks={
            "base_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(1, device=device, dtype=torch.bool),
        },
        state=state_tensor,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )

    return observation


def run_episode(model, tokenizer, env, task_description, initial_states, episode_idx, device,
                max_steps=220, resize_size=224, replan_steps=5, num_steps_wait=10, max_token_len=200,
                norm_stats=None):
    """Run a single episode."""
    env.reset()
    action_plan = collections.deque()
    obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])

    t = 0
    done = False
    success = False
    inference_times = []

    while t < max_steps + num_steps_wait:
        # Wait steps at the beginning
        if t < num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if not action_plan:
            # Prepare observation
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
            wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ))

            # Normalize state
            normalized_state = normalize_state(state, norm_stats)

            observation = prepare_observation(tokenizer, img, wrist_img, normalized_state,
                                              task_description, device, max_token_len)

            # Inference with timing (CUDA Events for accuracy)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.no_grad():
                actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
            end_event.record()
            torch.cuda.synchronize()

            inference_times.append(start_event.elapsed_time(end_event))

            # Extract actions: [1, horizon, 32] -> [horizon, 7]
            action_chunk = actions[0, :, :7].cpu().numpy()

            # Unnormalize actions
            for i in range(len(action_chunk)):
                action_chunk[i] = unnormalize_actions(action_chunk[i], norm_stats)

            action_plan.extend(action_chunk[:replan_steps])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())

        if done:
            success = True
            break
        t += 1

    avg_latency = np.mean(inference_times) if inference_times else 0
    return success, avg_latency, len(inference_times)


def main():
    parser = argparse.ArgumentParser(description="LIBERO W4A16 INT4 TVM Benchmark")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="~/.cache/openpi/checkpoints/pi05_libero",
                       help="Model checkpoint directory")
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--num_tasks", type=int, default=2, help="Number of tasks to test")
    parser.add_argument("--num_trials", type=int, default=3, help="Trials per task")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("LIBERO W4A16 INT4 TVM Benchmark (0.125ms kernel)")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Task Suite: {args.task_suite}")
    print(f"  Tasks: {args.num_tasks}")
    print(f"  Trials per task: {args.num_trials}")
    print("=" * 70)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda')

    # Create model with W4A16 INT4 TVM quantization
    model, pi0_config, tokenizer = create_model_w4a16_int4(args.checkpoint_dir)

    # Load normalization stats
    norm_stats = load_norm_stats(args.checkpoint_dir)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    max_steps_dict = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_dict.get(args.task_suite, 300)

    results = []
    all_latencies = []
    total_inferences = 0

    for task_id in range(min(args.num_tasks, task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        logger.info(f"\nTask {task_id + 1}/{args.num_tasks}: {task_description}")

        task_successes = 0
        task_latencies = []
        task_inferences = 0

        for episode_idx in range(args.num_trials):
            success, latency, num_inferences = run_episode(
                model, tokenizer, env, task_description, initial_states, episode_idx, device,
                max_steps=max_steps, resize_size=args.resize_size, replan_steps=args.replan_steps,
                max_token_len=pi0_config.max_token_len, norm_stats=norm_stats
            )
            task_successes += int(success)
            task_latencies.append(latency)
            task_inferences += num_inferences
            status = "SUCCESS" if success else "FAIL"
            logger.info(f"  Trial {episode_idx + 1}: {status} (latency: {latency:.1f} ms, inferences: {num_inferences})")

        env.close()

        success_rate = task_successes / args.num_trials
        avg_latency = np.mean(task_latencies)
        all_latencies.extend(task_latencies)
        total_inferences += task_inferences

        results.append({
            'task_id': task_id,
            'task': task_description,
            'success_rate': success_rate,
            'successes': task_successes,
            'trials': args.num_trials,
            'avg_latency_ms': avg_latency,
            'inferences': task_inferences,
        })

        logger.info(f"  Task success rate: {success_rate:.0%} ({task_successes}/{args.num_trials})")

    # Summary
    total_success = sum(r['successes'] for r in results)
    total_trials = sum(r['trials'] for r in results)
    overall_rate = total_success / total_trials if total_trials > 0 else 0
    overall_latency = np.mean(all_latencies) if all_latencies else 0
    overall_hz = 1000 / overall_latency if overall_latency > 0 else 0

    print("\n" + "=" * 70)
    print("W4A16 INT4 TVM RESULTS")
    print("=" * 70)
    print(f"  Tasks tested: {args.num_tasks}")
    print(f"  Trials per task: {args.num_trials}")
    print(f"  Total trials: {total_trials}")
    print(f"  Success rate: {overall_rate:.0%} ({total_success}/{total_trials})")
    print(f"  Avg latency: {overall_latency:.1f} ms")
    print(f"  Frequency: {overall_hz:.2f} Hz")
    print(f"  Total inferences: {total_inferences}")
    print("=" * 70)

    # Per-task breakdown
    print("\nPer-Task Results:")
    print("-" * 60)
    for r in results:
        print(f"  {r['task'][:50]:<50}")
        print(f"    Success: {r['success_rate']:.0%} ({r['successes']}/{r['trials']}), "
              f"Latency: {r['avg_latency_ms']:.1f} ms")
    print("-" * 60)

    # Save results
    if args.output:
        output = {
            'timestamp': datetime.now().isoformat(),
            'method': 'w4a16_int4_tvm',
            'config': vars(args),
            'results': results,
            'summary': {
                'success_rate': overall_rate,
                'total_success': total_success,
                'total_trials': total_trials,
                'avg_latency_ms': overall_latency,
                'hz': overall_hz,
                'total_inferences': total_inferences,
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
