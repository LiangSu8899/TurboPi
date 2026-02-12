#!/usr/bin/env python3
"""
LIBERO W4A16 TVM Benchmark
==========================

Evaluate W4A16 TVM kernel integration on LIBERO tasks.
Compare W4A16 TVM vs BF16 baseline on 2-3 tasks.

Usage:
    python scripts/libero_eval_w4a16_tvm.py --num_tasks 3 --num_trials 5

Author: Claude Code
Date: 2026-02-10
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


def create_model(checkpoint_dir, method='bf16'):
    """Create PI0 model with optional W4A16 TVM quantization."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config

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

    # Apply quantization
    if method == 'bf16':
        logger.info("BF16 baseline - no quantization")

    elif method == 'w4a16':
        from openpi.models_pytorch.w4a16_mlp import replace_paligemma_mlp_with_w4a16
        replaced = replace_paligemma_mlp_with_w4a16(model, cache_dequantized=True, use_tvm=False)
        logger.info(f"W4A16 PyTorch: Replaced {replaced} MLP layers (no TVM)")

    elif method == 'w4a16_tvm':
        from openpi.models_pytorch.w4a16_mlp import replace_paligemma_mlp_with_w4a16, _tvm_available
        replaced = replace_paligemma_mlp_with_w4a16(model, cache_dequantized=True, use_tvm=True)
        tvm_status = "with TVM" if _tvm_available else "TVM not available, using PyTorch"
        logger.info(f"W4A16 TVM: Replaced {replaced} MLP layers ({tvm_status})")

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

            # Inference with timing
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():
                actions = model.sample_actions(device, observation, num_steps=3, use_kv_cache=True)
            torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - start_time)

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

    avg_latency = np.mean(inference_times) * 1000 if inference_times else 0
    return success, avg_latency, len(inference_times)


def evaluate_method(method, checkpoint_dir, task_suite_name, num_tasks, num_trials, seed,
                    resize_size, replan_steps):
    """Evaluate a single method."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing: {method.upper()}")
    logger.info(f"{'='*70}")

    device = torch.device('cuda')

    # Create model and tokenizer
    model, pi0_config, tokenizer = create_model(checkpoint_dir, method)

    # Load normalization stats
    norm_stats = load_norm_stats(checkpoint_dir)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    max_steps_dict = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_dict.get(task_suite_name, 300)

    results = []
    all_latencies = []
    total_inferences = 0

    for task_id in range(min(num_tasks, task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

        logger.info(f"\nTask {task_id + 1}/{num_tasks}: {task_description}")

        task_successes = 0
        task_latencies = []
        task_inferences = 0

        for episode_idx in range(num_trials):
            success, latency, num_inferences = run_episode(
                model, tokenizer, env, task_description, initial_states, episode_idx, device,
                max_steps=max_steps, resize_size=resize_size, replan_steps=replan_steps,
                max_token_len=pi0_config.max_token_len, norm_stats=norm_stats
            )
            task_successes += int(success)
            task_latencies.append(latency)
            task_inferences += num_inferences
            status = "SUCCESS" if success else "FAIL"
            logger.info(f"  Trial {episode_idx + 1}: {status} (latency: {latency:.1f} ms, inferences: {num_inferences})")

        env.close()

        success_rate = task_successes / num_trials
        avg_latency = np.mean(task_latencies)
        all_latencies.extend(task_latencies)
        total_inferences += task_inferences

        results.append({
            'task_id': task_id,
            'task': task_description,
            'success_rate': success_rate,
            'successes': task_successes,
            'trials': num_trials,
            'avg_latency_ms': avg_latency,
            'inferences': task_inferences,
        })

        logger.info(f"  Task success rate: {success_rate:.0%} ({task_successes}/{num_trials})")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Summary
    total_success = sum(r['successes'] for r in results)
    total_trials = sum(r['trials'] for r in results)
    overall_rate = total_success / total_trials if total_trials > 0 else 0
    overall_latency = np.mean(all_latencies) if all_latencies else 0
    overall_hz = 1000 / overall_latency if overall_latency > 0 else 0

    return {
        'method': method,
        'results': results,
        'success_rate': overall_rate,
        'total_success': total_success,
        'total_trials': total_trials,
        'avg_latency_ms': overall_latency,
        'hz': overall_hz,
        'total_inferences': total_inferences,
    }


def save_results(all_results, output_path):
    """Save results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LIBERO W4A16 TVM Benchmark")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="~/.cache/openpi/checkpoints/pi05_libero",
                       help="Model checkpoint directory")
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--num_tasks", type=int, default=3, help="Number of tasks to test")
    parser.add_argument("--num_trials", type=int, default=5, help="Trials per task")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--methods", type=str, nargs='+',
                       default=['bf16', 'w4a16_tvm'],
                       choices=['bf16', 'w4a16', 'w4a16_tvm'],
                       help="Methods to test")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    args = parser.parse_args()

    print("=" * 70)
    print("LIBERO W4A16 TVM Benchmark")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Task Suite: {args.task_suite}")
    print(f"  Tasks: {args.num_tasks}")
    print(f"  Trials per task: {args.num_trials}")
    print(f"  Methods: {args.methods}")
    print("=" * 70)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_results = {}

    for method in args.methods:
        torch.cuda.empty_cache()

        result = evaluate_method(
            method,
            args.checkpoint_dir,
            args.task_suite,
            args.num_tasks,
            args.num_trials,
            args.seed,
            args.resize_size,
            args.replan_steps,
        )
        all_results[method] = result

        print(f"\n{method.upper()}: {result['success_rate']:.0%} ({result['total_success']}/{result['total_trials']}), "
              f"Latency: {result['avg_latency_ms']:.1f} ms ({result['hz']:.2f} Hz)")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<15} {'Success Rate':<15} {'Latency (ms)':<15} {'Hz':<10}")
    print("-" * 60)

    for method, data in all_results.items():
        rate_str = f"{data['success_rate']:.0%} ({data['total_success']}/{data['total_trials']})"
        print(f"{method.upper():<15} {rate_str:<15} {data['avg_latency_ms']:<15.1f} {data['hz']:<10.2f}")

    print("-" * 60)

    # Comparison
    if 'bf16' in all_results and 'w4a16_tvm' in all_results:
        bf16 = all_results['bf16']
        w4a16 = all_results['w4a16_tvm']

        success_diff = w4a16['success_rate'] - bf16['success_rate']
        speedup = bf16['avg_latency_ms'] / w4a16['avg_latency_ms'] if w4a16['avg_latency_ms'] > 0 else 0

        print(f"\nW4A16 TVM vs BF16:")
        print(f"  Success rate difference: {success_diff:+.0%}")
        print(f"  Speedup: {speedup:.2f}x")

    print("\n" + "=" * 70)

    # Save results if output path specified
    if args.output:
        save_results(all_results, args.output)


if __name__ == "__main__":
    main()
