#!/usr/bin/env python3
"""
Quick LIBERO W4A16 TVM Test
===========================

Fast test of W4A16 TVM quantization on 2-3 LIBERO tasks.
Validates accuracy and latency before full benchmark.

Usage:
    In Docker container (turbo_pi_eval):
    python /workspace/scripts/libero_quick_w4a16_test.py

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
    "/workspace/src",
    "/workspace/packages/openpi-client/src",
    "/workspace/third_party/libero",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# TVM paths
TVM_HOME = os.environ.get("TVM_HOME", "/workspace/external/tvm")
if TVM_HOME and os.path.exists(TVM_HOME):
    sys.path.insert(0, os.path.join(TVM_HOME, "python"))

# MuJoCo rendering
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch
torch.backends.cudnn.enabled = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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


def load_norm_stats(checkpoint_dir):
    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()
    norm_stats_path = checkpoint_path / "assets/physical-intelligence/libero/norm_stats.json"
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            data = json.load(f)
        return data.get("norm_stats", data)
    return None


def normalize_state(state, norm_stats):
    if norm_stats is None:
        return state
    mean = np.array(norm_stats["state"]["mean"])
    std = np.array(norm_stats["state"]["std"])
    return (state - mean) / (std + 1e-8)


def unnormalize_actions(actions, norm_stats):
    if norm_stats is None:
        return actions
    mean = np.array(norm_stats["actions"]["mean"])
    std = np.array(norm_stats["actions"]["std"])
    return actions * std + mean


def _quat2axisangle(quat):
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
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def load_tokenizer(max_token_len=200):
    import sentencepiece as spm
    tokenizer_paths = [
        pathlib.Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model"),
        pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model",
    ]
    for path in tokenizer_paths:
        if path.exists():
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.Load(str(path))
            return tokenizer, max_token_len
    raise FileNotFoundError(f"Tokenizer not found")


def create_model(checkpoint_dir, use_w4a16=True, use_tvm=True):
    """Create PI0 model with optional W4A16 TVM quantization."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from safetensors.torch import load_file

    checkpoint_path = pathlib.Path(checkpoint_dir).expanduser()

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

    weights_path = checkpoint_path / "model.safetensors"
    logger.info(f"Loading weights from {weights_path}")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    tokenizer, _ = load_tokenizer(max_token_len)

    # Apply W4A16 TVM if requested
    if use_w4a16:
        from openpi.models_pytorch.w4a16_mlp import replace_paligemma_mlp_with_w4a16, _tvm_available
        replaced = replace_paligemma_mlp_with_w4a16(model, cache_dequantized=True, use_tvm=use_tvm)
        tvm_status = "with TVM" if _tvm_available and use_tvm else "PyTorch fallback"
        logger.info(f"W4A16: Replaced {replaced} MLP layers ({tvm_status})")
    else:
        logger.info("BF16 baseline - no quantization")

    return model, pi0_config, tokenizer


def prepare_observation(tokenizer, img, wrist_img, state, prompt, device, max_token_len=200):
    from openpi.models_pytorch.pi0_pytorch import Observation

    def img_to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(device).to(torch.bfloat16)

    state_tensor = torch.zeros(1, 32, device=device, dtype=torch.bfloat16)
    state_tensor[0, :len(state)] = torch.tensor(state, dtype=torch.bfloat16)

    token_ids = tokenizer.Encode(prompt, add_bos=True)
    if len(token_ids) > max_token_len:
        token_ids = token_ids[:max_token_len]
    token_mask = [1] * len(token_ids)
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
                norm_stats=None, num_denoising_steps=3):
    from openpi_client import image_tools

    env.reset()
    action_plan = collections.deque()
    obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])

    t = 0
    done = False
    success = False
    inference_times = []

    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if not action_plan:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
            wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ))

            normalized_state = normalize_state(state, norm_stats)

            observation = prepare_observation(tokenizer, img, wrist_img, normalized_state,
                                              task_description, device, max_token_len)

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.no_grad():
                actions = model.sample_actions(device, observation, num_steps=num_denoising_steps, use_kv_cache=True)
            torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - start_time)

            action_chunk = actions[0, :, :7].cpu().numpy()

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


def main():
    parser = argparse.ArgumentParser(description="Quick LIBERO W4A16 TVM Test")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="/root/.cache/openpi/checkpoints/pi05_libero")
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--num_tasks", type=int, default=3, help="Number of tasks to test (2-3 for quick test)")
    parser.add_argument("--num_trials", type=int, default=5, help="Trials per task")
    parser.add_argument("--denoising_steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--compare_baseline", action="store_true", help="Also test BF16 baseline")
    args = parser.parse_args()

    print("=" * 70)
    print("Quick LIBERO W4A16 TVM Test")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Task Suite: {args.task_suite}")
    print(f"  Tasks: {args.num_tasks}")
    print(f"  Trials per task: {args.num_trials}")
    print(f"  Denoising Steps: {args.denoising_steps}")
    print(f"  Compare Baseline: {args.compare_baseline}")
    print("=" * 70)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check TVM availability
    try:
        import tvm
        print(f"\nTVM available: {tvm.__version__}")
    except ImportError:
        print("\nWARNING: TVM not available, will use PyTorch fallback")

    from libero.libero import benchmark

    device = torch.device('cuda')

    # Load model with W4A16 TVM
    print("\n" + "=" * 70)
    print("Loading W4A16 TVM Model...")
    print("=" * 70)

    model, pi0_config, tokenizer = create_model(args.checkpoint_dir, use_w4a16=True, use_tvm=True)
    norm_stats = load_norm_stats(args.checkpoint_dir)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    max_steps = MAX_STEPS_DICT.get(args.task_suite, 300)

    # Run evaluation
    results = []
    all_latencies = []

    for task_id in range(min(args.num_tasks, task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        print(f"\n{'='*60}")
        print(f"Task {task_id + 1}/{args.num_tasks}: {task_description}")
        print(f"{'='*60}")

        task_successes = 0
        task_latencies = []

        for episode_idx in range(args.num_trials):
            success, latency, num_inferences = run_episode(
                model, tokenizer, env, task_description, initial_states, episode_idx, device,
                max_steps=max_steps, resize_size=args.resize_size, replan_steps=args.replan_steps,
                max_token_len=pi0_config.max_token_len, norm_stats=norm_stats,
                num_denoising_steps=args.denoising_steps
            )
            task_successes += int(success)
            task_latencies.append(latency)
            status = "SUCCESS" if success else "FAIL"
            print(f"  Trial {episode_idx + 1}: {status} (latency: {latency:.1f} ms, inferences: {num_inferences})")

        env.close()

        success_rate = task_successes / args.num_trials
        avg_latency = np.mean(task_latencies)
        all_latencies.extend(task_latencies)

        results.append({
            'task_id': task_id,
            'task': task_description,
            'success_rate': success_rate,
            'successes': task_successes,
            'trials': args.num_trials,
            'avg_latency_ms': avg_latency,
        })

        print(f"\n  Task Success Rate: {success_rate:.0%} ({task_successes}/{args.num_trials})")
        print(f"  Average Latency: {avg_latency:.1f} ms")

    # Summary
    print("\n" + "=" * 70)
    print("W4A16 TVM RESULTS SUMMARY")
    print("=" * 70)

    total_success = sum(r['successes'] for r in results)
    total_trials = sum(r['trials'] for r in results)
    overall_rate = total_success / total_trials if total_trials > 0 else 0
    overall_latency = np.mean(all_latencies) if all_latencies else 0
    overall_hz = 1000 / overall_latency if overall_latency > 0 else 0

    print(f"\n{'Task':<50} {'Success':>10} {'Latency':>12}")
    print("-" * 72)
    for r in results:
        rate_str = f"{r['success_rate']:.0%}"
        lat_str = f"{r['avg_latency_ms']:.1f} ms"
        print(f"{r['task'][:50]:<50} {rate_str:>10} {lat_str:>12}")
    print("-" * 72)
    overall_str = f"{overall_rate:.0%} ({total_success}/{total_trials})"
    print(f"{'OVERALL':<50} {overall_str:>10} {overall_latency:.1f} ms")

    print(f"\n  Success Rate: {overall_rate:.0%} ({total_success}/{total_trials})")
    print(f"  Average Latency: {overall_latency:.1f} ms")
    print(f"  Throughput: {overall_hz:.2f} Hz")

    # Reference baselines
    print("\n--- Reference Baselines (from docs) ---")
    if args.denoising_steps == 3:
        trt_baseline = 120.6
        print("  TRT FP8 (3 steps): 120.6 ms, 8.3 Hz")
        print("  PyTorch BF16: ~180 ms, 5.6 Hz")
    elif args.denoising_steps == 1:
        trt_baseline = 83.5
        print("  TRT FP8 (1 step): 83.5 ms, 12.0 Hz")
    else:
        trt_baseline = 120.6  # Default

    improvement = (trt_baseline - overall_latency) / trt_baseline * 100
    print(f"\n  vs TRT FP8: {improvement:+.1f}% latency change")
    print(f"  vs TRT FP8: {trt_baseline / overall_latency:.2f}x speed ratio")

    print("\n" + "=" * 70)

    # Clean up
    del model
    torch.cuda.empty_cache()

    # Optionally test BF16 baseline
    if args.compare_baseline:
        print("\n" + "=" * 70)
        print("Loading BF16 Baseline Model...")
        print("=" * 70)

        model_bf16, _, _ = create_model(args.checkpoint_dir, use_w4a16=False, use_tvm=False)

        bf16_results = []
        bf16_latencies = []

        for task_id in range(min(args.num_tasks, task_suite.n_tasks)):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            print(f"\nTask {task_id + 1}/{args.num_tasks}: {task_description}")

            task_successes = 0
            task_latencies = []

            for episode_idx in range(args.num_trials):
                success, latency, _ = run_episode(
                    model_bf16, tokenizer, env, task_description, initial_states, episode_idx, device,
                    max_steps=max_steps, resize_size=args.resize_size, replan_steps=args.replan_steps,
                    max_token_len=pi0_config.max_token_len, norm_stats=norm_stats,
                    num_denoising_steps=args.denoising_steps
                )
                task_successes += int(success)
                task_latencies.append(latency)
                status = "SUCCESS" if success else "FAIL"
                print(f"  Trial {episode_idx + 1}: {status} (latency: {latency:.1f} ms)")

            env.close()

            success_rate = task_successes / args.num_trials
            avg_latency = np.mean(task_latencies)
            bf16_latencies.extend(task_latencies)

            bf16_results.append({
                'task_id': task_id,
                'success_rate': success_rate,
                'avg_latency_ms': avg_latency,
            })

        bf16_total_success = sum(r['successes'] if 'successes' in r else int(r['success_rate'] * args.num_trials) for r in bf16_results)
        bf16_overall_rate = sum(r['success_rate'] for r in bf16_results) / len(bf16_results)
        bf16_overall_latency = np.mean(bf16_latencies)

        print("\n" + "=" * 70)
        print("COMPARISON: W4A16 TVM vs BF16")
        print("=" * 70)
        print(f"\n{'Method':<20} {'Success Rate':>15} {'Latency':>15} {'Hz':>10}")
        print("-" * 60)
        w4a16_rate = f"{overall_rate:.0%}"
        w4a16_lat = f"{overall_latency:.1f} ms"
        w4a16_hz = f"{overall_hz:.2f}"
        print(f"{'W4A16 TVM':<20} {w4a16_rate:>15} {w4a16_lat:>15} {w4a16_hz:>10}")
        bf16_rate = f"{bf16_overall_rate:.0%}"
        bf16_lat = f"{bf16_overall_latency:.1f} ms"
        bf16_hz = f"{1000/bf16_overall_latency:.2f}"
        print(f"{'BF16 Baseline':<20} {bf16_rate:>15} {bf16_lat:>15} {bf16_hz:>10}")
        print("-" * 60)
        speedup = bf16_overall_latency / overall_latency
        print(f"\nW4A16 TVM Speedup: {speedup:.2f}x")

        del model_bf16
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Quick test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
