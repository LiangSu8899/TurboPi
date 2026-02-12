#!/usr/bin/env python3
"""
LIBERO W4A16 TVM vs TRT FP8 Benchmark
=====================================

Compare W4A16 TVM (with DLPack zero-copy) against TRT FP8 baseline.

Usage:
    python scripts/libero_eval_w4a16_vs_trt.py --denoising_steps 3 --num_tasks 3 --num_trials 5
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
from typing import Dict, Any, Optional

import numpy as np
import torch

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


def create_model(checkpoint_dir, method='trt_fp8', use_tvm=False):
    """Create PI0 model with specified optimization method."""
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
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device='cuda')
    model.eval()

    tokenizer, _ = load_tokenizer(max_token_len)

    # Apply W4A16 TVM if requested
    if use_tvm:
        try:
            from openpi.models_pytorch.w4a16_mlp import replace_paligemma_mlp_with_w4a16, _tvm_available
            replaced = replace_paligemma_mlp_with_w4a16(model, cache_dequantized=True, use_tvm=True)
            tvm_status = "with TVM DLPack" if _tvm_available else "PyTorch fallback"
            logger.info(f"W4A16 TVM: Replaced {replaced} MLP layers ({tvm_status})")
        except Exception as e:
            logger.error(f"W4A16 TVM replacement failed: {e}")
            raise

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


def evaluate_method(method, checkpoint_dir, task_suite_name, num_tasks, num_trials, seed,
                    resize_size, replan_steps, num_denoising_steps):
    from libero.libero import benchmark

    logger.info(f"\n{'='*70}")
    logger.info(f"Testing: {method.upper()} (denoising_steps={num_denoising_steps})")
    logger.info(f"{'='*70}")

    device = torch.device('cuda')

    # Create model
    use_tvm = (method == 'w4a16_tvm')
    model, pi0_config, tokenizer = create_model(checkpoint_dir, method, use_tvm=use_tvm)

    norm_stats = load_norm_stats(checkpoint_dir)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    max_steps = MAX_STEPS_DICT.get(task_suite_name, 300)

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
                max_token_len=pi0_config.max_token_len, norm_stats=norm_stats,
                num_denoising_steps=num_denoising_steps
            )
            task_successes += int(success)
            task_latencies.append(latency)
            task_inferences += num_inferences
            status = "SUCCESS" if success else "FAIL"
            logger.info(f"  Trial {episode_idx + 1}: {status} (latency: {latency:.1f} ms)")

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
        })

        logger.info(f"  Task success rate: {success_rate:.0%} ({task_successes}/{num_trials})")

    del model
    torch.cuda.empty_cache()

    total_success = sum(r['successes'] for r in results)
    total_trials = sum(r['trials'] for r in results)
    overall_rate = total_success / total_trials if total_trials > 0 else 0
    overall_latency = np.mean(all_latencies) if all_latencies else 0
    overall_hz = 1000 / overall_latency if overall_latency > 0 else 0

    return {
        'method': method,
        'denoising_steps': num_denoising_steps,
        'results': results,
        'success_rate': overall_rate,
        'total_success': total_success,
        'total_trials': total_trials,
        'avg_latency_ms': overall_latency,
        'hz': overall_hz,
    }


def main():
    parser = argparse.ArgumentParser(description="LIBERO W4A16 TVM vs TRT FP8 Benchmark")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="~/.cache/openpi/checkpoints/pi05_libero")
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--num_tasks", type=int, default=3)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--denoising_steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--method", type=str, default="w4a16_tvm",
                       choices=['bf16', 'w4a16_tvm'])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("LIBERO W4A16 TVM Benchmark (DLPack Zero-Copy)")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Task Suite: {args.task_suite}")
    print(f"  Tasks: {args.num_tasks}")
    print(f"  Trials per task: {args.num_trials}")
    print(f"  Denoising Steps: {args.denoising_steps}")
    print(f"  Method: {args.method}")
    print("=" * 70)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check TVM availability
    try:
        import tvm
        print(f"\nTVM available: {tvm.__version__}")
    except ImportError:
        print("\nWARNING: TVM not available, will use PyTorch fallback")

    result = evaluate_method(
        args.method,
        args.checkpoint_dir,
        args.task_suite,
        args.num_tasks,
        args.num_trials,
        args.seed,
        args.resize_size,
        args.replan_steps,
        args.denoising_steps,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    rate_str = f"{result['success_rate']:.0%} ({result['total_success']}/{result['total_trials']})"
    print(f"\n{result['method'].upper()}:")
    print(f"  Success Rate: {rate_str}")
    print(f"  Latency: {result['avg_latency_ms']:.1f} ms")
    print(f"  Throughput: {result['hz']:.2f} Hz")

    # Reference baselines from docs
    print("\n--- Reference Baselines (from docs) ---")
    if args.denoising_steps == 3:
        print("  TRT FP8 (3 steps): 120.6 ms, 8.3 Hz")
        print("  PyTorch BF16: ~180 ms, 5.6 Hz")
    elif args.denoising_steps == 1:
        print("  TRT FP8 (1 step): 83.5 ms, 12.0 Hz")

    # Calculate improvement
    if args.denoising_steps == 3:
        trt_baseline = 120.6
    else:
        trt_baseline = 83.5

    improvement = (trt_baseline - result['avg_latency_ms']) / trt_baseline * 100
    print(f"\n  vs TRT FP8: {improvement:+.1f}% latency change")

    print("\n" + "=" * 70)

    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'result': result,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
