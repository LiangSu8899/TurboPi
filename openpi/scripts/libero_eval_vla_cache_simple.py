#!/usr/bin/env python3
"""
Simplified VLA-Cache Evaluation.

Compares:
1. Baseline (no_reuse): Every frame computes fresh KV
2. VLA-Cache (full_reuse): Reuse all camera KV when similarity > threshold

Uses cosine similarity on images to decide reuse.
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "openpi-client", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "libero"))

for path in ["/workspace/src", "/workspace/packages/openpi-client/src", "/workspace/third_party/libero"]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import the base optimized policy
from libero_eval_full_optimized import (
    FullOptimizedPolicy,
    run_episode,
    _get_libero_env,
    MAX_STEPS_DICT,
    LIBERO_ENV_RESOLUTION,
    LIBERO_DUMMY_ACTION,
    logger,
)


class VLACachePolicy(FullOptimizedPolicy):
    """
    VLA-Cache Policy: extends FullOptimizedPolicy with KV cache reuse.

    When similarity between current and previous frame exceeds threshold,
    skip Vision + KV cache computation and reuse previous KV.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        mode: str = "no_reuse",  # "no_reuse" or "full_reuse"
        similarity_threshold: float = 0.98,
        **kwargs
    ):
        super().__init__(checkpoint_dir, **kwargs)

        self.mode = mode
        self.similarity_threshold = similarity_threshold

        # Cache state
        self.prev_base_image = None
        self.prev_wrist_image = None
        self.prev_prefix_keys = None
        self.prev_prefix_values = None
        self.prev_prefix_pad_masks = None

        # Statistics
        self.reuse_stats = {
            "total_frames": 0,
            "reused_frames": 0,
            "base_similarities": [],
            "wrist_similarities": [],
        }

        logger.info(f"VLACachePolicy initialized: mode={mode}, threshold={similarity_threshold}")

    def reset_episode(self):
        """Reset cache for new episode."""
        self.prev_base_image = None
        self.prev_wrist_image = None
        self.prev_prefix_keys = None
        self.prev_prefix_values = None
        self.prev_prefix_pad_masks = None

    def reset_stats(self):
        """Reset statistics."""
        self.reuse_stats = {
            "total_frames": 0,
            "reused_frames": 0,
            "base_similarities": [],
            "wrist_similarities": [],
        }
        self.latency_records = []
        for key in self.component_latencies:
            self.component_latencies[key] = []

    def _compute_similarity(self, img1, img2):
        """Compute cosine similarity between two image tensors."""
        if img1 is None or img2 is None:
            return 0.0
        v1 = img1.flatten().float()
        v2 = img2.flatten().float()
        return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

    def _should_reuse(self, current_base, current_wrist):
        """Decide whether to reuse cached KV."""
        if self.mode == "no_reuse":
            return False

        if self.prev_base_image is None:
            return False

        base_sim = self._compute_similarity(self.prev_base_image, current_base)
        wrist_sim = self._compute_similarity(self.prev_wrist_image, current_wrist)

        self.reuse_stats["base_similarities"].append(base_sim)
        self.reuse_stats["wrist_similarities"].append(wrist_sim)

        # Both cameras must pass threshold for full reuse
        return base_sim >= self.similarity_threshold and wrist_sim >= self.similarity_threshold

    def infer(self, observation, num_steps=None):
        """Run inference with optional KV cache reuse."""
        import math
        from openpi.models_pytorch.pi0_pytorch import Observation

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        self.reuse_stats["total_frames"] += 1

        # Preprocess observation
        obs = self._preprocess(observation)

        # Get current images for similarity check
        images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(
            obs, train=False
        )
        current_base = images[0]  # base camera
        current_wrist = images[1]  # wrist camera

        with torch.no_grad():
            # Check if we can reuse KV cache
            should_reuse = self._should_reuse(current_base, current_wrist)

            if should_reuse and self.prev_prefix_keys is not None:
                # ============== REUSE PATH ==============
                self.reuse_stats["reused_frames"] += 1

                # Skip Vision + KV Cache, directly use cached values
                torch.cuda.synchronize()
                vision_start = time.perf_counter()

                prefix_keys = self.prev_prefix_keys
                prefix_values = self.prev_prefix_values
                prefix_pad_masks = self.prev_prefix_pad_masks

                torch.cuda.synchronize()
                vision_time = (time.perf_counter() - vision_start) * 1000
                self.component_latencies['vision'].append(vision_time)

                # KV Cache time is essentially 0 for reuse
                torch.cuda.synchronize()
                kv_start = time.perf_counter()
                torch.cuda.synchronize()
                kv_time = (time.perf_counter() - kv_start) * 1000
                self.component_latencies['kv_cache'].append(kv_time)

            else:
                # ============== FRESH COMPUTATION PATH ==============
                # 1. Vision TRT
                torch.cuda.synchronize()
                vision_start = time.perf_counter()

                if self.use_vision_trt:
                    embs = []
                    pad_masks = []
                    att_masks = []

                    for img, img_mask in zip(images, img_masks, strict=True):
                        img_fp16 = img.half()
                        vision_out = self.vision_trt(img_fp16)
                        vision_out_bf16 = vision_out.to(torch.bfloat16)
                        img_emb = self.multi_modal_projector(vision_out_bf16)

                        bsize, num_img_embs = img_emb.shape[:2]
                        embs.append(img_emb)
                        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                        att_masks += [0] * num_img_embs

                    lang_emb = self.model.paligemma_with_expert.embed_language_tokens(lang_tokens)
                    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

                    embs.append(lang_emb)
                    pad_masks.append(lang_masks)
                    att_masks += [0] * lang_emb.shape[1]

                    prefix_embs = torch.cat(embs, dim=1)
                    prefix_pad_masks = torch.cat(pad_masks, dim=1)
                    att_masks_tensor = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
                    bsize = prefix_pad_masks.shape[0]
                    prefix_att_masks = att_masks_tensor[None, :].expand(bsize, len(att_masks))
                else:
                    prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                        images, img_masks, lang_tokens, lang_masks
                    )

                torch.cuda.synchronize()
                vision_time = (time.perf_counter() - vision_start) * 1000
                self.component_latencies['vision'].append(vision_time)

                # 2. KV Cache TRT
                torch.cuda.synchronize()
                kv_start = time.perf_counter()

                prefix_keys, prefix_values = self.kv_engine.compute_kv_cache(
                    prefix_embs, prefix_pad_masks, prefix_att_masks
                )

                torch.cuda.synchronize()
                kv_time = (time.perf_counter() - kv_start) * 1000
                self.component_latencies['kv_cache'].append(kv_time)

                # Update cache
                self.prev_base_image = current_base.clone()
                self.prev_wrist_image = current_wrist.clone()
                self.prev_prefix_keys = prefix_keys.clone()
                self.prev_prefix_values = prefix_values.clone()
                self.prev_prefix_pad_masks = prefix_pad_masks.clone()

            # 3. Denoising (always needed)
            torch.cuda.synchronize()
            denoise_start = time.perf_counter()

            self.denoise_graph.static_inputs['prefix_keys'].copy_(prefix_keys)
            self.denoise_graph.static_inputs['prefix_values'].copy_(prefix_values)
            self.denoise_graph.static_inputs['prefix_pad_masks'].copy_(prefix_pad_masks)

            x_t = torch.randn(1, self.action_horizon, self.action_dim,
                             device=self.device, dtype=torch.bfloat16)
            actions = self.denoise_graph.infer(x_t)

            torch.cuda.synchronize()
            denoise_time = (time.perf_counter() - denoise_start) * 1000
            self.component_latencies['denoise'].append(denoise_time)

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.latency_records.append(latency_ms)
        self.component_latencies['total'].append(latency_ms)

        # Post-process
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def get_reuse_stats(self):
        """Get reuse statistics."""
        total = max(self.reuse_stats["total_frames"], 1)
        return {
            "mode": self.mode,
            "threshold": self.similarity_threshold,
            "total_frames": self.reuse_stats["total_frames"],
            "reused_frames": self.reuse_stats["reused_frames"],
            "reuse_rate": self.reuse_stats["reused_frames"] / total * 100,
            "avg_base_similarity": np.mean(self.reuse_stats["base_similarities"]) if self.reuse_stats["base_similarities"] else 0,
            "avg_wrist_similarity": np.mean(self.reuse_stats["wrist_similarities"]) if self.reuse_stats["wrist_similarities"] else 0,
        }


def run_vla_cache_evaluation(args):
    """Run VLA-Cache comparison evaluation."""
    from libero.libero import benchmark

    # Modes to compare
    if args.mode == "all":
        modes = ["no_reuse", "full_reuse"]
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
            schedule_type=args.schedule_type,
        )

        # Get benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        tasks = task_suite.get_task_names()[:args.num_tasks]

        results = {
            "mode": mode,
            "task_suite": args.task_suite_name,
            "num_tasks": len(tasks),
            "num_trials": args.num_trials,
            "task_results": [],
            "total_successes": 0,
            "total_trials": 0,
        }

        for task_idx, task_name in enumerate(tasks):
            task_id = task_suite.get_task_names().index(task_name)
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)

            logger.info(f"[{task_idx+1}/{len(tasks)}] {task.language}")

            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            task_successes = 0

            for trial_idx in range(min(args.num_trials, len(init_states))):
                env.reset()
                env.set_init_state(init_states[trial_idx])

                # Reset policy for new episode
                policy.reset_episode()

                # Run episode
                success = run_episode(env, policy, task_description, args)

                if success:
                    task_successes += 1

                logger.info(f"  Trial {trial_idx+1}: {'Success' if success else 'Fail'}")

            results["task_results"].append({
                "task_name": task_name,
                "successes": task_successes,
                "trials": args.num_trials,
                "success_rate": task_successes / args.num_trials * 100,
            })
            results["total_successes"] += task_successes
            results["total_trials"] += args.num_trials

            env.close()

        results["overall_success_rate"] = results["total_successes"] / results["total_trials"] * 100
        results["reuse_stats"] = policy.get_reuse_stats()
        results["timing"] = policy.get_latency_stats()

        all_results[mode] = results

        # Print summary
        logger.info(f"\n{mode} Results:")
        logger.info(f"  Success Rate: {results['overall_success_rate']:.1f}%")
        logger.info(f"  Avg Latency: {results['timing'].get('mean_ms', 0):.1f}ms")
        logger.info(f"  Hz: {results['timing'].get('hz', 0):.1f}")
        logger.info(f"  Reuse Rate: {results['reuse_stats']['reuse_rate']:.1f}%")

        del policy
        torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + "="*80)
    print("VLA-Cache Comparison Results")
    print("="*80)
    print(f"{'Mode':<15} {'Accuracy':<12} {'Latency':<12} {'Hz':<8} {'Reuse Rate':<12}")
    print("-"*80)

    for mode, results in all_results.items():
        timing = results.get('timing', {})
        reuse = results.get('reuse_stats', {})
        print(f"{mode:<15} "
              f"{results['overall_success_rate']:.1f}%{'':<7} "
              f"{timing.get('mean_ms', 0):.1f}ms{'':<6} "
              f"{timing.get('hz', 0):.1f}{'':<4} "
              f"{reuse.get('reuse_rate', 0):.1f}%")

    print("="*80)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="VLA-Cache Comparison Evaluation")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="/root/.cache/openpi/checkpoints/pi05_libero")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["no_reuse", "full_reuse", "all"])
    parser.add_argument("--similarity_threshold", type=float, default=0.98)
    parser.add_argument("--denoising_steps", type=int, default=10)
    parser.add_argument("--schedule_type", type=str, default="linear")
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # Additional args needed by run_episode
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--replan_steps", type=int, default=10)
    parser.add_argument("--num_steps_wait", type=int, default=10)

    args = parser.parse_args()

    if args.quick:
        args.num_tasks = 3
        args.num_trials = 3

    results = run_vla_cache_evaluation(args)

    # Save results
    if args.output_file:
        output_path = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"vla_cache_results_{timestamp}.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
