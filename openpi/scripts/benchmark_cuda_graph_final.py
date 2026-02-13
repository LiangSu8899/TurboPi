#!/usr/bin/env python3
"""
Final CUDA Graph benchmark - compare with and without CUDA Graph optimization.

Usage:
    docker exec turbo_pi_eval python /workspace/scripts/benchmark_cuda_graph_final.py

Author: Turbo-Pi Team
Date: 2026-02-13
"""

import sys
import os
import time

os.environ["OPENPI_SKIP_TVM"] = "1"

sys.path.insert(0, "/workspace/src")
os.chdir("/workspace")

import torch
import numpy as np

# Disable cuDNN to avoid Thor SM110 compatibility issues
# This is a workaround for intermittent cuDNN sublibrary loading failures
torch.backends.cudnn.enabled = False


def benchmark_standard_pytorch():
    """Benchmark standard PyTorch inference."""
    print("=" * 60)
    print("Benchmark: Standard PyTorch Backend")
    print("=" * 60)

    from openpi.inference.unified_policy import UnifiedPolicy

    policy = UnifiedPolicy(
        checkpoint_dir="/root/.cache/openpi/pytorch_checkpoints/pi05_libero",
        backend="pytorch",
        num_denoising_steps=10,
        device="cuda",
    )
    policy.warmup(num_iterations=3)

    test_obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "pick up the red block",
    }

    # Warmup
    for _ in range(5):
        _ = policy.infer(test_obs)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = policy.infer(test_obs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    mean_ms = np.mean(times)
    std_ms = np.std(times)

    print(f"\nStandard PyTorch:")
    print(f"  Mean: {mean_ms:.2f} ms ({1000/mean_ms:.2f} Hz)")
    print(f"  Std:  {std_ms:.2f} ms")

    return mean_ms


def benchmark_with_cuda_graph():
    """Benchmark with CUDA Graph optimization."""
    print("\n" + "=" * 60)
    print("Benchmark: PyTorch + CUDA Graph")
    print("=" * 60)

    from openpi.inference.unified_policy import UnifiedPolicy
    from openpi.modules.graphed_denoise import ChainedDenoiseGraphs

    policy = UnifiedPolicy(
        checkpoint_dir="/root/.cache/openpi/pytorch_checkpoints/pi05_libero",
        backend="pytorch",
        num_denoising_steps=10,
        device="cuda",
    )
    policy.warmup(num_iterations=3)

    model = policy.backend.model
    device = torch.device("cuda")

    # Create test observation
    test_obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "pick up the red block",
    }

    # Process observation
    obs = policy.backend._preprocess(test_obs)

    # Compute prefix KV cache
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(obs, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_kv_cache = model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)

    # Create CUDA Graph wrapper
    graphed = ChainedDenoiseGraphs(
        model=model,
        num_steps=10,
        device=device,
    )

    # Capture graphs
    print("Capturing CUDA Graphs...")
    graphed.capture(
        state=state,
        prefix_kv_cache=prefix_kv_cache,
        prefix_pad_masks=prefix_pad_masks,
        warmup_iters=3,
    )
    print("  Done!")

    # Warmup
    noise = torch.randn(
        1, model.config.action_horizon, model.config.action_dim,
        device=device, dtype=torch.bfloat16
    )
    for _ in range(5):
        with torch.no_grad():
            _ = graphed(noise)
    torch.cuda.synchronize()

    # Benchmark denoise only (with graph)
    times_denoise = []
    for _ in range(30):
        noise.normal_()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = graphed(noise)
        torch.cuda.synchronize()
        times_denoise.append((time.perf_counter() - start) * 1000)

    mean_denoise = np.mean(times_denoise)

    print(f"\nDenoise with CUDA Graph:")
    print(f"  Mean: {mean_denoise:.2f} ms")

    # Benchmark full pipeline (vision + KV cache + graphed denoise)
    times_full = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Process observation
        obs = policy.backend._preprocess(test_obs)
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(obs, train=False)

        # Embed prefix (vision + language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Compute KV cache
        prefix_kv_cache = model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)

        # Update KV cache in graphed module
        graphed.update_kv_cache(prefix_kv_cache)

        # Run graphed denoise
        noise.normal_()
        with torch.no_grad():
            actions = graphed(noise)

        torch.cuda.synchronize()
        times_full.append((time.perf_counter() - start) * 1000)

    mean_full = np.mean(times_full)
    std_full = np.std(times_full)

    print(f"\nFull Pipeline with CUDA Graph:")
    print(f"  Mean: {mean_full:.2f} ms ({1000/mean_full:.2f} Hz)")
    print(f"  Std:  {std_full:.2f} ms")

    return mean_full, mean_denoise


def main():
    print("=" * 60)
    print("CUDA Graph Final Benchmark")
    print("=" * 60)

    props = torch.cuda.get_device_properties(0)
    print(f"\nDevice: {props.name}")
    print(f"Compute: {props.major}.{props.minor}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

    try:
        # Benchmark standard PyTorch
        pytorch_ms = benchmark_standard_pytorch()

        # Benchmark with CUDA Graph
        graph_full_ms, graph_denoise_ms = benchmark_with_cuda_graph()

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"""
  Standard PyTorch:       {pytorch_ms:.2f} ms ({1000/pytorch_ms:.2f} Hz)
  With CUDA Graph:        {graph_full_ms:.2f} ms ({1000/graph_full_ms:.2f} Hz)

  Improvement:            {pytorch_ms - graph_full_ms:.2f} ms ({(pytorch_ms - graph_full_ms) / pytorch_ms * 100:.1f}%)

  Denoise (graphed):      {graph_denoise_ms:.2f} ms
""")

        if graph_full_ms < pytorch_ms:
            print("  ✓ CUDA Graph optimization is effective!")
        else:
            print("  ⚠ CUDA Graph did not improve performance.")

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
