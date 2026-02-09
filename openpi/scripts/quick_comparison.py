#!/usr/bin/env python3
"""Quick comparison of baseline vs FP16 no-bounce engine."""

import time
import torch
import numpy as np
import json
import logging
from pathlib import Path
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load model
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config

    checkpoint_dir = '/root/.cache/openpi/checkpoints/pi05_libero'
    config_path = Path(checkpoint_dir) / 'config.json'
    with open(config_path) as f:
        model_config = json.load(f)
    config = Pi0Config(
        action_dim=model_config.get('action_dim', 32),
        action_horizon=model_config.get('action_horizon', 50),
    )

    model = PI0Pytorch(config)
    weights_path = Path(checkpoint_dir) / 'model.safetensors'
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to('cuda').eval()

    # Create test data
    batch_size = 1
    prefix_len = 968
    num_layers = 18
    device = 'cuda'

    prefix_kv_cache = []
    for _ in range(num_layers):
        k = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        prefix_kv_cache.append((k, v))

    prefix_pad_masks = torch.ones(batch_size, prefix_len, device=device, dtype=torch.bool)
    state = torch.randn(batch_size, 32, device=device, dtype=torch.bfloat16)

    num_steps = 10
    num_iterations = 20

    # Warmup
    logger.info("Warming up baseline...")
    for _ in range(3):
        x_t = torch.randn(1, config.action_horizon, config.action_dim, device='cuda', dtype=torch.bfloat16)
        dt = torch.tensor(-1.0 / num_steps, device='cuda')
        time_val = torch.tensor([1.0], device='cuda')
        for _ in range(num_steps):
            v_t = model.denoise_step_with_cache(state, prefix_kv_cache, prefix_pad_masks, x_t, time_val)
            x_t = x_t + dt * v_t
            time_val = time_val + dt
    torch.cuda.synchronize()

    # Benchmark baseline
    logger.info("Benchmarking baseline...")
    times = []
    for _ in range(num_iterations):
        x_t = torch.randn(1, config.action_horizon, config.action_dim, device='cuda', dtype=torch.bfloat16)
        dt = torch.tensor(-1.0 / num_steps, device='cuda')
        time_val = torch.tensor([1.0], device='cuda')

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_steps):
            v_t = model.denoise_step_with_cache(state, prefix_kv_cache, prefix_pad_masks, x_t, time_val)
            x_t = x_t + dt * v_t
            time_val = time_val + dt
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    baseline_ms = np.mean(times)
    logger.info(f"Baseline: {baseline_ms:.2f} ± {np.std(times):.2f} ms ({1000/baseline_ms:.1f} Hz)")

    # FP16 Engine
    from openpi.inference.fp8_no_bounce import FP16DenoiseEngine
    fp16_engine = FP16DenoiseEngine(model, device='cuda')

    # Warmup
    logger.info("Warming up FP16 no-bounce...")
    for _ in range(3):
        x_t = torch.randn(1, config.action_horizon, config.action_dim, device='cuda', dtype=torch.bfloat16)
        dt = torch.tensor(-1.0 / num_steps, device='cuda')
        time_val = torch.tensor([1.0], device='cuda')
        for _ in range(num_steps):
            v_t = fp16_engine.denoise_step_with_cache(state, prefix_kv_cache, prefix_pad_masks, x_t, time_val)
            x_t = x_t + dt * v_t
            time_val = time_val + dt
    torch.cuda.synchronize()

    # Benchmark FP16 no-bounce
    logger.info("Benchmarking FP16 no-bounce...")
    times = []
    for _ in range(num_iterations):
        x_t = torch.randn(1, config.action_horizon, config.action_dim, device='cuda', dtype=torch.bfloat16)
        dt = torch.tensor(-1.0 / num_steps, device='cuda')
        time_val = torch.tensor([1.0], device='cuda')

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_steps):
            v_t = fp16_engine.denoise_step_with_cache(state, prefix_kv_cache, prefix_pad_masks, x_t, time_val)
            x_t = x_t + dt * v_t
            time_val = time_val + dt
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    fp16_ms = np.mean(times)
    logger.info(f"FP16 No-Bounce: {fp16_ms:.2f} ± {np.std(times):.2f} ms ({1000/fp16_ms:.1f} Hz)")

    # Results
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Baseline (BF16):    {baseline_ms:.2f} ms ({1000/baseline_ms:.1f} Hz)")
    print(f"FP16 No-Bounce:     {fp16_ms:.2f} ms ({1000/fp16_ms:.1f} Hz)")
    print(f"Speedup:            {baseline_ms/fp16_ms:.2f}x")

if __name__ == "__main__":
    main()
