#!/usr/bin/env python3
"""Profile inference to identify bottlenecks."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

def profile():
    print("=" * 60)
    print("Pi0.5 Inference Profiling")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation

    config = Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        max_state_dim=32,
        pi05=True,
        dtype="bfloat16",
    )

    model = PI0Pytorch(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()

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
    with torch.no_grad():
        for _ in range(3):
            _ = model.sample_actions(device, observation, num_steps=1)
            torch.cuda.synchronize()

    # Profile individual components
    print("\n" + "-" * 60)
    print("Component Breakdown (single step inference)")
    print("-" * 60)

    with torch.no_grad():
        # 1. Preprocessing
        torch.cuda.synchronize()
        start = time.perf_counter()
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
        torch.cuda.synchronize()
        preprocess_time = time.perf_counter() - start
        print(f"  Preprocessing: {preprocess_time * 1000:.1f} ms")

        # 2. Prefix embedding (Vision + Language)
        torch.cuda.synchronize()
        start = time.perf_counter()
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        torch.cuda.synchronize()
        prefix_time = time.perf_counter() - start
        print(f"  Prefix embedding (Vision+Lang): {prefix_time * 1000:.1f} ms")

        # 3. Single denoise step
        noise = model.sample_noise((batch_size, 50, 32), device)
        x_t = noise.to(dtype)
        timestep = torch.tensor([1.0], device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        v_t = model.denoise_step_no_cache(
            state, prefix_embs, prefix_pad_masks, prefix_att_masks, x_t, timestep
        )
        torch.cuda.synchronize()
        denoise_time = time.perf_counter() - start
        print(f"  Denoise step: {denoise_time * 1000:.1f} ms")

    print("-" * 60)
    print(f"  Total (estimated): {(preprocess_time + prefix_time + denoise_time) * 1000:.1f} ms")

    # Profile with CUDA events for more accurate timing
    print("\n" + "-" * 60)
    print("CUDA Events Profiling (10 denoise steps)")
    print("-" * 60)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        actions = model.sample_actions(device, observation, num_steps=10)
        end_event.record()
        torch.cuda.synchronize()

    elapsed = start_event.elapsed_time(end_event)
    print(f"  Total (10 steps): {elapsed:.1f} ms")
    print(f"  Per step: {elapsed / 10:.1f} ms")
    print(f"  Throughput: {1000 / elapsed:.2f} Hz")

    # Check PyTorch profiler
    print("\n" + "-" * 60)
    print("PyTorch Profiler (top 10 ops)")
    print("-" * 60)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=1)
            torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    print(table)


if __name__ == "__main__":
    profile()
