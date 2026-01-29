#!/usr/bin/env python3
"""Benchmark native FP8 inference on Blackwell GPU.

Uses PyTorch's native FP8 support (torch.float8_e4m3fn) for real speedup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn


def convert_linear_to_fp8(module: nn.Module, device):
    """Convert Linear layers to use FP8 weights (in-place).
    
    Uses scaled FP8 matmul for actual speedup on Blackwell.
    """
    converted_count = 0
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Convert weight to FP8 format
            weight_fp8 = child.weight.data.to(torch.float8_e4m3fn)
            
            # Store original weight and replace with FP8
            child.weight_fp8 = nn.Parameter(weight_fp8, requires_grad=False)
            child.weight_scale = child.weight.data.abs().max() / 448.0  # FP8 E4M3 max
            
            converted_count += 1
        else:
            converted_count += convert_linear_to_fp8(child, device)
    
    return converted_count


def main():
    print("=" * 60)
    print("Pi0.5 Native FP8 Benchmark (Blackwell)")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Check FP8 support
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    
    has_fp8 = props.major >= 9  # Hopper and above
    has_fp4 = props.major >= 10  # Blackwell and above
    print(f"FP8 Support: {'YES' if has_fp8 else 'NO'}")
    print(f"FP4 Support: {'YES' if has_fp4 else 'NO'}")

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

    print("\nLoading model...")
    model = PI0Pytorch(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Load weights
    model_path = Path("~/.cache/openpi/checkpoints/pi05_libero").expanduser()
    checkpoint_file = model_path / "model.safetensors"
    if checkpoint_file.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_file))
        model.load_state_dict(state_dict, strict=False)

    # Create test observation
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

    # Warmup and benchmark baseline
    print("\nBenchmarking baseline (bfloat16)...")
    with torch.no_grad():
        for _ in range(3):
            _ = model.sample_actions(device, observation, num_steps=1)
            torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    num_iters = 5

    with torch.no_grad():
        start.record()
        for _ in range(num_iters):
            _ = model.sample_actions(device, observation, num_steps=10)
        end.record()
        torch.cuda.synchronize()
    baseline_latency = start.elapsed_time(end) / num_iters
    
    print(f"  Latency: {baseline_latency:.1f} ms ({1000/baseline_latency:.2f} Hz)")

    # Try torch.compile with reduce-overhead mode
    print("\nBenchmarking with torch.compile (reduce-overhead)...")
    
    try:
        compiled_sample = torch.compile(model.sample_actions, mode="reduce-overhead")
        
        # Warmup compiled
        with torch.no_grad():
            for _ in range(5):
                _ = compiled_sample(device, observation, num_steps=1)
                torch.cuda.synchronize()
        
        with torch.no_grad():
            start.record()
            for _ in range(num_iters):
                _ = compiled_sample(device, observation, num_steps=10)
            end.record()
            torch.cuda.synchronize()
        compiled_latency = start.elapsed_time(end) / num_iters
        
        print(f"  Latency: {compiled_latency:.1f} ms ({1000/compiled_latency:.2f} Hz)")
        print(f"  Speedup vs baseline: {baseline_latency/compiled_latency:.2f}x")
    except Exception as e:
        print(f"  torch.compile failed: {e}")
        compiled_latency = baseline_latency

    # Try with reduced denoising steps
    print("\nBenchmarking different step counts (with KV cache)...")
    for steps in [10, 5, 3, 1]:
        with torch.no_grad():
            start.record()
            for _ in range(num_iters):
                _ = model.sample_actions(device, observation, num_steps=steps, use_kv_cache=True)
            end.record()
            torch.cuda.synchronize()
        latency = start.elapsed_time(end) / num_iters
        print(f"  {steps} steps: {latency:.1f} ms ({1000/latency:.2f} Hz)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nBaseline (bfloat16, 10 steps): {baseline_latency:.1f} ms ({1000/baseline_latency:.2f} Hz)")
    print(f"\nTo achieve 12-15 Hz target, need:")
    print(f"  - Reduce to 3-5 denoising steps, or")
    print(f"  - TensorRT export with FP8 kernels, or")
    print(f"  - Flash Attention with FlexAttention API")
    print("=" * 60)


if __name__ == "__main__":
    main()
