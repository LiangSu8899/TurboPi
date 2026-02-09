#!/usr/bin/env python3
"""
Test torch.compile on Jetson Thor for Pi0.5 denoising.

This script tests whether torch.compile provides speedup on the Thor platform.
Thor uses Blackwell architecture (SM 11.0) which requires careful testing.

Usage:
    python scripts/test_torch_compile.py
"""

import sys
import os
import time
import json
import logging
import pathlib

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_torch_compile_support():
    """Check if torch.compile is supported on this platform."""
    print("=" * 60)
    print("TORCH.COMPILE COMPATIBILITY CHECK")
    print("=" * 60)

    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

    # Check if torch.compile is available
    try:
        # Simple function to compile
        def simple_fn(x):
            return x * 2 + 1

        compiled_fn = torch.compile(simple_fn, mode="default")
        test_input = torch.randn(10, device="cuda" if torch.cuda.is_available() else "cpu")
        _ = compiled_fn(test_input)
        print("\ntorch.compile: SUPPORTED ✓")
        return True
    except Exception as e:
        print(f"\ntorch.compile: NOT SUPPORTED ✗")
        print(f"Error: {e}")
        return False


def benchmark_denoise_step(model, observation, num_warmup=5, num_trials=20):
    """Benchmark a single denoising step."""
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.sample_actions(device, observation, num_steps=10)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_trials):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model.sample_actions(device, observation, num_steps=10)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
    }


def test_compile_modes():
    """Test different torch.compile modes."""
    print("\n" + "=" * 60)
    print("TORCH.COMPILE MODE COMPARISON")
    print("=" * 60)

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config, Observation
    from safetensors.torch import load_file

    device = "cuda"
    checkpoint_dir = pathlib.Path.home() / ".cache/openpi/checkpoints/pi05_libero"

    # Load model
    config_path = checkpoint_dir / "config.json"
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

    model = PI0Pytorch(pi0_config)
    weights_path = checkpoint_dir / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device)
    model.eval()

    logger.info("Model loaded successfully")

    # Create dummy observation
    dummy_obs = Observation(
        images={
            "base_0_rgb": torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16),
            "left_wrist_0_rgb": torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16),
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, device=device, dtype=torch.bfloat16) - 1,
        },
        image_masks={
            "base_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(1, device=device, dtype=torch.bool),
        },
        state=torch.randn(1, 32, device=device, dtype=torch.bfloat16),
        tokenized_prompt=torch.ones(1, 200, device=device, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(1, 200, device=device, dtype=torch.bool),
    )

    # Test baseline (no compile)
    print("\n[1/4] Testing BASELINE (no compile)...")
    baseline_stats = benchmark_denoise_step(model, dummy_obs)
    print(f"  Mean: {baseline_stats['mean_ms']:.2f} ms")
    print(f"  Std:  {baseline_stats['std_ms']:.2f} ms")

    # Test different compile modes
    compile_modes = ["default", "reduce-overhead"]  # Skip max-autotune (too slow)

    results = {"baseline": baseline_stats}

    for i, mode in enumerate(compile_modes, start=2):
        print(f"\n[{i}/4] Testing mode='{mode}'...")

        try:
            # Reload model to avoid interference
            model_compiled = PI0Pytorch(pi0_config)
            model_compiled.load_state_dict(state_dict, strict=False)
            model_compiled = model_compiled.to(device=device)
            model_compiled.eval()

            # Compile denoise_step_with_cache
            model_compiled.denoise_step_with_cache = torch.compile(
                model_compiled.denoise_step_with_cache,
                mode=mode,
                fullgraph=False,
            )

            print(f"  Compiling (this may take a few minutes for first run)...")

            # First run triggers compilation
            torch.cuda.synchronize()
            compile_start = time.perf_counter()
            with torch.no_grad():
                _ = model_compiled.sample_actions(device, dummy_obs, num_steps=10)
            torch.cuda.synchronize()
            compile_time = time.perf_counter() - compile_start
            print(f"  First run (incl. compilation): {compile_time:.2f}s")

            # Benchmark
            stats = benchmark_denoise_step(model_compiled, dummy_obs)
            speedup = baseline_stats['mean_ms'] / stats['mean_ms']

            print(f"  Mean: {stats['mean_ms']:.2f} ms (speedup: {speedup:.2f}x)")
            print(f"  Std:  {stats['std_ms']:.2f} ms")

            results[mode] = {**stats, "speedup": speedup, "compile_time": compile_time}

        except Exception as e:
            print(f"  FAILED: {e}")
            results[mode] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Mode':<20} | {'Latency (ms)':<15} | {'Speedup':<10}")
    print("-" * 50)

    print(f"{'baseline':<20} | {baseline_stats['mean_ms']:.2f} ms        | 1.00x")

    for mode in compile_modes:
        if mode in results and "error" not in results[mode]:
            r = results[mode]
            print(f"{mode:<20} | {r['mean_ms']:.2f} ms        | {r['speedup']:.2f}x")
        elif mode in results:
            print(f"{mode:<20} | ERROR: {results[mode]['error'][:30]}...")

    return results


def test_compile_components():
    """Test compiling individual components."""
    print("\n" + "=" * 60)
    print("COMPONENT-LEVEL COMPILATION TEST")
    print("=" * 60)

    # Test compiling smaller functions first
    components = [
        ("embed_suffix", "Suffix embedding (timestep + action)"),
        ("embed_prefix", "Prefix embedding (vision + language)"),
        ("denoise_step_with_cache", "Full denoise step with KV cache"),
    ]

    print("\nThis test compiles individual components to identify compatibility issues.")
    print("Components will be tested in order of complexity.\n")

    for comp_name, description in components:
        print(f"Testing: {comp_name}")
        print(f"  Description: {description}")

        try:
            # Create a simple test
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config

            config = Pi0Config(dtype="bfloat16")
            model = PI0Pytorch(config)
            model = model.to("cuda")
            model.eval()

            # Try to compile the component
            component = getattr(model, comp_name)
            compiled = torch.compile(component, mode="default", fullgraph=False)

            print(f"  Status: COMPILED ✓")

        except Exception as e:
            print(f"  Status: FAILED ✗")
            print(f"  Error: {str(e)[:100]}...")

        print()


def main():
    print("=" * 60)
    print("TORCH.COMPILE TEST FOR JETSON THOR")
    print("=" * 60)

    # Check basic support
    if not check_torch_compile_support():
        print("\ntorch.compile is not supported on this platform.")
        print("Skipping further tests.")
        return

    # Test component compilation
    test_compile_components()

    # Full benchmark (if model is available)
    checkpoint_dir = pathlib.Path.home() / ".cache/openpi/checkpoints/pi05_libero"
    if (checkpoint_dir / "model.safetensors").exists():
        test_compile_modes()
    else:
        print(f"\nModel not found at {checkpoint_dir}")
        print("Skipping full benchmark. Run with model checkpoint for complete test.")


if __name__ == "__main__":
    main()
