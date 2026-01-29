#!/usr/bin/env python3
"""
Performance Benchmark Script for Pi0.5 VLA Model on Jetson Thor

This script measures inference performance metrics including:
- Throughput (Hz)
- Latency per denoising step (ms)
- Vision encoder latency (ms)
- Memory usage (GB)

Usage:
    python scripts/benchmark_thor.py --model_path ~/.cache/openpi/checkpoints/pi05_libero --num_runs 100
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Add openpi to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_path: str = "~/.cache/openpi/checkpoints/pi05_libero"
    num_warmup: int = 3
    num_runs: int = 50
    batch_size: int = 1
    num_images: int = 2  # Number of camera views
    image_size: tuple = (224, 224)
    num_denoising_steps: int = 10
    action_horizon: int = 50
    action_dim: int = 32  # LIBERO uses 32
    state_dim: int = 32  # LIBERO uses 32
    precision: str = "float32"  # float32, float16, bfloat16
    device: str = "auto"  # auto, cuda, cpu
    profile_memory: bool = True
    save_results: bool = True
    output_dir: str = "benchmark_results"


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""
    # Timing metrics
    total_times: list = field(default_factory=list)
    vision_encoder_times: list = field(default_factory=list)
    denoise_step_times: list = field(default_factory=list)

    # Memory metrics
    peak_memory_gb: float = 0.0
    allocated_memory_gb: float = 0.0

    # Computed metrics
    throughput_hz: float = 0.0
    avg_latency_ms: float = 0.0
    avg_vision_latency_ms: float = 0.0
    avg_denoise_step_ms: float = 0.0

    # Configuration
    device: str = ""
    precision: str = ""
    num_denoising_steps: int = 10
    batch_size: int = 1

    def compute_statistics(self):
        """Compute summary statistics from collected timing data."""
        if self.total_times:
            avg_time = np.mean(self.total_times)
            self.throughput_hz = 1.0 / avg_time if avg_time > 0 else 0.0
            self.avg_latency_ms = avg_time * 1000

        if self.vision_encoder_times:
            self.avg_vision_latency_ms = np.mean(self.vision_encoder_times) * 1000

        if self.denoise_step_times:
            self.avg_denoise_step_ms = np.mean(self.denoise_step_times) * 1000

    def to_dict(self):
        """Convert results to dictionary for JSON serialization."""
        return {
            "throughput_hz": self.throughput_hz,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_vision_latency_ms": self.avg_vision_latency_ms,
            "avg_denoise_step_ms": self.avg_denoise_step_ms,
            "peak_memory_gb": self.peak_memory_gb,
            "allocated_memory_gb": self.allocated_memory_gb,
            "device": self.device,
            "precision": self.precision,
            "num_denoising_steps": self.num_denoising_steps,
            "batch_size": self.batch_size,
            "num_runs": len(self.total_times),
            "total_times": self.total_times,
            "vision_encoder_times": self.vision_encoder_times,
            "denoise_step_times": self.denoise_step_times,
        }


def get_device(config: BenchmarkConfig) -> torch.device:
    """Determine the device to use for benchmarking."""
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(config.device)


def get_dtype(precision: str) -> torch.dtype:
    """Get torch dtype from precision string."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(precision, torch.float32)


def create_dummy_observation(config: BenchmarkConfig, device: torch.device, dtype: torch.dtype):
    """Create dummy observation data for benchmarking."""
    from openpi.models_pytorch.pi0_pytorch import Observation

    batch_size = config.batch_size

    # Use LIBERO camera names
    image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

    # Create dummy images (B, C, H, W format)
    images = {}
    image_masks = {}
    for i, key in enumerate(image_keys[:config.num_images]):
        images[key] = torch.randn(
            batch_size, 3, config.image_size[0], config.image_size[1],
            device=device, dtype=dtype
        )
        image_masks[key] = torch.ones(batch_size, device=device, dtype=torch.bool)

    # Add third camera with mask=False if num_images is 2
    if config.num_images == 2:
        key = image_keys[2]
        images[key] = torch.zeros(batch_size, 3, config.image_size[0], config.image_size[1],
                                   device=device, dtype=dtype)
        image_masks[key] = torch.zeros(batch_size, device=device, dtype=torch.bool)

    # Create dummy state
    state = torch.randn(batch_size, config.state_dim, device=device, dtype=dtype)

    # Create dummy tokenized prompt
    max_token_len = 200
    tokenized_prompt = torch.zeros(batch_size, max_token_len, device=device, dtype=torch.long)
    tokenized_prompt_mask = torch.ones(batch_size, max_token_len, device=device, dtype=torch.bool)

    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )


def load_model(config: BenchmarkConfig, device: torch.device, dtype: torch.dtype):
    """Load the Pi0.5 model for benchmarking."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    model_path = Path(config.model_path).expanduser()

    # Support multiple weight formats
    weights_path = None
    for candidate in ["model.safetensors", "model_quantized.pt", "model.pt"]:
        p = model_path / candidate
        if p.exists():
            weights_path = p
            break

    if weights_path is None:
        raise FileNotFoundError(f"No model weights found in {model_path}")

    # Support multiple config formats
    config_path = model_path / "config.json"
    if not config_path.exists():
        config_path = model_path / "quant_config.json"

    # Load config with defaults
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
    else:
        print(f"  Warning: No config file found, using defaults")
        model_config = {}

    print(f"Loading model from {model_path}")
    print(f"  - Weights: {weights_path.name}")
    print(f"  - Variant: {model_config.get('paligemma_variant', 'gemma_2b')}")
    print(f"  - Action Expert: {model_config.get('action_expert_variant', 'gemma_300m')}")
    print(f"  - Precision: {config.precision}")

    # Create model configuration
    from openpi.models_pytorch.pi0_pytorch import Pi0Config

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        max_token_len=model_config.get("tokenizer_max_length", 200),
        max_state_dim=model_config.get("max_state_dim", 32),
        pi05=True,
        dtype=config.precision,  # Pass string not torch.dtype
    )

    # Create model
    model = PI0Pytorch(pi0_config)

    # Load weights based on format
    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"  - Missing keys: {len(missing)}")
    if unexpected:
        print(f"  - Unexpected keys: {len(unexpected)}")

    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model


def benchmark_inference(
    model,
    observation,
    config: BenchmarkConfig,
    device: torch.device,
    results: BenchmarkResults,
):
    """Run inference benchmark."""
    print(f"\nRunning {config.num_runs} inference iterations...")
    print(f"  - Warmup: {config.num_warmup} iterations")
    print(f"  - Denoising steps: {config.num_denoising_steps}")

    # Warmup
    with torch.no_grad():
        for i in range(config.num_warmup):
            _ = model.sample_actions(
                device,
                observation,
                num_steps=config.num_denoising_steps,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Reset memory stats after warmup
    if device.type == "cuda" and config.profile_memory:
        torch.cuda.reset_peak_memory_stats()

    # Benchmark runs
    with torch.no_grad():
        for i in range(config.num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            actions = model.sample_actions(
                device,
                observation,
                num_steps=config.num_denoising_steps,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            elapsed = end_time - start_time
            results.total_times.append(elapsed)

            if (i + 1) % 10 == 0:
                current_hz = 1.0 / elapsed
                print(f"  Iteration {i + 1}/{config.num_runs}: {current_hz:.2f} Hz ({elapsed * 1000:.1f} ms)")

    # Record memory usage
    if device.type == "cuda" and config.profile_memory:
        results.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        results.allocated_memory_gb = torch.cuda.memory_allocated() / 1e9

    return results


def benchmark_vision_encoder(
    model,
    observation,
    config: BenchmarkConfig,
    device: torch.device,
    results: BenchmarkResults,
):
    """Benchmark vision encoder separately."""
    print(f"\nBenchmarking Vision Encoder...")

    with torch.no_grad():
        for i in range(config.num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            # Run only the prefix embedding (Vision Encoder)
            # Note: embed_prefix expects (images, img_masks, lang_tokens, lang_masks)
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                observation.images,
                observation.image_masks,
                observation.tokenized_prompt,
                observation.tokenized_prompt_mask,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            results.vision_encoder_times.append(end_time - start_time)

    return results


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """Run the complete benchmark suite."""
    print("=" * 60)
    print("Pi0.5 VLA Model Performance Benchmark")
    print("=" * 60)

    # Setup
    device = get_device(config)
    dtype = get_dtype(config.precision)

    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Precision: {config.precision} ({dtype})")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Model path: {config.model_path}")

    # Initialize results
    results = BenchmarkResults(
        device=str(device),
        precision=config.precision,
        num_denoising_steps=config.num_denoising_steps,
        batch_size=config.batch_size,
    )

    # Load model
    try:
        model = load_model(config, device, dtype)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Ensure the model is downloaded to the specified path.")
        return results

    # Create dummy observation
    observation = create_dummy_observation(config, device, dtype)

    # Run benchmarks
    try:
        results = benchmark_inference(model, observation, config, device, results)
    except Exception as e:
        print(f"\nError during inference benchmark: {e}")
        import traceback
        traceback.print_exc()
        return results

    # Vision encoder benchmark is optional
    try:
        results = benchmark_vision_encoder(model, observation, config, device, results)
    except Exception as e:
        print(f"\nSkipping vision encoder benchmark: {e}")

    # Compute statistics
    results.compute_statistics()

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nThroughput: {results.throughput_hz:.3f} Hz")
    print(f"Average Latency: {results.avg_latency_ms:.1f} ms")
    print(f"Vision Encoder Latency: {results.avg_vision_latency_ms:.1f} ms")

    if device.type == "cuda":
        print(f"\nMemory Usage:")
        print(f"  Peak: {results.peak_memory_gb:.2f} GB")
        print(f"  Allocated: {results.allocated_memory_gb:.2f} GB")

    # Save results
    if config.save_results:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_{device.type}_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pi0.5 VLA Model on Jetson Thor")

    parser.add_argument(
        "--model_path",
        type=str,
        default="~/.cache/openpi/checkpoints/pi05_libero",
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=50,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=10,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model precision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        model_path=args.model_path,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        num_denoising_steps=args.num_denoising_steps,
        precision=args.precision,
        device=args.device,
        output_dir=args.output_dir,
    )

    results = run_benchmark(config)

    # Return exit code based on throughput
    # Phase 1 target: 3-4 Hz
    if results.throughput_hz >= 3.0:
        print("\n[PASS] Phase 1 target (3-4 Hz) achieved!")
        return 0
    elif results.throughput_hz > 0:
        print(f"\n[INFO] Current throughput: {results.throughput_hz:.3f} Hz")
        print("       Phase 1 target: 3-4 Hz")
        return 0
    else:
        print("\n[FAIL] Benchmark failed to produce valid results")
        return 1


if __name__ == "__main__":
    sys.exit(main())
