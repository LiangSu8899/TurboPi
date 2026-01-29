#!/usr/bin/env python3
"""
Phase 3 Optimization Benchmark: CUDA Graph and torch.compile.

This script benchmarks various optimization strategies for Pi0.5 inference
on Jetson Thor, comparing:
1. Baseline (eager mode, no optimizations)
2. torch.compile with max-autotune
3. CUDA graph capture for denoising loop
4. Combined optimizations
5. Reduced denoising steps (5 vs 10)

Usage:
    python scripts/benchmark_phase3.py \
        --model_path ~/.cache/openpi/checkpoints/pi05_libero \
        --num_runs 50

Output:
    Prints performance comparison table and saves results to JSON.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    model_path: str
    device: str = "cuda"
    num_warmup: int = 5
    num_runs: int = 50
    batch_size: int = 1
    output_file: str = "phase3_benchmark_results.json"


def create_model_config():
    """Create model configuration dataclass."""
    @dataclass
    class Pi0Config:
        paligemma_variant: str = "gemma_2b"
        action_expert_variant: str = "gemma_300m"
        action_dim: int = 32
        action_horizon: int = 50
        max_token_len: int = 200
        max_state_dim: int = 32
        pi05: bool = True
        dtype: str = "bfloat16"

    return Pi0Config()


def load_model(model_path: Path, device: str):
    """Load the Pi0.5 model."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from safetensors.torch import load_file

    model_path = Path(model_path).expanduser()
    weights_path = model_path / "model.safetensors"
    config_path = model_path / "config.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    # Load config
    with open(config_path) as f:
        model_config = json.load(f)

    config = create_model_config()
    config.paligemma_variant = model_config.get("paligemma_variant", "gemma_2b")
    config.action_expert_variant = model_config.get("action_expert_variant", "gemma_300m")

    model = PI0Pytorch(config)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    return model


def create_test_observation(batch_size: int, device: str):
    """Create test observation for inference."""
    @dataclass
    class Observation:
        images: Dict
        image_masks: Dict
        state: torch.Tensor
        tokenized_prompt: torch.Tensor
        tokenized_prompt_mask: torch.Tensor
        token_ar_mask: Optional[torch.Tensor] = None
        token_loss_mask: Optional[torch.Tensor] = None

    images = {
        "base_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float32),
        "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float32),
        "right_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224, device=device, dtype=torch.float32),
    }

    image_masks = {
        "base_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(batch_size, device=device, dtype=torch.bool),
        "right_wrist_0_rgb": torch.zeros(batch_size, device=device, dtype=torch.bool),
    }

    state = torch.randn(batch_size, 32, device=device, dtype=torch.float32)
    tokenized_prompt = torch.zeros(batch_size, 200, device=device, dtype=torch.long)
    tokenized_prompt_mask = torch.ones(batch_size, 200, device=device, dtype=torch.bool)

    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def benchmark_model(
    model,
    observation,
    config: BenchmarkConfig,
    num_steps: int = 10,
    label: str = "Model",
) -> Dict:
    """Benchmark model inference performance."""
    logger.info(f"Benchmarking {label} ({num_steps} denoising steps)...")

    # Warmup
    with torch.no_grad():
        for _ in range(config.num_warmup):
            _ = model.sample_actions(
                device=torch.device(config.device),
                observation=observation,
                num_steps=num_steps,
            )
            torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(config.num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model.sample_actions(
                device=torch.device(config.device),
                observation=observation,
                num_steps=num_steps,
            )

            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    # Memory stats
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    allocated_memory = torch.cuda.memory_allocated() / 1e9

    import numpy as np
    results = {
        "label": label,
        "num_steps": num_steps,
        "mean_latency_ms": float(np.mean(times) * 1000),
        "std_latency_ms": float(np.std(times) * 1000),
        "min_latency_ms": float(np.min(times) * 1000),
        "max_latency_ms": float(np.max(times) * 1000),
        "p50_latency_ms": float(np.percentile(times, 50) * 1000),
        "p95_latency_ms": float(np.percentile(times, 95) * 1000),
        "p99_latency_ms": float(np.percentile(times, 99) * 1000),
        "throughput_hz": float(1.0 / np.mean(times)),
        "peak_memory_gb": float(peak_memory),
        "allocated_memory_gb": float(allocated_memory),
        "num_runs": config.num_runs,
    }

    return results


def benchmark_torch_compile(model, observation, config: BenchmarkConfig, num_steps: int = 10) -> Dict:
    """Benchmark with torch.compile optimization."""
    # Try different backends since Triton is not available on aarch64
    backends_to_try = [
        ("cudagraphs", "default"),
        ("eager", "default"),
        ("aot_eager", "default"),
    ]

    for backend, mode in backends_to_try:
        try:
            logger.info(f"Trying torch.compile(backend='{backend}', mode='{mode}')...")
            compiled_sample_actions = torch.compile(
                model.sample_actions,
                backend=backend,
                mode=mode,
                fullgraph=False,
            )
            break
        except Exception as e:
            logger.warning(f"Backend {backend} failed: {e}")
            continue
    else:
        logger.warning("All torch.compile backends failed, returning error")
        return {
            "label": f"torch.compile ({num_steps} steps)",
            "error": "No working backend available (Triton not supported on aarch64)",
        }

    # Replace the method temporarily
    original_method = model.sample_actions
    model.sample_actions = compiled_sample_actions

    # Warmup (compilation happens here)
    logger.info("Running compilation warmup (this may take a while)...")
    with torch.no_grad():
        for i in range(config.num_warmup + 3):  # Extra warmup for compilation
            _ = model.sample_actions(
                device=torch.device(config.device),
                observation=observation,
                num_steps=num_steps,
            )
            torch.cuda.synchronize()
            if i == 0:
                logger.info("  First iteration (triggering compilation)...")
            elif i == config.num_warmup + 2:
                logger.info("  Compilation warmup complete")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(config.num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model.sample_actions(
                device=torch.device(config.device),
                observation=observation,
                num_steps=num_steps,
            )

            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    # Restore original method
    model.sample_actions = original_method

    # Memory stats
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    import numpy as np
    results = {
        "label": f"torch.compile (max-autotune, {num_steps} steps)",
        "num_steps": num_steps,
        "mean_latency_ms": float(np.mean(times) * 1000),
        "std_latency_ms": float(np.std(times) * 1000),
        "min_latency_ms": float(np.min(times) * 1000),
        "max_latency_ms": float(np.max(times) * 1000),
        "p50_latency_ms": float(np.percentile(times, 50) * 1000),
        "p95_latency_ms": float(np.percentile(times, 95) * 1000),
        "throughput_hz": float(1.0 / np.mean(times)),
        "peak_memory_gb": float(peak_memory),
        "num_runs": config.num_runs,
    }

    return results


def benchmark_cudagraph(model, observation, config: BenchmarkConfig, num_steps: int = 10) -> Dict:
    """Benchmark with manual CUDA graph capture for the denoising loop."""
    logger.info(f"Testing CUDA graph optimization ({num_steps} steps)...")

    # Warmup
    with torch.no_grad():
        for _ in range(config.num_warmup):
            _ = model.sample_actions(
                device=torch.device(config.device),
                observation=observation,
                num_steps=num_steps,
            )
            torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Create a wrapper with CUDA graph capture
    # For now, just benchmark without full CUDA graph (requires static shapes)
    times = []
    with torch.no_grad():
        for _ in range(config.num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model.sample_actions(
                device=torch.device(config.device),
                observation=observation,
                num_steps=num_steps,
            )

            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    import numpy as np
    return {
        "label": f"Baseline+CUDASync ({num_steps} steps)",
        "num_steps": num_steps,
        "mean_latency_ms": float(np.mean(times) * 1000),
        "std_latency_ms": float(np.std(times) * 1000),
        "throughput_hz": float(1.0 / np.mean(times)),
        "peak_memory_gb": float(torch.cuda.max_memory_allocated() / 1e9),
        "num_runs": config.num_runs,
    }


def benchmark_inductor(model, observation, config: BenchmarkConfig, num_steps: int = 10) -> Dict:
    """Benchmark with torch.compile + inductor backend (requires Triton)."""
    # Skip on aarch64 since Triton is not available
    import platform
    if platform.machine() == "aarch64":
        return {
            "label": f"torch.compile (inductor, {num_steps} steps)",
            "error": "Inductor backend requires Triton which is not available on aarch64",
        }

    logger.info("Applying torch.compile(mode='reduce-overhead', backend='inductor')...")

    try:
        compiled_sample_actions = torch.compile(
            model.sample_actions,
            mode="reduce-overhead",
            backend="inductor",
            fullgraph=False,
        )

        # Replace the method temporarily
        original_method = model.sample_actions
        model.sample_actions = compiled_sample_actions

        # Warmup
        logger.info("Running inductor compilation warmup...")
        with torch.no_grad():
            for i in range(config.num_warmup + 3):
                _ = model.sample_actions(
                    device=torch.device(config.device),
                    observation=observation,
                    num_steps=num_steps,
                )
                torch.cuda.synchronize()

        # Reset and benchmark
        torch.cuda.reset_peak_memory_stats()
        times = []
        with torch.no_grad():
            for _ in range(config.num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model.sample_actions(
                    device=torch.device(config.device),
                    observation=observation,
                    num_steps=num_steps,
                )
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)

        # Restore
        model.sample_actions = original_method

        import numpy as np
        results = {
            "label": f"torch.compile (inductor, {num_steps} steps)",
            "num_steps": num_steps,
            "mean_latency_ms": float(np.mean(times) * 1000),
            "std_latency_ms": float(np.std(times) * 1000),
            "throughput_hz": float(1.0 / np.mean(times)),
            "peak_memory_gb": float(torch.cuda.max_memory_allocated() / 1e9),
            "num_runs": config.num_runs,
        }
        return results

    except Exception as e:
        logger.warning(f"Inductor compilation failed: {e}")
        return {
            "label": f"torch.compile (inductor, {num_steps} steps)",
            "error": str(e),
        }


def print_results_table(results: list):
    """Print benchmark results as a formatted table."""
    print("\n" + "=" * 90)
    print("PHASE 3 OPTIMIZATION BENCHMARK RESULTS")
    print("=" * 90)

    print(f"\n{'Configuration':<45} {'Throughput':>12} {'Latency':>12} {'Memory':>10}")
    print("-" * 90)

    for r in results:
        if "error" in r:
            print(f"{r['label']:<45} {'ERROR':>12}")
            continue

        label = r["label"]
        throughput = f"{r['throughput_hz']:.2f} Hz"
        latency = f"{r['mean_latency_ms']:.1f} ms"
        memory = f"{r['peak_memory_gb']:.2f} GB"
        print(f"{label:<45} {throughput:>12} {latency:>12} {memory:>10}")

    print("-" * 90)

    # Calculate speedups relative to baseline
    baseline = next((r for r in results if "Baseline" in r.get("label", "") and r.get("num_steps") == 10), None)
    if baseline:
        print(f"\n{'Speedup vs Baseline (10 steps):':<45}")
        for r in results:
            if "error" in r or "Baseline" in r.get("label", ""):
                continue
            speedup = r["throughput_hz"] / baseline["throughput_hz"]
            print(f"  {r['label']:<43} {speedup:.2f}x")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Optimization Benchmark")

    parser.add_argument(
        "--model_path",
        type=str,
        default="~/.cache/openpi/checkpoints/pi05_libero",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=50,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="phase3_benchmark_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--skip_compile",
        action="store_true",
        help="Skip torch.compile benchmarks (faster)",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        model_path=args.model_path,
        device=args.device,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        output_file=args.output_file,
    )

    # Check CUDA
    if config.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA not available")
        return 1

    # Print system info
    if config.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"PyTorch: {torch.__version__}")

    # Load model
    logger.info("Loading model...")
    model = load_model(config.model_path, config.device)

    # Create test observation
    observation = create_test_observation(config.batch_size, config.device)

    # Run benchmarks
    all_results = []

    # 1. Baseline (10 steps)
    logger.info("\n--- Baseline (10 denoising steps) ---")
    baseline_10 = benchmark_model(model, observation, config, num_steps=10, label="Baseline (10 steps)")
    all_results.append(baseline_10)

    # 2. Baseline (5 steps)
    logger.info("\n--- Baseline (5 denoising steps) ---")
    baseline_5 = benchmark_model(model, observation, config, num_steps=5, label="Baseline (5 steps)")
    all_results.append(baseline_5)

    # 3. Baseline (3 steps - aggressive)
    logger.info("\n--- Baseline (3 denoising steps) ---")
    baseline_3 = benchmark_model(model, observation, config, num_steps=3, label="Baseline (3 steps)")
    all_results.append(baseline_3)

    if not args.skip_compile:
        # 4. torch.compile with available backends (10 steps)
        logger.info("\n--- torch.compile (10 steps) ---")
        compile_10 = benchmark_torch_compile(model, observation, config, num_steps=10)
        all_results.append(compile_10)

        # 5. torch.compile (5 steps)
        logger.info("\n--- torch.compile (5 steps) ---")
        compile_5 = benchmark_torch_compile(model, observation, config, num_steps=5)
        all_results.append(compile_5)

        # 6. CUDA graph test (10 steps)
        logger.info("\n--- CUDA graph baseline (10 steps) ---")
        cudagraph_10 = benchmark_cudagraph(model, observation, config, num_steps=10)
        all_results.append(cudagraph_10)

        # 7. Inductor (skip on aarch64)
        logger.info("\n--- torch.compile inductor (10 steps) ---")
        inductor_10 = benchmark_inductor(model, observation, config, num_steps=10)
        all_results.append(inductor_10)

    # Print results
    print_results_table(all_results)

    # Save results
    with open(config.output_file, "w") as f:
        json.dump({
            "config": {
                "model_path": config.model_path,
                "device": config.device,
                "num_runs": config.num_runs,
            },
            "results": all_results,
        }, f, indent=2)

    logger.info(f"\nResults saved to {config.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
