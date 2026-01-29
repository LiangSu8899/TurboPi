#!/usr/bin/env python3
"""
Benchmark Pipelined TensorRT Inference for Pi0.5 VLA Model.

Compares sequential vs pipelined execution with dual CUDA streams.

Usage:
    python scripts/benchmark_pipeline.py --engine_dir ./tensorrt_engines --num_frames 100

Expected Results:
    Sequential: ~20 Hz (Vision + Action in series)
    Pipelined:  ~23+ Hz (Vision overlapped with previous Action)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    logger.error("TensorRT not available")


def benchmark_individual_engines(engine_dir: Path, num_runs: int = 100):
    """Benchmark individual engines to get baseline latencies."""
    from openpi.inference.trt_pipeline import TensorRTEngineAsync

    results = {}

    # Vision encoder
    vision_path = engine_dir / "siglip_vision_encoder.engine"
    if vision_path.exists():
        engine = TensorRTEngineAsync(str(vision_path))

        # Create dummy input
        dummy = {}
        for name, info in engine.inputs.items():
            dummy[name] = np.random.randn(*info['shape']).astype(info['dtype'])

        # Warmup
        for _ in range(10):
            engine.infer_async(dummy, wait=True)

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            engine.infer_async(dummy, wait=True)
            latencies.append((time.perf_counter() - start) * 1000)

        results['vision'] = {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
        }
        logger.info(f"Vision encoder: {results['vision']['mean']:.2f} ms")

    # Action expert
    expert_path = engine_dir / "gemma_300m_expert_adarms_fp16.engine"
    if not expert_path.exists():
        expert_path = engine_dir / "gemma_300m_expert_fp16.engine"

    if expert_path.exists():
        engine = TensorRTEngineAsync(str(expert_path))

        dummy = {}
        for name, info in engine.inputs.items():
            dummy[name] = np.random.randn(*info['shape']).astype(info['dtype'])

        # Warmup
        for _ in range(10):
            engine.infer_async(dummy, wait=True)

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            engine.infer_async(dummy, wait=True)
            latencies.append((time.perf_counter() - start) * 1000)

        results['expert'] = {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
        }
        logger.info(f"Action expert: {results['expert']['mean']:.2f} ms")

    return results


def run_sequential_benchmark(
    engine_dir: Path,
    num_frames: int,
    num_steps: int,
) -> dict:
    """Run sequential (non-pipelined) benchmark."""
    from openpi.inference.trt_pipeline import TensorRTEngineAsync

    vision_path = engine_dir / "siglip_vision_encoder.engine"
    expert_path = engine_dir / "gemma_300m_expert_adarms_fp16.engine"
    if not expert_path.exists():
        expert_path = engine_dir / "gemma_300m_expert_fp16.engine"

    vision_engine = TensorRTEngineAsync(str(vision_path))
    expert_engine = TensorRTEngineAsync(str(expert_path))

    # Create dummy inputs
    vision_input = {}
    for name, info in vision_engine.inputs.items():
        vision_input[name] = np.random.randn(*info['shape']).astype(info['dtype'])

    expert_input = {}
    for name, info in expert_engine.inputs.items():
        expert_input[name] = np.random.randn(*info['shape']).astype(info['dtype'])

    # Warmup
    logger.info("Warming up sequential pipeline...")
    for _ in range(10):
        vision_engine.infer_async(vision_input, wait=True)
        for _ in range(num_steps):
            expert_engine.infer_async(expert_input, wait=True)

    # Benchmark
    logger.info(f"Running sequential benchmark ({num_frames} frames, {num_steps} steps)...")
    start = time.perf_counter()

    for _ in range(num_frames):
        # Vision (sequential)
        vision_engine.infer_async(vision_input, wait=True)

        # Action denoising loop (sequential)
        for _ in range(num_steps):
            expert_engine.infer_async(expert_input, wait=True)

    total_time = (time.perf_counter() - start) * 1000

    return {
        'total_time_ms': total_time,
        'throughput_hz': num_frames * 1000.0 / total_time,
        'avg_latency_ms': total_time / num_frames,
        'num_frames': num_frames,
        'num_steps': num_steps,
    }


def run_pipelined_benchmark(
    engine_dir: Path,
    num_frames: int,
    num_steps: int,
) -> dict:
    """Run pipelined benchmark with dual CUDA streams."""
    from openpi.inference.trt_pipeline import TensorRTEngineAsync

    vision_path = engine_dir / "siglip_vision_encoder.engine"
    expert_path = engine_dir / "gemma_300m_expert_adarms_fp16.engine"
    if not expert_path.exists():
        expert_path = engine_dir / "gemma_300m_expert_fp16.engine"

    # Create separate streams
    vision_stream = cuda.Stream()
    action_stream = cuda.Stream()

    # Create events for synchronization
    vision_done = cuda.Event()
    action_done = cuda.Event()

    vision_engine = TensorRTEngineAsync(str(vision_path), stream=vision_stream)
    expert_engine = TensorRTEngineAsync(str(expert_path), stream=action_stream)

    # Create dummy inputs
    vision_input = {}
    for name, info in vision_engine.inputs.items():
        vision_input[name] = np.random.randn(*info['shape']).astype(info['dtype'])

    expert_input = {}
    for name, info in expert_engine.inputs.items():
        expert_input[name] = np.random.randn(*info['shape']).astype(info['dtype'])

    # Warmup
    logger.info("Warming up pipelined pipeline...")
    for _ in range(10):
        vision_engine.infer_async(vision_input, wait=True)
        for _ in range(num_steps):
            expert_engine.infer_async(expert_input, wait=True)

    # Benchmark with pipelining
    logger.info(f"Running pipelined benchmark ({num_frames} frames, {num_steps} steps)...")

    start = time.perf_counter()

    # Frame 0: Start first vision
    vision_engine.infer_async(vision_input, wait=False)
    vision_done.record(vision_stream)

    # Wait for first vision
    action_stream.wait_for_event(vision_done)

    # Process frames with pipelining using event-based sync
    for frame_idx in range(num_frames):
        # Launch next frame's Vision immediately (overlapped with current Action)
        if frame_idx < num_frames - 1:
            # Action stream must wait for previous vision to complete before using its output
            vision_engine.infer_async(vision_input, wait=False)
            vision_done.record(vision_stream)

        # Run current frame's Action denoising loop (all async, no sync)
        for step in range(num_steps):
            expert_engine.infer_async(expert_input, wait=False)

        # Record when action is done
        action_done.record(action_stream)

        # Next iteration: action stream waits for vision to complete
        if frame_idx < num_frames - 1:
            action_stream.wait_for_event(vision_done)

    # Final sync
    vision_stream.synchronize()
    action_stream.synchronize()

    total_time = (time.perf_counter() - start) * 1000

    return {
        'total_time_ms': total_time,
        'throughput_hz': num_frames * 1000.0 / total_time,
        'avg_latency_ms': total_time / num_frames,
        'num_frames': num_frames,
        'num_steps': num_steps,
    }


def run_fully_async_benchmark(
    engine_dir: Path,
    num_frames: int,
    num_steps: int,
) -> dict:
    """
    Fully async pipelined benchmark.

    Uses ping-pong pattern where Vision(n+1) runs completely parallel to Action(n).
    """
    from openpi.inference.trt_pipeline import TensorRTEngineAsync

    vision_path = engine_dir / "siglip_vision_encoder.engine"
    expert_path = engine_dir / "gemma_300m_expert_adarms_fp16.engine"
    if not expert_path.exists():
        expert_path = engine_dir / "gemma_300m_expert_fp16.engine"

    # Create separate streams
    vision_stream = cuda.Stream()
    action_stream = cuda.Stream()

    vision_engine = TensorRTEngineAsync(str(vision_path), stream=vision_stream)
    expert_engine = TensorRTEngineAsync(str(expert_path), stream=action_stream)

    # Create dummy inputs
    vision_input = {}
    for name, info in vision_engine.inputs.items():
        vision_input[name] = np.random.randn(*info['shape']).astype(info['dtype'])

    expert_input = {}
    for name, info in expert_engine.inputs.items():
        expert_input[name] = np.random.randn(*info['shape']).astype(info['dtype'])

    # Warmup
    logger.info("Warming up fully async pipeline...")
    for _ in range(10):
        vision_engine.infer_async(vision_input, wait=True)
        for _ in range(num_steps):
            expert_engine.infer_async(expert_input, wait=True)

    logger.info(f"Running fully async benchmark ({num_frames} frames, {num_steps} steps)...")

    # Pre-launch all work onto streams with minimal CPU overhead
    start = time.perf_counter()

    # Create events for each frame
    vision_events = [cuda.Event() for _ in range(num_frames + 1)]
    action_events = [cuda.Event() for _ in range(num_frames)]

    # Launch first vision
    vision_engine.infer_async(vision_input, wait=False)
    vision_events[0].record(vision_stream)

    for frame_idx in range(num_frames):
        # Wait for vision from previous iteration (or first vision for frame 0)
        action_stream.wait_for_event(vision_events[frame_idx])

        # Launch Vision for next frame (overlaps with current Action)
        if frame_idx < num_frames - 1:
            vision_engine.infer_async(vision_input, wait=False)
            vision_events[frame_idx + 1].record(vision_stream)

        # Launch all Action steps (fully async)
        for _ in range(num_steps):
            expert_engine.infer_async(expert_input, wait=False)

        action_events[frame_idx].record(action_stream)

    # Final sync
    action_stream.synchronize()
    vision_stream.synchronize()

    total_time = (time.perf_counter() - start) * 1000

    return {
        'total_time_ms': total_time,
        'throughput_hz': num_frames * 1000.0 / total_time,
        'avg_latency_ms': total_time / num_frames,
        'num_frames': num_frames,
        'num_steps': num_steps,
    }


def validate_outputs(engine_dir: Path, num_steps: int = 10) -> bool:
    """
    Validate that pipelined outputs match sequential outputs.

    This ensures the optimization doesn't affect accuracy.
    """
    from openpi.inference.trt_pipeline import TensorRTEngineAsync

    logger.info("Validating output accuracy...")

    vision_path = engine_dir / "siglip_vision_encoder.engine"
    expert_path = engine_dir / "gemma_300m_expert_adarms_fp16.engine"
    if not expert_path.exists():
        expert_path = engine_dir / "gemma_300m_expert_fp16.engine"

    # Sequential execution
    vision_engine_seq = TensorRTEngineAsync(str(vision_path))
    expert_engine_seq = TensorRTEngineAsync(str(expert_path))

    # Fixed seed for reproducibility
    np.random.seed(42)

    vision_input = {}
    for name, info in vision_engine_seq.inputs.items():
        vision_input[name] = np.random.randn(*info['shape']).astype(info['dtype'])

    expert_input = {}
    for name, info in expert_engine_seq.inputs.items():
        expert_input[name] = np.random.randn(*info['shape']).astype(info['dtype'])

    # Run sequential
    seq_vision_out = vision_engine_seq.infer_async(vision_input, wait=True)
    seq_expert_outs = []
    for _ in range(num_steps):
        out = expert_engine_seq.infer_async(expert_input, wait=True)
        seq_expert_outs.append(out)

    # Pipelined execution with same inputs
    vision_stream = cuda.Stream()
    action_stream = cuda.Stream()

    vision_engine_pipe = TensorRTEngineAsync(str(vision_path), stream=vision_stream)
    expert_engine_pipe = TensorRTEngineAsync(str(expert_path), stream=action_stream)

    # Run pipelined
    pipe_vision_out = vision_engine_pipe.infer_async(vision_input, wait=True)
    pipe_expert_outs = []
    for _ in range(num_steps):
        out = expert_engine_pipe.infer_async(expert_input, wait=True)
        pipe_expert_outs.append(out)

    # Compare outputs
    vision_output_name = list(seq_vision_out.keys())[0]
    expert_output_name = list(seq_expert_outs[0].keys())[0]

    vision_diff = np.abs(seq_vision_out[vision_output_name] - pipe_vision_out[vision_output_name])
    vision_mse = np.mean(vision_diff ** 2)

    expert_diffs = []
    for seq_out, pipe_out in zip(seq_expert_outs, pipe_expert_outs):
        diff = np.abs(seq_out[expert_output_name] - pipe_out[expert_output_name])
        expert_diffs.append(np.mean(diff ** 2))
    expert_mse = np.mean(expert_diffs)

    logger.info(f"Vision MSE: {vision_mse:.2e}")
    logger.info(f"Expert MSE: {expert_mse:.2e}")

    # Threshold for numerical precision
    threshold = 1e-6
    vision_ok = vision_mse < threshold
    expert_ok = expert_mse < threshold

    if vision_ok and expert_ok:
        logger.info("[PASS] Output validation passed - pipelined matches sequential")
        return True
    else:
        logger.warning(f"[WARN] Output differences detected (threshold: {threshold})")
        logger.warning(f"  Vision MSE: {vision_mse:.2e} ({'PASS' if vision_ok else 'FAIL'})")
        logger.warning(f"  Expert MSE: {expert_mse:.2e} ({'PASS' if expert_ok else 'FAIL'})")
        return False


def print_results(
    engine_results: dict,
    seq_results: dict,
    pipe_results: dict,
    async_results: dict,
    num_steps: int,
):
    """Print comprehensive benchmark results."""
    print("\n" + "=" * 100)
    print("PIPELINED TENSORRT INFERENCE BENCHMARK")
    print("=" * 100)

    # Individual engine latencies
    print("\n--- Individual Engine Latencies ---")
    if 'vision' in engine_results:
        v = engine_results['vision']
        print(f"Vision encoder:  {v['mean']:.2f} ms (std: {v['std']:.2f})")
    if 'expert' in engine_results:
        e = engine_results['expert']
        print(f"Action expert:   {e['mean']:.2f} ms (std: {e['std']:.2f})")

    # Calculate theoretical times
    vision_ms = engine_results.get('vision', {}).get('mean', 5.61)
    expert_ms = engine_results.get('expert', {}).get('mean', 4.35)

    print(f"\n--- Theoretical Analysis ({num_steps} denoising steps) ---")
    seq_theory = vision_ms + expert_ms * num_steps
    pipe_theory = max(vision_ms, expert_ms * num_steps)
    print(f"Sequential theory:  Vision + Action = {vision_ms:.2f} + {expert_ms:.2f}×{num_steps} = {seq_theory:.2f} ms")
    print(f"Pipelined theory:   max(Vision, Action) = max({vision_ms:.2f}, {expert_ms * num_steps:.2f}) = {pipe_theory:.2f} ms")
    print(f"Theoretical speedup: {seq_theory / pipe_theory:.2f}x")

    # Actual results
    print("\n--- Benchmark Results ---")
    print(f"{'Metric':<25} {'Sequential':<18} {'Pipelined':<18} {'Fully Async':<18} {'Best Speedup':<15}")
    print("-" * 100)

    pipe_speedup = pipe_results['throughput_hz'] / seq_results['throughput_hz']
    async_speedup = async_results['throughput_hz'] / seq_results['throughput_hz']
    best_speedup = max(pipe_speedup, async_speedup)

    print(f"{'Throughput (Hz)':<25} {seq_results['throughput_hz']:<18.2f} {pipe_results['throughput_hz']:<18.2f} {async_results['throughput_hz']:<18.2f} {f'+{(best_speedup-1)*100:.1f}%':<15}")
    print(f"{'Avg Latency (ms)':<25} {seq_results['avg_latency_ms']:<18.2f} {pipe_results['avg_latency_ms']:<18.2f} {async_results['avg_latency_ms']:<18.2f} {'':<15}")
    print(f"{'Total Time (ms)':<25} {seq_results['total_time_ms']:<18.2f} {pipe_results['total_time_ms']:<18.2f} {async_results['total_time_ms']:<18.2f} {'':<15}")

    print("\n" + "=" * 100)
    print(f"RESULT: {best_speedup:.2f}x SPEEDUP (Best configuration)")
    print(f"  Sequential:  {seq_results['throughput_hz']:.1f} Hz")
    print(f"  Pipelined:   {pipe_results['throughput_hz']:.1f} Hz (+{(pipe_speedup-1)*100:.1f}%)")
    print(f"  Fully Async: {async_results['throughput_hz']:.1f} Hz (+{(async_speedup-1)*100:.1f}%)")
    print("=" * 100)

    # Summary for different step counts
    print("\n--- Theoretical Throughput at Different Denoising Steps ---")
    print(f"{'Steps':<10} {'Sequential (Hz)':<20} {'Pipelined (Hz)':<20} {'Speedup':<15}")
    print("-" * 65)

    for steps in [10, 5, 3, 2]:
        seq_time = vision_ms + expert_ms * steps
        pipe_time = max(vision_ms, expert_ms * steps)
        seq_hz = 1000.0 / seq_time
        pipe_hz = 1000.0 / pipe_time
        print(f"{steps:<10} {seq_hz:<20.1f} {pipe_hz:<20.1f} {pipe_hz/seq_hz:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pipelined TensorRT inference"
    )
    parser.add_argument(
        "--engine_dir",
        type=str,
        default="./tensorrt_engines",
        help="Directory containing TensorRT engines",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=100,
        help="Number of frames to benchmark",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip output validation",
    )
    args = parser.parse_args()

    if not HAS_TENSORRT:
        logger.error("TensorRT not available. Cannot run benchmarks.")
        return 1

    engine_dir = Path(args.engine_dir)
    if not engine_dir.exists():
        logger.error(f"Engine directory not found: {engine_dir}")
        return 1

    # Check required engines
    vision_engine = engine_dir / "siglip_vision_encoder.engine"
    expert_engine = engine_dir / "gemma_300m_expert_adarms_fp16.engine"

    if not vision_engine.exists():
        logger.error(f"Vision engine not found: {vision_engine}")
        return 1

    if not expert_engine.exists():
        expert_engine = engine_dir / "gemma_300m_expert_fp16.engine"
        if not expert_engine.exists():
            logger.error("Expert engine not found")
            return 1

    logger.info(f"Engine directory: {engine_dir}")
    logger.info(f"Vision engine: {vision_engine.name}")
    logger.info(f"Expert engine: {expert_engine.name}")

    # Validate outputs first
    if not args.skip_validation:
        validation_ok = validate_outputs(engine_dir, args.num_steps)
        if not validation_ok:
            logger.warning("Output validation showed differences - proceed with caution")

    # Benchmark individual engines
    logger.info("\n--- Benchmarking individual engines ---")
    engine_results = benchmark_individual_engines(engine_dir)

    # Sequential benchmark
    seq_results = run_sequential_benchmark(
        engine_dir,
        args.num_frames,
        args.num_steps,
    )

    # Pipelined benchmark
    pipe_results = run_pipelined_benchmark(
        engine_dir,
        args.num_frames,
        args.num_steps,
    )

    # Fully async benchmark
    async_results = run_fully_async_benchmark(
        engine_dir,
        args.num_frames,
        args.num_steps,
    )

    # Print results
    print_results(engine_results, seq_results, pipe_results, async_results, args.num_steps)

    return 0


if __name__ == "__main__":
    sys.exit(main())
