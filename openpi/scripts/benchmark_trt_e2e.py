#!/usr/bin/env python3
"""
End-to-End TensorRT Benchmark for Pi0.5 VLA Model.

Benchmarks individual TensorRT engines and estimates combined inference throughput.

Usage:
    python scripts/benchmark_trt_e2e.py --engine_dir ./tensorrt_engines
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    logger.warning("TensorRT not available")


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_hz: float


class TensorRTEngine:
    """TensorRT inference wrapper."""

    def __init__(self, engine_path: str):
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT not available")

        self.engine_path = engine_path
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        # Load engine
        logger.debug(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate buffers
        self.inputs = {}
        self.outputs = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            trt_dtype = self.engine.get_tensor_dtype(name)

            # Handle dtype conversion
            try:
                dtype = trt.nptype(trt_dtype)
            except TypeError:
                dtype = np.float32

            # Handle dynamic shapes
            shape = list(shape)
            for j, s in enumerate(shape):
                if s == -1:
                    shape[j] = 1  # Default batch size
            shape = tuple(shape)

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            tensor_info = {
                'name': name,
                'shape': shape,
                'dtype': dtype,
                'host': host_mem,
                'device': device_mem,
            }

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = tensor_info
            else:
                self.outputs[name] = tensor_info

    def infer(self, input_data: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Run inference."""
        # Copy inputs to device
        for name, tensor_info in self.inputs.items():
            if input_data and name in input_data:
                np.copyto(tensor_info['host'], input_data[name].ravel())
            cuda.memcpy_htod_async(tensor_info['device'], tensor_info['host'], self.stream)
            self.context.set_tensor_address(name, int(tensor_info['device']))

        # Set output addresses
        for name, tensor_info in self.outputs.items():
            self.context.set_tensor_address(name, int(tensor_info['device']))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs to host
        results = {}
        for name, tensor_info in self.outputs.items():
            cuda.memcpy_dtoh_async(tensor_info['host'], tensor_info['device'], self.stream)

        self.stream.synchronize()

        for name, tensor_info in self.outputs.items():
            results[name] = tensor_info['host'].reshape(tensor_info['shape']).copy()

        return results

    def __del__(self):
        if hasattr(self, 'stream'):
            self.stream.synchronize()


def benchmark_engine(
    engine_path: str,
    num_runs: int = 100,
    warmup: int = 10,
) -> Optional[BenchmarkResult]:
    """Benchmark a single TensorRT engine."""
    try:
        engine = TensorRTEngine(engine_path)
    except Exception as e:
        logger.error(f"Failed to load engine {engine_path}: {e}")
        return None

    # Warmup
    for _ in range(warmup):
        engine.infer()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        engine.infer()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)

    return BenchmarkResult(
        name=Path(engine_path).stem,
        mean_latency_ms=latencies.mean(),
        std_latency_ms=latencies.std(),
        min_latency_ms=latencies.min(),
        max_latency_ms=latencies.max(),
        throughput_hz=1000.0 / latencies.mean(),
    )


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results."""
    print("\n" + "=" * 90)
    print("TENSORRT ENGINE BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Engine':<40} {'Mean (ms)':<12} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Hz':<10}")
    print("-" * 90)

    total_latency = 0.0
    for r in results:
        print(f"{r.name:<40} {r.mean_latency_ms:<12.2f} {r.std_latency_ms:<10.2f} {r.min_latency_ms:<10.2f} {r.max_latency_ms:<10.2f} {r.throughput_hz:<10.1f}")
        total_latency += r.mean_latency_ms

    print("=" * 90)
    print(f"\nPipeline Analysis:")
    print(f"  Sum of individual latencies: {total_latency:.2f} ms")
    print(f"  Theoretical max throughput (sequential): {1000.0/total_latency:.1f} Hz")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TensorRT engines")
    parser.add_argument(
        "--engine_dir",
        type=str,
        default="./tensorrt_engines",
        help="Directory containing TensorRT engines",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs",
    )
    args = parser.parse_args()

    if not HAS_TENSORRT:
        logger.error("TensorRT not available. Cannot run benchmarks.")
        return

    engine_dir = Path(args.engine_dir)
    if not engine_dir.exists():
        logger.error(f"Engine directory not found: {engine_dir}")
        return

    # Find all engines
    engines = list(engine_dir.glob("*.engine"))
    if not engines:
        logger.error(f"No .engine files found in {engine_dir}")
        return

    logger.info(f"Found {len(engines)} TensorRT engines")

    # Benchmark each engine
    results = []
    for engine_path in sorted(engines):
        logger.info(f"Benchmarking: {engine_path.name}")
        result = benchmark_engine(str(engine_path), args.num_runs, args.warmup)
        if result:
            results.append(result)

    # Print results
    if results:
        print_results(results)

        # Estimate pipeline performance
        # Key components for inference:
        # 1. Vision encoder (once per frame)
        # 2. Action expert (multiple denoising steps)
        # 3. Projections (fast)

        vision_result = next((r for r in results if "siglip" in r.name.lower() and "fp8" not in r.name.lower()), None)
        vision_fp8_result = next((r for r in results if "siglip" in r.name.lower() and "fp8" in r.name.lower()), None)
        expert_result = next((r for r in results if "gemma_300m" in r.name.lower() and "adarms" in r.name.lower()), None)
        expert_no_adarms = next((r for r in results if "gemma_300m" in r.name.lower() and "adarms" not in r.name.lower()), None)

        print("\n" + "=" * 90)
        print("ESTIMATED END-TO-END PERFORMANCE")
        print("=" * 90)

        # Estimate with different configurations
        num_denoising_steps = 10  # Typical for diffusion policy

        if vision_result and expert_result:
            vision_ms = vision_result.mean_latency_ms
            expert_ms = expert_result.mean_latency_ms
            proj_ms = 1.0  # Projections are fast (~1ms total)

            # Full pipeline: vision + (expert * denoising_steps) + projections
            total_ms = vision_ms + (expert_ms * num_denoising_steps) + proj_ms
            throughput = 1000.0 / total_ms

            print(f"\nConfiguration: FP16 Vision + FP16 Expert (adaRMS)")
            print(f"  Vision encoder: {vision_ms:.2f} ms")
            print(f"  Expert per step: {expert_ms:.2f} ms x {num_denoising_steps} steps = {expert_ms * num_denoising_steps:.2f} ms")
            print(f"  Projections: ~{proj_ms:.1f} ms")
            print(f"  Total: {total_ms:.2f} ms")
            print(f"  Throughput: {throughput:.1f} Hz")

        if vision_fp8_result and expert_result:
            vision_ms = vision_fp8_result.mean_latency_ms
            expert_ms = expert_result.mean_latency_ms
            proj_ms = 1.0

            total_ms = vision_ms + (expert_ms * num_denoising_steps) + proj_ms
            throughput = 1000.0 / total_ms

            print(f"\nConfiguration: FP8 Vision + FP16 Expert (adaRMS)")
            print(f"  Vision encoder: {vision_ms:.2f} ms")
            print(f"  Expert per step: {expert_ms:.2f} ms x {num_denoising_steps} steps = {expert_ms * num_denoising_steps:.2f} ms")
            print(f"  Projections: ~{proj_ms:.1f} ms")
            print(f"  Total: {total_ms:.2f} ms")
            print(f"  Throughput: {throughput:.1f} Hz")

        # Reduced steps configurations
        for steps in [5, 3, 2]:
            if vision_result and expert_result:
                vision_ms = vision_result.mean_latency_ms
                expert_ms = expert_result.mean_latency_ms
                proj_ms = 1.0

                total_ms = vision_ms + (expert_ms * steps) + proj_ms
                throughput = 1000.0 / total_ms

                print(f"\nWith {steps} denoising steps (FP16):")
                print(f"  Total: {total_ms:.2f} ms | Throughput: {throughput:.1f} Hz")

        print("\n" + "=" * 90)
        print("Note: Actual throughput may vary based on:")
        print("  - Memory bandwidth")
        print("  - CPU-GPU data transfer")
        print("  - Batching efficiency")
        print("  - Actual denoising steps required")
        print("=" * 90)


if __name__ == "__main__":
    main()
