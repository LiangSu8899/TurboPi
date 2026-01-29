#!/usr/bin/env python3
"""
benchmark_tensorrt.py - Benchmark TensorRT engines vs PyTorch.

This script compares the performance of TensorRT engines with PyTorch inference
for the Pi0.5 model components.

Usage:
    python benchmark_tensorrt.py \
        --engine_dir ./onnx_exports \
        --num_runs 100
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_qps: float


class TensorRTInference:
    """TensorRT inference wrapper."""

    def __init__(self, engine_path: str):
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT not available")

        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # Load engine
        logger.info(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            trt_dtype = self.engine.get_tensor_dtype(name)

            # Handle dtype conversion (BF16 doesn't have numpy equivalent)
            try:
                dtype = trt.nptype(trt_dtype)
            except TypeError:
                # Fallback for BF16 and other unsupported types
                if trt_dtype == trt.bfloat16:
                    dtype = np.float32  # Use float32 as proxy
                else:
                    dtype = np.float32
                logger.warning(f"Tensor {name}: using float32 as proxy for {trt_dtype}")

            # Handle dynamic shapes
            if -1 in shape:
                # Use profile shape
                shape = list(shape)
                for j, s in enumerate(shape):
                    if s == -1:
                        shape[j] = 1  # Default batch size
                shape = tuple(shape)

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype,
                })
            else:
                self.outputs.append({
                    'name': name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype,
                })

        logger.info(f"Engine loaded with {len(self.inputs)} inputs, {len(self.outputs)} outputs")

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference."""
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output to host
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        self.stream.synchronize()

        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'stream'):
            self.stream.synchronize()


def benchmark_tensorrt(
    engine_path: str,
    input_shape: tuple,
    num_runs: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """Benchmark TensorRT engine."""
    logger.info(f"Benchmarking TensorRT: {engine_path}")

    trt_inference = TensorRTInference(engine_path)

    # Create dummy input
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        trt_inference.infer(input_data)

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        trt_inference.infer(input_data)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    return BenchmarkResult(
        name=Path(engine_path).stem,
        mean_latency_ms=latencies.mean(),
        std_latency_ms=latencies.std(),
        min_latency_ms=latencies.min(),
        max_latency_ms=latencies.max(),
        throughput_qps=1000.0 / latencies.mean(),
    )


def benchmark_pytorch_vision(
    model_path: str,
    input_shape: tuple,
    num_runs: int = 100,
    warmup: int = 10,
    device: str = "cuda",
) -> BenchmarkResult:
    """Benchmark PyTorch vision encoder."""
    import json
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models.pi0_config import Pi0Config
    from safetensors.torch import load_file

    logger.info(f"Benchmarking PyTorch vision encoder from: {model_path}")

    model_path = Path(model_path).expanduser()
    weights_path = model_path / "model.safetensors"
    config_path = model_path / "config.json"

    with open(config_path) as f:
        model_config = json.load(f)

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        pi05=True,
    )

    model = PI0Pytorch(pi0_config)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device)
    model.eval()

    vision_tower = model.paligemma_with_expert.paligemma.vision_tower

    # Create dummy input
    input_tensor = torch.randn(*input_shape, device=device, dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = vision_tower(input_tensor)
            torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = vision_tower(input_tensor)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    return BenchmarkResult(
        name="pytorch_vision_encoder",
        mean_latency_ms=latencies.mean(),
        std_latency_ms=latencies.std(),
        min_latency_ms=latencies.min(),
        max_latency_ms=latencies.max(),
        throughput_qps=1000.0 / latencies.mean(),
    )


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Name':<35} {'Mean (ms)':<12} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'QPS':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r.name:<35} {r.mean_latency_ms:<12.2f} {r.std_latency_ms:<10.2f} {r.min_latency_ms:<10.2f} {r.max_latency_ms:<10.2f} {r.throughput_qps:<10.1f}")

    print("=" * 80)

    # Calculate speedup
    if len(results) >= 2:
        pytorch_result = next((r for r in results if "pytorch" in r.name.lower()), None)
        trt_result = next((r for r in results if "tensorrt" in r.name.lower() or "fp16" in r.name.lower()), None)

        if pytorch_result and trt_result:
            speedup = pytorch_result.mean_latency_ms / trt_result.mean_latency_ms
            print(f"\nSpeedup (TensorRT vs PyTorch): {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TensorRT vs PyTorch")
    parser.add_argument(
        "--engine_dir",
        type=str,
        default="./onnx_exports",
        help="Directory containing TensorRT engines",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="~/.cache/openpi/checkpoints/pi05_libero",
        help="Path to Pi0.5 checkpoint (for PyTorch comparison)",
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
    parser.add_argument(
        "--skip_pytorch",
        action="store_true",
        help="Skip PyTorch benchmark (faster)",
    )
    args = parser.parse_args()

    results = []
    engine_dir = Path(args.engine_dir)

    # Vision encoder input shape: (batch, channels, height, width)
    vision_input_shape = (1, 3, 224, 224)

    # Benchmark TensorRT FP16
    fp16_engine = engine_dir / "siglip_vision_encoder.engine"
    if fp16_engine.exists() and HAS_TENSORRT:
        try:
            result = benchmark_tensorrt(
                str(fp16_engine),
                vision_input_shape,
                args.num_runs,
                args.warmup,
            )
            result.name = "TensorRT_FP16_vision"
            results.append(result)
        except Exception as e:
            logger.error(f"TensorRT FP16 benchmark failed: {e}")

    # Benchmark TensorRT FP8
    fp8_engine = engine_dir / "siglip_vision_encoder_fp8.engine"
    if fp8_engine.exists() and HAS_TENSORRT:
        try:
            result = benchmark_tensorrt(
                str(fp8_engine),
                vision_input_shape,
                args.num_runs,
                args.warmup,
            )
            result.name = "TensorRT_FP8_vision"
            results.append(result)
        except Exception as e:
            logger.error(f"TensorRT FP8 benchmark failed: {e}")

    # Benchmark PyTorch
    if not args.skip_pytorch:
        try:
            result = benchmark_pytorch_vision(
                args.model_path,
                vision_input_shape,
                args.num_runs,
                args.warmup,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"PyTorch benchmark failed: {e}")

    # Print results
    if results:
        print_results(results)
    else:
        logger.error("No benchmark results collected")


if __name__ == "__main__":
    main()
