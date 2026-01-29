#!/usr/bin/env python3
"""
Pipelined TensorRT Inference for Pi0.5 VLA Model.

Implements dual CUDA stream execution with double-buffering to overlap
Vision encoding with Action decoding, achieving higher throughput.

Architecture:
    Stream 0 (Vision):   [Vision n+1] -----> [Vision n+2] ----->
    Stream 1 (Action):        [Action n] --------> [Action n+1] ----->

    With ping-pong buffers, Vision(n+1) executes while Action(n) runs.

Performance Target:
    Sequential: Vision + Action = 5.61 + 43.45 = 49.06 ms → 20.4 Hz
    Pipelined:  max(Vision, Action) = 43.45 ms → 23.0 Hz (+12.7%)
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    import pycuda.autoinit  # noqa: F401
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    logger.warning("TensorRT not available")


@dataclass
class PipelineConfig:
    """Configuration for pipelined inference."""
    vision_engine_path: str
    expert_engine_path: str
    num_denoising_steps: int = 10
    action_horizon: int = 50
    action_dim: int = 32

    # Buffer configuration
    num_buffers: int = 2  # Double buffering


@dataclass
class PipelineStats:
    """Statistics from pipelined inference."""
    total_frames: int
    total_time_ms: float
    throughput_hz: float
    avg_latency_ms: float
    vision_time_ms: float
    action_time_ms: float
    overlap_efficiency: float  # 1.0 = perfect overlap


class TensorRTEngineAsync:
    """
    Async-capable TensorRT engine wrapper.

    Supports external CUDA stream injection for pipelined execution.
    """

    def __init__(self, engine_path: str, stream: Optional[cuda.Stream] = None):
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT not available")

        self.engine_path = engine_path
        self.name = Path(engine_path).stem
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        # Load engine
        logger.debug(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Use provided stream or create new one
        self.stream = stream if stream is not None else cuda.Stream()
        self.owns_stream = stream is None

        # Allocate buffers
        self._allocate_buffers()

        logger.info(f"Loaded engine: {self.name} "
                   f"(inputs: {list(self.inputs.keys())}, "
                   f"outputs: {list(self.outputs.keys())})")

    def _allocate_buffers(self):
        """Allocate host and device buffers."""
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
                'size': size,
            }

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = tensor_info
            else:
                self.outputs[name] = tensor_info

    def infer_async(
        self,
        input_data: Dict[str, np.ndarray] = None,
        wait: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run async inference.

        Args:
            input_data: Input tensors by name
            wait: If True, synchronize stream before returning

        Returns:
            Output tensors by name
        """
        # Copy inputs to device (async)
        for name, tensor_info in self.inputs.items():
            if input_data and name in input_data:
                data = input_data[name].astype(tensor_info['dtype']).ravel()
                np.copyto(tensor_info['host'][:len(data)], data)
            cuda.memcpy_htod_async(
                tensor_info['device'],
                tensor_info['host'],
                self.stream
            )
            self.context.set_tensor_address(name, int(tensor_info['device']))

        # Set output addresses
        for name, tensor_info in self.outputs.items():
            self.context.set_tensor_address(name, int(tensor_info['device']))

        # Execute (async)
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs to host (async)
        for name, tensor_info in self.outputs.items():
            cuda.memcpy_dtoh_async(
                tensor_info['host'],
                tensor_info['device'],
                self.stream
            )

        if wait:
            self.stream.synchronize()

        # Return results
        results = {}
        for name, tensor_info in self.outputs.items():
            results[name] = tensor_info['host'].reshape(tensor_info['shape']).copy()

        return results

    def synchronize(self):
        """Wait for all operations on this engine's stream to complete."""
        self.stream.synchronize()

    def get_output_shape(self, name: str) -> Tuple:
        """Get output tensor shape."""
        return self.outputs[name]['shape']

    def __del__(self):
        if hasattr(self, 'stream') and self.owns_stream:
            self.stream.synchronize()


class DoubleBuffer:
    """
    Double buffer for ping-pong latent storage.

    Allows Vision to write to buffer[i] while Action reads from buffer[1-i].
    """

    def __init__(self, shape: Tuple[int, ...], dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.buffers = [
            np.zeros(shape, dtype=dtype),
            np.zeros(shape, dtype=dtype),
        ]
        self.write_idx = 0  # Buffer being written by Vision
        self.read_idx = 1   # Buffer being read by Action

    def get_write_buffer(self) -> np.ndarray:
        """Get buffer for Vision to write to."""
        return self.buffers[self.write_idx]

    def get_read_buffer(self) -> np.ndarray:
        """Get buffer for Action to read from."""
        return self.buffers[self.read_idx]

    def swap(self):
        """Swap read and write buffers."""
        self.write_idx, self.read_idx = self.read_idx, self.write_idx


class TensorRTPipeline:
    """
    Pipelined TensorRT inference with Vision-Action overlap.

    Uses dual CUDA streams and double-buffering to maximize throughput.

    Execution pattern:
        Frame 0: Vision(0) → Action(0) [startup, sequential]
        Frame 1: Vision(1) || Action(0) → Action(1) [partial overlap]
        Frame N: Vision(N) || Action(N-1) → Action(N) [steady state]

    In steady state, Vision latency is hidden behind Action latency.
    """

    def __init__(self, config: PipelineConfig):
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT not available")

        self.config = config

        # Create CUDA streams
        self.vision_stream = cuda.Stream()
        self.action_stream = cuda.Stream()

        # Create CUDA events for synchronization
        self.vision_done_event = cuda.Event()
        self.action_done_event = cuda.Event()

        # Load engines with dedicated streams
        logger.info("Loading Vision encoder...")
        self.vision_engine = TensorRTEngineAsync(
            config.vision_engine_path,
            stream=self.vision_stream
        )

        logger.info("Loading Action expert...")
        self.action_engine = TensorRTEngineAsync(
            config.expert_engine_path,
            stream=self.action_stream
        )

        # Initialize double buffer for vision features
        # Shape depends on vision encoder output
        vision_output_name = list(self.vision_engine.outputs.keys())[0]
        vision_output_shape = self.vision_engine.get_output_shape(vision_output_name)
        logger.info(f"Vision output shape: {vision_output_shape}")

        self.vision_buffer = DoubleBuffer(vision_output_shape, dtype=np.float32)

        # State tracking
        self.frame_count = 0
        self.is_warmed_up = False

        logger.info(f"Pipeline initialized with {config.num_denoising_steps} denoising steps")

    def warmup(self, num_iterations: int = 5):
        """Warmup the pipeline to stabilize performance."""
        logger.info(f"Warming up pipeline ({num_iterations} iterations)...")

        # Create dummy inputs
        vision_input = self._create_dummy_vision_input()
        action_input = self._create_dummy_action_input()

        for i in range(num_iterations):
            # Run vision
            self.vision_engine.infer_async(vision_input, wait=True)

            # Run action for all denoising steps
            for _ in range(self.config.num_denoising_steps):
                self.action_engine.infer_async(action_input, wait=True)

        self.is_warmed_up = True
        logger.info("Warmup complete")

    def _create_dummy_vision_input(self) -> Dict[str, np.ndarray]:
        """Create dummy vision input for testing."""
        inputs = {}
        for name, info in self.vision_engine.inputs.items():
            inputs[name] = np.random.randn(*info['shape']).astype(info['dtype'])
        return inputs

    def _create_dummy_action_input(self) -> Dict[str, np.ndarray]:
        """Create dummy action input for testing."""
        inputs = {}
        for name, info in self.action_engine.inputs.items():
            inputs[name] = np.random.randn(*info['shape']).astype(info['dtype'])
        return inputs

    def infer_sequential(
        self,
        vision_input: Dict[str, np.ndarray],
        action_input: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Sequential inference (baseline for comparison).

        Vision → Action (no overlap)
        """
        # Vision
        vision_output = self.vision_engine.infer_async(vision_input, wait=True)

        # Action denoising loop
        for step in range(self.config.num_denoising_steps):
            action_output = self.action_engine.infer_async(action_input, wait=True)

        return action_output

    def infer_pipelined(
        self,
        vision_inputs: List[Dict[str, np.ndarray]],
        action_inputs: List[Dict[str, np.ndarray]],
    ) -> Tuple[List[Dict[str, np.ndarray]], PipelineStats]:
        """
        Pipelined inference for multiple frames.

        Overlaps Vision(n+1) with Action(n) for higher throughput.

        Args:
            vision_inputs: List of vision inputs for each frame
            action_inputs: List of action inputs for each frame

        Returns:
            Tuple of (action_outputs, stats)
        """
        num_frames = len(vision_inputs)
        action_outputs = []

        # Timing
        start_time = time.perf_counter()
        vision_times = []
        action_times = []

        for frame_idx in range(num_frames):
            frame_start = time.perf_counter()

            if frame_idx == 0:
                # First frame: sequential (no previous action to overlap with)
                vision_start = time.perf_counter()
                vision_output = self.vision_engine.infer_async(
                    vision_inputs[frame_idx],
                    wait=True
                )
                vision_times.append((time.perf_counter() - vision_start) * 1000)

                # Store in buffer
                output_name = list(vision_output.keys())[0]
                np.copyto(self.vision_buffer.get_write_buffer(), vision_output[output_name])
                self.vision_buffer.swap()

            else:
                # Subsequent frames: pipeline Vision with previous Action
                # Vision(n) runs on vision_stream while Action(n-1) runs on action_stream
                vision_start = time.perf_counter()

                # Launch Vision async (don't wait)
                self.vision_engine.infer_async(vision_inputs[frame_idx], wait=False)
                # Record event when vision is done
                self.vision_done_event.record(self.vision_stream)

                vision_times.append((time.perf_counter() - vision_start) * 1000)

            # Run Action denoising loop on action_stream
            action_start = time.perf_counter()

            for step in range(self.config.num_denoising_steps):
                action_output = self.action_engine.infer_async(
                    action_inputs[frame_idx],
                    wait=(step == self.config.num_denoising_steps - 1)
                )

            action_times.append((time.perf_counter() - action_start) * 1000)
            action_outputs.append(action_output)

            # Ensure vision is complete before next frame's action
            if frame_idx < num_frames - 1:
                self.vision_stream.synchronize()
                # Update buffer for next frame
                vision_output_name = list(self.vision_engine.outputs.keys())[0]
                vision_data = self.vision_engine.outputs[vision_output_name]['host']
                vision_shape = self.vision_engine.outputs[vision_output_name]['shape']
                np.copyto(
                    self.vision_buffer.get_write_buffer(),
                    vision_data.reshape(vision_shape)
                )
                self.vision_buffer.swap()

        # Final synchronization
        self.vision_stream.synchronize()
        self.action_stream.synchronize()

        total_time = (time.perf_counter() - start_time) * 1000

        # Calculate statistics
        avg_vision_ms = np.mean(vision_times[1:]) if len(vision_times) > 1 else vision_times[0]
        avg_action_ms = np.mean(action_times)

        # Theoretical sequential time
        sequential_time_ms = (avg_vision_ms + avg_action_ms) * num_frames

        # Overlap efficiency: 1.0 means perfect overlap, 0.0 means no overlap
        if sequential_time_ms > 0:
            overlap_efficiency = 1.0 - (total_time / sequential_time_ms)
        else:
            overlap_efficiency = 0.0

        stats = PipelineStats(
            total_frames=num_frames,
            total_time_ms=total_time,
            throughput_hz=num_frames * 1000.0 / total_time,
            avg_latency_ms=total_time / num_frames,
            vision_time_ms=avg_vision_ms,
            action_time_ms=avg_action_ms,
            overlap_efficiency=max(0, overlap_efficiency),
        )

        return action_outputs, stats

    def benchmark_sequential(self, num_frames: int = 100) -> PipelineStats:
        """Benchmark sequential inference."""
        vision_input = self._create_dummy_vision_input()
        action_input = self._create_dummy_action_input()

        # Warmup
        for _ in range(10):
            self.infer_sequential(vision_input, action_input)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_frames):
            self.infer_sequential(vision_input, action_input)
        total_time = (time.perf_counter() - start) * 1000

        return PipelineStats(
            total_frames=num_frames,
            total_time_ms=total_time,
            throughput_hz=num_frames * 1000.0 / total_time,
            avg_latency_ms=total_time / num_frames,
            vision_time_ms=0,  # Not measured separately
            action_time_ms=0,
            overlap_efficiency=0.0,
        )

    def benchmark_pipelined(self, num_frames: int = 100) -> PipelineStats:
        """Benchmark pipelined inference."""
        # Create inputs for all frames
        vision_inputs = [self._create_dummy_vision_input() for _ in range(num_frames)]
        action_inputs = [self._create_dummy_action_input() for _ in range(num_frames)]

        # Warmup
        warmup_vision = [self._create_dummy_vision_input() for _ in range(10)]
        warmup_action = [self._create_dummy_action_input() for _ in range(10)]
        self.infer_pipelined(warmup_vision, warmup_action)

        # Benchmark
        _, stats = self.infer_pipelined(vision_inputs, action_inputs)

        return stats

    def __del__(self):
        if hasattr(self, 'vision_stream'):
            self.vision_stream.synchronize()
        if hasattr(self, 'action_stream'):
            self.action_stream.synchronize()


def run_pipeline_benchmark(
    engine_dir: str,
    num_frames: int = 100,
    num_denoising_steps: int = 10,
) -> Tuple[PipelineStats, PipelineStats]:
    """
    Run comparative benchmark: sequential vs pipelined.

    Returns:
        Tuple of (sequential_stats, pipelined_stats)
    """
    engine_dir = Path(engine_dir)

    # Find engines
    vision_engine = engine_dir / "siglip_vision_encoder.engine"
    expert_engine = engine_dir / "gemma_300m_expert_adarms_fp16.engine"

    if not vision_engine.exists():
        raise FileNotFoundError(f"Vision engine not found: {vision_engine}")
    if not expert_engine.exists():
        # Try without adarms
        expert_engine = engine_dir / "gemma_300m_expert_fp16.engine"
        if not expert_engine.exists():
            raise FileNotFoundError(f"Expert engine not found")

    config = PipelineConfig(
        vision_engine_path=str(vision_engine),
        expert_engine_path=str(expert_engine),
        num_denoising_steps=num_denoising_steps,
    )

    pipeline = TensorRTPipeline(config)

    logger.info(f"Benchmarking with {num_frames} frames, {num_denoising_steps} steps")

    # Sequential benchmark
    logger.info("Running sequential benchmark...")
    seq_stats = pipeline.benchmark_sequential(num_frames)

    # Pipelined benchmark
    logger.info("Running pipelined benchmark...")
    pipe_stats = pipeline.benchmark_pipelined(num_frames)

    return seq_stats, pipe_stats


def print_comparison(seq_stats: PipelineStats, pipe_stats: PipelineStats):
    """Print benchmark comparison."""
    print("\n" + "=" * 80)
    print("PIPELINE BENCHMARK COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Sequential':<20} {'Pipelined':<20} {'Improvement':<15}")
    print("-" * 80)

    speedup = pipe_stats.throughput_hz / seq_stats.throughput_hz
    latency_reduction = (seq_stats.avg_latency_ms - pipe_stats.avg_latency_ms) / seq_stats.avg_latency_ms * 100

    print(f"{'Throughput (Hz)':<30} {seq_stats.throughput_hz:<20.2f} {pipe_stats.throughput_hz:<20.2f} {f'+{(speedup-1)*100:.1f}%':<15}")
    print(f"{'Avg Latency (ms)':<30} {seq_stats.avg_latency_ms:<20.2f} {pipe_stats.avg_latency_ms:<20.2f} {f'-{latency_reduction:.1f}%':<15}")
    print(f"{'Total Time (ms)':<30} {seq_stats.total_time_ms:<20.2f} {pipe_stats.total_time_ms:<20.2f} {'':<15}")
    print(f"{'Frames Processed':<30} {seq_stats.total_frames:<20} {pipe_stats.total_frames:<20} {'':<15}")

    print("\n" + "-" * 80)
    print(f"{'Vision Time (ms)':<30} {'-':<20} {pipe_stats.vision_time_ms:<20.2f}")
    print(f"{'Action Time (ms)':<30} {'-':<20} {pipe_stats.action_time_ms:<20.2f}")
    print(f"{'Overlap Efficiency':<30} {'-':<20} {f'{pipe_stats.overlap_efficiency*100:.1f}%':<20}")

    print("\n" + "=" * 80)
    print(f"SPEEDUP: {speedup:.2f}x ({seq_stats.throughput_hz:.1f} Hz → {pipe_stats.throughput_hz:.1f} Hz)")
    print("=" * 80)
