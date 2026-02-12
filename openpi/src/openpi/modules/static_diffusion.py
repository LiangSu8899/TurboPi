"""Static Denoising Loop for CUDA Graph capture.

This module provides unrolled/captured denoising loops that eliminate
Python control flow overhead during inference.

Key features:
- Unrolled N-step denoising as single CUDA Graph
- Static timestep schedule
- Zero-copy buffer management

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for diffusion denoising."""
    num_steps: int = 3
    action_horizon: int = 50
    action_dim: int = 32
    batch_size: int = 1
    dtype: torch.dtype = torch.bfloat16


class UnrolledDenoiseLoop(nn.Module):
    """Unrolled denoising loop that can be captured as CUDA Graph.

    Instead of a Python for-loop with N iterations, this module explicitly
    unrolls the loop into N sequential operations that can be captured
    as a single CUDA Graph.

    This eliminates:
    - Python loop overhead
    - Per-step CUDA kernel launch latency
    - Dynamic control flow that breaks graph capture
    """

    def __init__(
        self,
        denoise_step_fn: Callable,
        config: DiffusionConfig,
        device: torch.device,
    ):
        """
        Args:
            denoise_step_fn: Function that takes (state, x_t, timestep) -> v_t
            config: Diffusion configuration
            device: CUDA device
        """
        super().__init__()
        self.denoise_step_fn = denoise_step_fn
        self.config = config
        self.device = device

        # Pre-compute timestep schedule
        # For flow matching: t goes from 1.0 to 0.0
        dt = -1.0 / config.num_steps
        self.dt = dt

        # Pre-allocate timestep tensors for each step
        self.timesteps = nn.ParameterList([
            nn.Parameter(
                torch.tensor([1.0 + i * dt] * config.batch_size,
                            dtype=torch.float32, device=device),
                requires_grad=False
            )
            for i in range(config.num_steps)
        ])

        # Static buffers
        self._graph_captured = False
        self._loop_graph: Optional[torch.cuda.CUDAGraph] = None

        # Intermediate buffers for unrolled loop
        self._static_state: Optional[torch.Tensor] = None
        self._static_x_t: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None

    def _create_static_buffers(self):
        """Create static buffers for graph capture."""
        self._static_state = torch.zeros(
            self.config.batch_size, 32,  # max_state_dim
            dtype=self.config.dtype,
            device=self.device,
        )
        self._static_x_t = torch.zeros(
            self.config.batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            dtype=self.config.dtype,
            device=self.device,
        )
        self._static_output = torch.zeros_like(self._static_x_t)

    def _unrolled_forward(self) -> torch.Tensor:
        """Unrolled N-step denoising - this gets captured.

        Instead of:
            for i in range(N):
                v_t = denoise(x_t, timesteps[i])
                x_t = x_t + dt * v_t

        We have:
            v_0 = denoise(x_t, timesteps[0])
            x_t = x_t + dt * v_0
            v_1 = denoise(x_t, timesteps[1])
            x_t = x_t + dt * v_1
            ...
        """
        x_t = self._static_x_t

        # Step 0
        v_t = self.denoise_step_fn(self._static_state, x_t, self.timesteps[0])
        x_t = x_t + self.dt * v_t

        # Step 1
        if self.config.num_steps > 1:
            v_t = self.denoise_step_fn(self._static_state, x_t, self.timesteps[1])
            x_t = x_t + self.dt * v_t

        # Step 2
        if self.config.num_steps > 2:
            v_t = self.denoise_step_fn(self._static_state, x_t, self.timesteps[2])
            x_t = x_t + self.dt * v_t

        # Step 3
        if self.config.num_steps > 3:
            v_t = self.denoise_step_fn(self._static_state, x_t, self.timesteps[3])
            x_t = x_t + self.dt * v_t

        # Step 4
        if self.config.num_steps > 4:
            v_t = self.denoise_step_fn(self._static_state, x_t, self.timesteps[4])
            x_t = x_t + self.dt * v_t

        return x_t

    def capture_graph(self, warmup_iters: int = 3):
        """Capture the entire unrolled loop as CUDA Graph."""
        if self._graph_captured:
            return

        self._create_static_buffers()

        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = self._unrolled_forward()
        torch.cuda.synchronize()

        # Capture
        self._loop_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._loop_graph):
            output = self._unrolled_forward()
            self._static_output.copy_(output)

        torch.cuda.synchronize()
        self._graph_captured = True

    def forward(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Execute captured denoising loop.

        Args:
            state: (batch, state_dim)
            noise: (batch, action_horizon, action_dim) - initial noise

        Returns:
            Denoised actions (batch, action_horizon, action_dim)
        """
        if not self._graph_captured:
            raise RuntimeError("Graph not captured. Call capture_graph() first.")

        # Copy inputs
        self._static_state.copy_(state)
        self._static_x_t.copy_(noise)

        # Replay
        self._loop_graph.replay()

        return self._static_output.clone()


class CapturedDenoiseChain(nn.Module):
    """Capture N separate denoise steps as a chain of graphs.

    This is an alternative approach where each denoise step is captured
    separately, but they share the same input/output buffers for zero-copy
    chaining.

    Pros:
    - More flexible (can change num_steps)
    - Easier to debug

    Cons:
    - N graph replays instead of 1
    - Still has some Python overhead between steps
    """

    def __init__(
        self,
        denoise_step_fn: Callable,
        config: DiffusionConfig,
        device: torch.device,
    ):
        super().__init__()
        self.denoise_step_fn = denoise_step_fn
        self.config = config
        self.device = device

        self.dt = -1.0 / config.num_steps

        # Static buffers - shared across all steps
        self._static_state = torch.zeros(
            config.batch_size, 32,
            dtype=config.dtype,
            device=device,
        )
        self._static_x_t = torch.zeros(
            config.batch_size,
            config.action_horizon,
            config.action_dim,
            dtype=config.dtype,
            device=device,
        )
        self._static_v_t = torch.zeros_like(self._static_x_t)

        # Per-step timestep buffers
        self._static_timesteps = [
            torch.tensor([1.0 + i * self.dt] * config.batch_size,
                        dtype=torch.float32, device=device)
            for i in range(config.num_steps)
        ]

        # Per-step graphs
        self._step_graphs: List[torch.cuda.CUDAGraph] = []
        self._graphs_captured = False

    def _capture_single_step(self, step_idx: int, warmup_iters: int = 3):
        """Capture a single denoise step."""
        # Set timestep for this step
        timestep = self._static_timesteps[step_idx]

        # Warmup
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = self.denoise_step_fn(
                    self._static_state,
                    self._static_x_t,
                    timestep,
                )
        torch.cuda.synchronize()

        # Capture
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph):
            v_t = self.denoise_step_fn(
                self._static_state,
                self._static_x_t,
                timestep,
            )
            self._static_v_t.copy_(v_t)
            # Euler update in-place
            self._static_x_t.add_(self._static_v_t, alpha=self.dt)

        torch.cuda.synchronize()
        return graph

    def capture_graphs(self, warmup_iters: int = 3):
        """Capture all step graphs."""
        if self._graphs_captured:
            return

        self._step_graphs = []
        for step_idx in range(self.config.num_steps):
            graph = self._capture_single_step(step_idx, warmup_iters)
            self._step_graphs.append(graph)

        self._graphs_captured = True

    def forward(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Execute chain of captured denoise steps."""
        if not self._graphs_captured:
            raise RuntimeError("Graphs not captured. Call capture_graphs() first.")

        # Copy inputs
        self._static_state.copy_(state)
        self._static_x_t.copy_(noise)

        # Replay each step (minimal Python overhead)
        for graph in self._step_graphs:
            graph.replay()

        return self._static_x_t.clone()


class FullGraphedDenoise(nn.Module):
    """The ultimate: entire denoise loop as single CUDA Graph.

    This captures everything from noise input to action output in one graph:
    - embed_suffix for each step
    - expert forward for each step
    - action_out_proj for each step
    - Euler updates

    Requirements:
    - Fixed num_steps
    - Fixed batch_size
    - Static KV cache (prefix already computed)
    """

    def __init__(
        self,
        model,  # PI0Pytorch
        prefix_kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor,
        config: DiffusionConfig,
        device: torch.device,
    ):
        super().__init__()
        self.model = model
        self.prefix_kv_cache = prefix_kv_cache
        self.prefix_pad_masks = prefix_pad_masks
        self.config = config
        self.device = device

        self.dt = -1.0 / config.num_steps

        # Static buffers
        self._static_state = torch.zeros(
            config.batch_size, 32,
            dtype=config.dtype, device=device,
        )
        self._static_noise = torch.zeros(
            config.batch_size, config.action_horizon, config.action_dim,
            dtype=config.dtype, device=device,
        )
        self._static_output = torch.zeros_like(self._static_noise)

        # Working buffer for x_t
        self._working_x_t = torch.zeros_like(self._static_noise)

        # Pre-computed timesteps
        self._timesteps = [
            torch.tensor([1.0 + i * self.dt] * config.batch_size,
                        dtype=torch.float32, device=device)
            for i in range(config.num_steps)
        ]

        self._graph_captured = False
        self._full_graph: Optional[torch.cuda.CUDAGraph] = None

    def _denoise_step(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Single denoise step using model's denoise_step_with_cache."""
        return self.model.denoise_step_with_cache(
            self._static_state,
            self.prefix_kv_cache,
            self.prefix_pad_masks,
            x_t,
            timestep,
        )

    def _full_denoise_forward(self) -> torch.Tensor:
        """Full denoising loop - gets captured as single graph."""
        x_t = self._working_x_t.clone()

        for i in range(self.config.num_steps):
            v_t = self._denoise_step(x_t, self._timesteps[i])
            x_t = x_t + self.dt * v_t

        return x_t

    def capture_graph(self, warmup_iters: int = 3):
        """Capture the full denoise loop."""
        if self._graph_captured:
            return

        # Warmup
        for _ in range(warmup_iters):
            self._working_x_t.copy_(self._static_noise)
            with torch.no_grad():
                _ = self._full_denoise_forward()
        torch.cuda.synchronize()

        # Capture
        self._full_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._full_graph):
            self._working_x_t.copy_(self._static_noise)
            output = self._full_denoise_forward()
            self._static_output.copy_(output)

        torch.cuda.synchronize()
        self._graph_captured = True

    def forward(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Execute full captured denoise loop."""
        if not self._graph_captured:
            raise RuntimeError("Graph not captured. Call capture_graph() first.")

        self._static_state.copy_(state)
        self._static_noise.copy_(noise)

        self._full_graph.replay()

        return self._static_output.clone()
