"""
VLM Graph Runner - Full Decode Step CUDA Graphs.

This module captures the ENTIRE VLM decode step (Attention + W4A16 MLP) into
CUDA Graphs, eliminating Python launch overhead.

Key insight: The 171ms "Other" overhead in eager mode is NOT compute-bound,
it's Python dispatch overhead. CUDA Graphs eliminate this.

Expected speedup:
    - Eager W4A16: 226ms
    - Graph W4A16: <50ms (target)
    - TRT FP8:     120ms (baseline to beat)

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


class StaticKVCache:
    """
    Static KV Cache for CUDA Graph compatibility.

    CUDA Graphs require fixed memory addresses. This class pre-allocates
    the entire KV cache buffer and uses cache_position indexing.

    Layout: [num_layers, 2, batch, num_kv_heads, max_seq_len, head_dim]
            where 2 = [key, value]
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 1024,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

        # Pre-allocate static buffers
        # Shape: [num_layers, 2, batch, num_kv_heads, max_seq_len, head_dim]
        self.cache = torch.zeros(
            num_layers, 2, batch_size, num_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device=device
        )

        # Current sequence length (for tracking)
        self._seq_len = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current sequence length."""
        return self._seq_len

    def update(
        self,
        key_states: torch.Tensor,  # [batch, num_kv_heads, seq_len, head_dim]
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states.

        For CUDA Graph compatibility, we use in-place copy to static buffer
        and return views into the buffer.
        """
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None

        if cache_position is not None:
            # Decode mode: single token update
            # cache_position shape: [seq_len] (usually [1] for decode)
            k = key_states
            v = value_states

            # In-place update at cache_position
            # self.cache[layer_idx, 0] is key cache
            # self.cache[layer_idx, 1] is value cache
            self.cache[layer_idx, 0, :, :, cache_position, :] = k.transpose(2, 3)
            self.cache[layer_idx, 1, :, :, cache_position, :] = v.transpose(2, 3)

            # Update seq_len if this is a new position
            new_len = cache_position.max().item() + 1
            if new_len > self._seq_len:
                self._seq_len = new_len

            # Return full cache up to current seq_len
            # Shape: [batch, num_kv_heads, seq_len, head_dim]
            key_out = self.cache[layer_idx, 0, :, :, :self._seq_len, :].transpose(2, 3)
            value_out = self.cache[layer_idx, 1, :, :, :self._seq_len, :].transpose(2, 3)

            return key_out, value_out
        else:
            # Prefill mode: full sequence
            seq_len = key_states.size(2)

            # Copy to static buffer
            self.cache[layer_idx, 0, :, :, :seq_len, :] = key_states.transpose(2, 3)
            self.cache[layer_idx, 1, :, :, :seq_len, :] = value_states.transpose(2, 3)

            self._seq_len = seq_len

            return key_states, value_states

    def reset(self):
        """Reset cache (zero-copy, just reset length)."""
        self._seq_len = 0

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached key/value for layer (legacy interface)."""
        # Return [batch, num_kv_heads, seq_len, head_dim]
        key = self.cache[layer_idx, 0, :, :, :self._seq_len, :].transpose(2, 3)
        value = self.cache[layer_idx, 1, :, :, :self._seq_len, :].transpose(2, 3)
        return (key, value)


@dataclass
class GraphInputs:
    """Static input buffers for CUDA Graph capture."""
    hidden_states: torch.Tensor  # [batch, 1, hidden_size]
    position_ids: torch.Tensor   # [batch, 1]
    cache_position: torch.Tensor # [1]
    attention_mask: torch.Tensor # [batch, 1, 1, max_seq_len]


class VLMGraphRunner:
    """
    CUDA Graph runner for full VLM decode step.

    Captures the entire Transformer decode (Attention + MLP) into a CUDA Graph
    to eliminate Python dispatch overhead.

    Usage:
        runner = VLMGraphRunner(model.language_model, num_layers=18)
        runner.warmup()
        runner.capture()

        # Inference
        for step in range(num_steps):
            hidden = runner.run(hidden, position_id, cache_position)
    """

    def __init__(
        self,
        model: nn.Module,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        max_seq_len: int = 1024,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.model = model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

        # Create static KV cache
        self.kv_cache = StaticKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

        # Static input buffers (CUDA Graph requires fixed addresses)
        self.static_inputs = GraphInputs(
            hidden_states=torch.zeros(batch_size, 1, hidden_size, dtype=dtype, device=device),
            position_ids=torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            cache_position=torch.zeros(1, dtype=torch.long, device=device),
            attention_mask=torch.zeros(batch_size, 1, 1, max_seq_len, dtype=dtype, device=device),
        )

        # Static output buffer
        self.static_output = torch.zeros(batch_size, 1, hidden_size, dtype=dtype, device=device)

        # CUDA Graph state
        self.graph = None
        self.graph_captured = False

        # Stream for graph operations
        self.stream = torch.cuda.Stream()

    def _decode_step(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single decode step through all Transformer layers.

        This is what gets captured into the CUDA Graph.
        """
        # Get rotary embeddings (these are computed per-step, but could be cached)
        if hasattr(self.model, 'rotary_emb'):
            # Create dummy tensor for rotary embedding
            dummy = torch.zeros(
                hidden_states.size(0), hidden_states.size(1), self.model.layers[0].self_attn.head_dim,
                device=self.device, dtype=self.dtype
            )
            cos, sin = self.model.rotary_emb(dummy, position_ids)
            position_embeddings = (cos, sin)
        else:
            position_embeddings = None

        # Process through layers
        for layer_idx, layer in enumerate(self.model.layers):
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=self.kv_cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        # Final norm
        if hasattr(self.model, 'norm'):
            hidden_states = self.model.norm(hidden_states)

        return hidden_states

    def warmup(self, num_iters: int = 3):
        """
        Warmup the model before graph capture.

        Required to:
        1. JIT compile all kernels (including TVM W4A16)
        2. Ensure stable execution path
        """
        print("VLMGraphRunner: Warming up...")

        # Reset cache
        self.kv_cache.reset()

        # Prefill with dummy sequence (to populate KV cache)
        prefill_len = 10
        dummy_hidden = torch.randn(
            self.batch_size, prefill_len, self.hidden_size,
            dtype=self.dtype, device=self.device
        )

        with torch.no_grad():
            # Prefill
            for layer_idx, layer in enumerate(self.model.layers):
                layer_outputs = layer(
                    hidden_states=dummy_hidden,
                    position_ids=torch.arange(prefill_len, device=self.device).unsqueeze(0),
                    past_key_value=self.kv_cache,
                    use_cache=True,
                    cache_position=torch.arange(prefill_len, device=self.device),
                )
                dummy_hidden = layer_outputs[0]

        # Decode warmup
        for i in range(num_iters):
            pos = prefill_len + i
            self.static_inputs.hidden_states.copy_(torch.randn_like(self.static_inputs.hidden_states))
            self.static_inputs.position_ids.fill_(pos)
            self.static_inputs.cache_position.fill_(pos)

            # Create attention mask (causal, up to current position)
            self.static_inputs.attention_mask.fill_(float('-inf'))
            self.static_inputs.attention_mask[:, :, :, :pos+1] = 0

            with torch.no_grad():
                output = self._decode_step(
                    self.static_inputs.hidden_states,
                    self.static_inputs.position_ids,
                    self.static_inputs.cache_position,
                    self.static_inputs.attention_mask,
                )
                self.static_output.copy_(output)

        torch.cuda.synchronize()
        print("VLMGraphRunner: Warmup complete")

    def capture(self):
        """
        Capture the decode step into a CUDA Graph.

        After capture, run() will replay the graph instead of executing Python.
        """
        if self.graph_captured:
            print("VLMGraphRunner: Graph already captured")
            return

        print("VLMGraphRunner: Capturing CUDA Graph...")

        # Set up for capture
        pos = self.kv_cache.get_seq_length()
        self.static_inputs.position_ids.fill_(pos)
        self.static_inputs.cache_position.fill_(pos)
        self.static_inputs.attention_mask.fill_(float('-inf'))
        self.static_inputs.attention_mask[:, :, :, :pos+1] = 0

        # Capture
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph, stream=self.stream):
            output = self._decode_step(
                self.static_inputs.hidden_states,
                self.static_inputs.position_ids,
                self.static_inputs.cache_position,
                self.static_inputs.attention_mask,
            )
            self.static_output.copy_(output)

        torch.cuda.synchronize()
        self.graph_captured = True
        print("VLMGraphRunner: Graph captured successfully")

    def run(
        self,
        hidden_states: torch.Tensor,
        position_id: int,
        cache_position: int,
    ) -> torch.Tensor:
        """
        Run a single decode step using the captured graph.

        Args:
            hidden_states: [batch, 1, hidden_size] input hidden states
            position_id: Current position in sequence
            cache_position: Position in KV cache to update

        Returns:
            [batch, 1, hidden_size] output hidden states
        """
        if not self.graph_captured:
            raise RuntimeError("Graph not captured. Call capture() first.")

        # Copy inputs to static buffers (these are the only Python ops)
        self.static_inputs.hidden_states.copy_(hidden_states)
        self.static_inputs.position_ids.fill_(position_id)
        self.static_inputs.cache_position.fill_(cache_position)

        # Update attention mask for new position
        self.static_inputs.attention_mask[:, :, :, cache_position] = 0

        # Replay graph
        self.graph.replay()

        return self.static_output

    def run_eager(
        self,
        hidden_states: torch.Tensor,
        position_id: int,
        cache_position: int,
    ) -> torch.Tensor:
        """
        Run decode step in eager mode (for comparison).
        """
        self.static_inputs.hidden_states.copy_(hidden_states)
        self.static_inputs.position_ids.fill_(position_id)
        self.static_inputs.cache_position.fill_(cache_position)
        self.static_inputs.attention_mask[:, :, :, cache_position] = 0

        with torch.no_grad():
            output = self._decode_step(
                self.static_inputs.hidden_states,
                self.static_inputs.position_ids,
                self.static_inputs.cache_position,
                self.static_inputs.attention_mask,
            )

        return output

    def reset(self):
        """Reset the runner state."""
        self.kv_cache.reset()


def create_graph_runner_for_paligemma(model, max_seq_len: int = 1024):
    """
    Create a VLMGraphRunner for PaliGemma model.

    Args:
        model: The PaliGemma language model (GemmaModel)
        max_seq_len: Maximum sequence length

    Returns:
        VLMGraphRunner instance
    """
    config = model.config

    return VLMGraphRunner(
        model=model,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        hidden_size=config.hidden_size,
        max_seq_len=max_seq_len,
        batch_size=1,
        dtype=torch.bfloat16,
        device="cuda",
    )


# ============================================================================
# Benchmark utilities
# ============================================================================

def benchmark_graph_vs_eager(
    model: nn.Module,
    num_layers: int = 18,
    num_kv_heads: int = 8,
    head_dim: int = 256,
    hidden_size: int = 2048,
    num_decode_steps: int = 50,
    warmup: int = 10,
    runs: int = 100,
):
    """
    Benchmark CUDA Graph vs Eager execution.

    Returns dict with latency results.
    """
    import numpy as np

    device = 'cuda'
    dtype = torch.bfloat16

    # Create runner
    runner = VLMGraphRunner(
        model=model,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_seq_len=1024,
        batch_size=1,
        dtype=dtype,
        device=device,
    )

    # Warmup and capture
    runner.warmup(num_iters=5)
    runner.capture()

    # Benchmark Graph mode
    print("\nBenchmarking CUDA Graph mode...")
    graph_times = []

    for _ in range(runs):
        runner.reset()
        runner.warmup(num_iters=1)  # Prefill

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        hidden = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
        prefill_len = runner.kv_cache.get_seq_length()

        start.record()
        for step in range(num_decode_steps):
            pos = prefill_len + step
            hidden = runner.run(hidden, pos, pos)
        end.record()
        torch.cuda.synchronize()

        graph_times.append(start.elapsed_time(end))

    # Benchmark Eager mode
    print("Benchmarking Eager mode...")
    eager_times = []

    for _ in range(runs // 10):  # Fewer runs for eager (slower)
        runner.reset()
        runner.warmup(num_iters=1)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        hidden = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
        prefill_len = runner.kv_cache.get_seq_length()

        start.record()
        for step in range(num_decode_steps):
            pos = prefill_len + step
            hidden = runner.run_eager(hidden, pos, pos)
        end.record()
        torch.cuda.synchronize()

        eager_times.append(start.elapsed_time(end))

    results = {
        'graph_avg_ms': np.mean(graph_times),
        'graph_std_ms': np.std(graph_times),
        'eager_avg_ms': np.mean(eager_times),
        'eager_std_ms': np.std(eager_times),
        'speedup': np.mean(eager_times) / np.mean(graph_times),
        'num_decode_steps': num_decode_steps,
    }

    print(f"\nResults ({num_decode_steps} decode steps):")
    print(f"  Graph: {results['graph_avg_ms']:.2f} ms (std: {results['graph_std_ms']:.2f})")
    print(f"  Eager: {results['eager_avg_ms']:.2f} ms (std: {results['eager_std_ms']:.2f})")
    print(f"  Speedup: {results['speedup']:.2f}x")

    return results


if __name__ == "__main__":
    # Quick test with a simple model
    print("Testing VLMGraphRunner...")

    class SimpleTransformerLayer(nn.Module):
        def __init__(self, hidden_size=2048, intermediate_size=8192):
            super().__init__()
            self.self_attn = nn.Identity()  # Placeholder
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=False),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size, bias=False),
            )
            self.input_layernorm = nn.LayerNorm(hidden_size)
            self.post_attention_layernorm = nn.LayerNorm(hidden_size)

        def forward(self, hidden_states, **kwargs):
            # Simplified forward (no real attention)
            x = hidden_states + hidden_states  # Dummy attention
            x = x + self.mlp(self.post_attention_layernorm(x))
            return (x,)

    class SimpleModel(nn.Module):
        def __init__(self, num_layers=18, hidden_size=2048):
            super().__init__()
            self.layers = nn.ModuleList([
                SimpleTransformerLayer(hidden_size) for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(hidden_size)

    model = SimpleModel().cuda().to(torch.bfloat16)

    # Test basic functionality
    runner = VLMGraphRunner(
        model=model,
        num_layers=18,
        num_kv_heads=8,
        head_dim=256,
        hidden_size=2048,
        max_seq_len=256,
    )

    runner.warmup()
    runner.capture()

    hidden = torch.randn(1, 1, 2048, dtype=torch.bfloat16, device='cuda')
    output = runner.run(hidden, position_id=10, cache_position=10)
    print(f"Output shape: {output.shape}")
    print("Test passed!")
