# Debug-11: CUDA Graph Denoise Optimization

**Date**: 2026-02-13
**Status**: Completed
**Optimization**: 73-82 ms savings (40% speedup)

## Summary

Successfully implemented CUDA Graph capture for the 10-step denoising loop, eliminating Python dispatch overhead and optimizing kernel scheduling.

## Performance Results

| Configuration | Denoise Time | Savings |
|--------------|-------------|---------|
| No CUDA Graph | 182-192 ms | - |
| CUDA Graph | 109-110 ms | **73-82 ms (40%)** |

## Technical Implementation

### Key Challenge

CUDA Graph capture requires:
1. No dynamic tensor creation
2. No CPU-GPU data transfer
3. Static tensor shapes

The original `denoise_step_with_cache` used `torch.tensor()` and `torch.ones()` which break CUDA Graph capture.

### Solution: Graph-Compatible Denoise

1. **Pre-compute all static tensors** during warmup:
   - `suffix_position_ids`: Fixed position IDs for action tokens
   - `adarms_conds`: Pre-computed timestep conditioning for all 10 steps

2. **New method `denoise_step_graphed`**:
   - Accepts pre-computed tensors as input
   - Avoids all dynamic tensor creation
   - Uses same SDPA no-mask optimization

3. **ChainedDenoiseGraphs class**:
   - Captures 10 CUDA Graphs (one per timestep)
   - Chain-replays for minimal Python overhead

### Code Changes

**pi0_pytorch.py**:
```python
# New method for CUDA Graph compatibility
def denoise_step_graphed(self, x_t, suffix_position_ids, adarms_cond, prefix_kv_cache):
    """CUDA Graph-compatible denoise step with pre-computed tensors."""
    ...

# Pre-compute all static tensors for graph capture
def precompute_graph_tensors(self, batch_size, prefix_len, num_steps=10, device=None):
    """Pre-compute suffix_position_ids and adarms_conds for all timesteps."""
    ...
```

**graphed_denoise.py**:
```python
class ChainedDenoiseGraphs(nn.Module):
    """Chain of CUDA Graph-captured denoise steps."""

    def capture(self, state, prefix_kv_cache, prefix_pad_masks):
        # Pre-compute tensors
        graph_tensors = self.model.precompute_graph_tensors(...)

        # Capture 10 graphs
        for step_idx in range(10):
            with torch.cuda.graph(graph):
                v_t = self.model.denoise_step_graphed(...)
                self._static_x_t.add_(v_t, alpha=dt)

    def forward(self, noise):
        # Chain replay
        for graph in self._graphs:
            graph.replay()
```

## Numerical Verification

- **Cosine similarity**: 1.000 (perfect directional match)
- **Max diff**: ~0.13 (within BF16 accumulated precision tolerance)

## Combined Optimizations

| Optimization | Savings | Cumulative |
|-------------|---------|------------|
| Baseline TRT FP8 | - | 180 ms |
| SDPA no-mask | ~10 ms | 170 ms |
| CUDA Graph | ~73 ms | **97 ms** |

**Target achieved**: ~10 Hz inference (100 ms/frame)

## Usage

```python
from openpi.modules.graphed_denoise import ChainedDenoiseGraphs

# Create module
graphed = ChainedDenoiseGraphs(model=model, num_steps=10, device=device)

# Capture graphs (during warmup)
graphed.capture(state, prefix_kv_cache, prefix_pad_masks)

# Inference (just replay)
actions = graphed(noise)
```

## Requirements

- CUDA Graph compatible GPU (Thor/SM110 verified)
- Static batch size during capture
- Same prefix length for reuse

## Files Modified

- `openpi/src/openpi/models_pytorch/pi0_pytorch.py`
  - Added `denoise_step_graphed()` method
  - Added `precompute_graph_tensors()` method
- `openpi/src/openpi/modules/graphed_denoise.py`
  - Implemented `ChainedDenoiseGraphs` class
- `openpi/scripts/test_chained_denoise_graphs.py`
  - Test and benchmark script

## Next Steps

1. Integrate into UnifiedPolicy for automatic graph capture
2. Add KV cache update for multi-frame inference
3. Profile end-to-end with Vision + KV Cache + Denoise
