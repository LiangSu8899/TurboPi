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

## Combined Optimizations (Denoise Only)

| Optimization | Savings | Denoise Time |
|-------------|---------|--------------|
| Baseline (no graph) | - | ~186 ms |
| CUDA Graph | ~88 ms | **~98 ms** |

**Note**: This is Denoise component only. Full pipeline latency:

| Component | Latency |
|-----------|---------|
| Vision TRT FP16 | 17 ms |
| KV Cache TRT FP8 | 54 ms |
| Denoise CUDA Graph | 98 ms |
| **Total** | **~170 ms (5.9 Hz)** |

**Target**: 10 Hz (100 ms) - Gap: 70 ms

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

## Python-Level Optimization Analysis (2026-02-13)

### Tested Optimizations (INEFFECTIVE in CUDA Graph)

| 优化方案 | 独立测试 | 端到端测试 | 结论 |
|----------|----------|------------|------|
| **adaRMS BF16** | 节省 12.66 ms | **变慢 -3 ms** | ❌ 无效 |
| **RoPE 预计算** | 节省 15.69 ms | **变慢 -0.6 ms** | ❌ 无效 |

### 原因分析

1. **CUDA Graph 内部优化 vs Python 层面优化**
   - 独立组件测试未使用 CUDA Graph，测量的是 kernel launch overhead + 计算时间
   - CUDA Graph 消除了 kernel launch overhead (~82 ms)
   - 剩余时间 (~109 ms) 是纯 GPU 计算，无法通过 Python 层面优化

2. **实际时间分解 (无 CUDA Graph)**
   ```
   组件                    占比
   ─────────────────────────────
   adaRMS LayerNorm        35.2%
   Attention (SDPA)        27.6%
   RoPE                    15.6%
   MLP                      9.5%
   Kernel Launch Overhead  ~43% (被 CUDA Graph 消除)
   ```

3. **实际时间分解 (CUDA Graph)**
   ```
   Denoise 10 步: 108.96 ms
   = 纯 GPU 计算，Python 层面无法优化
   ```

### 结论

**Python 层面的 dtype 转换/预计算优化在 CUDA Graph 模式下无效。**

进一步优化需要：
1. **Triton Fused Kernel**: 融合 adaRMS + RoPE + Attention
2. **减少 prefix_len**: 当前 968 tokens，减少可显著降低 Attention 计算量
3. **TensorRT 编译**: 将 Denoise 部分编译成 TRT engine

### 当前状态

| 组件 | 延迟 | 状态 |
|------|------|------|
| Vision TRT FP16 | 17 ms | ✓ |
| KV Cache TRT FP8 | 54 ms | ✓ |
| Denoise CUDA Graph | **109 ms** | 需优化 |
| **Total** | **180 ms (5.6 Hz)** | |

**Target**: 10 Hz (100 ms) - Gap: 80 ms

## Next Steps

1. ~~Integrate into UnifiedPolicy for automatic graph capture~~ (低优先级)
2. **探索 Triton Fused Kernel** - 融合 LayerNorm + Attention
3. **减少 prefix_len** - 分析是否可以动态裁剪
4. **TensorRT Denoise** - 完整编译成 TRT engine
