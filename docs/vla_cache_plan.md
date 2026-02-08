# VLA-Cache Integration Plan for Pi0.5

## Overview

**Goal**: Integrate VLA-Cache (arXiv:2502.02175) into Pi0.5 to accelerate KV-cache computation by reusing static visual patches.

**Paper**: https://arxiv.org/pdf/2502.02175
**Reference**: https://github.com/siyuhsu/vla-cache

### Why VLA-Cache (Not Falcon)?

| Approach | What It Reuses | Observation Handling | Result on VLA |
|----------|---------------|---------------------|---------------|
| **Falcon** | Diffusion intermediate states | Ignores observation change | ❌ 0% accuracy |
| **VLA-Cache** | Vision patch KV-cache | Selective patch-level reuse | ✅ Expected to work |

### Expected Performance Gains

| Component | Current | With VLA-Cache | Speedup |
|-----------|---------|---------------|---------|
| Prefix KV Cache | 52 ms | ~25-30 ms | ~2x |
| **Total Pipeline** | **174 ms** | **~145-155 ms** | ~1.15x |
| **Frequency** | **5.7 Hz** | **~6.5-7 Hz** | +15% |

---

## Phase 1: Static Patch Detection (Day 1)

### Goal
Implement cosine similarity based detection of static patches between consecutive frames.

### Implementation

**File**: `openpi/src/openpi/inference/vla_cache/patch_detector.py`

```python
import torch
import torch.nn.functional as F

def patchify(image: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    """Convert image [B, C, H, W] to patches [B, num_patches, C*patch_size*patch_size]"""
    B, C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0

    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # [B, C, H//ps, W//ps, ps, ps] -> [B, num_patches, C*ps*ps]
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * patch_size * patch_size)
    return patches

def find_static_patches(
    img_0: torch.Tensor,  # [B, C, H, W] current frame
    img_1: torch.Tensor,  # [B, C, H, W] previous frame
    patch_size: int = 14,
    sim_threshold: float = 0.996,
    top_k: int = 150,
) -> torch.Tensor:
    """
    Find patches that are similar between consecutive frames.

    Returns:
        mask: [num_patches] boolean tensor indicating static patches
        indices: [k] indices of static patches
    """
    patches_0 = patchify(img_0, patch_size)  # [B, N, D]
    patches_1 = patchify(img_1, patch_size)  # [B, N, D]

    # L2 normalize
    patches_0_norm = F.normalize(patches_0, dim=-1)
    patches_1_norm = F.normalize(patches_1, dim=-1)

    # Cosine similarity per patch
    similarity = (patches_0_norm * patches_1_norm).sum(dim=-1)  # [B, N]

    # Find static patches above threshold
    static_mask = similarity[0] >= sim_threshold
    static_indices = torch.where(static_mask)[0]

    # Limit to top_k (sorted by similarity)
    if len(static_indices) > top_k:
        sim_values = similarity[0, static_indices]
        topk_idx = torch.argsort(sim_values, descending=True)[:top_k]
        static_indices = static_indices[topk_idx]

    return static_mask, static_indices
```

### Verification (V1)

**Script**: `openpi/scripts/verify_patch_detector.py`

```python
"""
Verification:
1. Load 10 consecutive frames from LIBERO rollout
2. Compute static patches between each pair
3. Visualize static patches on image (should be background/stable regions)
4. Verify statistics: expect 50-80% of patches are static
"""

# Expected metrics:
# - Static patch ratio: 50-80% (higher for slow robot movements)
# - Cosine similarity distribution: bimodal (static ~0.99+, dynamic ~0.8-0.95)
# - Visualization: static patches should cover non-robot/non-hand regions
```

### Pass Criteria
- [ ] Static patch ratio: 50-80% per frame pair
- [ ] Visualization shows static patches correctly identify background
- [ ] No memory leaks (run 1000 frames)
- [ ] Latency < 1ms per detection

---

## Phase 2: Attention-Based Task Relevance Filtering (Day 2)

### Goal
Filter static patches to keep only task-relevant ones using text-to-vision attention weights.

### Implementation

**File**: `openpi/src/openpi/inference/vla_cache/attention_filter.py`

```python
import torch

def compute_attention_importance(
    attention_weights: torch.Tensor,  # [num_heads, seq_len, seq_len]
    num_vision_tokens: int = 256,
    num_text_tokens: int = 50,
) -> torch.Tensor:
    """
    Compute importance score for each vision patch based on
    how much text tokens attend to them.

    Returns:
        importance: [num_vision_tokens] normalized importance scores
    """
    # Vision tokens: positions 1 to 1+256 (after BOS)
    # Text tokens: positions after vision
    v_start, v_end = 1, 1 + num_vision_tokens
    t_start, t_end = v_end, v_end + num_text_tokens

    # Extract text -> vision attention: [heads, text_tokens, vision_tokens]
    text_to_vision = attention_weights[:, t_start:t_end, v_start:v_end]

    # Average across heads and text tokens
    importance = text_to_vision.mean(dim=(0, 1))  # [num_vision_tokens]

    # Normalize
    importance = importance / importance.sum()

    return importance

def filter_task_relevant_patches(
    static_indices: torch.Tensor,    # [k] static patch indices
    importance_scores: torch.Tensor, # [num_vision_tokens]
    top_k: int = 120,
    min_importance: float = 0.001,
) -> torch.Tensor:
    """
    From static patches, keep only those with low task importance.
    (High importance patches should be recomputed for accuracy)

    Returns:
        reusable_indices: [m] indices of patches safe to reuse
    """
    # Get importance of static patches
    static_importance = importance_scores[static_indices]

    # Keep patches below importance threshold (task-irrelevant)
    irrelevant_mask = static_importance < min_importance
    reusable_indices = static_indices[irrelevant_mask]

    # Limit to top_k (sorted by lowest importance)
    if len(reusable_indices) > top_k:
        importance_values = importance_scores[reusable_indices]
        topk_idx = torch.argsort(importance_values)[:top_k]
        reusable_indices = reusable_indices[topk_idx]

    return reusable_indices
```

### Verification (V2)

**Script**: `openpi/scripts/verify_attention_filter.py`

```python
"""
Verification:
1. Run Pi0.5 forward pass on sample observation
2. Extract attention weights from Gemma layers
3. Compute importance scores for all vision patches
4. Visualize: high-importance patches should cover robot arm, target object
5. Verify filtering removes task-critical regions from reuse set
"""

# Expected metrics:
# - High importance patches: 10-30% (robot arm, gripper, target)
# - Low importance patches: 70-90% (background, furniture)
# - Filtering reduces reusable set by 10-30%
```

### Pass Criteria
- [ ] Attention extraction works on real Pi0.5 forward pass
- [ ] High importance patches correlate with robot/object regions
- [ ] Filtering correctly excludes task-critical areas
- [ ] No accuracy impact from excluding filtered patches

---

## Phase 3: Per-Layer KV Cache Scheduling (Day 3)

### Goal
Implement layer-wise cache reuse scheduling based on attention entropy.

### Implementation

**File**: `openpi/src/openpi/inference/vla_cache/layer_scheduler.py`

```python
import torch
import torch.nn.functional as F

def compute_layer_entropy(
    attention_weights: list[torch.Tensor],  # List of [heads, seq, seq] per layer
) -> torch.Tensor:
    """
    Compute attention entropy for each layer.
    High entropy = diffuse attention = safer to cache.
    Low entropy = focused attention = should recompute.

    Returns:
        entropy: [num_layers] normalized entropy scores
    """
    entropies = []
    for layer_attn in attention_weights:
        # Average across heads: [seq, seq]
        avg_attn = layer_attn.mean(dim=0)

        # Compute entropy for each query position
        # H = -sum(p * log(p))
        eps = 1e-10
        entropy = -torch.sum(avg_attn * torch.log(avg_attn + eps), dim=-1)

        # Average entropy across positions
        entropies.append(entropy.mean().item())

    entropy = torch.tensor(entropies)

    # Normalize to [0, 1]
    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-10)

    return entropy

def get_layer_reuse_schedule(
    layer_entropy: torch.Tensor,  # [num_layers]
    growth_factor: float = 0.55,
    min_reuse: float = 0.3,
    max_reuse: float = 0.95,
) -> torch.Tensor:
    """
    Convert entropy to reuse proportion per layer.
    Higher entropy -> higher reuse proportion.

    Returns:
        schedule: [num_layers] reuse proportions
    """
    # Apply growth factor smoothing
    smoothed = layer_entropy * growth_factor + 0.5 * (1 - growth_factor)

    # Clamp to [min_reuse, max_reuse]
    schedule = torch.clamp(smoothed, min=min_reuse, max=max_reuse)

    return schedule
```

### Verification (V3)

**Script**: `openpi/scripts/verify_layer_scheduler.py`

```python
"""
Verification:
1. Run Pi0.5 forward pass, collect attention from all layers
2. Compute entropy per layer
3. Plot entropy distribution across layers
4. Verify schedule: early layers (high entropy) -> high reuse
                    late layers (low entropy) -> lower reuse
"""

# Expected metrics:
# - Early layers (0-5): entropy ~0.8-0.95 (diffuse attention)
# - Middle layers (6-12): entropy ~0.5-0.7 (mixed)
# - Late layers (13-17): entropy ~0.3-0.5 (focused attention)
# - Schedule should reflect this pattern
```

### Pass Criteria
- [ ] Entropy computation matches expected pattern
- [ ] Schedule correctly maps entropy to reuse proportions
- [ ] Early layers get higher reuse, late layers lower

---

## Phase 4: KV Cache Integration (Day 4-5)

### Goal
Integrate patch-level KV cache reuse into Pi0.5's prefix computation.

### Implementation

**File**: `openpi/src/openpi/inference/vla_cache/kv_cache_manager.py`

```python
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class VLACacheState:
    """State for VLA-Cache across frames."""
    prev_image: Optional[torch.Tensor] = None      # Previous frame
    prev_kv_cache: Optional[dict] = None           # Previous prefix KV cache
    prev_attention: Optional[list] = None          # Previous attention weights
    reusable_indices: Optional[torch.Tensor] = None # Patch indices to reuse
    layer_schedule: Optional[torch.Tensor] = None   # Per-layer reuse proportions

class VLACacheManager:
    """Manages KV cache reuse across frames."""

    def __init__(
        self,
        sim_threshold: float = 0.996,
        static_top_k: int = 150,
        relevance_top_k: int = 120,
        importance_threshold: float = 0.001,
        growth_factor: float = 0.55,
    ):
        self.sim_threshold = sim_threshold
        self.static_top_k = static_top_k
        self.relevance_top_k = relevance_top_k
        self.importance_threshold = importance_threshold
        self.growth_factor = growth_factor

        self.state = VLACacheState()

    def update(
        self,
        current_image: torch.Tensor,
        current_kv_cache: dict,
        current_attention: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache state and compute what to reuse for next frame.

        Returns:
            reusable_indices: Patch indices safe to reuse
            layer_schedule: Per-layer reuse proportions
        """
        if self.state.prev_image is None:
            # First frame: no reuse possible
            self.state.prev_image = current_image.clone()
            self.state.prev_kv_cache = current_kv_cache
            self.state.prev_attention = current_attention
            return None, None

        # Step 1: Find static patches
        static_mask, static_indices = find_static_patches(
            current_image, self.state.prev_image,
            sim_threshold=self.sim_threshold,
            top_k=self.static_top_k,
        )

        # Step 2: Filter by task relevance
        importance = compute_attention_importance(
            self.state.prev_attention[-1],  # Use last layer attention
        )
        reusable_indices = filter_task_relevant_patches(
            static_indices, importance,
            top_k=self.relevance_top_k,
            min_importance=self.importance_threshold,
        )

        # Step 3: Compute layer schedule
        layer_entropy = compute_layer_entropy(self.state.prev_attention)
        layer_schedule = get_layer_reuse_schedule(
            layer_entropy, growth_factor=self.growth_factor,
        )

        # Update state for next frame
        self.state.prev_image = current_image.clone()
        self.state.prev_kv_cache = current_kv_cache
        self.state.prev_attention = current_attention
        self.state.reusable_indices = reusable_indices
        self.state.layer_schedule = layer_schedule

        return reusable_indices, layer_schedule

    def get_reusable_kv(
        self,
        layer_idx: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Get reusable K,V for specified layer.
        Only returns KV for reusable patch positions.
        """
        if self.state.prev_kv_cache is None or self.state.reusable_indices is None:
            return None

        # Get layer's cached K,V
        layer_kv = self.state.prev_kv_cache[layer_idx]
        k_cache, v_cache = layer_kv  # [B, heads, seq, head_dim]

        # Extract only reusable positions (add 1 for BOS offset)
        reuse_pos = self.state.reusable_indices + 1  # [m]
        k_reuse = k_cache[:, :, reuse_pos, :]
        v_reuse = v_cache[:, :, reuse_pos, :]

        return k_reuse, v_reuse, reuse_pos

    def reset(self):
        """Reset state for new episode."""
        self.state = VLACacheState()
```

### Modified Prefix Computation

**Changes to**: `openpi/src/openpi/models_pytorch/pi0_pytorch.py`

```python
def compute_prefix_kv_cache_with_reuse(
    self,
    prefix_embeddings: torch.Tensor,  # [B, seq, hidden]
    reusable_indices: torch.Tensor,   # [m] indices to reuse
    layer_schedule: torch.Tensor,     # [num_layers] reuse proportions
    cache_manager: VLACacheManager,
) -> dict:
    """
    Compute prefix KV cache with selective reuse.
    """
    B, seq_len, hidden = prefix_embeddings.shape
    all_layer_kv = {}

    x = prefix_embeddings
    for layer_idx, layer in enumerate(self.gemma_layers):
        # Get reuse proportion for this layer
        reuse_ratio = layer_schedule[layer_idx].item() if layer_schedule is not None else 0

        # Try to get reusable KV
        reusable_kv = cache_manager.get_reusable_kv(layer_idx)

        if reusable_kv is not None and reuse_ratio > 0:
            k_reuse, v_reuse, reuse_pos = reusable_kv

            # Compute K,V only for non-reusable positions
            all_positions = torch.arange(seq_len, device=x.device)
            compute_mask = ~torch.isin(all_positions, reuse_pos)
            compute_pos = all_positions[compute_mask]

            # Compute K,V for remaining positions
            x_compute = x[:, compute_pos, :]
            k_new, v_new = layer.self_attn.compute_kv(x_compute)

            # Merge reused and new K,V
            k_full = torch.zeros(B, self.num_heads, seq_len, self.head_dim, device=x.device)
            v_full = torch.zeros_like(k_full)

            k_full[:, :, reuse_pos, :] = k_reuse
            v_full[:, :, reuse_pos, :] = v_reuse
            k_full[:, :, compute_pos, :] = k_new
            v_full[:, :, compute_pos, :] = v_new

            all_layer_kv[layer_idx] = (k_full, v_full)
        else:
            # No reuse: compute everything
            k, v = layer.self_attn.compute_kv(x)
            all_layer_kv[layer_idx] = (k, v)

        # Forward through layer with full K,V
        k, v = all_layer_kv[layer_idx]
        x = layer.forward_with_kv(x, k, v)

    return all_layer_kv
```

### Verification (V4)

**Script**: `openpi/scripts/verify_kv_integration.py`

```python
"""
Verification (CRITICAL for accuracy):

1. Numerical correctness:
   - Run 10 frames with VLA-Cache
   - Run same 10 frames WITHOUT VLA-Cache (baseline)
   - Compare final actions: max L2 difference should be < 0.01

2. Speedup validation:
   - Measure prefix KV computation time with/without reuse
   - Expected: 40-60% reduction in prefix KV time

3. Memory check:
   - Ensure no memory leaks across 1000 frames
   - Peak memory should not exceed baseline + 10%
"""

def test_numerical_correctness():
    # Load model
    model = load_pi05_model()
    cache_manager = VLACacheManager()

    # Load test trajectory
    frames = load_libero_trajectory(num_frames=20)

    actions_with_cache = []
    actions_baseline = []

    for i, frame in enumerate(frames):
        # With VLA-Cache
        action_cache = model.infer_with_vla_cache(frame, cache_manager)
        actions_with_cache.append(action_cache)

        # Baseline (reset cache each time)
        cache_manager.reset()
        action_base = model.infer_with_vla_cache(frame, cache_manager)
        actions_baseline.append(action_base)
        cache_manager.reset()

    # Compare actions
    for i, (a_cache, a_base) in enumerate(zip(actions_with_cache, actions_baseline)):
        diff = torch.norm(a_cache - a_base).item()
        print(f"Frame {i}: L2 diff = {diff:.6f}")
        assert diff < 0.01, f"Action divergence too large at frame {i}!"

    print("✓ Numerical correctness verified")
```

### Pass Criteria
- [ ] Action L2 difference < 0.01 vs baseline
- [ ] Prefix KV computation speedup > 30%
- [ ] No memory leaks (1000 frames)
- [ ] Integration works with existing TRT FP8 pipeline

---

## Phase 5: LIBERO Accuracy Validation (Day 6)

### Goal
Validate that VLA-Cache maintains task success rate on LIBERO benchmark.

### Test Configuration

```python
# Quick validation: 3 tasks, 5 trials each
QUICK_TASKS = [
    "libero_spatial/pick_up_the_black_bowl",
    "libero_spatial/put_the_bowl_on_the_plate",
    "libero_goal/open_the_top_drawer",
]
NUM_TRIALS = 5
```

### Verification Script

**File**: `openpi/scripts/libero_eval_vla_cache.py`

```python
"""
LIBERO Evaluation with VLA-Cache

Metrics to collect:
1. Task success rate (should match baseline within 5%)
2. Mean latency (should be lower than baseline)
3. Prefix KV cache time (target: 30-50% reduction)
4. Cache hit rate (static patches reused)
5. Per-layer reuse statistics
"""

@dataclass
class VLACacheMetrics:
    success_rate: float
    mean_latency_ms: float
    prefix_kv_time_ms: float
    static_patch_ratio: float
    reusable_patch_ratio: float
    layer_reuse_avg: float

def run_evaluation():
    # Setup
    model = load_pi05_with_vla_cache()

    results = {}
    for task in QUICK_TASKS:
        task_results = []
        for trial in range(NUM_TRIALS):
            success, metrics = run_single_episode(model, task, trial)
            task_results.append((success, metrics))

        results[task] = task_results

    # Aggregate
    total_success = sum(r[0] for task_results in results.values() for r in task_results)
    total_trials = len(QUICK_TASKS) * NUM_TRIALS

    print(f"Overall Success Rate: {total_success}/{total_trials} = {100*total_success/total_trials:.1f}%")

    return results
```

### Expected Results

| Metric | Baseline | With VLA-Cache | Acceptable |
|--------|----------|---------------|------------|
| Success Rate | 100% | ≥ 95% | ± 5% |
| Mean Latency | 174 ms | ~150-155 ms | ≤ 160 ms |
| Prefix KV Time | 52 ms | ~25-35 ms | ≤ 40 ms |
| Static Patch Ratio | N/A | 50-80% | ≥ 40% |
| Reusable Ratio | N/A | 40-70% | ≥ 30% |

### Pass Criteria
- [ ] Success rate ≥ 95% (within 5% of baseline)
- [ ] Prefix KV time reduced by ≥ 30%
- [ ] No NaN/Inf warnings in simulation
- [ ] Stable across all 3 test tasks

---

## Summary Timeline

| Phase | Task | Duration | Deliverable |
|-------|------|----------|-------------|
| **1** | Static Patch Detection | Day 1 | `patch_detector.py` + verification |
| **2** | Attention Filtering | Day 2 | `attention_filter.py` + verification |
| **3** | Layer Scheduling | Day 3 | `layer_scheduler.py` + verification |
| **4** | KV Cache Integration | Day 4-5 | `kv_cache_manager.py` + model integration |
| **5** | LIBERO Validation | Day 6 | Success rate ≥ 95%, speedup verified |

## Key Differences from VLA-Cache Paper

| Aspect | Original VLA-Cache | Our Adaptation |
|--------|-------------------|----------------|
| Base Model | OpenVLA (7B, Llama) | Pi0.5 (Gemma Expert) |
| Vision Backbone | SigLIP | PaliGemma |
| Token Structure | 256 vision + text | 968 prefix (multi-image) |
| Denoising | None (autoregressive) | Flow matching (10 steps) |
| KV Format | HuggingFace Cache | Custom dict format |

## Risk Mitigation

1. **Accuracy Degradation**
   - Mitigation: Conservative thresholds, verify at each phase
   - Fallback: Reduce reuse ratio or skip specific layers

2. **Latency Overhead**
   - Mitigation: Profile patch detection, use efficient tensor ops
   - Fallback: Simplify to per-frame caching without filtering

3. **Memory Issues**
   - Mitigation: Only cache last frame, clear on episode reset
   - Fallback: Reduce cache size or disable per-layer scheduling

## References

- [VLA-Cache Paper](https://arxiv.org/pdf/2502.02175)
- [VLA-Cache GitHub](https://github.com/siyuhsu/vla-cache)
- [Falcon Paper](https://arxiv.org/abs/2503.00339) - Why it fails on VLA
- [Pi0.5 Documentation](./pi0_docs.md)
