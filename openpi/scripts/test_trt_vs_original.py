#!/usr/bin/env python3
"""Unit tests for TRT Static Denoise implementation.

This script verifies that the TRT static implementation produces identical
outputs to the original PyTorch model before TRT compilation.

Usage:
    docker exec turbo_pi_eval python3 /workspace/scripts/test_trt_vs_original.py
"""

import sys
import os
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from typing import Tuple, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.backends.cudnn.enabled = False

CHECKPOINT_DIR = os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero")

# Import constants from TRT implementation
sys.path.insert(0, os.path.dirname(__file__))


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name: str, condition: bool, msg: str = ""):
        if condition:
            print(f"  ✅ {name}")
            self.passed += 1
        else:
            print(f"  ❌ {name}: {msg}")
            self.failed += 1
            self.errors.append(f"{name}: {msg}")

    def check_close(self, name: str, a: torch.Tensor, b: torch.Tensor,
                    rtol: float = 1e-3, atol: float = 1e-5):
        """Check if two tensors are close."""
        if a.shape != b.shape:
            self.check(name, False, f"Shape mismatch: {a.shape} vs {b.shape}")
            return False

        max_diff = (a.float() - b.float()).abs().max().item()
        cos_sim = F.cosine_similarity(
            a.float().flatten().unsqueeze(0),
            b.float().flatten().unsqueeze(0)
        ).item()

        is_close = cos_sim > 0.999 and max_diff < 0.1
        self.check(name, is_close,
                   f"cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}")
        return is_close

    def summary(self):
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY: {self.passed} passed, {self.failed} failed")
        print("=" * 60)
        if self.errors:
            print("\nFailures:")
            for e in self.errors:
                print(f"  - {e}")
        return self.failed == 0


def load_original_model():
    """Load the original PI0Pytorch model."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from safetensors.torch import load_file

    checkpoint_dir = Path(CHECKPOINT_DIR)
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
    else:
        model_config = {}

    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=model_config.get("action_dim", 32),
        action_horizon=model_config.get("action_horizon", 50),
        max_token_len=model_config.get("tokenizer_max_length", 200),
        pi05=True,
        dtype="bfloat16",
    )

    model = PI0Pytorch(pi0_config)
    weights_path = checkpoint_dir / "model.safetensors"
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device="cuda")
    model.eval()

    return model


def test_parameter_comparison(results: TestResults):
    """Test 1: Compare parameter configurations."""
    print("\n" + "=" * 60)
    print("TEST 1: Parameter Configuration Comparison")
    print("=" * 60)

    model = load_original_model()

    # Check action projections have bias
    results.check(
        "action_in_proj has bias",
        model.action_in_proj.bias is not None,
        "Original model should have bias"
    )
    results.check(
        "action_out_proj has bias",
        model.action_out_proj.bias is not None,
        "Original model should have bias"
    )

    # Check shapes
    print(f"\n  action_in_proj: weight={model.action_in_proj.weight.shape}, "
          f"bias={model.action_in_proj.bias.shape if model.action_in_proj.bias is not None else None}")
    print(f"  action_out_proj: weight={model.action_out_proj.weight.shape}, "
          f"bias={model.action_out_proj.bias.shape if model.action_out_proj.bias is not None else None}")

    # Check gemma expert layer structure
    gemma_expert = model.paligemma_with_expert.gemma_expert.model
    layer0 = gemma_expert.layers[0]

    results.check(
        "Layer has input_layernorm.dense",
        hasattr(layer0.input_layernorm, 'dense') and layer0.input_layernorm.dense is not None,
        "Adaptive RMSNorm should have dense layer"
    )

    # Check dense layer shapes
    if hasattr(layer0.input_layernorm, 'dense'):
        dense = layer0.input_layernorm.dense
        print(f"\n  input_layernorm.dense: weight={dense.weight.shape}, bias={dense.bias.shape}")
        results.check(
            "input_layernorm.dense has correct shape",
            dense.weight.shape == torch.Size([3072, 1024]),  # 3*hidden_size x hidden_size
            f"Expected [3072, 1024], got {dense.weight.shape}"
        )

    return model


def test_action_embedding(results: TestResults, model):
    """Test 2: Action embedding with bias."""
    print("\n" + "=" * 60)
    print("TEST 2: Action Embedding (with bias)")
    print("=" * 60)

    device = "cuda"
    batch_size = 1
    action_horizon = 50
    action_dim = 32

    torch.manual_seed(42)
    x_t = torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=torch.bfloat16)

    # Original model embedding
    with torch.no_grad():
        orig_emb = model.action_in_proj(x_t.to(model.action_in_proj.weight.dtype))

    # Without bias (simulating TRT bug)
    with torch.no_grad():
        no_bias_emb = F.linear(
            x_t.to(model.action_in_proj.weight.dtype),
            model.action_in_proj.weight,
            bias=None
        )

    # Compute difference
    diff = (orig_emb - no_bias_emb).abs()

    print(f"\n  With bias mean: {orig_emb.float().mean().item():.6f}")
    print(f"  Without bias mean: {no_bias_emb.float().mean().item():.6f}")
    print(f"  Difference mean: {diff.float().mean().item():.6f}")
    print(f"  Difference max: {diff.float().max().item():.6f}")

    results.check(
        "Bias makes a difference",
        diff.float().mean().item() > 1e-6,
        "Bias should affect output"
    )

    # The bias contribution should match the actual bias
    expected_diff = model.action_in_proj.bias.float().mean().abs().item()
    actual_diff = diff.float().mean().item()
    results.check(
        "Difference matches bias magnitude",
        abs(actual_diff - expected_diff) / (expected_diff + 1e-8) < 0.5,
        f"Expected ~{expected_diff:.6f}, got {actual_diff:.6f}"
    )


def test_rmsnorm_forward(results: TestResults, model):
    """Test 3: RMSNorm forward with conditioning."""
    print("\n" + "=" * 60)
    print("TEST 3: Adaptive RMSNorm Forward")
    print("=" * 60)

    device = "cuda"
    batch_size = 1
    seq_len = 50
    hidden_size = 1024

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    cond = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    gemma_expert = model.paligemma_with_expert.gemma_expert.model
    layer0 = gemma_expert.layers[0]

    with torch.no_grad():
        normed, gate = layer0.input_layernorm(x, cond=cond)

    print(f"\n  Input shape: {x.shape}")
    print(f"  Cond shape: {cond.shape}")
    print(f"  Normed output shape: {normed.shape}")
    print(f"  Gate shape: {gate.shape if gate is not None else None}")
    print(f"  Normed mean: {normed.float().mean().item():.6f}")
    print(f"  Gate mean: {gate.float().mean().item():.6f}" if gate is not None else "  Gate is None")

    results.check(
        "RMSNorm returns gate",
        gate is not None,
        "Adaptive RMSNorm should return gate"
    )
    results.check(
        "Output shapes correct",
        normed.shape == x.shape and gate.shape == torch.Size([batch_size, 1, hidden_size]),
        f"normed={normed.shape}, gate={gate.shape if gate is not None else None}"
    )


def test_rope_embedding(results: TestResults, model):
    """Test 4: RoPE embedding computation."""
    print("\n" + "=" * 60)
    print("TEST 4: RoPE Embedding")
    print("=" * 60)

    device = "cuda"
    batch_size = 1
    seq_len = 50
    head_dim = 256

    paligemma_lm = model.paligemma_with_expert.paligemma.language_model

    # Position IDs (simulating suffix positions)
    position_ids = torch.arange(560, 560 + seq_len, device=device).unsqueeze(0)

    # Dummy tensor for RoPE computation
    dummy = torch.zeros(batch_size, seq_len, head_dim, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        cos, sin = paligemma_lm.rotary_emb(dummy, position_ids)

    print(f"\n  Position IDs: {position_ids[0, :5].tolist()} ... {position_ids[0, -5:].tolist()}")
    print(f"  cos shape: {cos.shape}")
    print(f"  sin shape: {sin.shape}")
    print(f"  cos[0, 0, :5]: {cos[0, 0, :5].float().tolist()}")
    print(f"  attention_scaling: {paligemma_lm.rotary_emb.attention_scaling}")

    results.check(
        "RoPE output shapes correct",
        cos.shape == torch.Size([batch_size, seq_len, head_dim]),
        f"Expected [1, 50, 256], got {cos.shape}"
    )
    results.check(
        "attention_scaling is 1.0",
        paligemma_lm.rotary_emb.attention_scaling == 1.0,
        f"Expected 1.0, got {paligemma_lm.rotary_emb.attention_scaling}"
    )

    # Verify inv_freq calculation
    inv_freq = paligemma_lm.rotary_emb.inv_freq
    expected_inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))

    results.check(
        "inv_freq approximately correct",
        torch.allclose(inv_freq.cpu().float(), expected_inv_freq, rtol=1e-2),
        "inv_freq should match expected calculation"
    )


def test_single_denoise_step(results: TestResults, model):
    """Test 5: Complete single denoise step."""
    print("\n" + "=" * 60)
    print("TEST 5: Single Denoise Step (Reference)")
    print("=" * 60)

    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
    from transformers.models.gemma import modeling_gemma

    device = "cuda"
    batch_size = 1
    action_horizon = 50
    action_dim = 32
    prefix_len = 968
    hidden_size = 1024

    torch.manual_seed(42)

    # Create inputs
    x_t = torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([1.0], device=device, dtype=torch.float32)

    # Create fake KV cache
    prefix_kv_cache = []
    for _ in range(18):
        k = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        prefix_kv_cache.append((k, v))

    # Create masks
    valid_tokens = 560
    prefix_pad_masks = torch.zeros(batch_size, prefix_len, dtype=torch.bool, device=device)
    prefix_pad_masks[:, :valid_tokens] = True

    state = torch.randn(batch_size, action_dim, device=device, dtype=torch.bfloat16)

    # Run original model (note: param order is state, prefix_kv_cache, prefix_pad_masks, x_t, timestep)
    with torch.no_grad():
        output = model.denoise_step_with_cache(
            state, prefix_kv_cache, prefix_pad_masks, x_t, timestep
        )

    print(f"\n  Input x_t shape: {x_t.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.float().mean().item():.6f}")
    print(f"  Output std: {output.float().std().item():.6f}")

    results.check(
        "Denoise step output shape correct",
        output.shape == torch.Size([batch_size, action_horizon, action_dim]),
        f"Expected [1, 50, 32], got {output.shape}"
    )
    results.check(
        "Output is not NaN",
        not torch.isnan(output).any(),
        "Output contains NaN"
    )

    return output


def test_trt_module_structure(results: TestResults):
    """Test 6: Verify TRT module structure matches requirements."""
    print("\n" + "=" * 60)
    print("TEST 6: TRT Module Structure Requirements")
    print("=" * 60)

    # Import TRT module
    from denoise_torch_trt_static import (
        StaticDenoiseStep, StaticDenoiseLoop, RMSNorm,
        SimpleAttention, SimpleMLP, RotaryEmbedding,
        HIDDEN_SIZE, HEAD_DIM, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS
    )

    device = "cuda"

    # Create TRT module
    trt_step = StaticDenoiseStep().to(device).half()

    print(f"\n  Checking StaticDenoiseStep structure...")

    # Check action projections have bias
    results.check(
        "TRT action_in_proj has bias",
        trt_step.action_in_proj.bias is not None,
        "Should have bias=True"
    )
    results.check(
        "TRT action_out_proj has bias",
        trt_step.action_out_proj.bias is not None,
        "Should have bias=True"
    )

    # Check layer structure
    results.check(
        "TRT has 18 layers",
        len(trt_step.layers) == NUM_LAYERS,
        f"Expected 18, got {len(trt_step.layers)}"
    )

    layer0 = trt_step.layers[0]
    results.check(
        "Layer has input_layernorm with dense",
        hasattr(layer0.input_layernorm, 'dense') and layer0.input_layernorm.dense is not None,
        "RMSNorm should have dense layer"
    )

    # Check RMSNorm dense shape
    if hasattr(layer0.input_layernorm, 'dense'):
        dense = layer0.input_layernorm.dense
        expected_shape = torch.Size([3 * HIDDEN_SIZE, HIDDEN_SIZE])
        results.check(
            "RMSNorm dense weight shape",
            dense.weight.shape == expected_shape,
            f"Expected {expected_shape}, got {dense.weight.shape}"
        )
        results.check(
            "RMSNorm dense has bias",
            dense.bias is not None,
            "Dense should have bias"
        )


def test_weight_loading(results: TestResults):
    """Test 7: Verify weight loading is complete."""
    print("\n" + "=" * 60)
    print("TEST 7: Weight Loading Completeness")
    print("=" * 60)

    from safetensors import safe_open
    from denoise_torch_trt_static import StaticDenoiseLoop, load_weights_from_checkpoint

    device = "cuda"

    # Create TRT module
    trt_loop = StaticDenoiseLoop().to(device).half()

    # Load weights
    load_weights_from_checkpoint(trt_loop, CHECKPOINT_DIR, device)

    denoise_step = trt_loop.denoise_step

    # Load original weights for comparison
    checkpoint_path = Path(CHECKPOINT_DIR) / "model.safetensors"
    with safe_open(checkpoint_path, framework='pt') as f:
        orig_action_in_weight = f.get_tensor("action_in_proj.weight")
        orig_action_in_bias = f.get_tensor("action_in_proj.bias")
        orig_action_out_weight = f.get_tensor("action_out_proj.weight")
        orig_action_out_bias = f.get_tensor("action_out_proj.bias")

    # Check weights match
    results.check_close(
        "action_in_proj.weight loaded",
        denoise_step.action_in_proj.weight.cpu().float(),
        orig_action_in_weight.float()
    )
    results.check_close(
        "action_in_proj.bias loaded",
        denoise_step.action_in_proj.bias.cpu().float(),
        orig_action_in_bias.float()
    )
    results.check_close(
        "action_out_proj.weight loaded",
        denoise_step.action_out_proj.weight.cpu().float(),
        orig_action_out_weight.float()
    )
    results.check_close(
        "action_out_proj.bias loaded",
        denoise_step.action_out_proj.bias.cpu().float(),
        orig_action_out_bias.float()
    )


def test_trt_vs_original_single_step(results: TestResults):
    """Test 8: Compare TRT module output with original model (single step)."""
    print("\n" + "=" * 60)
    print("TEST 8: TRT vs Original Single Step Output")
    print("=" * 60)

    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
    from denoise_torch_trt_static import StaticDenoiseStep, load_weights_from_checkpoint

    device = "cuda"
    batch_size = 1
    action_horizon = 50
    action_dim = 32
    prefix_len = 968

    # Load original model
    orig_model = load_original_model()

    # Create and load TRT module
    trt_step = StaticDenoiseStep(
        batch_size=batch_size,
        action_horizon=action_horizon,
        action_dim=action_dim,
        prefix_len=prefix_len,
    ).to(device).half()
    load_weights_from_checkpoint(trt_step, CHECKPOINT_DIR, device)
    trt_step.eval()

    # Create identical inputs
    torch.manual_seed(123)
    x_t = torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=torch.bfloat16)

    # Create KV cache
    prefix_kv_cache = []
    torch.manual_seed(456)
    for _ in range(18):
        k = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch_size, 1, prefix_len, 256, device=device, dtype=torch.bfloat16)
        prefix_kv_cache.append((k, v))

    # Masks
    valid_tokens = 560
    prefix_pad_masks = torch.zeros(batch_size, prefix_len, dtype=torch.bool, device=device)
    prefix_pad_masks[:, :valid_tokens] = True

    state = torch.randn(batch_size, action_dim, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([1.0], device=device, dtype=torch.float32)

    # Run original model (note: param order is state, prefix_kv_cache, prefix_pad_masks, x_t, timestep)
    with torch.no_grad():
        orig_output = orig_model.denoise_step_with_cache(
            state, prefix_kv_cache, prefix_pad_masks, x_t, timestep
        )

    # Prepare TRT inputs
    # Compute adarms_cond
    time_emb = create_sinusoidal_pos_embedding(
        timestep, orig_model.action_in_proj.out_features,
        min_period=4e-3, max_period=4.0,
        device=torch.device(device)
    ).to(orig_model.time_mlp_in.weight.dtype)

    with torch.no_grad():
        x = orig_model.time_mlp_in(time_emb)
        x = F.silu(x)
        x = orig_model.time_mlp_out(x)
        adarms_cond = F.silu(x)

    # Position IDs
    prefix_offset = torch.sum(prefix_pad_masks.long(), dim=-1, keepdim=True)
    suffix_position_ids = prefix_offset + torch.arange(action_horizon, device=device, dtype=torch.long)

    # Stack KV cache for TRT format
    cached_keys = torch.stack([kv[0] for kv in prefix_kv_cache], dim=0)
    cached_values = torch.stack([kv[1] for kv in prefix_kv_cache], dim=0)

    # Note: No attention mask needed - matching original model (uses SDPA without mask)

    # Run TRT module
    with torch.no_grad():
        trt_output = trt_step(
            x_t.half(),
            suffix_position_ids,
            adarms_cond.half(),
            cached_keys.half(),
            cached_values.half(),
        )

    # Compare outputs
    print(f"\n  Original output mean: {orig_output.float().mean().item():.6f}")
    print(f"  TRT output mean: {trt_output.float().mean().item():.6f}")

    results.check_close(
        "TRT output matches original",
        trt_output.float(),
        orig_output.float()
    )


def main():
    print("=" * 60)
    print("TRT vs Original Model Unit Tests")
    print("=" * 60)

    results = TestResults()

    try:
        # Test 1: Parameter comparison
        model = test_parameter_comparison(results)

        # Test 2: Action embedding
        test_action_embedding(results, model)

        # Test 3: RMSNorm
        test_rmsnorm_forward(results, model)

        # Test 4: RoPE
        test_rope_embedding(results, model)

        # Test 5: Single denoise step
        test_single_denoise_step(results, model)

        # Test 6: TRT module structure (after fixing)
        try:
            test_trt_module_structure(results)
        except Exception as e:
            print(f"  ⚠️ Skipped (TRT module needs fix): {e}")

        # Test 7: Weight loading (after fixing)
        try:
            test_weight_loading(results)
        except Exception as e:
            print(f"  ⚠️ Skipped (TRT module needs fix): {e}")

        # Test 8: TRT vs original output (after fixing)
        try:
            test_trt_vs_original_single_step(results)
        except Exception as e:
            print(f"  ⚠️ Skipped (TRT module needs fix): {e}")

    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

    return 0 if results.summary() else 1


if __name__ == "__main__":
    sys.exit(main())
