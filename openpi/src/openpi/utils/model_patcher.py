"""
PaliGemma Model Patcher - Hot-swap Linear layers with W4A16 quantization.

Features:
- Replaces nn.Linear in PaliGemma with W4A16Linear for Decode optimization
- Uses TVM 128-bit vectorized kernel (0.125ms for 16384x2048)
- Hybrid forward: seq_len > 1 uses F.linear, seq_len == 1 uses TVM kernel
- CUDA Graph safe, torch.compile compatible

Usage:
    from openpi.utils.model_patcher import patch_paligemma_decode_path

    model = PI0Pytorch(config)
    stats = patch_paligemma_decode_path(model)
    print(f"Replaced {stats['replaced']} layers, saved {stats['memory_saved_mb']:.1f} MB")

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import gc

import sys
import os

# Ensure module paths are available
_utils_dir = os.path.dirname(os.path.abspath(__file__))
_openpi_dir = os.path.dirname(_utils_dir)
_src_dir = os.path.dirname(_openpi_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Direct imports to avoid circular dependency
from openpi.modules.w4a16_linear import W4A16Linear, replace_linear_with_w4a16
from openpi.ops.w4a16_gemv import precompile_kernels
QUANT_BLOCK = 32  # Constant to avoid import cycle


def get_model_memory_mb(model: nn.Module) -> float:
    """Calculate model parameter memory in MB."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / (1024 * 1024)


def patch_paligemma_decode_path(
    model: nn.Module,
    min_features: int = 256,  # Lowered to include K/V projections (256 output features)
    quantize_attention: bool = False,  # Explicit attention quantization flag
    skip_vision: bool = True,
    skip_lm_head: bool = True,
    precompile: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Patch PaliGemma model to use W4A16 quantization for Decode optimization.

    This function replaces nn.Linear layers in the language model with
    W4A16Linear, which uses our TVM 128-bit vectorized kernel for batch=1
    inference (0.125ms latency).

    Args:
        model: PI0Pytorch or similar model containing paligemma_with_expert
        min_features: Minimum in_features to replace (smaller layers not worth it)
        quantize_attention: Enable attention projection quantization (q, k, v, o)
                           WARNING: May affect accuracy. Test thoroughly.
        skip_vision: Skip vision tower layers
        skip_lm_head: Skip language model head (vocab projection)
        precompile: Precompile TVM kernels for common dimensions
        verbose: Print progress information

    Returns:
        Dictionary with patching statistics:
        - replaced: Number of layers replaced
        - skipped: Number of layers skipped
        - memory_before_mb: Memory before patching
        - memory_after_mb: Memory after patching
        - memory_saved_mb: Memory saved
        - layers: List of replaced layer names
        - mlp_layers: List of replaced MLP layer names
        - attention_layers: List of replaced attention layer names

    Example:
        >>> from openpi.utils.model_patcher import patch_paligemma_decode_path
        >>> # MLP only (default, safe)
        >>> stats = patch_paligemma_decode_path(model)
        >>> # Full coverage (MLP + Attention)
        >>> stats = patch_paligemma_decode_path(model, quantize_attention=True)
    """
    mode_str = "Full Coverage (MLP + Attention)" if quantize_attention else "MLP Only"
    if verbose:
        print("=" * 60)
        print(f"PaliGemma W4A16 Decode Path Patcher [{mode_str}]")
        print("=" * 60)

    # Get memory before patching
    memory_before = get_model_memory_mb(model)

    # Track statistics
    replaced_layers = []
    skipped_layers = []
    mlp_layers = []
    attention_layers = []

    # Get PaliGemma language model
    try:
        paligemma_lm = model.paligemma_with_expert.paligemma.language_model
    except AttributeError:
        # Try alternative paths
        try:
            paligemma_lm = model.paligemma.language_model
        except AttributeError:
            raise ValueError(
                "Could not find PaliGemma language model. "
                "Expected model.paligemma_with_expert.paligemma.language_model "
                "or model.paligemma.language_model"
            )

    # Collect dimensions for precompilation
    dimensions_to_compile = set()

    # Iterate through transformer layers
    for layer_idx, layer in enumerate(paligemma_lm.layers):
        layer_prefix = f"layer_{layer_idx}"

        # Process MLP layers (primary target)
        if hasattr(layer, 'mlp') and layer.mlp is not None:
            mlp = layer.mlp

            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, proj_name):
                    linear = getattr(mlp, proj_name)

                    if isinstance(linear, nn.Linear):
                        # Check if eligible for replacement
                        if (linear.in_features >= min_features and
                            linear.in_features % QUANT_BLOCK == 0):

                            # Replace with W4A16Linear
                            device = linear.weight.device
                            w4a16_linear = W4A16Linear.from_linear(linear, device)
                            setattr(mlp, proj_name, w4a16_linear)

                            layer_name = f"{layer_prefix}.mlp.{proj_name}"
                            replaced_layers.append(layer_name)
                            mlp_layers.append(layer_name)

                            # Record dimensions for precompilation
                            dimensions_to_compile.add(
                                (linear.out_features, linear.in_features)
                            )

                            if verbose:
                                print(f"  [MLP] Replaced: {layer_name} "
                                      f"({linear.in_features} → {linear.out_features})")
                        else:
                            skipped_layers.append(
                                f"{layer_prefix}.mlp.{proj_name} (too small or not aligned)"
                            )

        # Process attention layers (if enabled)
        if quantize_attention and hasattr(layer, 'self_attn'):
            attn = layer.self_attn

            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(attn, proj_name):
                    linear = getattr(attn, proj_name)

                    if isinstance(linear, nn.Linear):
                        # Check both in_features and out_features alignment
                        in_aligned = linear.in_features % QUANT_BLOCK == 0
                        out_aligned = linear.out_features % QUANT_BLOCK == 0

                        if in_aligned and out_aligned and linear.in_features >= min_features:
                            device = linear.weight.device
                            w4a16_linear = W4A16Linear.from_linear(linear, device)
                            setattr(attn, proj_name, w4a16_linear)

                            layer_name = f"{layer_prefix}.self_attn.{proj_name}"
                            replaced_layers.append(layer_name)
                            attention_layers.append(layer_name)

                            dimensions_to_compile.add(
                                (linear.out_features, linear.in_features)
                            )

                            if verbose:
                                print(f"  [ATT] Replaced: {layer_name} "
                                      f"({linear.in_features} → {linear.out_features})")
                        else:
                            reason = []
                            if not in_aligned:
                                reason.append(f"in_features {linear.in_features} not aligned")
                            if not out_aligned:
                                reason.append(f"out_features {linear.out_features} not aligned")
                            skipped_layers.append(f"{layer_prefix}.self_attn.{proj_name} ({', '.join(reason)})")

    # Precompile TVM kernels
    if precompile and dimensions_to_compile:
        if verbose:
            print(f"\nPrecompiling TVM kernels for {len(dimensions_to_compile)} dimensions...")

        try:
            precompile_kernels(list(dimensions_to_compile))
            if verbose:
                print("  Kernel precompilation complete!")
        except Exception as e:
            print(f"  Warning: Kernel precompilation failed: {e}")

    # Force garbage collection to reclaim memory
    gc.collect()
    torch.cuda.empty_cache()

    # Get memory after patching
    memory_after = get_model_memory_mb(model)
    memory_saved = memory_before - memory_after

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("Patching Summary")
        print("=" * 60)
        print(f"  Total replaced:  {len(replaced_layers)}")
        print(f"    - MLP layers:      {len(mlp_layers)}")
        print(f"    - Attention layers:{len(attention_layers)}")
        print(f"  Layers skipped:  {len(skipped_layers)}")
        print(f"  Memory before:   {memory_before:.1f} MB")
        print(f"  Memory after:    {memory_after:.1f} MB")
        print(f"  Memory saved:    {memory_saved:.1f} MB ({memory_saved/memory_before*100:.1f}%)")
        print("=" * 60)

    return {
        'replaced': len(replaced_layers),
        'skipped': len(skipped_layers),
        'memory_before_mb': memory_before,
        'memory_after_mb': memory_after,
        'memory_saved_mb': memory_saved,
        'layers': replaced_layers,
        'skipped_layers': skipped_layers,
        'mlp_layers': mlp_layers,
        'attention_layers': attention_layers,
    }


def patch_expert_decode_path(
    model: nn.Module,
    min_features: int = 256,  # Lowered for attention projections
    quantize_attention: bool = False,  # Enable attention quantization
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Patch Expert model (Gemma transformer) with W4A16 quantization.

    Similar to patch_paligemma_decode_path but targets the expert model.

    Args:
        model: PI0Pytorch model containing gemma_expert
        min_features: Minimum in_features to replace
        quantize_attention: Enable attention projection quantization (q, k, v, o)
        verbose: Print progress

    Returns:
        Patching statistics
    """
    mode_str = "Full Coverage (MLP + Attention)" if quantize_attention else "MLP Only"
    if verbose:
        print("=" * 60)
        print(f"Expert (Gemma) W4A16 Decode Path Patcher [{mode_str}]")
        print("=" * 60)

    memory_before = get_model_memory_mb(model)
    replaced_layers = []
    mlp_layers = []
    attention_layers = []

    # Get Expert model
    try:
        expert = model.paligemma_with_expert.gemma_expert.model
    except AttributeError:
        try:
            expert = model.gemma_expert.model
        except AttributeError:
            raise ValueError("Could not find Gemma expert model")

    dimensions_to_compile = set()

    # Iterate through transformer layers
    for layer_idx, layer in enumerate(expert.layers):
        layer_prefix = f"expert_layer_{layer_idx}"

        # Process MLP
        if hasattr(layer, 'mlp') and layer.mlp is not None:
            mlp = layer.mlp

            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, proj_name):
                    linear = getattr(mlp, proj_name)

                    if isinstance(linear, nn.Linear):
                        if (linear.in_features >= min_features and
                            linear.in_features % QUANT_BLOCK == 0):

                            device = linear.weight.device
                            w4a16_linear = W4A16Linear.from_linear(linear, device)
                            setattr(mlp, proj_name, w4a16_linear)

                            layer_name = f"{layer_prefix}.mlp.{proj_name}"
                            replaced_layers.append(layer_name)
                            mlp_layers.append(layer_name)
                            dimensions_to_compile.add(
                                (linear.out_features, linear.in_features)
                            )

                            if verbose:
                                print(f"  [MLP] Replaced: {layer_name} "
                                      f"({linear.in_features} → {linear.out_features})")

        # Process attention layers (if enabled)
        if quantize_attention and hasattr(layer, 'self_attn'):
            attn = layer.self_attn

            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(attn, proj_name):
                    linear = getattr(attn, proj_name)

                    if isinstance(linear, nn.Linear):
                        in_aligned = linear.in_features % QUANT_BLOCK == 0
                        out_aligned = linear.out_features % QUANT_BLOCK == 0

                        if in_aligned and out_aligned and linear.in_features >= min_features:
                            device = linear.weight.device
                            w4a16_linear = W4A16Linear.from_linear(linear, device)
                            setattr(attn, proj_name, w4a16_linear)

                            layer_name = f"{layer_prefix}.self_attn.{proj_name}"
                            replaced_layers.append(layer_name)
                            attention_layers.append(layer_name)
                            dimensions_to_compile.add(
                                (linear.out_features, linear.in_features)
                            )

                            if verbose:
                                print(f"  [ATT] Replaced: {layer_name} "
                                      f"({linear.in_features} → {linear.out_features})")

    # Precompile kernels
    if dimensions_to_compile:
        if verbose:
            print(f"\nPrecompiling TVM kernels for {len(dimensions_to_compile)} dimensions...")
        try:
            precompile_kernels(list(dimensions_to_compile))
            if verbose:
                print("  Kernel precompilation complete!")
        except Exception as e:
            print(f"  Warning: Kernel precompilation failed: {e}")

    gc.collect()
    torch.cuda.empty_cache()

    memory_after = get_model_memory_mb(model)
    memory_saved = memory_before - memory_after

    if verbose:
        print("\n" + "=" * 60)
        print("Expert Patching Summary")
        print("=" * 60)
        print(f"  Total replaced:  {len(replaced_layers)}")
        print(f"    - MLP layers:      {len(mlp_layers)}")
        print(f"    - Attention layers:{len(attention_layers)}")
        print(f"  Memory saved: {memory_saved:.1f} MB")

    return {
        'replaced': len(replaced_layers),
        'memory_before_mb': memory_before,
        'memory_after_mb': memory_after,
        'memory_saved_mb': memory_saved,
        'layers': replaced_layers,
        'mlp_layers': mlp_layers,
        'attention_layers': attention_layers,
    }


def patch_all_decode_paths(
    model: nn.Module,
    min_features: int = 256,
    quantize_attention: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Patch both PaliGemma and Expert models with W4A16 quantization.

    This is a convenience function that calls both patch_paligemma_decode_path
    and patch_expert_decode_path.

    Args:
        model: PI0Pytorch model
        min_features: Minimum in_features to replace
        quantize_attention: Enable attention projection quantization
        verbose: Print progress

    Returns:
        Combined patching statistics
    """
    paligemma_stats = patch_paligemma_decode_path(
        model, min_features=min_features, quantize_attention=quantize_attention, verbose=verbose
    )

    expert_stats = patch_expert_decode_path(
        model, min_features=min_features, quantize_attention=quantize_attention, verbose=verbose
    )

    total_mlp = len(paligemma_stats.get('mlp_layers', [])) + len(expert_stats.get('mlp_layers', []))
    total_attn = len(paligemma_stats.get('attention_layers', [])) + len(expert_stats.get('attention_layers', []))

    return {
        'paligemma': paligemma_stats,
        'expert': expert_stats,
        'total_replaced': paligemma_stats['replaced'] + expert_stats['replaced'],
        'total_mlp_layers': total_mlp,
        'total_attention_layers': total_attn,
        'total_memory_saved_mb': (
            paligemma_stats['memory_saved_mb'] + expert_stats['memory_saved_mb']
        ),
    }


def patch_full_coverage(
    model: nn.Module,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Patch model with FULL W4A16 coverage: MLP + Attention projections.

    This is the recommended function for maximum performance optimization.
    Replaces all eligible nn.Linear layers in both PaliGemma and Expert models.

    WARNING: Attention quantization may affect accuracy. Test thoroughly
    before deploying in production.

    Args:
        model: PI0Pytorch model
        verbose: Print progress

    Returns:
        Combined patching statistics

    Example:
        >>> from openpi.utils.model_patcher import patch_full_coverage
        >>> stats = patch_full_coverage(model)
        >>> print(f"Total replaced: {stats['total_replaced']} layers")
        >>> print(f"  MLP: {stats['total_mlp_layers']}")
        >>> print(f"  Attention: {stats['total_attention_layers']}")
    """
    if verbose:
        print("\n" + "=" * 70)
        print("   W4A16 FULL COVERAGE MODE")
        print("   Quantizing: MLP + Attention (Q, K, V, O projections)")
        print("=" * 70 + "\n")

    return patch_all_decode_paths(
        model,
        min_features=256,  # Include all projections
        quantize_attention=True,
        verbose=verbose,
    )


def verify_patching(model: nn.Module) -> Dict[str, int]:
    """
    Verify W4A16 patching by counting layer types.

    Args:
        model: Patched model

    Returns:
        Dictionary with counts of each layer type
    """
    counts = {
        'W4A16Linear': 0,
        'nn.Linear': 0,
        'other': 0,
    }

    for name, module in model.named_modules():
        if isinstance(module, W4A16Linear):
            counts['W4A16Linear'] += 1
        elif isinstance(module, nn.Linear):
            counts['nn.Linear'] += 1

    return counts


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Model Patcher Test")
    print("=" * 60)

    # Test with a simple model
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(2048, 16384, bias=False)
            self.up_proj = nn.Linear(2048, 16384, bias=False)
            self.down_proj = nn.Linear(16384, 2048, bias=False)

        def forward(self, x):
            return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))

    class SimpleTransformer(nn.Module):
        def __init__(self, num_layers=2):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({'mlp': SimpleMLP()})
                for _ in range(num_layers)
            ])

    class MockPaliGemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = SimpleTransformer(num_layers=4)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma_with_expert = nn.Module()
            self.paligemma_with_expert.paligemma = MockPaliGemma()

    # Create mock model
    model = MockModel()
    model.cuda()

    print("\nBefore patching:")
    counts = verify_patching(model)
    print(f"  W4A16Linear: {counts['W4A16Linear']}")
    print(f"  nn.Linear:   {counts['nn.Linear']}")

    # Patch
    stats = patch_paligemma_decode_path(model, verbose=True)

    print("\nAfter patching:")
    counts = verify_patching(model)
    print(f"  W4A16Linear: {counts['W4A16Linear']}")
    print(f"  nn.Linear:   {counts['nn.Linear']}")

    # Test forward pass
    print("\n--- Forward Pass Test ---")
    x = torch.randn(1, 2048, dtype=torch.float16, device='cuda')

    with torch.no_grad():
        for i, layer_dict in enumerate(model.paligemma_with_expert.paligemma.language_model.layers):
            mlp = layer_dict['mlp']
            y = mlp.gate_proj(x)
            print(f"Layer {i} gate_proj output: {y.shape}, dtype: {y.dtype}")

    print("\nTest complete!")
