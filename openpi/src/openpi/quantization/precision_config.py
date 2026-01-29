"""Precision configuration for Pi0.5 FP4/FP8 quantization.

This module defines which layers should be quantized to FP4, FP8, or kept at FP16.

Layer Classification:
- FP4: MLP layers (gate_proj, up_proj, down_proj, fc1, fc2) - ~90% of compute
- FP8: Attention projections (q_proj, k_proj, v_proj, o_proj)
- FP16: Normalization, embeddings, residuals (precision-sensitive)
"""

import re
from dataclasses import dataclass, field


@dataclass
class PrecisionConfig:
    """Configuration for mixed-precision quantization."""

    # FP4 layer patterns (MLP layers - biggest compute savings)
    fp4_patterns: list[str] = field(default_factory=lambda: [
        r".*\.gate_proj$",
        r".*\.up_proj$",
        r".*\.down_proj$",
        r".*\.fc1$",
        r".*\.fc2$",
        r".*action_in_proj$",
        r".*action_out_proj$",
        r".*time_mlp\.0$",  # time_mlp Linear layers
        r".*time_mlp\.2$",
    ])

    # FP8 layer patterns (Attention projections)
    fp8_patterns: list[str] = field(default_factory=lambda: [
        r".*\.q_proj$",
        r".*\.k_proj$",
        r".*\.v_proj$",
        r".*\.o_proj$",
    ])

    # Skip patterns (keep at FP16/FP32)
    skip_patterns: list[str] = field(default_factory=lambda: [
        r".*norm.*",           # RMSNorm, LayerNorm
        r".*embed.*",          # Embeddings
        r".*rotary.*",         # RoPE
        r".*lm_head.*",        # Language model head
        r".*vision_tower.*head.*",  # Vision classifier head
    ])

    def get_layer_precision(self, layer_name: str) -> str:
        """Get the precision for a given layer name.

        Args:
            layer_name: Full name of the layer (e.g., "paligemma.language_model.layers.0.mlp.gate_proj")

        Returns:
            "fp4", "fp8", or "skip"
        """
        # Check skip patterns first (highest priority)
        for pattern in self.skip_patterns:
            if re.match(pattern, layer_name):
                return "skip"

        # Check FP4 patterns
        for pattern in self.fp4_patterns:
            if re.match(pattern, layer_name):
                return "fp4"

        # Check FP8 patterns
        for pattern in self.fp8_patterns:
            if re.match(pattern, layer_name):
                return "fp8"

        # Default: skip (keep original precision)
        return "skip"


def get_fp4_layers(model, config: PrecisionConfig | None = None) -> list[str]:
    """Get list of layer names that should be quantized to FP4.

    Args:
        model: PyTorch model
        config: PrecisionConfig instance (uses default if None)

    Returns:
        List of layer names for FP4 quantization
    """
    if config is None:
        config = PrecisionConfig()

    fp4_layers = []
    for name, _ in model.named_modules():
        if config.get_layer_precision(name) == "fp4":
            fp4_layers.append(name)

    return fp4_layers


def get_fp8_layers(model, config: PrecisionConfig | None = None) -> list[str]:
    """Get list of layer names that should be quantized to FP8.

    Args:
        model: PyTorch model
        config: PrecisionConfig instance (uses default if None)

    Returns:
        List of layer names for FP8 quantization
    """
    if config is None:
        config = PrecisionConfig()

    fp8_layers = []
    for name, _ in model.named_modules():
        if config.get_layer_precision(name) == "fp8":
            fp8_layers.append(name)

    return fp8_layers


def get_skip_layers(model, config: PrecisionConfig | None = None) -> list[str]:
    """Get list of layer names that should be kept at original precision.

    Args:
        model: PyTorch model
        config: PrecisionConfig instance (uses default if None)

    Returns:
        List of layer names to skip quantization
    """
    if config is None:
        config = PrecisionConfig()

    skip_layers = []
    for name, _ in model.named_modules():
        if config.get_layer_precision(name) == "skip":
            skip_layers.append(name)

    return skip_layers


def print_quantization_summary(model, config: PrecisionConfig | None = None):
    """Print a summary of the quantization plan.

    Args:
        model: PyTorch model
        config: PrecisionConfig instance (uses default if None)
    """
    if config is None:
        config = PrecisionConfig()

    fp4_layers = get_fp4_layers(model, config)
    fp8_layers = get_fp8_layers(model, config)
    skip_layers = get_skip_layers(model, config)

    print("=" * 60)
    print("Pi0.5 Quantization Plan")
    print("=" * 60)
    print(f"\nFP4 Layers ({len(fp4_layers)}):")
    for name in fp4_layers[:10]:
        print(f"  - {name}")
    if len(fp4_layers) > 10:
        print(f"  ... and {len(fp4_layers) - 10} more")

    print(f"\nFP8 Layers ({len(fp8_layers)}):")
    for name in fp8_layers[:10]:
        print(f"  - {name}")
    if len(fp8_layers) > 10:
        print(f"  ... and {len(fp8_layers) - 10} more")

    print(f"\nSkipped Layers ({len(skip_layers)})")
    print("=" * 60)

    total_linear = len(fp4_layers) + len(fp8_layers)
    if total_linear > 0:
        fp4_pct = len(fp4_layers) / total_linear * 100
        fp8_pct = len(fp8_layers) / total_linear * 100
        print(f"FP4 Coverage: {fp4_pct:.1f}% of quantized layers")
        print(f"FP8 Coverage: {fp8_pct:.1f}% of quantized layers")
