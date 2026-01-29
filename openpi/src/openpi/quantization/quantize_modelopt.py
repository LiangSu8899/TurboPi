"""ModelOpt-based PTQ quantization for Pi0.5.

This module provides FP4/FP8 quantization using NVIDIA ModelOpt library.
Optimized for Blackwell (SM110) GPUs with native FP4 support.
"""

import re
import torch
from pathlib import Path
from typing import Callable

from .precision_config import PrecisionConfig, get_fp4_layers, get_fp8_layers
from .calibration_data import SyntheticCalibrationDataset, calibration_forward_loop


def check_modelopt_available() -> bool:
    """Check if ModelOpt is installed and available."""
    try:
        import modelopt
        return True
    except ImportError:
        return False


def get_modelopt_quant_config(precision_config: PrecisionConfig | None = None):
    """Create ModelOpt quantization configuration.

    Args:
        precision_config: PrecisionConfig instance

    Returns:
        ModelOpt QuantizeConfig dict
    """
    try:
        import modelopt.torch.quantization as mtq
    except ImportError:
        raise ImportError(
            "ModelOpt not found. Install with: pip install nvidia-modelopt[torch]>=0.17.0"
        )

    if precision_config is None:
        precision_config = PrecisionConfig()

    # Build layer-specific quantization config
    quant_config = {
        "quant_cfg": {
            # Default: skip quantization
            "*": {"enable": False},
        },
        "algorithm": "max",  # Use max calibration algorithm
    }

    # Add FP4 patterns for MLP layers
    for pattern in precision_config.fp4_patterns:
        # Convert regex pattern to ModelOpt glob pattern
        glob_pattern = pattern.replace(r".*\.", "*").replace(r"$", "")
        quant_config["quant_cfg"][glob_pattern] = {
            "num_bits": 4,
            "axis": None,
            "enable": True,
        }

    # Add FP8 patterns for attention projections
    for pattern in precision_config.fp8_patterns:
        glob_pattern = pattern.replace(r".*\.", "*").replace(r"$", "")
        quant_config["quant_cfg"][glob_pattern] = {
            "num_bits": 8,
            "axis": None,
            "enable": True,
        }

    return quant_config


def quantize_model_fp4fp8(
    model: torch.nn.Module,
    calibration_loader: Callable | None = None,
    num_calibration_samples: int = 512,
    precision_config: PrecisionConfig | None = None,
    device: str = "cuda",
) -> torch.nn.Module:
    """Quantize Pi0.5 model with FP4 (MLP) and FP8 (Attention) precision.

    Args:
        model: PI0Pytorch model instance
        calibration_loader: Optional calibration data loader function
        num_calibration_samples: Number of calibration samples
        precision_config: Layer precision configuration
        device: Device for quantization

    Returns:
        Quantized model
    """
    try:
        import modelopt.torch.quantization as mtq
    except ImportError:
        raise ImportError(
            "ModelOpt not found. Install with: pip install nvidia-modelopt[torch]>=0.17.0"
        )

    if precision_config is None:
        precision_config = PrecisionConfig()

    print("=" * 60)
    print("Pi0.5 FP4/FP8 Quantization with ModelOpt")
    print("=" * 60)

    # Get quantization config
    quant_config = get_modelopt_quant_config(precision_config)

    # Count layers to be quantized
    fp4_layers = get_fp4_layers(model, precision_config)
    fp8_layers = get_fp8_layers(model, precision_config)
    print(f"\nTarget FP4 layers: {len(fp4_layers)}")
    print(f"Target FP8 layers: {len(fp8_layers)}")

    # Set up calibration
    if calibration_loader is None:
        print(f"\nGenerating {num_calibration_samples} synthetic calibration samples...")
        dataset = SyntheticCalibrationDataset(
            num_samples=num_calibration_samples,
            device=device,
        )

        def calibration_loader():
            calibration_forward_loop(model, dataset, num_calibration_samples)

    # Apply quantization
    print("\nApplying quantization...")
    model.eval()

    # Use ModelOpt PTQ
    with torch.no_grad():
        quantized_model = mtq.quantize(
            model,
            quant_config,
            forward_loop=calibration_loader,
        )

    print("\nQuantization complete!")
    return quantized_model


def quantize_model_simple(
    model: torch.nn.Module,
    precision_config: PrecisionConfig | None = None,
) -> torch.nn.Module:
    """Simple weight-only quantization without ModelOpt.

    This is a fallback when ModelOpt is not available.
    Uses PyTorch's native int8 dynamic quantization.

    Args:
        model: PI0Pytorch model instance
        precision_config: Layer precision configuration

    Returns:
        Quantized model (int8 dynamic)
    """
    if precision_config is None:
        precision_config = PrecisionConfig()

    print("=" * 60)
    print("Pi0.5 Simple Quantization (Fallback)")
    print("=" * 60)
    print("WARNING: Using PyTorch dynamic quantization (int8)")
    print("         For FP4/FP8, install ModelOpt")

    # Get layers to quantize
    fp4_layers = get_fp4_layers(model, precision_config)
    fp8_layers = get_fp8_layers(model, precision_config)

    modules_to_quantize = set(fp4_layers + fp8_layers)
    print(f"\nTarget layers: {len(modules_to_quantize)}")

    # Apply dynamic quantization to Linear layers
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    print("\nSimple quantization complete!")
    return quantized_model


def save_quantized_model(
    model: torch.nn.Module,
    output_dir: str | Path,
    config: dict | None = None,
):
    """Save quantized model to disk.

    Args:
        model: Quantized model
        output_dir: Output directory path
        config: Optional config dict to save alongside model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    model_path = output_dir / "quantized_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved quantized model to: {model_path}")

    # Save config if provided
    if config is not None:
        import json
        config_path = output_dir / "quantization_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to: {config_path}")


def load_quantized_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> torch.nn.Module:
    """Load quantized weights into model.

    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to quantized model checkpoint

    Returns:
        Model with quantized weights loaded
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "quantized_model.pt"

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded quantized weights from: {checkpoint_path}")

    return model
