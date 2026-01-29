#!/usr/bin/env python3
"""Quantize Pi0.5 model with FP4/FP8 for Blackwell GPU optimization.

Usage:
    python scripts/quantize_model.py \
        --model_path ~/.cache/openpi/checkpoints/pi05_libero \
        --output_dir ./quantized_models/pi05_fp4fp8 \
        --num_calibration_samples 512

Requirements:
    - NVIDIA ModelOpt: pip install nvidia-modelopt[torch]>=0.17.0
    - Blackwell GPU (SM110) for native FP4 support
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Pi0.5 model with FP4/FP8 precision"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="~/.cache/openpi/checkpoints/pi05_libero",
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_models/pi05_fp4fp8",
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=512,
        help="Number of calibration samples for PTQ",
    )
    parser.add_argument(
        "--use_modelopt",
        action="store_true",
        default=True,
        help="Use NVIDIA ModelOpt for quantization (default: True)",
    )
    parser.add_argument(
        "--no_modelopt",
        action="store_true",
        help="Use simple PyTorch quantization instead of ModelOpt",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only show quantization plan without applying",
    )

    args = parser.parse_args()

    # Resolve paths
    model_path = Path(args.model_path).expanduser()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Pi0.5 Model Quantization")
    print("=" * 60)
    print(f"\nModel path: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Calibration samples: {args.num_calibration_samples}")

    # Check GPU
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available!")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")

    # Import model and quantization modules
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from openpi.quantization.precision_config import (
        PrecisionConfig,
        print_quantization_summary,
    )

    # Create model config
    config = Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        max_state_dim=32,
        pi05=True,
        dtype="bfloat16",
    )

    print("\nLoading model...")
    model = PI0Pytorch(config)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # Load checkpoint weights if available
    checkpoint_file = model_path / "model.safetensors"
    if checkpoint_file.exists():
        print(f"Loading weights from {checkpoint_file}...")
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_file))
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_file}")
        print("         Proceeding with randomly initialized weights")

    # Show quantization plan
    precision_config = PrecisionConfig()
    print_quantization_summary(model, precision_config)

    if args.dry_run:
        print("\n[DRY RUN] Quantization plan shown. Exiting without applying.")
        return

    # Apply quantization
    use_modelopt = args.use_modelopt and not args.no_modelopt

    if use_modelopt:
        from openpi.quantization.quantize_modelopt import (
            check_modelopt_available,
            quantize_model_fp4fp8,
            save_quantized_model,
        )

        if not check_modelopt_available():
            print("\nWARNING: ModelOpt not available!")
            print("         Install with: pip install nvidia-modelopt[torch]>=0.17.0")
            print("         Falling back to simple quantization...")
            use_modelopt = False

    if use_modelopt:
        quantized_model = quantize_model_fp4fp8(
            model,
            num_calibration_samples=args.num_calibration_samples,
            precision_config=precision_config,
            device="cuda",
        )
    else:
        from openpi.quantization.quantize_modelopt import (
            quantize_model_simple,
            save_quantized_model,
        )
        quantized_model = quantize_model_simple(model, precision_config)

    # Save quantized model
    save_quantized_model(
        quantized_model,
        output_dir,
        config={
            "model_path": str(model_path),
            "num_calibration_samples": args.num_calibration_samples,
            "use_modelopt": use_modelopt,
            "precision_config": {
                "fp4_patterns": precision_config.fp4_patterns,
                "fp8_patterns": precision_config.fp8_patterns,
            },
        },
    )

    print("\n" + "=" * 60)
    print("Quantization Complete!")
    print("=" * 60)
    print(f"\nQuantized model saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Validate accuracy: python scripts/validate_quantization.py")
    print("  2. Benchmark: python scripts/benchmark_thor.py --precision mixed")


if __name__ == "__main__":
    main()
