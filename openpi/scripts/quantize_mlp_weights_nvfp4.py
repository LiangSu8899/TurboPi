#!/usr/bin/env python3
"""
Pre-quantize MLP weights to nvFP4 format.

This script quantizes all 18 layers' MLP weights (gate_proj, up_proj, down_proj)
from the pi0.5 checkpoint to packed nvFP4 format with per-block scales.

Output format:
- W_packed: [N, K//2] uint8 (2 FP4 values per byte)
- scales: [N, num_blocks] float32

Usage:
    python quantize_mlp_weights_nvfp4.py --checkpoint /path/to/checkpoint --output /path/to/output

In Docker:
    python /workspace/scripts/quantize_mlp_weights_nvfp4.py \
        --checkpoint /root/.cache/openpi/checkpoints/pi05_libero \
        --output /workspace/quantized_weights
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import time

# Model constants
NUM_LAYERS = 18
HIDDEN_SIZE = 2048
MLP_DIM = 16384
BLOCK_SIZE = 32

# nvFP4 E2M1 lookup table
NVFP4_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # positive (0-7)
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0  # negative (8-15)
], dtype=np.float32)


def quantize_to_nvfp4_packed_fast(weight: np.ndarray, block_size: int = BLOCK_SIZE):
    """
    Fast vectorized quantization to packed nvFP4 format.

    Args:
        weight: FP32 weight array [N, K]
        block_size: Block size for scaling (default 32)

    Returns:
        W_packed: uint8 array [N, K//2] (2 FP4 values per byte)
        scales: FP32 array [N, num_blocks]
    """
    N, K = weight.shape
    num_blocks = (K + block_size - 1) // block_size

    # Pad K to multiple of block_size if needed
    K_padded = num_blocks * block_size
    if K_padded > K:
        weight = np.pad(weight, ((0, 0), (0, K_padded - K)), mode='constant')

    # Reshape for block processing: [N, num_blocks, block_size]
    weight_blocks = weight.reshape(N, num_blocks, block_size)

    # Compute scales per block (max abs value / 6.0)
    block_max = np.abs(weight_blocks).max(axis=2)
    scales = np.where(block_max > 0, block_max / 6.0, 1.0).astype(np.float32)

    # Normalize values by scale
    scales_expanded = scales[:, :, np.newaxis]  # [N, num_blocks, 1]
    normalized = weight_blocks / scales_expanded  # [N, num_blocks, block_size]
    normalized = normalized.reshape(N, K_padded)[:, :K]  # Back to [N, K]

    # Quantize to nvFP4 indices using vectorized lookup
    # Find closest nvFP4 value for each element
    # Expand dims for broadcasting: normalized [N, K, 1] vs LUT [16]
    diffs = np.abs(normalized[:, :, np.newaxis] - NVFP4_LUT[np.newaxis, np.newaxis, :])
    W_quant = np.argmin(diffs, axis=2).astype(np.uint8)  # [N, K]

    # Pack to uint8 (2 FP4 per byte): low nibble + high nibble
    K_packed = K // 2
    W_packed = np.zeros((N, K_packed), dtype=np.uint8)
    W_packed = (W_quant[:, 0::2] & 0xF) | ((W_quant[:, 1::2] & 0xF) << 4)

    return W_packed, scales


def quantize_checkpoint(checkpoint_dir: str, output_dir: str):
    """
    Quantize all MLP weights from checkpoint to nvFP4 format.

    Args:
        checkpoint_dir: Path to pi0.5 checkpoint
        output_dir: Path to save quantized weights
    """
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weights_path = checkpoint_path / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    print(f"Loading checkpoint from: {weights_path}")
    print(f"Output directory: {output_path}")
    print(f"Block size: {BLOCK_SIZE}")
    print()

    # Load weights
    state_dict = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            if "mlp" in key:
                state_dict[key] = f.get_tensor(key)

    print(f"Found {len(state_dict)} MLP weight tensors")
    print()

    # Quantize each layer
    quantized_weights = {}
    total_fp16_bytes = 0
    total_nvfp4_bytes = 0

    for layer_idx in range(NUM_LAYERS):
        prefix = f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}"

        print(f"Layer {layer_idx:2d}: ", end="", flush=True)
        start_time = time.time()

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            key = f"{prefix}.mlp.{proj_name}.weight"

            if key not in state_dict:
                print(f"Warning: {key} not found")
                continue

            weight = state_dict[key].float().numpy()
            N, K = weight.shape

            # Track original size
            total_fp16_bytes += N * K * 2  # FP16 = 2 bytes

            # Quantize
            W_packed, scales = quantize_to_nvfp4_packed_fast(weight)

            # Track quantized size
            total_nvfp4_bytes += W_packed.nbytes + scales.nbytes

            # Save to quantized dict
            out_key_w = f"layer.{layer_idx}.{proj_name}.weight_packed"
            out_key_s = f"layer.{layer_idx}.{proj_name}.scales"
            quantized_weights[out_key_w] = torch.from_numpy(W_packed)
            quantized_weights[out_key_s] = torch.from_numpy(scales)

        elapsed = time.time() - start_time
        print(f"quantized in {elapsed:.2f}s")

    # Save quantized weights
    output_file = output_path / "mlp_weights_nvfp4.safetensors"
    print(f"\nSaving quantized weights to: {output_file}")
    save_file(quantized_weights, str(output_file))

    # Print summary
    print()
    print("=" * 60)
    print("QUANTIZATION SUMMARY")
    print("=" * 60)
    print(f"Layers:           {NUM_LAYERS}")
    print(f"Block size:       {BLOCK_SIZE}")
    print(f"Original (FP16):  {total_fp16_bytes / 1024 / 1024:.1f} MB")
    print(f"Quantized (FP4):  {total_nvfp4_bytes / 1024 / 1024:.1f} MB")
    print(f"Compression:      {total_fp16_bytes / total_nvfp4_bytes:.1f}x")
    print(f"Output file:      {output_file}")
    print(f"Output size:      {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    print()

    # Verify quantization accuracy for one layer
    print("Verifying quantization accuracy (layer 0, gate_proj)...")
    verify_quantization(state_dict, quantized_weights, layer_idx=0, proj_name="gate_proj")

    return str(output_file)


def verify_quantization(state_dict, quantized_weights, layer_idx=0, proj_name="gate_proj"):
    """Verify quantization accuracy by comparing dequantized output."""
    prefix = f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}"
    key = f"{prefix}.mlp.{proj_name}.weight"

    original = state_dict[key].float().numpy()
    N, K = original.shape

    # Load quantized
    W_packed = quantized_weights[f"layer.{layer_idx}.{proj_name}.weight_packed"].numpy()
    scales = quantized_weights[f"layer.{layer_idx}.{proj_name}.scales"].numpy()

    # Dequantize
    num_blocks = scales.shape[1]
    K_packed = W_packed.shape[1]

    # Unpack
    W_quant = np.zeros((N, K), dtype=np.int32)
    W_quant[:, 0::2] = W_packed & 0xF
    W_quant[:, 1::2] = (W_packed >> 4) & 0xF

    # Dequantize using LUT
    W_dequant = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            block_idx = k // BLOCK_SIZE
            fp4_idx = W_quant[n, k]
            W_dequant[n, k] = NVFP4_LUT[fp4_idx] * scales[n, block_idx]

    # Compute error
    abs_error = np.abs(original - W_dequant)
    rel_error = abs_error / (np.abs(original) + 1e-8)

    print(f"  Shape: {original.shape}")
    print(f"  Mean abs error:  {abs_error.mean():.6f}")
    print(f"  Max abs error:   {abs_error.max():.6f}")
    print(f"  Mean rel error:  {rel_error.mean() * 100:.2f}%")
    print(f"  Max rel error:   {rel_error.max() * 100:.2f}%")

    # Check if values are within expected range
    original_range = (original.min(), original.max())
    dequant_range = (W_dequant.min(), W_dequant.max())
    print(f"  Original range:  [{original_range[0]:.4f}, {original_range[1]:.4f}]")
    print(f"  Dequant range:   [{dequant_range[0]:.4f}, {dequant_range[1]:.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Quantize MLP weights to nvFP4 format")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/.cache/openpi/checkpoints/pi05_libero",
        help="Path to pi0.5 checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/quantized_weights",
        help="Output directory for quantized weights"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("nvFP4 Weight Quantization for W4A16 TVM Plugin")
    print("=" * 60)
    print()

    quantize_checkpoint(args.checkpoint, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
