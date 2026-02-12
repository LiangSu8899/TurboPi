#!/usr/bin/env python3
"""
Debug NVFP4 Persistent MLP Kernel.

Find the root cause of NaN outputs.

Author: Claude Code
Date: 2026-02-10
"""

import torch
import torch.nn.functional as F
import numpy as np

# ============================================================================
# Constants
# ============================================================================

HIDDEN_SIZE = 2048
MLP_DIM = 16384
BLOCK_SIZE = 32

NVFP4_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
NVFP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def quantize_to_nvfp4(weight: torch.Tensor, block_size: int = 32):
    """Quantize FP32 weight to NVFP4 format."""
    N, K = weight.shape
    device = weight.device
    weight = weight.to(torch.float32)

    num_blocks = K // block_size
    weight_blocked = weight.view(N, num_blocks, block_size)

    scales = weight_blocked.abs().amax(dim=-1) / 6.0
    scales = scales.clamp(min=1e-8)

    weight_normalized = weight_blocked / scales.unsqueeze(-1)

    nvfp4_magnitudes = torch.tensor(NVFP4_MAGNITUDES, device=device, dtype=torch.float32)
    signs = (weight_normalized < 0).to(torch.uint8) * 8
    abs_vals = weight_normalized.abs()

    diffs = (abs_vals.unsqueeze(-1) - nvfp4_magnitudes).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)

    fp4_vals = (signs + indices).view(N, K)

    low = fp4_vals[:, 0::2]
    high = fp4_vals[:, 1::2]
    packed = (high << 4) | low

    return packed.to(torch.uint8), scales.to(torch.float32)


def main():
    print("=" * 70)
    print("Debug NVFP4 Persistent Kernel")
    print("=" * 70)

    device = torch.device('cuda')

    # Try to import extension
    try:
        import nvfp4_persistent
        print("[OK] Extension loaded")
    except ImportError as e:
        print(f"[ERROR] {e}")
        return

    # Create minimal test: 1 layer
    print("\n--- Single Layer Test ---")

    gate = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32) * 0.01
    up = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32) * 0.01
    down = torch.randn(HIDDEN_SIZE, MLP_DIM, device=device, dtype=torch.float32) * 0.01

    gate_p, gate_s = quantize_to_nvfp4(gate)
    up_p, up_s = quantize_to_nvfp4(up)
    down_p, down_s = quantize_to_nvfp4(down)

    print(f"gate_packed shape: {gate_p.shape}, dtype: {gate_p.dtype}")
    print(f"gate_scale shape: {gate_s.shape}, dtype: {gate_s.dtype}")
    print(f"gate_packed sample: {gate_p[0, :10].tolist()}")
    print(f"gate_scale sample: {gate_s[0, :10].tolist()}")

    # Simple input
    x = torch.ones(1, HIDDEN_SIZE, device=device, dtype=torch.float32)
    print(f"\nInput: all ones, shape {x.shape}")

    # This should use 4-layer kernel
    num_layers = 4
    output = nvfp4_persistent.forward(
        x,
        [gate_p] * num_layers,
        [gate_s] * num_layers,
        [up_p] * num_layers,
        [up_s] * num_layers,
        [down_p] * num_layers,
        [down_s] * num_layers,
        num_layers
    )
    torch.cuda.synchronize()

    print(f"\nOutput shape: {output.shape}")
    print(f"Output stats:")
    print(f"  min: {output.min().item()}")
    print(f"  max: {output.max().item()}")
    print(f"  mean: {output.mean().item()}")
    print(f"  has nan: {output.isnan().any().item()}")
    print(f"  has inf: {output.isinf().any().item()}")
    print(f"  sample: {output[0, :10].tolist()}")

    # Check if kernel even runs
    print("\n--- Kernel Launch Check ---")

    # Use CUDA events to measure actual kernel time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    output2 = nvfp4_persistent.forward(
        x,
        [gate_p] * num_layers,
        [gate_s] * num_layers,
        [up_p] * num_layers,
        [up_s] * num_layers,
        [down_p] * num_layers,
        [down_s] * num_layers,
        num_layers
    )
    end_event.record()
    torch.cuda.synchronize()

    elapsed = start_event.elapsed_time(end_event)
    print(f"Kernel time: {elapsed:.3f} ms")

    # Check memory consistency
    print("\n--- Memory Layout Check ---")
    print(f"gate_packed is contiguous: {gate_p.is_contiguous()}")
    print(f"gate_scale is contiguous: {gate_s.is_contiguous()}")

    # Expected shapes
    print(f"\nExpected packed shape: [{MLP_DIM}, {HIDDEN_SIZE//2}]")
    print(f"Actual packed shape: {list(gate_p.shape)}")
    print(f"Expected scale shape: [{MLP_DIM}, {HIDDEN_SIZE//BLOCK_SIZE}]")
    print(f"Actual scale shape: {list(gate_s.shape)}")

    # Verify data pointers
    print(f"\nData pointers:")
    print(f"  gate_packed: {gate_p.data_ptr():#x}")
    print(f"  gate_scale: {gate_s.data_ptr():#x}")


if __name__ == "__main__":
    main()
