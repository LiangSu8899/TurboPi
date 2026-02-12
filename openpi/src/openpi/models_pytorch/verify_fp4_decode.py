#!/usr/bin/env python3
"""
Verify FP4 decode logic matches between CUDA kernel and Python.

Author: Claude Code
Date: 2026-02-10
"""

import torch
import torch.nn.functional as F

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


def dequantize_from_nvfp4(packed: torch.Tensor, scales: torch.Tensor, block_size: int = 32):
    """Dequantize NVFP4 back to FP32."""
    N = packed.shape[0]
    K = packed.shape[1] * 2
    device = packed.device

    full_lut = torch.tensor(NVFP4_LUT, device=device, dtype=torch.float32)

    low = packed & 0xF
    high = (packed >> 4) & 0xF

    fp4_vals = torch.zeros(N, K, dtype=torch.uint8, device=device)
    fp4_vals[:, 0::2] = low
    fp4_vals[:, 1::2] = high

    decoded = full_lut[fp4_vals.to(torch.int64)]

    num_blocks = K // block_size
    decoded_blocked = decoded.view(N, num_blocks, block_size)
    weight = (decoded_blocked * scales.unsqueeze(-1)).view(N, K)

    return weight


def single_layer_mlp_reference(x, gate_p, gate_s, up_p, up_s, down_p, down_s):
    """Reference single layer MLP."""
    gate = dequantize_from_nvfp4(gate_p, gate_s)
    up = dequantize_from_nvfp4(up_p, up_s)
    down = dequantize_from_nvfp4(down_p, down_s)

    g = F.linear(x, gate)
    u = F.linear(x, up)
    inter = F.silu(g) * u
    out = F.linear(inter, down)

    return out, gate, up, down, g, u, inter


def main():
    print("=" * 70)
    print("FP4 Decode Verification")
    print("=" * 70)

    device = torch.device('cuda')

    # Create simple weights
    print("\n--- Simple Weight Test ---")

    # Use small weights for testing
    gate = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32) * 0.02
    up = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32) * 0.02
    down = torch.randn(HIDDEN_SIZE, MLP_DIM, device=device, dtype=torch.float32) * 0.02

    print(f"Original gate stats: mean={gate.mean().item():.6f}, std={gate.std().item():.6f}")

    # Quantize
    gate_p, gate_s = quantize_to_nvfp4(gate)
    up_p, up_s = quantize_to_nvfp4(up)
    down_p, down_s = quantize_to_nvfp4(down)

    # Dequantize
    gate_dq = dequantize_from_nvfp4(gate_p, gate_s)
    up_dq = dequantize_from_nvfp4(up_p, up_s)
    down_dq = dequantize_from_nvfp4(down_p, down_s)

    print(f"Dequant gate stats: mean={gate_dq.mean().item():.6f}, std={gate_dq.std().item():.6f}")

    # Quantization error
    gate_error = (gate - gate_dq).abs()
    print(f"Quantization error: max={gate_error.max().item():.6f}, mean={gate_error.mean().item():.6f}")

    # Test single layer reference
    print("\n--- Single Layer Reference Test ---")
    x = torch.ones(1, HIDDEN_SIZE, device=device, dtype=torch.float32)

    out, gate_dq, up_dq, down_dq, g, u, inter = single_layer_mlp_reference(
        x, gate_p, gate_s, up_p, up_s, down_p, down_s
    )

    print(f"Input: all ones, shape {x.shape}")
    print(f"gate projection (g): mean={g.mean().item():.6f}, std={g.std().item():.6f}")
    print(f"up projection (u): mean={u.mean().item():.6f}, std={u.std().item():.6f}")
    print(f"intermediate (silu(g)*u): mean={inter.mean().item():.6f}, std={inter.std().item():.6f}")
    print(f"output: mean={out.mean().item():.6f}, std={out.std().item():.6f}")
    print(f"output sample: {out[0, :10].tolist()}")

    # Multi-layer test
    print("\n--- Multi-Layer Reference Test (4 layers) ---")
    activation = x.clone()
    for layer_idx in range(4):
        gate = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32) * 0.02
        up = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32) * 0.02
        down = torch.randn(HIDDEN_SIZE, MLP_DIM, device=device, dtype=torch.float32) * 0.02

        gate_p, gate_s = quantize_to_nvfp4(gate)
        up_p, up_s = quantize_to_nvfp4(up)
        down_p, down_s = quantize_to_nvfp4(down)

        gate_dq = dequantize_from_nvfp4(gate_p, gate_s)
        up_dq = dequantize_from_nvfp4(up_p, up_s)
        down_dq = dequantize_from_nvfp4(down_p, down_s)

        g = F.linear(activation, gate_dq)
        u = F.linear(activation, up_dq)
        inter = F.silu(g) * u
        activation = F.linear(inter, down_dq)

        print(f"Layer {layer_idx}: mean={activation.mean().item():.6e}, std={activation.std().item():.6e}")

    print(f"\nFinal output sample: {activation[0, :10].tolist()}")

    # Compare with extension
    print("\n--- Compare with Extension ---")
    try:
        import nvfp4_persistent

        gate_p_list = []
        gate_s_list = []
        up_p_list = []
        up_s_list = []
        down_p_list = []
        down_s_list = []
        layers_for_ref = []

        for _ in range(4):
            gate = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32) * 0.02
            up = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32) * 0.02
            down = torch.randn(HIDDEN_SIZE, MLP_DIM, device=device, dtype=torch.float32) * 0.02

            gate_p, gate_s = quantize_to_nvfp4(gate)
            up_p, up_s = quantize_to_nvfp4(up)
            down_p, down_s = quantize_to_nvfp4(down)

            gate_p_list.append(gate_p)
            gate_s_list.append(gate_s)
            up_p_list.append(up_p)
            up_s_list.append(up_s)
            down_p_list.append(down_p)
            down_s_list.append(down_s)

            layers_for_ref.append((gate_p, gate_s, up_p, up_s, down_p, down_s))

        x = torch.ones(1, HIDDEN_SIZE, device=device, dtype=torch.float32)

        # Extension output
        ext_out = nvfp4_persistent.forward(
            x,
            gate_p_list, gate_s_list,
            up_p_list, up_s_list,
            down_p_list, down_s_list,
            4
        )
        torch.cuda.synchronize()

        # Reference output
        ref_activation = x.clone()
        for gate_p, gate_s, up_p, up_s, down_p, down_s in layers_for_ref:
            gate_dq = dequantize_from_nvfp4(gate_p, gate_s)
            up_dq = dequantize_from_nvfp4(up_p, up_s)
            down_dq = dequantize_from_nvfp4(down_p, down_s)

            g = F.linear(ref_activation, gate_dq)
            u = F.linear(ref_activation, up_dq)
            inter = F.silu(g) * u
            ref_activation = F.linear(inter, down_dq)

        print(f"Extension output: mean={ext_out.mean().item():.6e}, std={ext_out.std().item():.6e}")
        print(f"Reference output: mean={ref_activation.mean().item():.6e}, std={ref_activation.std().item():.6e}")
        print(f"Extension sample: {ext_out[0, :5].tolist()}")
        print(f"Reference sample: {ref_activation[0, :5].tolist()}")

        diff = (ext_out - ref_activation).abs()
        print(f"Difference: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    except ImportError as e:
        print(f"Extension not available: {e}")


if __name__ == "__main__":
    main()
