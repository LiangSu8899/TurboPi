"""
W4A16 Weight Packer - Offline weight quantization and packing.

Converts BF16/FP32 weights to INT4 block-interleaved format for
128-bit vectorized loads.

Output format:
    weight_packed: (num_scale_blocks, N, 4) int32
    scales: (num_scale_blocks, N) float16

Each quant block = 32 INT4 values = 4 uint32 = 128 bits

Author: Claude Code
Date: 2026-02-11
"""

import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import time


# Constants
QUANT_BLOCK = 32
INT4_OFFSET = 8  # Offset to map [-8, 7] to [0, 15]
INT4_RANGE = 7   # Max absolute value in symmetric INT4


@dataclass
class PackedWeight:
    """Container for packed W4A16 weight data."""
    weight_packed: torch.Tensor  # (num_scale_blocks, N, 4) int32
    scales: torch.Tensor         # (num_scale_blocks, N) float16
    original_shape: Tuple[int, int]  # (out_features, in_features)

    @property
    def out_features(self) -> int:
        return self.original_shape[0]

    @property
    def in_features(self) -> int:
        return self.original_shape[1]

    def to(self, device: torch.device) -> "PackedWeight":
        """Move packed weight to device."""
        return PackedWeight(
            weight_packed=self.weight_packed.to(device),
            scales=self.scales.to(device),
            original_shape=self.original_shape
        )

    def cuda(self) -> "PackedWeight":
        """Move to CUDA."""
        return self.to(torch.device("cuda"))

    def cpu(self) -> "PackedWeight":
        """Move to CPU."""
        return self.to(torch.device("cpu"))


class W4A16Packer:
    """
    Weight packer for W4A16 quantization.

    Converts [out_features, in_features] weights to block-interleaved
    INT4 format optimized for 128-bit vectorized loads.

    Example:
        >>> packer = W4A16Packer()
        >>> weight = torch.randn(16384, 2048, dtype=torch.bfloat16)
        >>> packed = packer.pack(weight)
        >>> print(packed.weight_packed.shape)  # (64, 16384, 4)
        >>> print(packed.scales.shape)         # (64, 16384)
    """

    def __init__(self, block_size: int = QUANT_BLOCK):
        """
        Initialize packer.

        Args:
            block_size: Quantization block size (default 32)
        """
        self.block_size = block_size

    def pack(
        self,
        weight: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> PackedWeight:
        """
        Pack weight tensor to INT4 block-interleaved format.

        Args:
            weight: (out_features, in_features) BF16/FP16/FP32 weight
            device: Target device for packed weights (default: same as input)

        Returns:
            PackedWeight containing packed weights and scales
        """
        assert weight.dim() == 2, "Weight must be 2D"

        out_features, in_features = weight.shape
        assert in_features % self.block_size == 0, \
            f"in_features ({in_features}) must be divisible by block_size ({self.block_size})"

        # Convert to float32 for quantization accuracy
        weight_fp32 = weight.float().cpu().numpy()

        # Pack
        weight_packed, scales = self._pack_numpy(weight_fp32)

        # Convert to torch tensors
        weight_packed = torch.from_numpy(weight_packed).to(torch.int32)
        scales = torch.from_numpy(scales).to(torch.float16)

        # Move to target device
        if device is None:
            device = weight.device
        weight_packed = weight_packed.to(device)
        scales = scales.to(device)

        return PackedWeight(
            weight_packed=weight_packed,
            scales=scales,
            original_shape=(out_features, in_features)
        )

    def _pack_numpy(
        self,
        weight: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pack weight using NumPy (CPU optimized).

        Args:
            weight: (N, K) float32 weight array

        Returns:
            weight_packed: (num_scale_blocks, N, 4) uint32
            scales: (num_scale_blocks, N) float16
        """
        N, K = weight.shape
        num_scale_blocks = K // self.block_size

        # Reshape to (N, num_scale_blocks, block_size) for vectorized ops
        weight_blocks = weight.reshape(N, num_scale_blocks, self.block_size)

        # Compute per-block scales: max(abs(block)) / 7
        block_max_abs = np.max(np.abs(weight_blocks), axis=2)  # (N, num_scale_blocks)
        scales = (block_max_abs / INT4_RANGE).astype(np.float16)
        scales = np.where(scales == 0, 1.0, scales)  # Avoid division by zero

        # Transpose scales to (num_scale_blocks, N) for coalesced access
        scales_T = scales.T.copy()

        # Quantize: round(weight / scale) + 8, clamp to [0, 15]
        scales_expanded = scales[:, :, np.newaxis]  # (N, num_scale_blocks, 1)
        quantized = np.round(weight_blocks / scales_expanded + INT4_OFFSET)
        quantized = np.clip(quantized, 0, 15).astype(np.uint8)

        # Pack to block-interleaved uint32 layout
        # Each quant block = 32 INT4 = 4 uint32
        weight_packed = np.zeros((num_scale_blocks, N, 4), dtype=np.uint32)

        for qb in range(num_scale_blocks):
            for u_idx in range(4):  # 4 uint32 per quant block
                # Pack 8 INT4 values into one uint32
                for i in range(8):
                    k_in_block = u_idx * 8 + i
                    int4_vals = quantized[:, qb, k_in_block].astype(np.uint32)
                    weight_packed[qb, :, u_idx] |= int4_vals << (i * 4)

        return weight_packed, scales_T

    def unpack(
        self,
        packed: PackedWeight,
    ) -> torch.Tensor:
        """
        Unpack INT4 weights back to float16 (for verification).

        Args:
            packed: PackedWeight to unpack

        Returns:
            (out_features, in_features) float16 weight
        """
        weight_packed = packed.weight_packed.cpu().numpy().astype(np.uint32)
        scales = packed.scales.cpu().numpy()

        num_scale_blocks, N, _ = weight_packed.shape
        K = num_scale_blocks * self.block_size

        weight = np.zeros((N, K), dtype=np.float32)

        for qb in range(num_scale_blocks):
            k_base = qb * self.block_size

            for u_idx in range(4):
                u_vals = weight_packed[qb, :, u_idx]

                for i in range(8):
                    int4_vals = (u_vals >> (i * 4)) & 0xF
                    k_idx = k_base + u_idx * 8 + i

                    # Dequantize: (int4 - 8) * scale
                    weight[:, k_idx] = (int4_vals.astype(np.float32) - INT4_OFFSET) * scales[qb, :]

        return torch.from_numpy(weight).to(torch.float16)

    def verify(
        self,
        original: torch.Tensor,
        packed: PackedWeight,
        rtol: float = 0.1,
        atol: float = 0.1,
    ) -> Tuple[bool, float]:
        """
        Verify packed weights by comparing dequantized output.

        Args:
            original: Original weight tensor
            packed: Packed weight
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            (is_close, cosine_similarity)
        """
        unpacked = self.unpack(packed)
        original_fp16 = original.float().cpu()
        unpacked_fp16 = unpacked.float().cpu()

        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            original_fp16.flatten().unsqueeze(0),
            unpacked_fp16.flatten().unsqueeze(0)
        ).item()

        # Check closeness
        is_close = torch.allclose(original_fp16, unpacked_fp16, rtol=rtol, atol=atol)

        return is_close, cos_sim


# ============================================================================
# Convenience Functions
# ============================================================================

def pack_linear_weight(
    weight: torch.Tensor,
    device: Optional[torch.device] = None,
) -> PackedWeight:
    """
    Pack nn.Linear weight to W4A16 format.

    Convenience function that creates a packer and packs the weight.

    Args:
        weight: (out_features, in_features) weight tensor
        device: Target device

    Returns:
        PackedWeight
    """
    packer = W4A16Packer()
    return packer.pack(weight, device)


def benchmark_packing(
    out_features: int = 16384,
    in_features: int = 2048,
    runs: int = 10,
) -> float:
    """
    Benchmark weight packing performance.

    Returns:
        Average packing time in milliseconds
    """
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    packer = W4A16Packer()

    # Warmup
    _ = packer.pack(weight)

    # Benchmark
    start = time.time()
    for _ in range(runs):
        _ = packer.pack(weight)
    avg_ms = (time.time() - start) / runs * 1000

    return avg_ms


# ============================================================================
# Vectorized Packer (Optimized)
# ============================================================================

class W4A16PackerFast:
    """
    Optimized weight packer using vectorized operations.

    Significantly faster than W4A16Packer for large weights.
    """

    def __init__(self, block_size: int = QUANT_BLOCK):
        self.block_size = block_size

    def pack(
        self,
        weight: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> PackedWeight:
        """Pack weight using vectorized PyTorch operations."""
        assert weight.dim() == 2, "Weight must be 2D"

        N, K = weight.shape
        assert K % self.block_size == 0

        num_scale_blocks = K // self.block_size

        # Work on CPU for packing
        weight_cpu = weight.float().cpu()

        # Reshape to (N, num_scale_blocks, block_size)
        weight_blocks = weight_cpu.reshape(N, num_scale_blocks, self.block_size)

        # Compute per-block scales
        block_max_abs = weight_blocks.abs().max(dim=2).values  # (N, num_scale_blocks)
        scales = block_max_abs / INT4_RANGE
        scales = scales.clamp(min=1e-8)  # Avoid division by zero

        # Quantize
        scales_expanded = scales.unsqueeze(2)  # (N, num_scale_blocks, 1)
        quantized = (weight_blocks / scales_expanded + INT4_OFFSET).round()
        quantized = quantized.clamp(0, 15).to(torch.uint8)

        # Reshape for packing: (N, num_scale_blocks, 4, 8)
        # 4 uint32 per block, 8 INT4 per uint32
        quantized = quantized.reshape(N, num_scale_blocks, 4, 8)

        # Pack into uint32
        # Create shift amounts: [0, 4, 8, 12, 16, 20, 24, 28]
        shifts = torch.arange(8, dtype=torch.int32).unsqueeze(0).unsqueeze(0).unsqueeze(0) * 4

        # Pack: sum of (int4 << shift) for each uint32
        quantized_int32 = quantized.to(torch.int32)
        packed = (quantized_int32 << shifts).sum(dim=3)  # (N, num_scale_blocks, 4)

        # Transpose to (num_scale_blocks, N, 4) for coalesced access
        weight_packed = packed.permute(1, 0, 2).contiguous()

        # Transpose scales to (num_scale_blocks, N)
        scales_T = scales.T.contiguous().to(torch.float16)

        # Move to device
        if device is None:
            device = weight.device
        weight_packed = weight_packed.to(device)
        scales_T = scales_T.to(device)

        return PackedWeight(
            weight_packed=weight_packed,
            scales=scales_T,
            original_shape=(N, K)
        )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W4A16 Packer Test")
    print("=" * 60)

    # Test dimensions
    out_features, in_features = 16384, 2048

    # Create test weight
    print(f"\nCreating test weight: ({out_features}, {in_features})")
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    # Test basic packer
    print("\n--- Basic Packer ---")
    packer = W4A16Packer()

    start = time.time()
    packed = packer.pack(weight)
    pack_time = (time.time() - start) * 1000
    print(f"Pack time: {pack_time:.2f} ms")
    print(f"weight_packed shape: {packed.weight_packed.shape}")
    print(f"scales shape: {packed.scales.shape}")

    # Verify
    is_close, cos_sim = packer.verify(weight, packed)
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Verification: {'PASS' if cos_sim > 0.99 else 'FAIL'}")

    # Test fast packer
    print("\n--- Fast Packer ---")
    packer_fast = W4A16PackerFast()

    start = time.time()
    packed_fast = packer_fast.pack(weight)
    pack_time_fast = (time.time() - start) * 1000
    print(f"Pack time: {pack_time_fast:.2f} ms")
    print(f"Speedup: {pack_time / pack_time_fast:.1f}x")

    # Memory comparison
    print("\n--- Memory Comparison ---")
    original_bytes = weight.numel() * 2  # BF16
    packed_bytes = packed.weight_packed.numel() * 4 + packed.scales.numel() * 2
    print(f"Original (BF16): {original_bytes / 1e6:.2f} MB")
    print(f"Packed (INT4):   {packed_bytes / 1e6:.2f} MB")
    print(f"Compression:     {original_bytes / packed_bytes:.1f}x")
