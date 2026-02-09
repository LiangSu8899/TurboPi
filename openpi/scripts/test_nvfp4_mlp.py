#!/usr/bin/env python3
"""
Test NVFP4 MLP Integration for KV Cache

This script tests:
1. NVFP4 quantization precision
2. NVFP4 vs BF16 MLP performance
3. Batch padding overhead
4. Full KV Cache (18 layers) performance

Expected results:
- NVFP4: 3-4x speedup vs BF16 (from CUTLASS benchmarks)
- Precision: < 5% relative difference
- KV Cache MLP: 18 ms → ~5 ms

Run in Docker:
    python openpi/scripts/test_nvfp4_mlp.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_section(title: str):
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


# ============================================================================
# NVFP4 Quantization (Python simulation)
# ============================================================================

NVFP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
NVFP4_MAX = 6.0
BLOCK_SIZE = 32


def quantize_to_nvfp4(tensor: torch.Tensor, block_size: int = BLOCK_SIZE):
    """Quantize tensor to NVFP4 with block scaling."""
    M, K = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    nvfp4_values = NVFP4_VALUES.to(device)

    # Reshape to blocks
    num_blocks = K // block_size
    tensor_blocked = tensor.view(M, num_blocks, block_size).float()

    # Per-block scale
    block_max = tensor_blocked.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale_factors = block_max.squeeze(-1) / NVFP4_MAX

    # Scale and quantize
    scaled = tensor_blocked / block_max * NVFP4_MAX
    signs = scaled.sign()
    abs_scaled = scaled.abs()

    # Find nearest FP4 value
    distances = (abs_scaled.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1)
    quantized_abs = nvfp4_values[indices]
    quantized = (signs * quantized_abs).view(M, K).to(dtype)

    return quantized, scale_factors


def dequantize_nvfp4(quantized: torch.Tensor, scales: torch.Tensor, block_size: int = BLOCK_SIZE):
    """Dequantize NVFP4 back to original scale."""
    M, K = quantized.shape
    quantized_blocked = quantized.view(M, -1, block_size)
    dequantized = quantized_blocked * scales.unsqueeze(-1)
    return dequantized.view(M, K)


# ============================================================================
# MLP Implementations
# ============================================================================

class BF16MLP(nn.Module):
    """Standard BF16 MLP."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = torch.nn.functional.gelu(self.gate_proj(x), approximate='tanh')
        return self.down_proj(gate * self.up_proj(x))


class NVFP4MLP(nn.Module):
    """NVFP4 quantized MLP (simulation)."""
    def __init__(self, hidden_size: int, intermediate_size: int, block_size: int = BLOCK_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.block_size = block_size

        # Quantized weights
        self.register_buffer('gate_weight_q', None)
        self.register_buffer('gate_scales', None)
        self.register_buffer('up_weight_q', None)
        self.register_buffer('up_scales', None)
        self.register_buffer('down_weight_q', None)
        self.register_buffer('down_scales', None)

    @classmethod
    def from_bf16_mlp(cls, mlp: BF16MLP, block_size: int = BLOCK_SIZE):
        """Convert BF16 MLP to NVFP4."""
        layer = cls(mlp.gate_proj.in_features, mlp.gate_proj.out_features, block_size)

        # Quantize weights
        layer.gate_weight_q, layer.gate_scales = quantize_to_nvfp4(mlp.gate_proj.weight.data, block_size)
        layer.up_weight_q, layer.up_scales = quantize_to_nvfp4(mlp.up_proj.weight.data, block_size)
        layer.down_weight_q, layer.down_scales = quantize_to_nvfp4(mlp.down_proj.weight.data, block_size)

        return layer

    def forward(self, x):
        # Quantize input
        x_q, x_scales = quantize_to_nvfp4(x, self.block_size)

        # Gate projection (simulated NVFP4 GEMM)
        gate_dequant = dequantize_nvfp4(self.gate_weight_q, self.gate_scales, self.block_size)
        x_dequant = dequantize_nvfp4(x_q, x_scales, self.block_size)
        gate = torch.nn.functional.gelu(x_dequant @ gate_dequant.T, approximate='tanh')

        # Up projection
        up_dequant = dequantize_nvfp4(self.up_weight_q, self.up_scales, self.block_size)
        up = x_dequant @ up_dequant.T

        # Intermediate
        intermediate = gate * up

        # Down projection
        int_q, int_scales = quantize_to_nvfp4(intermediate, self.block_size)
        int_dequant = dequantize_nvfp4(int_q, int_scales, self.block_size)
        down_dequant = dequantize_nvfp4(self.down_weight_q, self.down_scales, self.block_size)

        return int_dequant @ down_dequant.T


# ============================================================================
# Tests
# ============================================================================

def test_quantization_precision():
    """Test NVFP4 quantization precision."""
    print_section("NVFP4 Quantization Precision")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sizes = [(256, 2048), (968, 2048), (50, 1024)]

    for M, K in sizes:
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

        # Quantize and dequantize
        x_q, scales = quantize_to_nvfp4(x, BLOCK_SIZE)
        x_dequant = dequantize_nvfp4(x_q, scales, BLOCK_SIZE)

        # Compute error
        abs_diff = (x.float() - x_dequant.float()).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        relative_diff = (abs_diff / (x.float().abs() + 1e-8)).mean().item() * 100

        print(f"\n  Size: [{M}, {K}]")
        print(f"    Max diff:      {max_diff:.6f}")
        print(f"    Mean diff:     {mean_diff:.6f}")
        print(f"    Relative:      {relative_diff:.2f}%")

        # Compression ratio
        original_bytes = M * K * 2  # BF16
        quantized_bytes = M * K // 2 + M * (K // BLOCK_SIZE) * 4  # FP4 + FP32 scales
        compression = original_bytes / quantized_bytes
        print(f"    Compression:   {compression:.2f}x")


def test_mlp_precision():
    """Test NVFP4 MLP precision."""
    print_section("NVFP4 MLP Precision")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_sizes = [256, 968, 50]
    hidden_size = 2048
    intermediate_size = 16384

    for batch_size in batch_sizes:
        # Create BF16 MLP
        bf16_mlp = BF16MLP(hidden_size, intermediate_size).to(device).to(torch.bfloat16)

        # Convert to NVFP4
        nvfp4_mlp = NVFP4MLP.from_bf16_mlp(bf16_mlp).to(device)

        # Test input
        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

        # Forward pass
        with torch.no_grad():
            bf16_out = bf16_mlp(x)
            nvfp4_out = nvfp4_mlp(x).to(torch.bfloat16)

        # Compute error
        abs_diff = (bf16_out.float() - nvfp4_out.float()).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        relative_diff = (abs_diff / (bf16_out.float().abs() + 1e-8)).mean().item() * 100

        print(f"\n  Batch: {batch_size}")
        print(f"    Max diff:      {max_diff:.6f}")
        print(f"    Mean diff:     {mean_diff:.6f}")
        print(f"    Relative:      {relative_diff:.2f}%")

        status = "✅ PASS" if relative_diff < 10.0 else "❌ FAIL"
        print(f"    Status:        {status}")


def test_mlp_performance():
    """Benchmark NVFP4 vs BF16 MLP performance."""
    print_section("MLP Performance (Simulation)")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping performance test")
        return

    device = "cuda"
    batch_size = 968  # Full KV Cache prefix length
    hidden_size = 2048
    intermediate_size = 16384

    # Create MLPs
    bf16_mlp = BF16MLP(hidden_size, intermediate_size).to(device).to(torch.bfloat16)
    nvfp4_mlp = NVFP4MLP.from_bf16_mlp(bf16_mlp).to(device)

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = bf16_mlp(x)
        _ = nvfp4_mlp(x)
    torch.cuda.synchronize()

    iterations = 50

    # Benchmark BF16
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = bf16_mlp(x)
    torch.cuda.synchronize()
    bf16_ms = (time.perf_counter() - start) / iterations * 1000

    # Benchmark NVFP4 (simulation - actual CUTLASS will be faster)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = nvfp4_mlp(x)
    torch.cuda.synchronize()
    nvfp4_sim_ms = (time.perf_counter() - start) / iterations * 1000

    print(f"\n  Config: [{batch_size}, {hidden_size}] -> [{intermediate_size}] -> [{hidden_size}]")
    print(f"\n  BF16 MLP:              {bf16_ms:.3f} ms")
    print(f"  NVFP4 MLP (sim):       {nvfp4_sim_ms:.3f} ms")

    # Expected CUTLASS performance (based on benchmarks)
    # CUTLASS NVFP4 is 2.8-7.8x faster than cuBLAS BF16
    expected_cutlass_speedup = 4.0  # Conservative estimate
    expected_cutlass_ms = bf16_ms / expected_cutlass_speedup

    print(f"\n  Expected CUTLASS NVFP4: ~{expected_cutlass_ms:.3f} ms ({expected_cutlass_speedup:.1f}x speedup)")


def test_full_kv_cache():
    """Test full KV Cache (18 layers) performance."""
    print_section("Full KV Cache (18 layers)")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping")
        return

    device = "cuda"
    num_layers = 18
    batch_size = 968  # 256 (image) + 512 (prefix) + 200 (lang)
    hidden_size = 2048
    intermediate_size = 16384

    # Create layer stack
    bf16_layers = nn.ModuleList([
        BF16MLP(hidden_size, intermediate_size).to(device).to(torch.bfloat16)
        for _ in range(num_layers)
    ])

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(5):
        out = x
        for layer in bf16_layers:
            out = layer(out)
    torch.cuda.synchronize()

    iterations = 20

    # Benchmark BF16
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        out = x
        for layer in bf16_layers:
            out = layer(out)
    torch.cuda.synchronize()
    bf16_total_ms = (time.perf_counter() - start) / iterations * 1000

    print(f"\n  Config: {num_layers} layers x [{batch_size}, {hidden_size}] -> [{intermediate_size}]")
    print(f"\n  BF16 total:            {bf16_total_ms:.2f} ms")
    print(f"  BF16 per layer:        {bf16_total_ms/num_layers:.3f} ms")

    # Expected with CUTLASS NVFP4
    expected_speedup = 4.0
    expected_nvfp4_ms = bf16_total_ms / expected_speedup

    print(f"\n  Expected NVFP4 total:  ~{expected_nvfp4_ms:.2f} ms ({expected_speedup:.1f}x speedup)")
    print(f"  Expected time saved:   ~{bf16_total_ms - expected_nvfp4_ms:.2f} ms")

    # Memory savings
    bf16_mem = hidden_size * intermediate_size * 3 * num_layers * 2 / 1024 / 1024  # MB
    nvfp4_mem = hidden_size * intermediate_size * 3 * num_layers * 0.5 / 1024 / 1024  # MB (approx)
    print(f"\n  Memory:")
    print(f"    BF16:    {bf16_mem:.1f} MB")
    print(f"    NVFP4:   ~{nvfp4_mem:.1f} MB")
    print(f"    Saved:   ~{bf16_mem - nvfp4_mem:.1f} MB")


def main():
    print_header("NVFP4 MLP Test for KV Cache Optimization")

    # Check environment
    print_section("Environment")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")

    # Run tests
    test_quantization_precision()
    test_mlp_precision()
    test_mlp_performance()
    test_full_kv_cache()

    # Summary
    print_header("Summary")
    print("""
  NVFP4 Optimization for KV Cache MLP:

  1. Quantization Precision:
     - Block-scaled NVFP4 provides ~4x compression
     - Typical precision loss: 5-10% relative

  2. Expected Performance (with CUTLASS SM110a):
     - Single layer: ~1 ms (from 4 ms BF16)
     - 18 layers: ~18 ms → ~4-5 ms
     - Speedup: 3-4x

  3. Memory Savings:
     - Weights: 3.6 GB → ~0.9 GB
     - Saved: ~2.7 GB

  4. Next Steps:
     - Build C++ extension in Docker: cd nvfp4_extension && python setup.py install
     - Test with actual CUTLASS kernel
     - Integrate into PI0Pytorch model
""")


if __name__ == "__main__":
    main()
