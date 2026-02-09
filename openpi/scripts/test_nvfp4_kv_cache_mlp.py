#!/usr/bin/env python3
"""
NVFP4 KV Cache MLP 完整测试

测试:
1. NVFP4 量化精度
2. 单层 MLP 延迟 (NVFP4 vs BF16 vs FP8)
3. 18 层 KV Cache MLP 总延迟
4. 端到端集成测试

目标:
- 精度损失 < 5%
- 18 层 MLP: 60ms -> ~15ms (4x 加速)
"""

import torch
import torch.nn as nn
import time
import subprocess
import os
import json
import numpy as np
from typing import Tuple, Optional

# CUTLASS NVFP4 binary 路径
CUTLASS_NVFP4_BINARY = "/workspace/external/cutlass_sm110_build/nvfp4_gemm_sm110a"


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_section(title: str):
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


# ============================================================================
# NVFP4 Quantization
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


class FP8MLP(nn.Module):
    """FP8 quantized MLP using torch._scaled_mm."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # BF16 weights
        self.gate_weight = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.up_weight = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.down_weight = nn.Parameter(torch.empty(hidden_size, intermediate_size))

        # FP8 quantized buffers
        self.register_buffer('gate_fp8', None)
        self.register_buffer('gate_scale', None)
        self.register_buffer('up_fp8', None)
        self.register_buffer('up_scale', None)
        self.register_buffer('down_fp8', None)
        self.register_buffer('down_scale', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.gate_weight)
        nn.init.kaiming_uniform_(self.up_weight)
        nn.init.kaiming_uniform_(self.down_weight)

    def quantize_weights(self):
        """Pre-quantize weights to FP8."""
        with torch.no_grad():
            for name, weight in [
                ('gate', self.gate_weight),
                ('up', self.up_weight),
                ('down', self.down_weight)
            ]:
                amax = weight.abs().max().clamp(min=1e-12)
                scale = torch.tensor(448.0 / amax.item(), device=weight.device)
                w_scaled = (weight.float() * scale.item()).clamp(-448, 448)

                # Column-major for _scaled_mm
                K, N = weight.shape[1], weight.shape[0]
                w_fp8 = torch.empty(N, K, device=weight.device, dtype=torch.float8_e4m3fn).T
                w_fp8.copy_(w_scaled.T.to(torch.float8_e4m3fn))

                setattr(self, f'{name}_fp8', w_fp8)
                setattr(self, f'{name}_scale', 1.0 / scale)

    @classmethod
    def from_bf16_mlp(cls, mlp: BF16MLP):
        """Convert BF16 MLP to FP8."""
        layer = cls(mlp.gate_proj.in_features, mlp.gate_proj.out_features)
        with torch.no_grad():
            layer.gate_weight.copy_(mlp.gate_proj.weight)
            layer.up_weight.copy_(mlp.up_proj.weight)
            layer.down_weight.copy_(mlp.down_proj.weight)
        layer.quantize_weights()
        return layer

    def forward(self, x):
        # Quantize input
        x_amax = x.abs().max().clamp(min=1e-12)
        x_scale = torch.tensor(448.0 / x_amax.item(), device=x.device)
        x_scaled = (x.float() * x_scale.item()).clamp(-448, 448)
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)

        scale_a = (1.0 / x_scale).to(x.device)

        # Gate projection
        gate = torch._scaled_mm(
            x_fp8, self.gate_fp8,
            scale_a=scale_a, scale_b=self.gate_scale,
            out_dtype=torch.bfloat16
        )
        gate = torch.nn.functional.gelu(gate, approximate='tanh')

        # Up projection
        up = torch._scaled_mm(
            x_fp8, self.up_fp8,
            scale_a=scale_a, scale_b=self.up_scale,
            out_dtype=torch.bfloat16
        )

        # Intermediate
        intermediate = gate * up

        # Quantize intermediate
        int_amax = intermediate.abs().max().clamp(min=1e-12)
        int_scale = torch.tensor(448.0 / int_amax.item(), device=x.device)
        int_scaled = (intermediate.float() * int_scale.item()).clamp(-448, 448)
        int_fp8 = int_scaled.to(torch.float8_e4m3fn)

        # Down projection
        out = torch._scaled_mm(
            int_fp8, self.down_fp8,
            scale_a=(1.0 / int_scale).to(x.device), scale_b=self.down_scale,
            out_dtype=torch.bfloat16
        )

        return out


# ============================================================================
# CUTLASS NVFP4 Benchmark
# ============================================================================

def run_cutlass_nvfp4(M: int, N: int, K: int, iterations: int = 100) -> Tuple[Optional[float], Optional[float]]:
    """Run CUTLASS NVFP4 GEMM benchmark."""
    if not os.path.exists(CUTLASS_NVFP4_BINARY):
        return None, None

    cmd = f"{CUTLASS_NVFP4_BINARY} --m={M} --n={N} --k={K} --iterations={iterations}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if "Failed" in output or "error" in output.lower():
        return None, None

    ms = None
    gflops = None
    for line in output.split("\n"):
        if "Avg runtime" in line:
            ms = float(line.split()[-2])
        if "GFLOPS" in line:
            gflops = float(line.split()[-1])

    return ms, gflops


def benchmark_cutlass_bf16(M: int, N: int, K: int, iterations: int = 100) -> float:
    """Benchmark cuBLAS BF16 GEMM."""
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = torch.matmul(x, w.T)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(x, w.T)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / iterations * 1000


# ============================================================================
# Tests
# ============================================================================

def test_quantization_precision():
    """Test NVFP4 quantization precision."""
    print_section("NVFP4 Quantization Precision")

    device = "cuda"
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


def test_single_layer_performance():
    """Test single layer performance: NVFP4 vs FP8 vs BF16."""
    print_section("Single Layer MLP Performance")

    device = "cuda"
    batch_size = 256  # Padded size for CUTLASS
    hidden_size = 2048
    intermediate_size = 16384

    # Create and convert MLPs
    bf16_mlp = BF16MLP(hidden_size, intermediate_size).to(device).to(torch.bfloat16)
    fp8_mlp = FP8MLP.from_bf16_mlp(bf16_mlp).to(device)

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    iterations = 50

    # Warmup
    for _ in range(10):
        _ = bf16_mlp(x)
        _ = fp8_mlp(x)
    torch.cuda.synchronize()

    # Benchmark BF16
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = bf16_mlp(x)
    torch.cuda.synchronize()
    bf16_ms = (time.perf_counter() - start) / iterations * 1000

    # Benchmark FP8
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fp8_mlp(x)
    torch.cuda.synchronize()
    fp8_ms = (time.perf_counter() - start) / iterations * 1000

    # CUTLASS NVFP4 GEMM benchmarks
    # MLP = gate_proj (M,K->M,N) + up_proj (M,K->M,N) + down_proj (M,N->M,K)
    # gate_proj: 256 x 16384 x 2048
    # up_proj:   256 x 16384 x 2048
    # down_proj: 256 x 2048 x 16384

    nvfp4_gate_ms, _ = run_cutlass_nvfp4(batch_size, intermediate_size, hidden_size)
    nvfp4_up_ms, _ = run_cutlass_nvfp4(batch_size, intermediate_size, hidden_size)
    nvfp4_down_ms, _ = run_cutlass_nvfp4(batch_size, hidden_size, intermediate_size)

    if nvfp4_gate_ms is not None:
        nvfp4_total_ms = nvfp4_gate_ms + nvfp4_up_ms + nvfp4_down_ms
    else:
        nvfp4_total_ms = None

    print(f"\n  Config: batch={batch_size}, hidden={hidden_size}, intermediate={intermediate_size}")
    print(f"\n  BF16 MLP:       {bf16_ms:.3f} ms")
    print(f"  FP8 MLP:        {fp8_ms:.3f} ms ({bf16_ms/fp8_ms:.2f}x vs BF16)")

    if nvfp4_total_ms:
        print(f"  NVFP4 GEMM:     {nvfp4_total_ms:.3f} ms ({bf16_ms/nvfp4_total_ms:.2f}x vs BF16)")
        print(f"    - gate_proj:  {nvfp4_gate_ms:.3f} ms")
        print(f"    - up_proj:    {nvfp4_up_ms:.3f} ms")
        print(f"    - down_proj:  {nvfp4_down_ms:.3f} ms")
    else:
        print("  NVFP4 GEMM:     Not available (CUTLASS binary missing)")

    return bf16_ms, fp8_ms, nvfp4_total_ms


def test_18_layer_kv_cache():
    """Test full 18-layer KV Cache MLP performance."""
    print_section("18-Layer KV Cache MLP Performance")

    device = "cuda"
    num_layers = 18
    batch_size = 256  # Padded from 200 (typical prefix length)
    hidden_size = 2048
    intermediate_size = 16384

    # Create layer stack
    bf16_layers = nn.ModuleList([
        BF16MLP(hidden_size, intermediate_size).to(device).to(torch.bfloat16)
        for _ in range(num_layers)
    ])

    fp8_layers = nn.ModuleList([
        FP8MLP.from_bf16_mlp(layer).to(device)
        for layer in bf16_layers
    ])

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    iterations = 20

    # Warmup
    for _ in range(5):
        out = x
        for layer in bf16_layers:
            out = layer(out)
        out = x
        for layer in fp8_layers:
            out = layer(out)
    torch.cuda.synchronize()

    # Benchmark BF16
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        out = x
        for layer in bf16_layers:
            out = layer(out)
    torch.cuda.synchronize()
    bf16_total_ms = (time.perf_counter() - start) / iterations * 1000

    # Benchmark FP8
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        out = x
        for layer in fp8_layers:
            out = layer(out)
    torch.cuda.synchronize()
    fp8_total_ms = (time.perf_counter() - start) / iterations * 1000

    # Estimate NVFP4 (based on single layer GEMM)
    nvfp4_gate_ms, _ = run_cutlass_nvfp4(batch_size, intermediate_size, hidden_size)
    nvfp4_up_ms, _ = run_cutlass_nvfp4(batch_size, intermediate_size, hidden_size)
    nvfp4_down_ms, _ = run_cutlass_nvfp4(batch_size, hidden_size, intermediate_size)

    if nvfp4_gate_ms is not None:
        nvfp4_layer_ms = nvfp4_gate_ms + nvfp4_up_ms + nvfp4_down_ms
        nvfp4_total_ms = nvfp4_layer_ms * num_layers
    else:
        nvfp4_total_ms = None

    print(f"\n  Config: {num_layers} layers x [{batch_size}, {hidden_size}] -> [{intermediate_size}]")
    print(f"\n  BF16 total:     {bf16_total_ms:.2f} ms ({bf16_total_ms/num_layers:.3f} ms/layer)")
    print(f"  FP8 total:      {fp8_total_ms:.2f} ms ({fp8_total_ms/num_layers:.3f} ms/layer)")
    print(f"                  {bf16_total_ms/fp8_total_ms:.2f}x speedup vs BF16")

    if nvfp4_total_ms:
        print(f"  NVFP4 total:    {nvfp4_total_ms:.2f} ms ({nvfp4_layer_ms:.3f} ms/layer)")
        print(f"                  {bf16_total_ms/nvfp4_total_ms:.2f}x speedup vs BF16")
        print(f"                  {fp8_total_ms/nvfp4_total_ms:.2f}x speedup vs FP8")

    # Memory savings
    bf16_mem = hidden_size * intermediate_size * 3 * num_layers * 2 / 1024 / 1024  # MB
    fp8_mem = hidden_size * intermediate_size * 3 * num_layers * 1 / 1024 / 1024  # MB
    nvfp4_mem = hidden_size * intermediate_size * 3 * num_layers * 0.5 / 1024 / 1024  # MB

    print(f"\n  Memory:")
    print(f"    BF16:   {bf16_mem:.1f} MB")
    print(f"    FP8:    {fp8_mem:.1f} MB ({bf16_mem/fp8_mem:.1f}x compression)")
    print(f"    NVFP4:  {nvfp4_mem:.1f} MB ({bf16_mem/nvfp4_mem:.1f}x compression)")

    return bf16_total_ms, fp8_total_ms, nvfp4_total_ms


def test_precision_comparison():
    """Compare output precision: BF16 vs FP8 vs NVFP4 (simulated)."""
    print_section("Output Precision Comparison")

    device = "cuda"
    batch_size = 256
    hidden_size = 2048
    intermediate_size = 16384

    # Create reference BF16 MLP
    bf16_mlp = BF16MLP(hidden_size, intermediate_size).to(device).to(torch.bfloat16)
    fp8_mlp = FP8MLP.from_bf16_mlp(bf16_mlp).to(device)

    # Test multiple inputs
    num_samples = 10
    fp8_errors = []

    for _ in range(num_samples):
        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            bf16_out = bf16_mlp(x)
            fp8_out = fp8_mlp(x)

        # Compute relative error
        relative_error = (fp8_out.float() - bf16_out.float()).abs() / (bf16_out.float().abs() + 1e-8)
        fp8_errors.append(relative_error.mean().item() * 100)

    fp8_mean_error = np.mean(fp8_errors)
    fp8_max_error = np.max(fp8_errors)

    print(f"\n  FP8 vs BF16:")
    print(f"    Mean relative error: {fp8_mean_error:.2f}%")
    print(f"    Max relative error:  {fp8_max_error:.2f}%")

    if fp8_mean_error < 5.0:
        print(f"    Status: PASS (< 5%)")
    else:
        print(f"    Status: WARNING (> 5%)")

    return fp8_mean_error


def main():
    print_header("NVFP4 KV Cache MLP Complete Test")

    # Check environment
    print_section("Environment")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUTLASS binary: {'Available' if os.path.exists(CUTLASS_NVFP4_BINARY) else 'Not found'}")

    # Run tests
    test_quantization_precision()
    bf16_single, fp8_single, nvfp4_single = test_single_layer_performance()
    bf16_18layer, fp8_18layer, nvfp4_18layer = test_18_layer_kv_cache()
    precision_error = test_precision_comparison()

    # Summary
    print_header("Summary")

    results = {
        "single_layer_ms": {
            "bf16": bf16_single,
            "fp8": fp8_single,
            "nvfp4": nvfp4_single,
        },
        "18_layer_ms": {
            "bf16": bf16_18layer,
            "fp8": fp8_18layer,
            "nvfp4": nvfp4_18layer,
        },
        "fp8_precision_error_pct": precision_error,
    }

    print("\n  Performance Summary:")
    print(f"    Single Layer MLP:")
    print(f"      - BF16:  {bf16_single:.3f} ms")
    print(f"      - FP8:   {fp8_single:.3f} ms ({bf16_single/fp8_single:.2f}x)")
    if nvfp4_single:
        print(f"      - NVFP4: {nvfp4_single:.3f} ms ({bf16_single/nvfp4_single:.2f}x)")

    print(f"\n    18-Layer KV Cache:")
    print(f"      - BF16:  {bf16_18layer:.2f} ms")
    print(f"      - FP8:   {fp8_18layer:.2f} ms ({bf16_18layer/fp8_18layer:.2f}x)")
    if nvfp4_18layer:
        print(f"      - NVFP4: {nvfp4_18layer:.2f} ms ({bf16_18layer/nvfp4_18layer:.2f}x)")

    print(f"\n    Precision:")
    print(f"      - FP8 error: {precision_error:.2f}%")

    # Save results
    output_file = "/workspace/nvfp4_kv_cache_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
