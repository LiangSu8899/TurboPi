#!/usr/bin/env python3
"""
TorchAO Quantization Test on Jetson Thor (SM110)
Test W4A16, W4A8, W4A4 configurations for precision and latency
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, Optional

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")

# Import torchao quantization
import torchao
from torchao.quantization import (
    quantize_,
    int4_weight_only,
    int8_weight_only,
    int8_dynamic_activation_int4_weight,
)

print(f"\nTorchAO version: {torchao.__version__}")


class SimpleLinear(nn.Module):
    """Simple linear layer for testing"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class ErrorStats:
    """Error statistics for precision verification"""
    def __init__(self, ref: torch.Tensor, test: torch.Tensor):
        diff = (test - ref).float()
        ref_f = ref.float()

        self.max_abs_error = diff.abs().max().item()
        self.mean_abs_error = diff.abs().mean().item()
        self.rmse = torch.sqrt((diff ** 2).mean()).item()

        # Relative error (avoid div by zero)
        mask = ref_f.abs() > 1e-6
        if mask.any():
            rel_err = (diff[mask] / ref_f[mask]).abs()
            self.max_rel_error = rel_err.max().item()
            self.mean_rel_error = rel_err.mean().item()
        else:
            self.max_rel_error = float('inf')
            self.mean_rel_error = float('inf')

    def __str__(self):
        return (f"MaxAbs: {self.max_abs_error:.6f}, MeanAbs: {self.mean_abs_error:.6f}, "
                f"RMSE: {self.rmse:.6f}, MaxRel: {self.max_rel_error:.4%}, MeanRel: {self.mean_rel_error:.4%}")


def benchmark_latency(model: nn.Module, input_tensor: torch.Tensor,
                      warmup: int = 10, iterations: int = 100) -> Dict[str, float]:
    """Benchmark model latency"""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)

    torch.cuda.synchronize()

    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iterations):
        start_event.record()
        with torch.no_grad():
            _ = model(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
    }


def test_quantization(name: str, quant_func, M: int, K: int, N: int,
                      dtype: torch.dtype = torch.bfloat16) -> Optional[Dict[str, Any]]:
    """Test a specific quantization configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Matrix size: M={M}, K={K}, N={N}, dtype={dtype}")
    print('='*60)

    try:
        # Create baseline FP16/BF16 model
        model_ref = SimpleLinear(K, N).to(device).to(dtype)

        # Clone for quantization
        model_quant = SimpleLinear(K, N).to(device).to(dtype)
        model_quant.load_state_dict(model_ref.state_dict())

        # Apply quantization
        print("Applying quantization...")
        quantize_(model_quant, quant_func())

        # Create input
        x = torch.randn(M, K, dtype=dtype, device=device)

        # Reference output
        with torch.no_grad():
            ref_output = model_ref(x)

        # Quantized output
        with torch.no_grad():
            quant_output = model_quant(x)

        # Compute errors
        errors = ErrorStats(ref_output, quant_output)
        print(f"Precision: {errors}")

        # Benchmark
        print("Benchmarking...")
        ref_latency = benchmark_latency(model_ref, x)
        quant_latency = benchmark_latency(model_quant, x)

        # Compute TFLOPS
        flops = 2 * M * N * K  # GEMM FLOPs
        ref_tflops = flops / (ref_latency['mean_ms'] * 1e-3) / 1e12
        quant_tflops = flops / (quant_latency['mean_ms'] * 1e-3) / 1e12
        speedup = ref_latency['mean_ms'] / quant_latency['mean_ms']

        print(f"\nLatency Results:")
        print(f"  Reference ({dtype}): {ref_latency['mean_ms']:.3f} ± {ref_latency['std_ms']:.3f} ms ({ref_tflops:.2f} TFLOPS)")
        print(f"  Quantized ({name}): {quant_latency['mean_ms']:.3f} ± {quant_latency['std_ms']:.3f} ms ({quant_tflops:.2f} TFLOPS)")
        print(f"  Speedup: {speedup:.2f}x")

        return {
            'name': name,
            'M': M, 'K': K, 'N': N,
            'ref_latency_ms': ref_latency['mean_ms'],
            'quant_latency_ms': quant_latency['mean_ms'],
            'ref_tflops': ref_tflops,
            'quant_tflops': quant_tflops,
            'speedup': speedup,
            'max_abs_error': errors.max_abs_error,
            'mean_abs_error': errors.mean_abs_error,
            'rmse': errors.rmse,
            'max_rel_error': errors.max_rel_error,
            'mean_rel_error': errors.mean_rel_error,
            'success': True
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'name': name,
            'M': M, 'K': K, 'N': N,
            'error': str(e),
            'success': False
        }


def main():
    print("="*60)
    print("TorchAO Quantization Benchmark on Jetson Thor (SM110)")
    print("="*60)

    # Test sizes (typical for transformer models)
    test_sizes = [
        # (M, K, N) - batch, in_features, out_features
        (1, 2048, 2048),      # Single token, medium model
        (8, 2048, 2048),      # Small batch
        (1, 4096, 4096),      # Larger model
        (8, 4096, 4096),      # Larger batch
        (1, 2048, 8192),      # MLP up projection
        (1, 8192, 2048),      # MLP down projection
    ]

    # Quantization configurations to test
    quant_configs = [
        ("W4A16 (int4_weight_only)", int4_weight_only),
        ("W8A16 (int8_weight_only)", int8_weight_only),
        ("W4A8 (int8_dynamic_activation_int4_weight)", int8_dynamic_activation_int4_weight),
    ]

    # Try to import additional configs
    try:
        from torchao.quantization import int4_dynamic_activation_int4_weight
        quant_configs.append(("W4A4 (int4_dynamic_activation_int4_weight)", int4_dynamic_activation_int4_weight))
    except ImportError:
        print("Note: int4_dynamic_activation_int4_weight not available")

    try:
        from torchao.quantization import Float8DynamicActivationInt4WeightConfig
        # This needs different API, skip for now
        pass
    except ImportError:
        pass

    results = []

    # Test each configuration with different sizes
    for M, K, N in test_sizes:
        for name, quant_func in quant_configs:
            result = test_quantization(name, quant_func, M, K, N)
            if result:
                results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if successful:
        print("\n✅ Successful configurations:")
        for r in successful:
            print(f"  {r['name']} ({r['M']}×{r['K']}×{r['N']}): "
                  f"{r['quant_tflops']:.2f} TFLOPS, "
                  f"Speedup: {r['speedup']:.2f}x, "
                  f"MaxRelErr: {r['max_rel_error']:.4%}")

    if failed:
        print("\n❌ Failed configurations:")
        for r in failed:
            print(f"  {r['name']} ({r['M']}×{r['K']}×{r['N']}): {r.get('error', 'Unknown error')}")

    # Best performing
    if successful:
        best = max(successful, key=lambda x: x['quant_tflops'])
        print(f"\n🏆 Best performing: {best['name']} @ {best['M']}×{best['K']}×{best['N']}")
        print(f"   {best['quant_tflops']:.2f} TFLOPS, {best['speedup']:.2f}x speedup")


if __name__ == "__main__":
    main()
