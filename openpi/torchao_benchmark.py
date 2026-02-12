#!/usr/bin/env python3
"""
TorchAO Quantization Benchmark on Jetson Thor (SM110)
Test all available quantization methods for precision and latency
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, Optional, Callable
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")

import torchao
from torchao.quantization import quantize_
print(f"TorchAO version: {torchao.__version__}")


class ErrorStats:
    """Error statistics for precision verification"""
    def __init__(self, ref: torch.Tensor, test: torch.Tensor):
        diff = (test.float() - ref.float())
        ref_f = ref.float()

        self.max_abs_error = diff.abs().max().item()
        self.mean_abs_error = diff.abs().mean().item()
        self.rmse = torch.sqrt((diff ** 2).mean()).item()

        mask = ref_f.abs() > 1e-6
        if mask.any():
            rel_err = (diff[mask] / ref_f[mask]).abs()
            self.max_rel_error = rel_err.max().item()
            self.mean_rel_error = rel_err.mean().item()
        else:
            self.max_rel_error = float('inf')
            self.mean_rel_error = float('inf')


def benchmark_latency(model: nn.Module, x: torch.Tensor,
                      warmup: int = 10, iterations: int = 100) -> Dict[str, float]:
    """Benchmark model latency"""
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iterations):
        start_event.record()
        with torch.no_grad():
            _ = model(x)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
    }


def test_config(name: str, config_fn: Callable, M: int, K: int, N: int,
                dtype: torch.dtype = torch.bfloat16) -> Dict[str, Any]:
    """Test a quantization configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {name} | M={M}, K={K}, N={N}")
    print('='*60)

    try:
        # Create model
        model_ref = nn.Linear(K, N, bias=False).to(device).to(dtype)
        model_quant = copy.deepcopy(model_ref)

        # Input
        x = torch.randn(M, K, dtype=dtype, device=device)

        # Reference output
        with torch.no_grad():
            ref_output = model_ref(x)

        # Apply quantization
        print("Applying quantization...")
        config = config_fn()
        quantize_(model_quant, config)

        # Quantized output
        with torch.no_grad():
            quant_output = model_quant(x)

        # Errors
        errors = ErrorStats(ref_output, quant_output)
        print(f"Precision: MaxAbs={errors.max_abs_error:.6f}, MeanAbs={errors.mean_abs_error:.6f}, "
              f"MaxRel={errors.max_rel_error:.4%}")

        # Benchmark
        print("Benchmarking...")
        ref_latency = benchmark_latency(model_ref, x)
        quant_latency = benchmark_latency(model_quant, x)

        # TFLOPS
        flops = 2 * M * N * K
        ref_tflops = flops / (ref_latency['mean_ms'] * 1e-3) / 1e12
        quant_tflops = flops / (quant_latency['mean_ms'] * 1e-3) / 1e12
        speedup = ref_latency['mean_ms'] / quant_latency['mean_ms']

        print(f"Latency: Ref={ref_latency['mean_ms']:.3f}ms ({ref_tflops:.2f} TFLOPS), "
              f"Quant={quant_latency['mean_ms']:.3f}ms ({quant_tflops:.2f} TFLOPS), "
              f"Speedup={speedup:.2f}x")

        return {
            'name': name, 'M': M, 'K': K, 'N': N,
            'success': True,
            'ref_ms': ref_latency['mean_ms'],
            'quant_ms': quant_latency['mean_ms'],
            'ref_tflops': ref_tflops,
            'quant_tflops': quant_tflops,
            'speedup': speedup,
            'max_abs_error': errors.max_abs_error,
            'mean_abs_error': errors.mean_abs_error,
            'max_rel_error': errors.max_rel_error,
        }

    except Exception as e:
        print(f"❌ Failed: {e}")
        return {'name': name, 'M': M, 'K': K, 'N': N, 'success': False, 'error': str(e)}


def main():
    print("="*70)
    print("TorchAO Quantization Benchmark on Jetson Thor (SM110)")
    print("="*70)

    # Import available configs
    from torchao.quantization import (
        Int8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Float8DynamicActivationFloat8WeightConfig,
        Float8WeightOnlyConfig,
    )

    # Test sizes (typical for Pi0 model)
    # Pi0.5: hidden_dim=1536, mlp_dim=6144, heads=24, head_dim=64
    test_sizes = [
        (1, 1536, 1536),    # Attention projection
        (1, 1536, 6144),    # MLP up projection
        (1, 6144, 1536),    # MLP down projection
        (8, 1536, 1536),    # Batch=8
        (8, 1536, 6144),
        (8, 6144, 1536),
    ]

    # Available quantization configs
    configs = [
        ("W8A16 (Int8WeightOnly)", lambda: Int8WeightOnlyConfig()),
        ("W8A8 (Int8DynActInt8Weight)", lambda: Int8DynamicActivationInt8WeightConfig()),
        ("FP8 Weight-Only", lambda: Float8WeightOnlyConfig()),
        ("FP8 Dynamic (W8A8-FP8)", lambda: Float8DynamicActivationFloat8WeightConfig()),
    ]

    # Try to add FP8xINT4 if available
    try:
        from torchao.quantization import Float8DynamicActivationInt4WeightConfig
        configs.append(("W4A8-FP8 (Float8DynActInt4Weight)",
                       lambda: Float8DynamicActivationInt4WeightConfig()))
    except ImportError:
        print("Note: Float8DynamicActivationInt4WeightConfig not available")

    results = []

    for M, K, N in test_sizes:
        for name, config_fn in configs:
            result = test_config(name, config_fn, M, K, N)
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if successful:
        print(f"\n✅ Successful: {len(successful)} tests")
        print("-"*70)
        print(f"{'Config':<30} {'Size':<20} {'TFLOPS':>10} {'Speedup':>10} {'MaxRelErr':>12}")
        print("-"*70)
        for r in successful:
            size = f"{r['M']}x{r['K']}x{r['N']}"
            print(f"{r['name']:<30} {size:<20} {r['quant_tflops']:>10.2f} "
                  f"{r['speedup']:>9.2f}x {r['max_rel_error']:>11.4%}")

    if failed:
        print(f"\n❌ Failed: {len(failed)} tests")
        for r in failed:
            print(f"  {r['name']}: {r.get('error', 'Unknown')[:50]}")

    # Best per config
    print("\n" + "="*70)
    print("BEST RESULTS PER CONFIG")
    print("="*70)

    config_best = {}
    for r in successful:
        name = r['name']
        if name not in config_best or r['quant_tflops'] > config_best[name]['quant_tflops']:
            config_best[name] = r

    for name, r in sorted(config_best.items(), key=lambda x: -x[1]['quant_tflops']):
        size = f"{r['M']}x{r['K']}x{r['N']}"
        print(f"  {name:<35} {r['quant_tflops']:>8.2f} TFLOPS @ {size} | Speedup: {r['speedup']:.2f}x")

    # Best overall
    if successful:
        best = max(successful, key=lambda x: x['quant_tflops'])
        print(f"\n🏆 Overall Best: {best['name']}")
        print(f"   {best['quant_tflops']:.2f} TFLOPS @ {best['M']}x{best['K']}x{best['N']}")
        print(f"   Speedup: {best['speedup']:.2f}x, MaxRelError: {best['max_rel_error']:.4%}")


if __name__ == "__main__":
    main()
