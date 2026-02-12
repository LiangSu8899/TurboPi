#!/usr/bin/env python3
"""
TensorRT Native Quantization Benchmark on Jetson Thor (SM110)
Test INT8 and FP8 precision with TensorRT
"""

import torch
import torch.nn as nn
import torch_tensorrt
import tensorrt as trt
import time
import numpy as np
from typing import Dict

print(f"TensorRT version: {trt.__version__}")
print(f"torch_tensorrt version: {torch_tensorrt.__version__}")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")


class SimpleLinear(nn.Module):
    """Simple linear layer for testing"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class ErrorStats:
    def __init__(self, ref: torch.Tensor, test: torch.Tensor):
        diff = (test.float() - ref.float())
        ref_f = ref.float()
        self.max_abs_error = diff.abs().max().item()
        self.mean_abs_error = diff.abs().mean().item()
        self.rmse = torch.sqrt((diff ** 2).mean()).item()
        mask = ref_f.abs() > 1e-6
        if mask.any():
            self.max_rel_error = (diff[mask] / ref_f[mask]).abs().max().item()
        else:
            self.max_rel_error = float('inf')


def benchmark_latency(model, x, warmup=20, iters=100) -> Dict[str, float]:
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        start.record()
        with torch.no_grad():
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
    }


def test_trt_precision(M: int, K: int, N: int, precision: str):
    """Test TensorRT with specific precision"""
    print(f"\n{'='*60}")
    print(f"Testing TRT {precision} | M={M}, K={K}, N={N}")
    print('='*60)

    dtype = torch.float16
    enabled_precisions = {torch.float16}

    if precision == "INT8":
        enabled_precisions.add(torch.int8)
    elif precision == "FP8":
        if hasattr(torch, 'float8_e4m3fn'):
            enabled_precisions.add(torch.float8_e4m3fn)
        else:
            print("FP8 dtype not available in PyTorch")
            return None

    try:
        # Create model
        model = SimpleLinear(K, N).cuda().to(dtype).eval()
        x = torch.randn(M, K, dtype=dtype, device='cuda')

        # Reference output
        with torch.no_grad():
            ref_output = model(x)

        # Compile with TRT
        print(f"Compiling with TRT ({precision})...")

        compile_settings = {
            "inputs": [torch_tensorrt.Input(shape=(M, K), dtype=dtype)],
            "enabled_precisions": enabled_precisions,
            "truncate_long_and_double": True,
            "device": torch_tensorrt.Device("cuda:0"),
        }

        if precision == "INT8":
            # For INT8, need calibration or PTQ
            # Using post-training quantization with default calibration
            compile_settings["calibrator"] = None  # Use default

        trt_model = torch_tensorrt.compile(model, **compile_settings)

        # TRT output
        with torch.no_grad():
            trt_output = trt_model(x)

        # Errors
        errors = ErrorStats(ref_output, trt_output)
        print(f"Precision: MaxAbs={errors.max_abs_error:.6f}, MaxRel={errors.max_rel_error:.4%}")

        # Benchmark
        print("Benchmarking...")
        ref_lat = benchmark_latency(model, x)
        trt_lat = benchmark_latency(trt_model, x)

        # TFLOPS
        flops = 2 * M * N * K
        ref_tflops = flops / (ref_lat['mean_ms'] * 1e-3) / 1e12
        trt_tflops = flops / (trt_lat['mean_ms'] * 1e-3) / 1e12
        speedup = ref_lat['mean_ms'] / trt_lat['mean_ms']

        print(f"Latency: PyTorch={ref_lat['mean_ms']:.3f}ms ({ref_tflops:.2f} TFLOPS)")
        print(f"         TRT {precision}={trt_lat['mean_ms']:.3f}ms ({trt_tflops:.2f} TFLOPS)")
        print(f"Speedup: {speedup:.2f}x")

        return {
            'precision': precision,
            'M': M, 'K': K, 'N': N,
            'pytorch_ms': ref_lat['mean_ms'],
            'trt_ms': trt_lat['mean_ms'],
            'pytorch_tflops': ref_tflops,
            'trt_tflops': trt_tflops,
            'speedup': speedup,
            'max_rel_error': errors.max_rel_error,
            'success': True
        }

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return {'precision': precision, 'M': M, 'K': K, 'N': N, 'success': False, 'error': str(e)}


def main():
    print("="*70)
    print("TensorRT Native Quantization Benchmark on Jetson Thor")
    print("="*70)

    # Test sizes
    test_sizes = [
        (1, 1536, 1536),
        (8, 1536, 1536),
        (1, 1536, 6144),
        (8, 1536, 6144),
        (1, 6144, 1536),
        (8, 6144, 1536),
    ]

    # Precisions to test
    precisions = ["FP16", "INT8", "FP8"]

    results = []

    for M, K, N in test_sizes:
        for precision in precisions:
            result = test_trt_precision(M, K, N, precision)
            if result:
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
        print(f"{'Precision':<10} {'Size':<20} {'TRT TFLOPS':>12} {'Speedup':>10} {'MaxRelErr':>12}")
        print("-"*70)
        for r in successful:
            size = f"{r['M']}x{r['K']}x{r['N']}"
            print(f"{r['precision']:<10} {size:<20} {r['trt_tflops']:>12.2f} "
                  f"{r['speedup']:>9.2f}x {r['max_rel_error']:>11.4%}")

    if failed:
        print(f"\n❌ Failed: {len(failed)} tests")
        for r in failed:
            print(f"  {r['precision']} @ {r['M']}x{r['K']}x{r['N']}: {r.get('error', 'Unknown')[:60]}")

    # Best per precision
    if successful:
        print("\n" + "="*70)
        print("BEST RESULTS PER PRECISION")
        print("="*70)

        for prec in precisions:
            prec_results = [r for r in successful if r['precision'] == prec]
            if prec_results:
                best = max(prec_results, key=lambda x: x['trt_tflops'])
                size = f"{best['M']}x{best['K']}x{best['N']}"
                print(f"  {prec:<10} {best['trt_tflops']:>8.2f} TFLOPS @ {size} | "
                      f"Speedup: {best['speedup']:.2f}x | MaxRelErr: {best['max_rel_error']:.4%}")


if __name__ == "__main__":
    main()
