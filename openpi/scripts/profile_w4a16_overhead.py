#!/usr/bin/env python3
"""
Profile W4A16 overhead sources.

Identifies:
1. dtype conversion overhead (bf16 <-> fp16)
2. Python dispatch overhead
3. TVM kernel JIT compilation
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import torch
import numpy as np

# TVM path
TVM_PATH = "/opt/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)


def benchmark_dtype_conversion():
    """Measure dtype conversion overhead."""
    print("=" * 60)
    print("1. DTYPE CONVERSION OVERHEAD")
    print("=" * 60)

    device = 'cuda'
    shape = (1, 2048)

    # Create tensors
    x_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)
    x_fp16 = torch.randn(shape, dtype=torch.float16, device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(100):
        _ = x_bf16.to(torch.float16)
        _ = x_fp16.to(torch.bfloat16)
    torch.cuda.synchronize()

    # Benchmark bf16 -> fp16
    times = []
    for _ in range(1000):
        start.record()
        _ = x_bf16.to(torch.float16)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    print(f"  bf16 -> fp16: {np.mean(times):.4f} ms (std: {np.std(times):.4f})")

    # Benchmark fp16 -> bf16
    times = []
    for _ in range(1000):
        start.record()
        _ = x_fp16.to(torch.bfloat16)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    print(f"  fp16 -> bf16: {np.mean(times):.4f} ms (std: {np.std(times):.4f})")

    # Benchmark output conversion (1, 16384) -> typical MLP output
    x_fp32 = torch.randn(1, 16384, dtype=torch.float32, device=device)
    times = []
    for _ in range(1000):
        start.record()
        _ = x_fp32.to(torch.bfloat16)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    print(f"  fp32 -> bf16 (output): {np.mean(times):.4f} ms")


def benchmark_raw_tvm_kernel():
    """Measure raw TVM kernel latency."""
    print("\n" + "=" * 60)
    print("2. RAW TVM KERNEL LATENCY")
    print("=" * 60)

    from openpi.ops.w4a16_gemv import w4a16_gemv, precompile_kernels, QUANT_BLOCK

    device = 'cuda'
    N, K = 16384, 2048
    num_scale_blocks = K // QUANT_BLOCK

    # Create test data
    x = torch.randn(1, K, dtype=torch.float16, device=device)
    weight_packed = torch.randint(0, 2**31-1, (num_scale_blocks, N, 4),
                                   dtype=torch.int32, device=device)
    scales = torch.randn(num_scale_blocks, N, dtype=torch.float16, device=device).abs() + 0.1

    # Precompile
    print("  Precompiling kernel...")
    precompile_kernels([(N, K)])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(50):
        _ = w4a16_gemv(x, weight_packed, scales)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(200):
        start.record()
        _ = w4a16_gemv(x, weight_packed, scales)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print(f"  Raw w4a16_gemv: {np.mean(times):.4f} ms (std: {np.std(times):.4f})")
    print(f"  Target: 0.125 ms")


def benchmark_w4a16linear():
    """Measure W4A16Linear wrapper overhead."""
    print("\n" + "=" * 60)
    print("3. W4A16LINEAR WRAPPER OVERHEAD")
    print("=" * 60)

    from openpi.modules.w4a16_linear import W4A16Linear

    device = 'cuda'
    K, N = 2048, 16384

    # Create W4A16Linear
    linear = torch.nn.Linear(K, N, bias=False).to(device).to(torch.float16)
    w4a16 = W4A16Linear.from_linear(linear)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Test with float16 input (no conversion)
    x_fp16 = torch.randn(1, 1, K, dtype=torch.float16, device=device)

    # Warmup
    for _ in range(50):
        _ = w4a16(x_fp16)
    torch.cuda.synchronize()

    times = []
    for _ in range(200):
        start.record()
        _ = w4a16(x_fp16)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    print(f"  W4A16Linear (fp16 input): {np.mean(times):.4f} ms")

    # Test with bfloat16 input (WITH conversion)
    x_bf16 = torch.randn(1, 1, K, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(50):
        _ = w4a16(x_bf16)
    torch.cuda.synchronize()

    times = []
    for _ in range(200):
        start.record()
        _ = w4a16(x_bf16)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    print(f"  W4A16Linear (bf16 input): {np.mean(times):.4f} ms")
    print(f"  Overhead from dtype conv: {np.mean(times) - 0.125:.4f} ms")


def benchmark_flinear_baseline():
    """Measure F.linear baseline for comparison."""
    print("\n" + "=" * 60)
    print("4. F.LINEAR BASELINE")
    print("=" * 60)

    import torch.nn.functional as F

    device = 'cuda'
    K, N = 2048, 16384

    weight = torch.randn(N, K, dtype=torch.float16, device=device)
    x = torch.randn(1, 1, K, dtype=torch.float16, device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(50):
        _ = F.linear(x, weight)
    torch.cuda.synchronize()

    times = []
    for _ in range(200):
        start.record()
        _ = F.linear(x, weight)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print(f"  F.linear (fp16): {np.mean(times):.4f} ms")


if __name__ == "__main__":
    benchmark_dtype_conversion()
    benchmark_raw_tvm_kernel()
    benchmark_w4a16linear()
    benchmark_flinear_baseline()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
If W4A16Linear (bf16) >> Raw TVM kernel:
  -> dtype conversion overhead is the culprit
  -> Solution: Keep all tensors in fp16 OR rewrite kernel to accept bf16

If Raw TVM kernel >> 0.125ms:
  -> TVM kernel is not running optimally
  -> Check if JIT compilation is happening every call
""")
