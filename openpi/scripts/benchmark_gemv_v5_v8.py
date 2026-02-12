#!/usr/bin/env python3
"""
NVFP4 GEMV V5-V8 Kernel Benchmark

测试新优化版本 vs 原版本 vs cuBLAS BF16
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path


def load_cuda_extensions():
    from torch.utils.cpp_extension import load

    plugin_dir = Path(__file__).parent.parent / 'nvfp4_packed_plugin' / 'src'

    # 原始 GEMV
    gemv_orig = load(
        name='nvfp4_gemv_orig',
        sources=[str(plugin_dir / 'nvfp4_packed_torch.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=True
    )

    # V5-V8 优化版本
    gemv_v5 = load(
        name='nvfp4_gemv_v5_ext',
        sources=[str(plugin_dir / 'nvfp4_gemv_v5.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=True
    )

    return gemv_orig, gemv_v5


def quantize_to_nvfp4(tensor: torch.Tensor, block_size: int = 32):
    """量化到 NVFP4 格式"""
    NVFP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=tensor.device)
    NVFP4_MAX = 6.0

    original_shape = tensor.shape
    tensor = tensor.view(-1)

    pad_size = (block_size - len(tensor) % block_size) % block_size
    if pad_size > 0:
        tensor = F.pad(tensor, (0, pad_size))

    tensor = tensor.view(-1, block_size)
    abs_max = tensor.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scales = abs_max / NVFP4_MAX
    tensor_scaled = tensor / scales

    tensor_abs = tensor_scaled.abs()
    signs = (tensor_scaled < 0).to(torch.int32)

    distances = (tensor_abs.unsqueeze(-1) - NVFP4_VALUES.unsqueeze(0).unsqueeze(0)).abs()
    indices = distances.argmin(dim=-1)
    indices = indices + signs * 8

    indices = indices.to(torch.uint8)
    indices_reshaped = indices.view(-1, block_size // 2, 2)
    packed = indices_reshaped[:, :, 0] | (indices_reshaped[:, :, 1] << 4)
    packed = packed.view(-1)

    num_original_elements = np.prod(original_shape)
    num_blocks_needed = (num_original_elements + block_size - 1) // block_size
    scales = scales[:num_blocks_needed].view(-1)

    return packed, scales


def benchmark_kernel(func, *args, warmup=20, runs=100):
    """Benchmark a kernel function"""
    if len(args) == 0:
        for _ in range(warmup):
            out = func()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            out = func()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / runs
        return elapsed, out
    else:
        for _ in range(warmup):
            out = func(*args)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            out = func(*args)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / runs
        return elapsed, out


def main():
    print("=" * 80)
    print("NVFP4 GEMV V5-V8 Kernel Benchmark")
    print("=" * 80)

    device = torch.device('cuda')

    print("\nCompiling CUDA extensions...")
    gemv_orig, gemv_v5 = load_cuda_extensions()
    print("Done!\n")

    # Pi0 MLP 典型维度
    test_configs = [
        (1, 2048, 2048),
        (1, 8192, 2048),
        (1, 16384, 2048),
        (1, 2048, 8192),
        (50, 2048, 2048),
    ]

    print("Testing configurations:")
    print("-" * 80)
    print(f"{'M':>4} {'N':>6} {'K':>5} | {'Orig':>8} {'V5':>8} {'V6':>8} {'V7':>8} {'V8':>8} | {'Best':>6} {'vs Orig':>8}")
    print("-" * 80)

    for M, N, K in test_configs:
        activation = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)

        weight_packed, weight_scales = quantize_to_nvfp4(weight)
        weight_packed = weight_packed.view(N, K // 2)
        weight_scales = weight_scales.view(N, -1)

        # 原始版本
        try:
            orig_ms, _ = benchmark_kernel(
                gemv_orig.nvfp4_gemv_w4a16,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
        except Exception as e:
            orig_ms = float('inf')
            print(f"Orig error: {e}")

        # V5
        try:
            v5_ms, _ = benchmark_kernel(
                gemv_v5.nvfp4_gemv_v5,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
        except Exception as e:
            v5_ms = float('inf')
            print(f"V5 error: {e}")

        # V6
        try:
            v6_ms, _ = benchmark_kernel(
                gemv_v5.nvfp4_gemv_v6,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
        except Exception as e:
            v6_ms = float('inf')
            print(f"V6 error: {e}")

        # V7
        try:
            v7_ms, _ = benchmark_kernel(
                gemv_v5.nvfp4_gemv_v7,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
        except Exception as e:
            v7_ms = float('inf')
            print(f"V7 error: {e}")

        # V8
        try:
            if K % 8 == 0:
                v8_ms, _ = benchmark_kernel(
                    gemv_v5.nvfp4_gemv_v8,
                    activation, weight_packed, weight_scales, None,
                    M, N, K, 0
                )
            else:
                v8_ms = float('inf')
        except Exception as e:
            v8_ms = float('inf')
            print(f"V8 error: {e}")

        times = {'Orig': orig_ms, 'V5': v5_ms, 'V6': v6_ms, 'V7': v7_ms, 'V8': v8_ms}
        best_name = min(times, key=times.get)
        best_time = times[best_name]
        speedup = orig_ms / best_time if best_time > 0 else 0

        print(f"{M:>4} {N:>6} {K:>5} | {orig_ms:>7.3f}ms {v5_ms:>7.3f}ms {v6_ms:>7.3f}ms {v7_ms:>7.3f}ms {v8_ms:>7.3f}ms | {best_name:>6} {speedup:>7.2f}x")

    # 与 cuBLAS BF16 对比
    print("\n" + "=" * 80)
    print("Comparison with cuBLAS BF16")
    print("=" * 80)
    print(f"{'M':>4} {'N':>6} {'K':>5} | {'NVFP4 Best':>12} {'cuBLAS BF16':>12} | {'Ratio':>8} {'BW (GB/s)':>10}")
    print("-" * 80)

    for M, N, K in test_configs:
        activation = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)
        weight_packed, weight_scales = quantize_to_nvfp4(weight)
        weight_packed = weight_packed.view(N, K // 2)
        weight_scales = weight_scales.view(N, -1)

        # 找最佳 NVFP4 版本
        times = []
        for name, func in [
            ('v5', gemv_v5.nvfp4_gemv_v5),
            ('v6', gemv_v5.nvfp4_gemv_v6),
            ('v7', gemv_v5.nvfp4_gemv_v7),
            ('v8', gemv_v5.nvfp4_gemv_v8 if K % 8 == 0 else None),
        ]:
            if func is None:
                continue
            try:
                ms, _ = benchmark_kernel(func, activation, weight_packed, weight_scales, None, M, N, K, 0)
                times.append(ms)
            except:
                pass

        nvfp4_best = min(times) if times else float('inf')

        # cuBLAS BF16
        activation_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        weight_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)

        cublas_ms, _ = benchmark_kernel(
            lambda: F.linear(activation_bf16, weight_bf16),
        )

        ratio = nvfp4_best / cublas_ms if cublas_ms > 0 else float('inf')

        # 计算带宽 (NVFP4 kernel)
        # 数据量: 权重 N*K/2 bytes + scales N*(K/32)*4 bytes + 激活 M*K*4 bytes
        data_bytes = N * K / 2 + N * (K // 32) * 4 + M * K * 4
        bandwidth = data_bytes / (nvfp4_best / 1000) / 1e9 if nvfp4_best > 0 else 0

        print(f"{M:>4} {N:>6} {K:>5} | {nvfp4_best:>11.3f}ms {cublas_ms:>11.3f}ms | {ratio:>7.2f}x {bandwidth:>9.1f}")

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
