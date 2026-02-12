#!/usr/bin/env python3
"""
NVFP4 GEMV 全版本性能对比
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path


def load_cuda_extensions():
    from torch.utils.cpp_extension import load

    plugin_dir = Path(__file__).parent.parent / 'nvfp4_packed_plugin' / 'src'

    # 原始版本
    gemv_orig = load(
        name='nvfp4_orig',
        sources=[str(plugin_dir / 'nvfp4_packed_torch.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=True
    )

    # V5-V8
    gemv_v5 = load(
        name='nvfp4_v5',
        sources=[str(plugin_dir / 'nvfp4_gemv_v5.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=True
    )

    # V9-V11
    gemv_v9 = load(
        name='nvfp4_v9',
        sources=[str(plugin_dir / 'nvfp4_gemv_v9.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=True
    )

    return gemv_orig, gemv_v5, gemv_v9


def quantize_to_nvfp4(tensor: torch.Tensor, block_size: int = 32):
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


def benchmark_kernel(func, *args, warmup=50, runs=200):
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
    print("=" * 100)
    print("NVFP4 GEMV All Versions Benchmark")
    print("=" * 100)

    device = torch.device('cuda')

    print("\nCompiling CUDA extensions...")
    gemv_orig, gemv_v5, gemv_v9 = load_cuda_extensions()
    print("Done!\n")

    # Pi0 MLP典型维度 - 只测试 K=2048
    test_configs = [
        (1, 2048, 2048),
        (1, 8192, 2048),
        (1, 16384, 2048),
    ]

    print("Testing M=1 configurations (bs=1):")
    print("-" * 100)
    print(f"{'Config':>18} | {'Orig':>8} {'V6':>8} {'V9':>8} {'V10':>8} {'V11':>8} | {'Best':>6} {'Speed':>7} {'BW':>10}")
    print("-" * 100)

    for M, N, K in test_configs:
        activation = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)

        weight_packed, weight_scales = quantize_to_nvfp4(weight)
        weight_packed = weight_packed.view(N, K // 2)
        weight_scales = weight_scales.view(N, -1)

        results = {}

        # 原始版本
        try:
            ms, _ = benchmark_kernel(
                gemv_orig.nvfp4_gemv_w4a16,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
            results['Orig'] = ms
        except:
            results['Orig'] = float('inf')

        # V6
        try:
            ms, _ = benchmark_kernel(
                gemv_v5.nvfp4_gemv_v6,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
            results['V6'] = ms
        except:
            results['V6'] = float('inf')

        # V9
        try:
            ms, _ = benchmark_kernel(
                gemv_v9.nvfp4_gemv_v9,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
            results['V9'] = ms
        except Exception as e:
            results['V9'] = float('inf')
            print(f"V9 error: {e}")

        # V10
        try:
            ms, _ = benchmark_kernel(
                gemv_v9.nvfp4_gemv_v10,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
            results['V10'] = ms
        except Exception as e:
            results['V10'] = float('inf')

        # V11
        try:
            ms, _ = benchmark_kernel(
                gemv_v9.nvfp4_gemv_v11,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
            results['V11'] = ms
        except Exception as e:
            results['V11'] = float('inf')

        best_name = min(results, key=results.get)
        best_time = results[best_name]
        speedup = results['Orig'] / best_time if best_time > 0 else 0

        # 计算带宽
        data_bytes = N * K / 2 + N * (K // 32) * 4 + M * K * 4
        bandwidth = data_bytes / (best_time / 1000) / 1e9 if best_time > 0 else 0

        config_str = f"M={M},N={N},K={K}"
        print(f"{config_str:>18} | {results['Orig']:>7.3f}ms {results['V6']:>7.3f}ms {results['V9']:>7.3f}ms {results['V10']:>7.3f}ms {results['V11']:>7.3f}ms | {best_name:>6} {speedup:>6.2f}x {bandwidth:>9.1f}")

    # 与 cuBLAS 对比
    print("\n" + "=" * 100)
    print("Best NVFP4 vs cuBLAS BF16")
    print("=" * 100)
    print(f"{'Config':>18} | {'NVFP4 Best':>12} {'cuBLAS':>12} | {'Ratio':>8} {'Notes':>20}")
    print("-" * 100)

    for M, N, K in test_configs:
        activation = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)
        weight_packed, weight_scales = quantize_to_nvfp4(weight)
        weight_packed = weight_packed.view(N, K // 2)
        weight_scales = weight_scales.view(N, -1)

        # 找最佳 NVFP4
        times = []
        for func in [gemv_v5.nvfp4_gemv_v6, gemv_v9.nvfp4_gemv_v9, gemv_v9.nvfp4_gemv_v10, gemv_v9.nvfp4_gemv_v11]:
            try:
                ms, _ = benchmark_kernel(func, activation, weight_packed, weight_scales, None, M, N, K, 0)
                times.append(ms)
            except:
                pass

        nvfp4_best = min(times) if times else float('inf')

        # cuBLAS BF16
        activation_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        weight_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        cublas_ms, _ = benchmark_kernel(lambda: F.linear(activation_bf16, weight_bf16))

        ratio = nvfp4_best / cublas_ms if cublas_ms > 0 else float('inf')

        # 理论分析
        nvfp4_data = N * K / 2 + N * (K // 32) * 4  # FP4权重 + scales
        bf16_data = N * K * 2  # BF16权重
        data_ratio = bf16_data / nvfp4_data

        config_str = f"M={M},N={N},K={K}"
        notes = f"data {data_ratio:.1f}x smaller"
        print(f"{config_str:>18} | {nvfp4_best:>11.3f}ms {cublas_ms:>11.3f}ms | {ratio:>7.2f}x {notes:>20}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
