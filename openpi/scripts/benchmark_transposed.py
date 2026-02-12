#!/usr/bin/env python3
"""
测试转置权重布局的性能
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
        name='nvfp4_orig_t',
        sources=[str(plugin_dir / 'nvfp4_packed_torch.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=True
    )

    # V6
    gemv_v5 = load(
        name='nvfp4_v5_t',
        sources=[str(plugin_dir / 'nvfp4_gemv_v5.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=True
    )

    # 转置版本
    gemv_T = load(
        name='nvfp4_T',
        sources=[str(plugin_dir / 'nvfp4_gemv_transposed.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=True
    )

    return gemv_orig, gemv_v5, gemv_T


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
    print("NVFP4 GEMV Transposed Layout Benchmark")
    print("=" * 100)

    device = torch.device('cuda')

    print("\nCompiling CUDA extensions...")
    gemv_orig, gemv_v5, gemv_T = load_cuda_extensions()
    print("Done!\n")

    test_configs = [
        (1, 2048, 2048),
        (1, 8192, 2048),
        (1, 16384, 2048),
    ]

    print("Testing M=1 configurations:")
    print("-" * 100)
    print(f"{'Config':>18} | {'Orig':>8} {'V6':>8} {'Trans':>8} | {'Best':>6} {'vs Orig':>8} {'BW (GB/s)':>10}")
    print("-" * 100)

    for M, N, K in test_configs:
        activation = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)

        # 量化权重
        weight_packed, weight_scales = quantize_to_nvfp4(weight)
        weight_packed = weight_packed.view(N, K // 2)
        weight_scales = weight_scales.view(N, -1)

        # 转置权重和scales
        weight_T = gemv_T.transpose_weight_packed(weight_packed, N, K)
        scale_T = weight_scales.t().contiguous()

        results = {}

        # 原始版本
        try:
            ms, _ = benchmark_kernel(
                gemv_orig.nvfp4_gemv_w4a16,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
            results['Orig'] = ms
        except Exception as e:
            results['Orig'] = float('inf')
            print(f"Orig error: {e}")

        # V6
        try:
            ms, _ = benchmark_kernel(
                gemv_v5.nvfp4_gemv_v6,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
            results['V6'] = ms
        except Exception as e:
            results['V6'] = float('inf')
            print(f"V6 error: {e}")

        # 转置版本
        try:
            ms, _ = benchmark_kernel(
                gemv_T.nvfp4_gemv_transposed,
                activation, weight_T, scale_T, None,
                M, N, K, 0
            )
            results['Trans'] = ms
        except Exception as e:
            results['Trans'] = float('inf')
            print(f"Trans error: {e}")

        best_name = min(results, key=results.get)
        best_time = results[best_name]
        speedup = results['Orig'] / best_time if best_time > 0 else 0

        # 计算带宽
        data_bytes = N * K / 2 + N * (K // 32) * 4 + M * K * 4
        bandwidth = data_bytes / (best_time / 1000) / 1e9 if best_time > 0 else 0

        config_str = f"M={M},N={N},K={K}"
        print(f"{config_str:>18} | {results['Orig']:>7.3f}ms {results['V6']:>7.3f}ms {results['Trans']:>7.3f}ms | {best_name:>6} {speedup:>7.2f}x {bandwidth:>9.1f}")

    # 验证正确性
    print("\n" + "=" * 100)
    print("Correctness Verification")
    print("=" * 100)

    M, N, K = 1, 2048, 2048
    activation = torch.randn(M, K, device=device, dtype=torch.float32)
    weight = torch.randn(N, K, device=device, dtype=torch.float32)

    weight_packed, weight_scales = quantize_to_nvfp4(weight)
    weight_packed = weight_packed.view(N, K // 2)
    weight_scales = weight_scales.view(N, -1)

    weight_T = gemv_T.transpose_weight_packed(weight_packed, N, K)
    scale_T = weight_scales.t().contiguous()

    # 计算结果
    out_orig = gemv_orig.nvfp4_gemv_w4a16(activation, weight_packed, weight_scales, None, M, N, K, 0)
    out_v6 = gemv_v5.nvfp4_gemv_v6(activation, weight_packed, weight_scales, None, M, N, K, 0)
    out_T = gemv_T.nvfp4_gemv_transposed(activation, weight_T, scale_T, None, M, N, K, 0)

    # 比较
    diff_v6 = (out_orig - out_v6).abs().max().item()
    diff_T = (out_orig - out_T).abs().max().item()
    cos_v6 = F.cosine_similarity(out_orig.flatten().unsqueeze(0), out_v6.flatten().unsqueeze(0)).item()
    cos_T = F.cosine_similarity(out_orig.flatten().unsqueeze(0), out_T.flatten().unsqueeze(0)).item()

    print(f"V6 vs Orig:    max_diff={diff_v6:.6f}, cos_sim={cos_v6:.6f}")
    print(f"Trans vs Orig: max_diff={diff_T:.6f}, cos_sim={cos_T:.6f}")

    if cos_T > 0.999:
        print("\n[PASS] Transposed version is correct!")
    else:
        print("\n[WARN] Transposed version may have issues")

    # vs cuBLAS
    print("\n" + "=" * 100)
    print("Best NVFP4 vs cuBLAS BF16")
    print("=" * 100)

    for M, N, K in test_configs:
        activation = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)

        weight_packed, weight_scales = quantize_to_nvfp4(weight)
        weight_packed = weight_packed.view(N, K // 2)
        weight_scales = weight_scales.view(N, -1)

        weight_T = gemv_T.transpose_weight_packed(weight_packed, N, K)
        scale_T = weight_scales.t().contiguous()

        # Best NVFP4
        nvfp4_best = min(
            benchmark_kernel(gemv_v5.nvfp4_gemv_v6, activation, weight_packed, weight_scales, None, M, N, K, 0)[0],
            benchmark_kernel(gemv_T.nvfp4_gemv_transposed, activation, weight_T, scale_T, None, M, N, K, 0)[0]
        )

        # cuBLAS BF16
        activation_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        weight_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        cublas_ms, _ = benchmark_kernel(lambda: F.linear(activation_bf16, weight_bf16))

        ratio = nvfp4_best / cublas_ms if cublas_ms > 0 else float('inf')

        config_str = f"M={M},N={N},K={K}"
        print(f"{config_str:>18} | NVFP4={nvfp4_best:.3f}ms, cuBLAS={cublas_ms:.3f}ms, ratio={ratio:.2f}x")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
