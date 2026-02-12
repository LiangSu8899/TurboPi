#!/usr/bin/env python3
"""
NVFP4 GEMV Kernel 性能对比测试

对比不同优化版本的 kernel 性能，针对 bs=1 场景。
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path

# 编译 CUDA extensions
def load_cuda_extensions():
    from torch.utils.cpp_extension import load

    plugin_dir = Path(__file__).parent.parent / 'nvfp4_packed_plugin' / 'src'

    # 原始 GEMV
    gemv_orig = load(
        name='nvfp4_gemv_orig',
        sources=[str(plugin_dir / 'nvfp4_packed_torch.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
        verbose=True
    )

    # 优化版本
    gemv_opt = load(
        name='nvfp4_gemv_opt',
        sources=[str(plugin_dir / 'nvfp4_gemv_optimized.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
        verbose=True
    )

    return gemv_orig, gemv_opt


def quantize_to_nvfp4(tensor: torch.Tensor, block_size: int = 32):
    """量化到 NVFP4 格式"""
    NVFP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=tensor.device)
    NVFP4_MAX = 6.0

    original_shape = tensor.shape
    tensor = tensor.view(-1)

    # Pad to block_size
    pad_size = (block_size - len(tensor) % block_size) % block_size
    if pad_size > 0:
        tensor = F.pad(tensor, (0, pad_size))

    # Reshape to blocks
    tensor = tensor.view(-1, block_size)

    # Per-block scaling
    abs_max = tensor.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scales = abs_max / NVFP4_MAX
    tensor_scaled = tensor / scales

    # Quantize
    tensor_abs = tensor_scaled.abs()
    signs = (tensor_scaled < 0).to(torch.int32)

    # Find nearest NVFP4 value
    distances = (tensor_abs.unsqueeze(-1) - NVFP4_VALUES.unsqueeze(0).unsqueeze(0)).abs()
    indices = distances.argmin(dim=-1)
    indices = indices + signs * 8  # Add sign bit

    # Pack to uint8 (2 values per byte)
    indices = indices.to(torch.uint8)
    indices_reshaped = indices.view(-1, block_size // 2, 2)
    packed = indices_reshaped[:, :, 0] | (indices_reshaped[:, :, 1] << 4)
    packed = packed.view(-1)

    # Remove padding from scales
    num_original_elements = np.prod(original_shape)
    num_blocks_needed = (num_original_elements + block_size - 1) // block_size
    scales = scales[:num_blocks_needed].view(-1)

    return packed, scales


def benchmark_kernel(func, *args, warmup=20, runs=100, name="kernel"):
    """Benchmark a kernel function"""
    # Check if func is callable without args (lambda case)
    if len(args) == 0:
        # Lambda function case
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
        # Normal case with args
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
    print("=" * 70)
    print("NVFP4 GEMV Kernel Benchmark")
    print("=" * 70)

    device = torch.device('cuda')

    # 加载 CUDA extensions
    print("\nCompiling CUDA extensions...")
    gemv_orig, gemv_opt = load_cuda_extensions()
    print("Done!\n")

    # 测试配置 - Pi0 模型的典型维度
    test_configs = [
        # (M, N, K) - 实际推理场景
        (1, 2048, 2048),      # Small layer, bs=1
        (1, 8192, 2048),      # Medium layer, bs=1
        (1, 16384, 2048),     # Large layer (typical MLP), bs=1
        (1, 2048, 8192),      # Up projection
        (50, 2048, 2048),     # Denoising steps
        (455, 2048, 2048),    # Prefix pass
    ]

    print("Testing configurations:")
    print("-" * 70)
    print(f"{'M':>6} {'N':>8} {'K':>6} | {'Orig':>10} {'V2':>10} {'V3':>10} {'V4':>10} | {'Best':>8} {'Speedup':>8}")
    print("-" * 70)

    for M, N, K in test_configs:
        # 准备数据
        activation = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)

        # 量化权重
        weight_packed, weight_scales = quantize_to_nvfp4(weight)
        weight_packed = weight_packed.view(N, K // 2)
        weight_scales = weight_scales.view(N, -1)

        # 测试原始版本
        try:
            orig_ms, _ = benchmark_kernel(
                gemv_orig.nvfp4_gemv_w4a16,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
        except Exception as e:
            orig_ms = float('inf')

        # 测试 V2
        try:
            v2_ms, _ = benchmark_kernel(
                gemv_opt.nvfp4_gemv_v2,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
        except Exception as e:
            v2_ms = float('inf')

        # 测试 V3
        try:
            v3_ms, _ = benchmark_kernel(
                gemv_opt.nvfp4_gemv_v3,
                activation, weight_packed, weight_scales, None,
                M, N, K, 0
            )
        except Exception as e:
            v3_ms = float('inf')

        # 测试 V4
        try:
            if K % 8 == 0:
                v4_ms, _ = benchmark_kernel(
                    gemv_opt.nvfp4_gemv_v4,
                    activation, weight_packed, weight_scales, None,
                    M, N, K, 0
                )
            else:
                v4_ms = float('inf')
        except Exception as e:
            v4_ms = float('inf')

        # 找最佳版本
        times = {'Orig': orig_ms, 'V2': v2_ms, 'V3': v3_ms, 'V4': v4_ms}
        best_name = min(times, key=times.get)
        best_time = times[best_name]
        speedup = orig_ms / best_time if best_time < float('inf') else 0

        print(f"{M:>6} {N:>8} {K:>6} | {orig_ms:>9.3f}ms {v2_ms:>9.3f}ms {v3_ms:>9.3f}ms {v4_ms:>9.3f}ms | {best_name:>8} {speedup:>7.2f}x")

    # BF16 cuBLAS 对比
    print("\n" + "=" * 70)
    print("Comparison with cuBLAS BF16")
    print("=" * 70)
    print(f"{'M':>6} {'N':>8} {'K':>6} | {'NVFP4 Best':>12} {'cuBLAS BF16':>12} | {'Ratio':>10}")
    print("-" * 70)

    for M, N, K in test_configs:
        # NVFP4 最佳时间
        activation = torch.randn(M, K, device=device, dtype=torch.float32)
        weight = torch.randn(N, K, device=device, dtype=torch.float32)
        weight_packed, weight_scales = quantize_to_nvfp4(weight)
        weight_packed = weight_packed.view(N, K // 2)
        weight_scales = weight_scales.view(N, -1)

        times = []
        for func in [gemv_opt.nvfp4_gemv_v2, gemv_opt.nvfp4_gemv_v3, gemv_opt.nvfp4_gemv_v4]:
            try:
                if func == gemv_opt.nvfp4_gemv_v4 and K % 8 != 0:
                    continue
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
            warmup=20, runs=100
        )

        ratio = nvfp4_best / cublas_ms if cublas_ms > 0 else float('inf')

        print(f"{M:>6} {N:>8} {K:>6} | {nvfp4_best:>11.3f}ms {cublas_ms:>11.3f}ms | {ratio:>9.2f}x")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
