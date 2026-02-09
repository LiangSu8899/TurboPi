#!/usr/bin/env python3
"""
Benchmark NVFuser NVFP4 GEMM 在 Thor SM110 上的性能

使用 nvfuser_direct.nvf_cutlass.nvfp4_scaled_mm
"""

import sys
import os

# 添加 PyTorch fuser 测试路径
sys.path.insert(0, '/opt/pytorch/fuser/tests')

import torch
import time


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# 复制必要的工具函数 (来自 python/direct_utils.py)
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def pytorch_nvfp4_quantize(x, global_scale, block_size=16):
    """量化到 NVFP4 (e2m1) 格式"""
    shape = x.shape
    k = shape[-1]

    # 确保 K 是 block_size 的倍数
    padded_k = round_up(k, block_size)
    if padded_k != k:
        x = torch.nn.functional.pad(x, (0, padded_k - k))

    # Reshape for block-wise quantization
    x_reshaped = x.reshape(-1, padded_k // block_size, block_size)

    # 计算每个 block 的 amax
    block_amax = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True)
    block_amax = torch.clamp(block_amax, min=1e-12)

    # 计算 scale factor (FP8 格式)
    sf = (block_amax / FLOAT4_E2M1_MAX).to(torch.float8_e4m3fn)
    sf_float = sf.to(torch.float32)

    # 量化到 FP4 范围
    x_scaled = x_reshaped * global_scale / sf_float
    x_clipped = torch.clamp(x_scaled, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX)

    # Round to nearest FP4 value
    # e2m1: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (正值)
    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device, dtype=x.dtype)
    fp4_values = torch.cat([-fp4_values.flip(0)[:-1], fp4_values])  # 包含负值

    # 找最近的 FP4 值
    x_flat = x_clipped.reshape(-1, 1)
    distances = torch.abs(x_flat - fp4_values)
    indices = torch.argmin(distances, dim=1)
    x_quantized = fp4_values[indices].reshape(x_clipped.shape)

    # Pack to int8 (每个 byte 存 2 个 FP4)
    x_int = (indices.reshape(-1, 2)[:, 0] | (indices.reshape(-1, 2)[:, 1] << 4)).to(torch.uint8)
    x_packed = x_int.reshape(shape[0], padded_k // 2)

    # Scale factor reshape
    sf_reshaped = sf.reshape(shape[0], -1)

    return x_packed, sf_reshaped


def linear_to_swizzled_128_4(scales):
    """Convert linear scale layout to swizzled 128x4 layout for CUTLASS"""
    # 简化版本 - 可能需要根据实际 CUTLASS 要求调整
    return scales.contiguous()


def dequantize_to_dtype(x_fp4, sf, global_scale, dtype, device, block_size=16):
    """从 FP4 解量化回原始类型"""
    # 这是简化版本，用于验证
    pass


def benchmark_cublas_baseline():
    """cuBLAS BF16 baseline"""
    print_header("cuBLAS BF16 Baseline")

    M, K, N = 712, 2048, 16384

    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)

    # Warmup
    for _ in range(20):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 100 * 1000

    flops = 2 * M * K * N
    tflops = flops / (elapsed / 1000) / 1e12

    print(f"Shape: ({M}, {K}) @ ({K}, {N})")
    print(f"Latency: {elapsed:.3f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")

    return elapsed


def benchmark_nvfuser_nvfp4():
    """Benchmark NVFuser NVFP4 GEMM"""
    print_header("NVFuser NVFP4 GEMM Benchmark")

    try:
        from nvfuser_direct import nvf_cutlass
        print("✅ nvf_cutlass 模块加载成功")

        # 检查 GPU compute capability
        props = torch.cuda.get_device_properties(0)
        cc = (props.major, props.minor)
        print(f"GPU Compute Capability: {cc}")

        if cc < (10, 0) or cc >= (12, 0):
            print(f"⚠️ NVFP4 需要 compute capability 10.x, 当前是 {cc}")
            # Thor 是 SM 11.0，应该可以

        M, K, N = 712, 2048, 16384

        # 创建输入 (FP16)
        a_dtype = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b_dtype = torch.randn(N, K, device='cuda', dtype=torch.float16)

        # 计算 global scale
        a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten())
        b_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten())
        alpha = (1.0 / (a_global_scale * b_global_scale)).to(torch.float32)

        print(f"a_global_scale: {a_global_scale.item():.4f}")
        print(f"b_global_scale: {b_global_scale.item():.4f}")
        print(f"alpha: {alpha.item():.6f}")

        # 使用原始测试中的工具函数
        try:
            from python.direct_utils import (
                pytorch_nvfp4_quantize as orig_quantize,
                linear_to_swizzled_128_4 as orig_swizzle,
            )
            print("✅ 使用原始 direct_utils")
            quantize_fn = orig_quantize
            swizzle_fn = orig_swizzle
        except ImportError:
            print("⚠️ 使用简化的量化函数")
            quantize_fn = pytorch_nvfp4_quantize
            swizzle_fn = linear_to_swizzled_128_4

        # FP4 量化
        a_fp4, a_scale_linear = quantize_fn(a_dtype, a_global_scale)
        b_fp4, b_scale_linear = quantize_fn(b_dtype, b_global_scale)

        print(f"a_fp4 shape: {a_fp4.shape}, dtype: {a_fp4.dtype}")
        print(f"b_fp4 shape: {b_fp4.shape}, dtype: {b_fp4.dtype}")
        print(f"a_scale shape: {a_scale_linear.shape}")

        # Swizzle scales
        a_scale = swizzle_fn(a_scale_linear)
        b_scale = swizzle_fn(b_scale_linear)

        print(f"Swizzled a_scale shape: {a_scale.shape}")

        # 运行 NVFP4 GEMM
        print("\n正在运行 NVFP4 GEMM...")

        try:
            out = nvf_cutlass.nvfp4_scaled_mm(
                a_fp4, b_fp4, a_scale, b_scale, alpha, torch.float16
            )
            print(f"✅ NVFP4 GEMM 成功!")
            print(f"   Output shape: {out.shape}, dtype: {out.dtype}")

            # Warmup
            for _ in range(20):
                out = nvf_cutlass.nvfp4_scaled_mm(
                    a_fp4, b_fp4, a_scale, b_scale, alpha, torch.float16
                )
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                out = nvf_cutlass.nvfp4_scaled_mm(
                    a_fp4, b_fp4, a_scale, b_scale, alpha, torch.float16
                )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 100 * 1000

            flops = 2 * M * K * N
            tflops = flops / (elapsed / 1000) / 1e12

            print(f"\nNVFP4 GEMM 性能:")
            print(f"  Latency: {elapsed:.3f} ms")
            print(f"  Throughput: {tflops:.2f} TFLOPS")

            return elapsed

        except RuntimeError as e:
            print(f"❌ NVFP4 GEMM 运行失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    except Exception as e:
        print(f"❌ NVFuser NVFP4 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("Thor SM110 NVFuser NVFP4 GEMM Benchmark")
    print("=" * 60)

    # GPU 信息
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Compute Capability: SM {props.major}.{props.minor}")
    print(f"Memory: {props.total_memory / 1024**3:.1f} GB")

    # 1. Baseline
    baseline_ms = benchmark_cublas_baseline()

    # 2. NVFP4 benchmark
    nvfp4_ms = benchmark_nvfuser_nvfp4()

    # 3. Summary
    print_header("总结")

    print(f"cuBLAS BF16: {baseline_ms:.3f} ms")

    if nvfp4_ms:
        speedup = baseline_ms / nvfp4_ms
        print(f"NVFP4 GEMM: {nvfp4_ms:.3f} ms")
        print(f"加速比: {speedup:.2f}x")

        # 估算完整模型
        print(f"\n完整 MLP 估算 (3 GEMM × 18 层):")
        print(f"  BF16:  {baseline_ms * 3 * 18:.1f} ms")
        print(f"  NVFP4: {nvfp4_ms * 3 * 18:.1f} ms")
        print(f"  节省:  {(baseline_ms - nvfp4_ms) * 3 * 18:.1f} ms")

        # 预期频率
        current_total = 173  # ms (当前总延迟)
        mlp_time = baseline_ms * 3 * 18
        other_time = current_total - mlp_time
        new_total = nvfp4_ms * 3 * 18 + other_time

        print(f"\n推理频率估算:")
        print(f"  当前 (BF16): {1000/current_total:.1f} Hz")
        print(f"  NVFP4 后: {1000/new_total:.1f} Hz")
    else:
        print("❌ NVFP4 GEMM benchmark 失败")


if __name__ == "__main__":
    main()
