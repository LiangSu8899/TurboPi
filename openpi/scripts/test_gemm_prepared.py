#!/usr/bin/env python3
"""
测试 nvfp4_gemm.gemm_prepared() 函数

验证:
1. Python 预处理的 FP8 scales 是否正确
2. CUTLASS GEMM 输出精度
3. 与 BF16 参考结果对比
"""

import torch
import torch.nn.functional as F
import sys
import time

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    swizzle_scales_for_cutlass,
    convert_scales_to_fp8,
    prepare_scales_for_cutlass,
    pack_nvfp4_data,
    BLOCK_SIZE,
)


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_section(title):
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


def test_gemm_prepared():
    """测试 gemm_prepared 函数"""
    print_section("Testing gemm_prepared()")

    try:
        import nvfp4_gemm
        print(f"Module loaded: {nvfp4_gemm}")
        print(f"Available functions: {[f for f in dir(nvfp4_gemm) if not f.startswith('_')]}")

        if not hasattr(nvfp4_gemm, 'gemm_prepared'):
            print("ERROR: gemm_prepared not found! Need to rebuild extension.")
            return False

    except ImportError as e:
        print(f"Failed to import: {e}")
        return False

    # 测试维度
    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE

    print(f"\nTest dimensions: M={M}, K={K}, N={N}, block_size={block_size}")

    # 创建测试数据
    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)

    # BF16 参考结果
    ref = torch.matmul(x.float(), w.float().T)
    print(f"Reference shape: {ref.shape}")
    print(f"Reference stats: mean={ref.mean().item():.4f}, std={ref.std().item():.4f}")

    # 使用 Python 量化
    print("\nQuantizing with Python simulation...")
    x_q, x_scales = quantize_to_nvfp4_sim(x.float(), block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w.float(), block_size)

    print(f"x_q: {x_q.shape}, x_scales: {x_scales.shape}")
    print(f"w_q: {w_q.shape}, w_scales: {w_scales.shape}")

    # 打包 NVFP4 数据
    print("\nPacking NVFP4 data...")
    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    print(f"x_packed: {x_packed.shape}, dtype={x_packed.dtype}")
    print(f"w_packed: {w_packed.shape}, dtype={w_packed.dtype}")

    # 准备 scale factors (FP8 + CUTLASS layout)
    print("\nPreparing scale factors for CUTLASS...")
    num_k_blocks = K // block_size

    x_scales_fp8 = prepare_scales_for_cutlass(
        x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True
    )
    w_scales_fp8 = prepare_scales_for_cutlass(
        w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True
    )

    print(f"x_scales_fp8: {x_scales_fp8.shape}, dtype={x_scales_fp8.dtype}")
    print(f"w_scales_fp8: {w_scales_fp8.shape}, dtype={w_scales_fp8.dtype}")

    # 确保所有数据在 CUDA 上
    x_packed = x_packed.cuda().contiguous()
    w_packed = w_packed.cuda().contiguous()
    x_scales_fp8 = x_scales_fp8.cuda().contiguous()
    w_scales_fp8 = w_scales_fp8.cuda().contiguous()

    # 调用 gemm_prepared
    print("\nCalling nvfp4_gemm.gemm_prepared()...")
    try:
        output = nvfp4_gemm.gemm_prepared(
            x_packed,
            w_packed,
            x_scales_fp8,
            w_scales_fp8,
            M, N, K,
            None,  # no bias
            1.0,   # alpha
            0.0    # beta
        )

        print(f"Output shape: {output.shape}, dtype={output.dtype}")
        print(f"Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")

        # 计算精度指标
        output_ratio = output.abs().mean().item() / (ref.abs().mean().item() + 1e-8)
        print(f"\nOutput/Ref magnitude ratio: {output_ratio:.4f}")

        cos_sim = F.cosine_similarity(
            output.flatten().float().unsqueeze(0),
            ref.flatten().unsqueeze(0)
        ).item()
        print(f"Cosine similarity: {cos_sim:.6f}")

        mse = F.mse_loss(output.float(), ref.float()).item()
        rmse = mse ** 0.5
        print(f"RMSE: {rmse:.4f}")

        # 相对误差
        rel_error = (output.float() - ref.float()).abs() / (ref.float().abs() + 1e-8)
        mean_rel_error = rel_error.mean().item() * 100
        print(f"Mean relative error: {mean_rel_error:.2f}%")

        if cos_sim > 0.95:
            print("\n[SUCCESS] High precision achieved!")
            return True
        elif cos_sim > 0.8:
            print("\n[PARTIAL] Moderate precision, layout may need adjustment")
            return False
        else:
            print("\n[FAIL] Poor precision, check data format")
            return False

    except Exception as e:
        print(f"GEMM failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_gemm_prepared():
    """性能测试"""
    print_section("Performance Benchmark")

    try:
        import nvfp4_gemm
        if not hasattr(nvfp4_gemm, 'gemm_prepared'):
            print("gemm_prepared not available")
            return
    except ImportError:
        print("Extension not available")
        return

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE
    iterations = 100

    # 准备数据
    x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)

    x_q, x_scales = quantize_to_nvfp4_sim(x.float(), block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w.float(), block_size)

    x_packed = pack_nvfp4_data(x_q, block_size).cuda()
    w_packed = pack_nvfp4_data(w_q, block_size).cuda()

    num_k_blocks = K // block_size
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks).contiguous()
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks).contiguous()

    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = torch.matmul(x.float(), w.float().T)
        try:
            _ = nvfp4_gemm.gemm_prepared(x_packed, w_packed, x_scales_fp8, w_scales_fp8, M, N, K)
        except:
            print("gemm_prepared failed during warmup")
            return
    torch.cuda.synchronize()

    # BF16 baseline
    print("\nBenchmarking BF16 matmul...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(x.float(), w.float().T)
    torch.cuda.synchronize()
    bf16_ms = (time.perf_counter() - start) / iterations * 1000

    # NVFP4 GEMM
    print("Benchmarking NVFP4 gemm_prepared...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = nvfp4_gemm.gemm_prepared(x_packed, w_packed, x_scales_fp8, w_scales_fp8, M, N, K)
    torch.cuda.synchronize()
    nvfp4_ms = (time.perf_counter() - start) / iterations * 1000

    print(f"\nResults:")
    print(f"  BF16 matmul:     {bf16_ms:.3f} ms")
    print(f"  NVFP4 GEMM:      {nvfp4_ms:.3f} ms")
    print(f"  Speedup:         {bf16_ms/nvfp4_ms:.2f}x")


def main():
    print_header("NVFP4 gemm_prepared() Integration Test")

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    success = test_gemm_prepared()

    if success:
        benchmark_gemm_prepared()
    else:
        print("\nSkipping benchmark due to test failure")


if __name__ == "__main__":
    main()
