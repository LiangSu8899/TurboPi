#!/usr/bin/env python3
"""
测试 MSE Search 量化优化的精度提升

比较:
1. Min-Max 量化 (原始方法)
2. MSE Search 量化 (优化方法)
"""

import torch
import torch.nn.functional as F
import sys
import time

sys.path.insert(0, '/workspace/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    prepare_scales_for_cutlass,
    pack_nvfp4_data,
    BLOCK_SIZE,
)


def test_quantization_precision():
    """比较 Min-Max 和 MSE Search 的量化精度"""
    print("=" * 60)
    print("Quantization Precision Comparison")
    print("=" * 60)

    device = torch.device('cuda')
    M, K = 256, 2048
    block_size = BLOCK_SIZE

    # 测试多种数据分布
    test_cases = [
        ("Uniform Random", lambda: torch.randn(M, K, device=device)),
        ("With Outliers", lambda: torch.randn(M, K, device=device) +
         torch.where(torch.rand(M, K, device=device) < 0.02,
                     torch.randn(M, K, device=device) * 5,
                     torch.zeros(M, K, device=device))),
        ("Heavy Tail", lambda: torch.randn(M, K, device=device) ** 3),
        ("Gemma-like (scaled)", lambda: torch.randn(M, K, device=device) * 0.1),
    ]

    for name, data_fn in test_cases:
        print(f"\n--- {name} ---")
        torch.manual_seed(42)
        x = data_fn()

        # Min-Max 量化
        x_q_minmax, scales_minmax = quantize_to_nvfp4_sim(x, block_size, use_mse_search=False)
        x_deq_minmax = dequantize_nvfp4_sim(x_q_minmax, scales_minmax, block_size)

        # MSE Search 量化
        x_q_mse, scales_mse = quantize_to_nvfp4_sim(x, block_size, use_mse_search=True)
        x_deq_mse = dequantize_nvfp4_sim(x_q_mse, scales_mse, block_size)

        # 计算误差
        mse_minmax = ((x - x_deq_minmax) ** 2).mean().item()
        mse_mse = ((x - x_deq_mse) ** 2).mean().item()

        cos_minmax = F.cosine_similarity(x.flatten().unsqueeze(0),
                                         x_deq_minmax.flatten().unsqueeze(0)).item()
        cos_mse = F.cosine_similarity(x.flatten().unsqueeze(0),
                                      x_deq_mse.flatten().unsqueeze(0)).item()

        improvement = (mse_minmax - mse_mse) / mse_minmax * 100 if mse_minmax > 0 else 0

        print(f"  Min-Max: MSE={mse_minmax:.6f}, Cosine={cos_minmax:.6f}")
        print(f"  MSE Search: MSE={mse_mse:.6f}, Cosine={cos_mse:.6f}")
        print(f"  MSE Improvement: {improvement:.1f}%")


def test_gemm_precision():
    """测试 GEMM 精度"""
    print("\n" + "=" * 60)
    print("GEMM Precision Test")
    print("=" * 60)

    try:
        import nvfp4_gemm
    except ImportError:
        print("Warning: nvfp4_gemm not available, skipping GEMM test")
        return

    device = torch.device('cuda')
    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)

    # BF16 参考
    bf16_ref = torch.matmul(x, w.T)

    results = {}

    for use_mse, name in [(False, "Min-Max"), (True, "MSE Search")]:
        # 量化
        x_q, x_scales = quantize_to_nvfp4_sim(x, block_size, use_mse_search=use_mse)
        w_q, w_scales = quantize_to_nvfp4_sim(w, block_size, use_mse_search=use_mse)

        # Python 模拟参考
        x_dequant = dequantize_nvfp4_sim(x_q, x_scales, block_size)
        w_dequant = dequantize_nvfp4_sim(w_q, w_scales, block_size)
        sim_output = torch.matmul(x_dequant, w_dequant.T)

        # CUTLASS GEMM
        x_packed = pack_nvfp4_data(x_q, block_size)
        w_packed = pack_nvfp4_data(w_q, block_size)

        x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks,
                                                   convert_to_fp8=True, K=K)
        w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks,
                                                   convert_to_fp8=True, K=K)

        cutlass_output = nvfp4_gemm.gemm_prepared(
            x_packed.cuda(), w_packed.cuda(),
            x_scales_fp8.cuda(), w_scales_fp8.cuda(),
            M, N, K
        )

        # 计算各种精度指标
        cos_sim_bf16 = F.cosine_similarity(
            sim_output.flatten().unsqueeze(0),
            bf16_ref.flatten().unsqueeze(0)
        ).item()

        cos_cutlass_sim = F.cosine_similarity(
            cutlass_output.flatten().float().unsqueeze(0),
            sim_output.flatten().unsqueeze(0)
        ).item()

        cos_cutlass_bf16 = F.cosine_similarity(
            cutlass_output.flatten().float().unsqueeze(0),
            bf16_ref.flatten().unsqueeze(0)
        ).item()

        results[name] = {
            'cos_sim_bf16': cos_sim_bf16,
            'cos_cutlass_sim': cos_cutlass_sim,
            'cos_cutlass_bf16': cos_cutlass_bf16,
        }

        print(f"\n{name}:")
        print(f"  Python Sim vs BF16:    {cos_sim_bf16:.6f}")
        print(f"  CUTLASS vs Python Sim: {cos_cutlass_sim:.6f}")
        print(f"  CUTLASS vs BF16:       {cos_cutlass_bf16:.6f}")

    # 对比
    print("\n--- Improvement ---")
    for metric in ['cos_sim_bf16', 'cos_cutlass_sim', 'cos_cutlass_bf16']:
        minmax = results['Min-Max'][metric]
        mse = results['MSE Search'][metric]
        improvement = (mse - minmax) * 100  # 以百分点计
        print(f"  {metric}: +{improvement:.3f} percentage points")


def benchmark_quantization_speed():
    """测试量化速度"""
    print("\n" + "=" * 60)
    print("Quantization Speed Benchmark")
    print("=" * 60)

    device = torch.device('cuda')
    M, K = 256, 2048
    block_size = BLOCK_SIZE
    iterations = 100

    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = quantize_to_nvfp4_sim(x, block_size, use_mse_search=False)
        _ = quantize_to_nvfp4_sim(x, block_size, use_mse_search=True)
    torch.cuda.synchronize()

    # Min-Max
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = quantize_to_nvfp4_sim(x, block_size, use_mse_search=False)
    torch.cuda.synchronize()
    minmax_ms = (time.perf_counter() - start) / iterations * 1000

    # MSE Search (10 steps)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = quantize_to_nvfp4_sim(x, block_size, use_mse_search=True, mse_search_steps=10)
    torch.cuda.synchronize()
    mse_ms = (time.perf_counter() - start) / iterations * 1000

    print(f"  Min-Max:      {minmax_ms:.3f} ms")
    print(f"  MSE Search:   {mse_ms:.3f} ms")
    print(f"  Overhead:     {mse_ms/minmax_ms:.1f}x")
    print(f"\n  Note: MSE Search only done once during weight quantization,")
    print(f"        not during inference. The overhead is acceptable.")


def main():
    print("NVFP4 MSE Search Precision Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    test_quantization_precision()
    test_gemm_precision()
    benchmark_quantization_speed()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
MSE Search 优化通过搜索最佳 scale 来减少量化误差。

对于有 outliers 的分布特别有效:
- Min-Max 会让一个大值决定整个 block 的 scale
- MSE Search 可以选择牺牲 outlier 以获得更好的整体精度

如果 MSE Search 提升不够 (< 0.98 cosine):
1. 考虑增加 mse_search_steps (默认 10)
2. 对 down_proj 层使用 BF16 (最敏感)
3. 使用 LoRA 矫正
""")


if __name__ == "__main__":
    main()
