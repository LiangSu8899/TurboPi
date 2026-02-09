#!/usr/bin/env python3
"""
深入分析 FP8 Scale 精度损失问题

关键发现：
- Python 模拟精度: ~0.99 cosine (NVFP4 量化误差有限)
- CUTLASS GEMM: ~0.93 cosine (额外 6% 误差来自 FP8 Scale)

FP8 E4M3 的限制:
- 4 bit exponent, 3 bit mantissa
- 可表示范围: ~±448
- 精度: 尾数只有 8 个值 (1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875)
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

sys.path.insert(0, '/workspace/src')


def analyze_fp8_e4m3_precision():
    """分析 FP8 E4M3 的表示精度"""
    print("=" * 60)
    print("FP8 E4M3 Precision Analysis")
    print("=" * 60)

    # FP8 E4M3 的所有可表示值 (正数部分)
    # Format: 1 sign + 4 exponent + 3 mantissa
    # 可表示的尾数值: 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875

    # 生成所有 FP8 E4M3 可表示的正值
    fp8_values = []
    for exp in range(-6, 8):  # exponent range
        for mant in range(8):  # 3-bit mantissa
            val = (1.0 + mant / 8.0) * (2 ** exp)
            if val <= 448:  # E4M3 max
                fp8_values.append(val)

    fp8_values = sorted(set(fp8_values))
    print(f"FP8 E4M3 可表示的正值数量: {len(fp8_values)}")

    # 分析常见 scale 范围内的精度
    print("\n常见 scale 范围的 FP8 精度:")
    for range_start, range_end in [(0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0)]:
        values_in_range = [v for v in fp8_values if range_start <= v < range_end]
        if values_in_range:
            step = np.mean(np.diff(values_in_range)) if len(values_in_range) > 1 else 0
            print(f"  [{range_start}, {range_end}): {len(values_in_range)} 个值, "
                  f"平均间隔 ~{step:.4f}")


def analyze_scale_quantization_error():
    """分析实际 Scale 的 FP8 量化误差"""
    print("\n" + "=" * 60)
    print("Scale FP8 Quantization Error")
    print("=" * 60)

    from openpi.models_pytorch.nvfp4_mlp import (
        quantize_to_nvfp4_sim,
        convert_scales_to_fp8,
        BLOCK_SIZE,
    )

    device = torch.device('cuda')
    M, K = 256, 2048
    block_size = BLOCK_SIZE

    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.float32)

    # 量化得到 scales
    _, scales = quantize_to_nvfp4_sim(x, block_size)
    scales_flat = scales.flatten()

    # 转换为 FP8 并转回
    scales_fp8 = convert_scales_to_fp8(scales_flat)
    scales_back = scales_fp8.view(torch.float8_e4m3fn).to(torch.float32)

    # 计算误差
    abs_error = (scales_back - scales_flat).abs()
    rel_error = abs_error / (scales_flat.abs() + 1e-8)

    print(f"Scale 值统计:")
    print(f"  Min: {scales_flat.min().item():.6f}")
    print(f"  Max: {scales_flat.max().item():.6f}")
    print(f"  Mean: {scales_flat.mean().item():.6f}")

    print(f"\nFP8 转换误差:")
    print(f"  绝对误差 Mean: {abs_error.mean().item():.6f}")
    print(f"  绝对误差 Max:  {abs_error.max().item():.6f}")
    print(f"  相对误差 Mean: {rel_error.mean().item()*100:.2f}%")
    print(f"  相对误差 Max:  {rel_error.max().item()*100:.2f}%")

    # 分析误差分布
    print(f"\n相对误差分布:")
    for threshold in [1, 5, 10, 20, 50]:
        pct = (rel_error > threshold/100).float().mean().item() * 100
        print(f"  > {threshold}%: {pct:.1f}% 的 scales")


def analyze_gemm_error_breakdown():
    """分解 GEMM 误差来源"""
    print("\n" + "=" * 60)
    print("GEMM Error Breakdown")
    print("=" * 60)

    try:
        import nvfp4_gemm
    except ImportError:
        print("nvfp4_gemm not available")
        return

    from openpi.models_pytorch.nvfp4_mlp import (
        quantize_to_nvfp4_sim,
        dequantize_nvfp4_sim,
        prepare_scales_for_cutlass,
        pack_nvfp4_data,
        convert_scales_to_fp8,
        BLOCK_SIZE,
    )

    device = torch.device('cuda')
    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    # BF16 参考
    bf16_ref = torch.matmul(x, w.T)

    # 1. Python 模拟 (使用 FP32 scales)
    x_deq_fp32 = dequantize_nvfp4_sim(x_q, x_scales, block_size)
    w_deq_fp32 = dequantize_nvfp4_sim(w_q, w_scales, block_size)
    sim_fp32 = torch.matmul(x_deq_fp32, w_deq_fp32.T)

    cos_sim_fp32 = F.cosine_similarity(
        sim_fp32.flatten().unsqueeze(0),
        bf16_ref.flatten().unsqueeze(0)
    ).item()

    # 2. Python 模拟 (使用 FP8 scales - 模拟 CUTLASS 精度)
    x_scales_fp8 = convert_scales_to_fp8(x_scales.flatten())
    w_scales_fp8 = convert_scales_to_fp8(w_scales.flatten())

    x_scales_back = x_scales_fp8.view(torch.float8_e4m3fn).to(torch.float32).view(M, num_k_blocks)
    w_scales_back = w_scales_fp8.view(torch.float8_e4m3fn).to(torch.float32).view(N, num_k_blocks)

    x_deq_fp8 = dequantize_nvfp4_sim(x_q, x_scales_back, block_size)
    w_deq_fp8 = dequantize_nvfp4_sim(w_q, w_scales_back, block_size)
    sim_fp8 = torch.matmul(x_deq_fp8, w_deq_fp8.T)

    cos_sim_fp8 = F.cosine_similarity(
        sim_fp8.flatten().unsqueeze(0),
        bf16_ref.flatten().unsqueeze(0)
    ).item()

    # 3. CUTLASS GEMM
    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    x_scales_cutlass = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks,
                                                   convert_to_fp8=True, K=K)
    w_scales_cutlass = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks,
                                                   convert_to_fp8=True, K=K)

    cutlass_output = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(), w_packed.cuda(),
        x_scales_cutlass.cuda(), w_scales_cutlass.cuda(),
        M, N, K
    )

    cos_cutlass = F.cosine_similarity(
        cutlass_output.flatten().float().unsqueeze(0),
        bf16_ref.flatten().unsqueeze(0)
    ).item()

    # 误差分解
    print("误差分解 (Cosine Similarity vs BF16):")
    print(f"  [1] NVFP4 量化 (FP32 scales):  {cos_sim_fp32:.6f}")
    print(f"  [2] NVFP4 量化 (FP8 scales):   {cos_sim_fp8:.6f}")
    print(f"  [3] CUTLASS GEMM:              {cos_cutlass:.6f}")

    print(f"\n误差来源:")
    nvfp4_error = 1 - cos_sim_fp32
    fp8_error = cos_sim_fp32 - cos_sim_fp8
    other_error = cos_sim_fp8 - cos_cutlass

    total_error = 1 - cos_cutlass
    print(f"  NVFP4 量化误差:     {nvfp4_error*100:.2f}% ({nvfp4_error/total_error*100:.0f}% of total)")
    print(f"  FP8 Scale 误差:     {fp8_error*100:.2f}% ({fp8_error/total_error*100:.0f}% of total)")
    print(f"  其他误差 (layout):  {other_error*100:.2f}% ({other_error/total_error*100:.0f}% of total)")
    print(f"  总误差:             {total_error*100:.2f}%")


def suggest_solutions():
    """提出解决方案"""
    print("\n" + "=" * 60)
    print("Suggested Solutions")
    print("=" * 60)

    print("""
基于误差分解分析，主要误差来源是 FP8 Scale 转换。

解决方案 (按优先级):

1. [推荐] 优化 Scale 使其对 FP8 友好
   - 在量化时考虑 FP8 的量化格点
   - 让 scale 尽量落在 FP8 可精确表示的值上
   - 预期收益: 减少 FP8 转换误差

2. [推荐] 使用更大的 block size
   - 更大的 block = 更少的 scales = 更少的 FP8 误差累积
   - 但 CUTLASS SM110 固定 block_size=32，无法修改

3. [备选] 混合精度策略
   - Down_Proj 层不使用 NVFP4，保持 BF16
   - Down_Proj 是汇聚层，对精度最敏感
   - 代价: 失去部分加速

4. [终极方案] LoRA 矫正
   - 冻结 NVFP4 模型，训练小 LoRA 补偿误差
   - 可恢复到接近 BF16 精度
   - 代价: 需要额外训练

5. [实验性] 检查 CUTLASS 是否支持其他 Scale 格式
   - 如 float_ue8m0_t 虽然只有 2 的幂次
   - 但某些 scale 分布下可能误差更小
""")


def main():
    print("FP8 Scale Precision Deep Analysis")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    analyze_fp8_e4m3_precision()
    analyze_scale_quantization_error()
    analyze_gemm_error_breakdown()
    suggest_solutions()


if __name__ == "__main__":
    main()
