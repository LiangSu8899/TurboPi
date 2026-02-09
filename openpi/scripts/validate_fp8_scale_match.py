#!/usr/bin/env python3
"""
验证使用 FP8 Scales 后 CUTLASS 和 Python 的输出匹配

发现的问题:
- FP8 E4M3 对 scale=0.167 转换为 0.172 (偏大 3.12%)
- 两个 scale 相乘后累积误差 6.35%
- 这解释了 CUTLASS 输出系统性偏高的原因

解决方案:
- 在 Python 模拟中使用 FP8 转换后的 scale
- 这样 Python 参考值和 CUTLASS 使用相同的 scale
- 预期 cosine similarity 应该接近 0.99+
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/workspace/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    pack_nvfp4_data,
    prepare_scales_for_cutlass,
    BLOCK_SIZE,
)


def test_fp8_scale_match():
    """测试 FP8 scale 匹配"""
    print("=" * 60)
    print("FP8 Scale Match Validation")
    print("=" * 60)

    try:
        import nvfp4_gemm
    except ImportError:
        print("nvfp4_gemm not available")
        return

    device = torch.device('cuda')
    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size, use_mse_search=False)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size, use_mse_search=False)

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # BF16 参考 (无量化)
    bf16_ref = torch.matmul(x, w.T)

    # Python 模拟 - 使用 FP32 scales
    x_deq_fp32 = dequantize_nvfp4_sim(x_q, x_scales, block_size, use_fp8_scales=False)
    w_deq_fp32 = dequantize_nvfp4_sim(w_q, w_scales, block_size, use_fp8_scales=False)
    sim_fp32 = torch.matmul(x_deq_fp32, w_deq_fp32.T)

    # Python 模拟 - 使用 FP8 scales (模拟 CUTLASS 行为)
    x_deq_fp8 = dequantize_nvfp4_sim(x_q, x_scales, block_size, use_fp8_scales=True)
    w_deq_fp8 = dequantize_nvfp4_sim(w_q, w_scales, block_size, use_fp8_scales=True)
    sim_fp8 = torch.matmul(x_deq_fp8, w_deq_fp8.T)

    # CUTLASS GEMM
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=K)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=K)

    cutlass_output = nvfp4_gemm.gemm_prepared(
        x_packed, w_packed,
        x_scales_fp8, w_scales_fp8,
        M, N, K
    )

    # 计算 Cosine Similarity
    cos_sim_fp32_bf16 = F.cosine_similarity(
        sim_fp32.flatten().unsqueeze(0),
        bf16_ref.flatten().unsqueeze(0)
    ).item()

    cos_sim_fp8_bf16 = F.cosine_similarity(
        sim_fp8.flatten().unsqueeze(0),
        bf16_ref.flatten().unsqueeze(0)
    ).item()

    cos_cutlass_sim_fp32 = F.cosine_similarity(
        cutlass_output.flatten().float().unsqueeze(0),
        sim_fp32.flatten().unsqueeze(0)
    ).item()

    cos_cutlass_sim_fp8 = F.cosine_similarity(
        cutlass_output.flatten().float().unsqueeze(0),
        sim_fp8.flatten().unsqueeze(0)
    ).item()

    cos_cutlass_bf16 = F.cosine_similarity(
        cutlass_output.flatten().float().unsqueeze(0),
        bf16_ref.flatten().unsqueeze(0)
    ).item()

    print("\nCosine Similarity Results:")
    print("-" * 50)
    print(f"Python (FP32 scales) vs BF16:    {cos_sim_fp32_bf16:.6f}")
    print(f"Python (FP8 scales) vs BF16:     {cos_sim_fp8_bf16:.6f}")
    print(f"CUTLASS vs Python (FP32 scales): {cos_cutlass_sim_fp32:.6f}")
    print(f"CUTLASS vs Python (FP8 scales):  {cos_cutlass_sim_fp8:.6f}")
    print(f"CUTLASS vs BF16:                 {cos_cutlass_bf16:.6f}")

    print("\n分析:")
    if cos_cutlass_sim_fp8 > 0.98:
        print(f"  ✓ CUTLASS 和 Python (FP8 scales) 高度一致!")
        print(f"    这证明误差主要来自 FP8 scale 转换，而不是 layout 问题。")
    else:
        print(f"  ✗ CUTLASS 和 Python (FP8 scales) 仍有差异")
        print(f"    可能还有其他因素影响精度")

    # 检查 FP8 转换对 scale 的影响
    print("\n\nFP8 Scale 转换影响:")
    print("-" * 50)

    x_scales_flat = x_scales.flatten()
    x_scales_fp8_back = x_scales_flat.to(torch.float8_e4m3fn).to(torch.float32)
    scale_ratio = (x_scales_fp8_back / x_scales_flat).mean().item()
    print(f"  平均 scale 放大比例: {scale_ratio:.4f} ({(scale_ratio-1)*100:.2f}%)")

    # 对于 GEMM，两个 scale 相乘
    print(f"  预期 output 放大比例: {scale_ratio**2:.4f} ({(scale_ratio**2-1)*100:.2f}%)")


def test_uniform_input():
    """测试 uniform 输入的精确匹配"""
    print("\n" + "=" * 60)
    print("Uniform Input Test")
    print("=" * 60)

    try:
        import nvfp4_gemm
    except ImportError:
        print("nvfp4_gemm not available")
        return

    device = torch.device('cuda')
    M, K, N = 256, 64, 256  # 小尺寸便于精确验证
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # 使用全 1 输入
    x = torch.ones(M, K, device=device, dtype=torch.float32)
    w = torch.ones(N, K, device=device, dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size, use_mse_search=False)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size, use_mse_search=False)

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # Python 模拟 - FP8 scales
    x_deq_fp8 = dequantize_nvfp4_sim(x_q, x_scales, block_size, use_fp8_scales=True)
    w_deq_fp8 = dequantize_nvfp4_sim(w_q, w_scales, block_size, use_fp8_scales=True)
    sim_fp8 = torch.matmul(x_deq_fp8, w_deq_fp8.T)

    # CUTLASS
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=K)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=K)

    cutlass_output = nvfp4_gemm.gemm_prepared(
        x_packed, w_packed,
        x_scales_fp8, w_scales_fp8,
        M, N, K
    )

    print(f"\nUniform input (all 1s), K={K}:")
    print(f"  Python (FP8 scales) [0,0]: {sim_fp8[0,0].item():.4f}")
    print(f"  CUTLASS output [0,0]:      {cutlass_output[0,0].item():.4f}")
    print(f"  Ratio: {cutlass_output[0,0].item() / sim_fp8[0,0].item():.4f}")

    # 检查是否所有元素相同
    sim_unique = sim_fp8.unique().numel()
    cutlass_unique = cutlass_output.unique().numel()
    print(f"  Python unique values: {sim_unique}")
    print(f"  CUTLASS unique values: {cutlass_unique}")


def main():
    print("FP8 Scale Match Validation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    test_fp8_scale_match()
    test_uniform_input()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
关键发现:
1. FP8 E4M3 对常见 scale 值 (如 0.167) 转换时有 3% 左右的误差
2. 两个 scale 相乘后，误差累积到 6%
3. 这解释了 CUTLASS 输出系统性偏高的原因

解决方案:
1. 在 Python 模拟中使用 use_fp8_scales=True 来匹配 CUTLASS 行为
2. 这样精度对比更公平
3. 最终精度受限于 FP8 scale 表示能力

对于 Diffusion Policy 的建议:
1. 如果需要 0.99+ 精度，考虑:
   - 混合精度: 敏感层 (down_proj) 保持 BF16
   - LoRA 矫正: 用小型 LoRA 补偿量化误差
   - 校准: 使用真实数据找到更好的 scale
""")


if __name__ == "__main__":
    main()
