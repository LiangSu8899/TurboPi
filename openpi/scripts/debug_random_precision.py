#!/usr/bin/env python3
"""
调试随机数据的精度问题
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/workspace/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    prepare_scales_for_cutlass,
    pack_nvfp4_data,
    BLOCK_SIZE,
    CUTLASS_ROW_TILE,
)


def compare_scale_layouts():
    """比较不同的 scale 布局"""
    print("=" * 60)
    print("Scale Layout Comparison")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w = torch.randn(N, K, device='cuda', dtype=torch.float32)

    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # Python 模拟参考
    x_dequant = dequantize_nvfp4_sim(x_q, x_scales, block_size)
    w_dequant = dequantize_nvfp4_sim(w_q, w_scales, block_size)
    ref = torch.matmul(x_dequant.cuda(), w_dequant.cuda().T)

    print(f"Reference output: mean={ref.mean().item():.4f}, std={ref.std().item():.4f}")

    # 方法 A: 使用 K 参数扩展 scales
    x_scales_a = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=K)
    w_scales_a = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=K)

    output_a = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(), w_packed.cuda(),
        x_scales_a.cuda(), w_scales_a.cuda(),
        M, N, K
    )

    cos_a = F.cosine_similarity(
        output_a.flatten().float().unsqueeze(0),
        ref.flatten().unsqueeze(0)
    ).item()

    print(f"\nMethod A (with K expansion):")
    print(f"  x_scales size: {x_scales_a.numel()}")
    print(f"  Cosine sim: {cos_a:.6f}")
    print(f"  Output mean: {output_a.mean().item():.4f}")

    # 方法 B: 不使用 K 扩展，看看效果
    x_scales_b = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=None)
    w_scales_b = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=None)

    # 手动 padding 到足够大
    # 这是错误的方法，但用于对比
    target_size_x = M * K
    target_size_w = N * K

    if x_scales_b.numel() < target_size_x:
        x_scales_b = torch.cat([
            x_scales_b,
            torch.zeros(target_size_x - x_scales_b.numel(), dtype=torch.uint8, device='cuda')
        ])

    if w_scales_b.numel() < target_size_w:
        w_scales_b = torch.cat([
            w_scales_b,
            torch.zeros(target_size_w - w_scales_b.numel(), dtype=torch.uint8, device='cuda')
        ])

    output_b = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(), w_packed.cuda(),
        x_scales_b.cuda(), w_scales_b.cuda(),
        M, N, K
    )

    cos_b = F.cosine_similarity(
        output_b.flatten().float().unsqueeze(0),
        ref.flatten().unsqueeze(0)
    ).item()

    print(f"\nMethod B (padded with zeros):")
    print(f"  x_scales size: {x_scales_b.numel()}")
    print(f"  Cosine sim: {cos_b:.6f}")
    print(f"  Output mean: {output_b.mean().item():.4f}")


def analyze_per_block_error():
    """分析每个 block 的误差"""
    print("\n" + "=" * 60)
    print("Per-Block Error Analysis")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w = torch.randn(N, K, device='cuda', dtype=torch.float32)

    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    # 检查 scale 分布
    print(f"x_scales: min={x_scales.min():.4f}, max={x_scales.max():.4f}, mean={x_scales.mean():.4f}")
    print(f"w_scales: min={w_scales.min():.4f}, max={w_scales.max():.4f}, mean={w_scales.mean():.4f}")

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # 检查扩展后的 scales
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=K)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=K)

    print(f"\nExpanded scales:")
    print(f"  x_scales_fp8 unique values: {x_scales_fp8.unique().numel()}")
    print(f"  w_scales_fp8 unique values: {w_scales_fp8.unique().numel()}")

    # 检查 repeat_interleave 是否正确
    # 原始 scales 形状: [M, num_k_blocks]
    # 扩展后应该是: [M, K] 其中每个 scale 重复 block_size 次

    # 反向检查
    from openpi.models_pytorch.nvfp4_mlp import convert_scales_to_fp8

    x_scales_expanded = x_scales.cuda().repeat_interleave(block_size, dim=1)
    K_padded = ((K + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    M_padded = 256

    print(f"\nExpanded shape: {x_scales_expanded.shape}")
    print(f"Expected shape: [{M_padded}, {K_padded}]")

    # 验证第一行的 scales
    print("\nFirst row scales (first 64 elements):")
    print(f"  Original (block 0-3): {x_scales[0, :4].tolist()}")

    expanded_row = x_scales_expanded[0]
    print(f"  Expanded (elem 0-3): {expanded_row[:4].tolist()}")
    print(f"  Expanded (elem 32-35): {expanded_row[32:36].tolist()}")


def test_without_scale_expansion():
    """测试不扩展 scales（假设 CUTLASS 内部处理）"""
    print("\n" + "=" * 60)
    print("Test Without Scale Expansion")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w = torch.randn(N, K, device='cuda', dtype=torch.float32)

    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # Python 模拟参考
    x_dequant = dequantize_nvfp4_sim(x_q, x_scales, block_size)
    w_dequant = dequantize_nvfp4_sim(w_q, w_scales, block_size)
    ref = torch.matmul(x_dequant.cuda(), w_dequant.cuda().T)

    # 尝试直接使用 [M, num_k_blocks] 形状的 scales
    # 只进行 padding 和 FP8 转换，不扩展
    from openpi.models_pytorch.nvfp4_mlp import convert_scales_to_fp8, CUTLASS_K_TILE

    M_padded = 256
    K_blocks_padded = ((num_k_blocks + CUTLASS_K_TILE - 1) // CUTLASS_K_TILE) * CUTLASS_K_TILE

    x_scales_padded = torch.zeros(M_padded, K_blocks_padded, device='cuda', dtype=torch.float32)
    x_scales_padded[:M, :num_k_blocks] = x_scales.cuda()
    x_scales_flat = x_scales_padded.flatten()
    x_scales_fp8 = convert_scales_to_fp8(x_scales_flat)

    w_scales_padded = torch.zeros(N, K_blocks_padded, device='cuda', dtype=torch.float32)
    w_scales_padded[:N, :num_k_blocks] = w_scales.cuda()
    w_scales_flat = w_scales_padded.flatten()
    w_scales_fp8 = convert_scales_to_fp8(w_scales_flat)

    # 需要 padding 到 CUTLASS 期望的大小
    # 这可能不工作，因为 CUTLASS 期望更大的 tensor
    K_padded = ((K + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE

    target_x = M_padded * K_padded
    target_w = N_padded * K_padded

    if x_scales_fp8.numel() < target_x:
        x_scales_fp8 = torch.cat([
            x_scales_fp8,
            torch.zeros(target_x - x_scales_fp8.numel(), dtype=torch.uint8, device='cuda')
        ])

    if w_scales_fp8.numel() < target_w:
        w_scales_fp8 = torch.cat([
            w_scales_fp8,
            torch.zeros(target_w - w_scales_fp8.numel(), dtype=torch.uint8, device='cuda')
        ])

    print(f"Scale sizes: x={x_scales_fp8.numel()}, w={w_scales_fp8.numel()}")

    try:
        output = nvfp4_gemm.gemm_prepared(
            x_packed.cuda(), w_packed.cuda(),
            x_scales_fp8.cuda(), w_scales_fp8.cuda(),
            M, N, K
        )

        cos = F.cosine_similarity(
            output.flatten().float().unsqueeze(0),
            ref.flatten().unsqueeze(0)
        ).item()

        print(f"Cosine sim: {cos:.6f}")
        print(f"Output mean: {output.mean().item():.4f}")
        print(f"Ref mean: {ref.mean().item():.4f}")

    except Exception as e:
        print(f"ERROR: {e}")


def main():
    print("Random Data Precision Debug")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    compare_scale_layouts()
    analyze_per_block_error()
    test_without_scale_expansion()


if __name__ == "__main__":
    main()
