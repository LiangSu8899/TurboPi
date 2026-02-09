#!/usr/bin/env python3
"""
Scale Factor Reordering 验证脚本

验证 CUTLASS SM110a NVFP4 GEMM 所需的 Scale Factor 内存布局。

基于 sm100_blockscaled_layout.hpp 分析:
- Blk_MN = 128 (128-row tiles)
- Blk_SF = 4 (4 scale factors per unit)
- 32-row groups within each tile
- Stride pattern: <_16, _4>, <_0, _1>
"""

import torch
import numpy as np
from typing import Tuple, Optional


def swizzle_scales_for_cutlass_v1(
    scales: torch.Tensor,
    rows: int,
    k_blocks: int,
    row_tile: int = 128,
    k_tile: int = 4,
    row_group: int = 32
) -> torch.Tensor:
    """
    Version 1: 基于文档描述的 reshape + permute 方法

    将 Row-Major scales 重排为 CUTLASS interleaved 布局

    CUTLASS 布局:
    - 每 128 行为一个 tile
    - 每 4 个 k-blocks 为一组
    - tile 内部按 32-row groups 组织
    """
    device = scales.device
    dtype = scales.dtype

    # 确保输入是 2D
    if scales.dim() == 1:
        scales = scales.view(rows, k_blocks)

    # 1. Padding 到 tile 边界
    rows_padded = ((rows + row_tile - 1) // row_tile) * row_tile
    k_padded = ((k_blocks + k_tile - 1) // k_tile) * k_tile

    if rows_padded != rows or k_padded != k_blocks:
        scales_padded = torch.zeros(rows_padded, k_padded, device=device, dtype=dtype)
        scales_padded[:rows, :k_blocks] = scales
        scales = scales_padded

    # 2. Reshape 到 tile 结构
    num_row_tiles = rows_padded // row_tile
    num_k_tiles = k_padded // k_tile

    # [num_row_tiles, row_tile, num_k_tiles, k_tile]
    scales = scales.view(num_row_tiles, row_tile, num_k_tiles, k_tile)

    # 3. 拆分 row_tile 为 groups
    # [num_row_tiles, num_groups, group_size, num_k_tiles, k_tile]
    num_groups = row_tile // row_group
    scales = scales.view(num_row_tiles, num_groups, row_group, num_k_tiles, k_tile)

    # 4. Permute: 尝试多种排列
    # 目标: 让 CUTLASS 读取时能正确映射
    # 原始索引: [rt, g, r, kt, k] -> 重排后应该是什么？

    # 基于 Stride<_16, _4> 分析:
    # - 行方向 stride = 16 (不是 1，说明行不连续)
    # - k方向 stride = 4 (k-blocks 连续)

    # 尝试: [num_row_tiles, num_k_tiles, num_groups, k_tile, row_group]
    scales = scales.permute(0, 3, 1, 4, 2)

    return scales.contiguous().flatten()


def swizzle_scales_for_cutlass_v2(
    scales: torch.Tensor,
    rows: int,
    k_blocks: int,
    row_tile: int = 128,
    k_tile: int = 4,
    row_group: int = 32
) -> torch.Tensor:
    """
    Version 2: 基于 Stride 模式的手动索引计算

    CUTLASS Stride: Stride<Stride<_16,_4>, Stride<_0, _1>>

    这意味着:
    - 外层 32x4 block 中，行 stride=16，k stride=4
    - 内层是向量化访问
    """
    device = scales.device
    dtype = scales.dtype

    if scales.dim() == 1:
        scales = scales.view(rows, k_blocks)

    # Padding
    rows_padded = ((rows + row_tile - 1) // row_tile) * row_tile
    k_padded = ((k_blocks + k_tile - 1) // k_tile) * k_tile

    if rows_padded != rows or k_padded != k_blocks:
        scales_padded = torch.zeros(rows_padded, k_padded, device=device, dtype=dtype)
        scales_padded[:rows, :k_blocks] = scales
        scales = scales_padded

    num_row_tiles = rows_padded // row_tile
    num_k_tiles = k_padded // k_tile

    # 输出大小
    output_size = rows_padded * k_padded
    output = torch.zeros(output_size, device=device, dtype=dtype)

    # 基于 Stride<_16, _4> 模式计算
    # 每个 128x4 tile 内部:
    # - 4 个 32-row groups
    # - 每个 group 有 32 行 x 4 k-blocks = 128 个 scale factors
    # - stride pattern: row_in_group * 16 + k_in_tile * 4 + ???

    idx = 0
    for rt in range(num_row_tiles):
        for kt in range(num_k_tiles):
            # 处理一个 128x4 tile
            tile_base = rt * row_tile * k_padded + kt * k_tile

            for g in range(4):  # 4 groups of 32 rows
                for k in range(4):  # 4 k-blocks
                    for r in range(32):  # 32 rows in group
                        src_row = rt * row_tile + g * row_group + r
                        src_k = kt * k_tile + k
                        src_idx = src_row * k_padded + src_k

                        # 目标索引基于 stride 模式
                        # Stride<_16, _4> 意味着:
                        # - row (0-31) 的 stride 是 16
                        # - k (0-3) 的 stride 是 4
                        # 但这不是简单的 r*16 + k*4，因为还有 group 维度

                        # 尝试: tile_offset + group*128 + r*4 + k
                        # (每个 group 有 128 个元素，行内 k 连续)
                        tile_offset = (rt * num_k_tiles + kt) * (row_tile * k_tile)
                        dst_idx = tile_offset + g * 128 + r * 4 + k

                        if dst_idx < output_size:
                            output[dst_idx] = scales[src_row, src_k]

    return output


def swizzle_scales_for_cutlass_v3(
    scales: torch.Tensor,
    rows: int,
    k_blocks: int
) -> torch.Tensor:
    """
    Version 3: 简化版 - 直接 reshape 为 CUTLASS 期望的顺序

    假设 CUTLASS 期望: [tile_m, tile_k, k_in_tile, row_in_tile]
    即 K 维度在最内层
    """
    device = scales.device
    dtype = scales.dtype

    row_tile = 128
    k_tile = 4

    if scales.dim() == 1:
        scales = scales.view(rows, k_blocks)

    # Padding
    rows_padded = ((rows + row_tile - 1) // row_tile) * row_tile
    k_padded = ((k_blocks + k_tile - 1) // k_tile) * k_tile

    if rows_padded != rows or k_padded != k_blocks:
        scales_padded = torch.zeros(rows_padded, k_padded, device=device, dtype=dtype)
        scales_padded[:rows, :k_blocks] = scales
        scales = scales_padded

    num_row_tiles = rows_padded // row_tile
    num_k_tiles = k_padded // k_tile

    # Reshape: [rows, k_blocks] -> [num_row_tiles, row_tile, num_k_tiles, k_tile]
    scales = scales.view(num_row_tiles, row_tile, num_k_tiles, k_tile)

    # Permute to: [num_row_tiles, num_k_tiles, k_tile, row_tile]
    # 这样 row_tile 在最内层连续
    scales = scales.permute(0, 2, 3, 1)

    return scales.contiguous().flatten()


def analyze_cutlass_layout():
    """分析 CUTLASS 72a example 中的 scale factor 布局"""
    print("=" * 70)
    print("CUTLASS Scale Factor Layout Analysis")
    print("=" * 70)

    # 从 sm100_blockscaled_layout.hpp 提取的信息
    print("""
基于 CUTLASS 源码分析:

1. Sm1xxBlkScaledConfig 定义:
   - using Blk_MN = _128;     // 128-row blocks
   - using Blk_SF = _4;       // 4 scale factors per unit

2. SfKMajorAtom Layout:
   Shape: Shape<Shape<_32,_4>, Shape<Int<SFVecSize>, _4>>
   Stride: Stride<Stride<_16,_4>, Stride<_0, _1>>

   解读:
   - 外层 32x4: 32 rows × 4 k-blocks
   - Stride<_16, _4>: row stride=16, k stride=4
   - 内层是向量化 (SFVecSize, 通常是 4 或 8)

3. 这意味着:
   - 每 128 行 (4 个 32-row groups) × 4 k-blocks 形成一个 tile
   - tile 内部，同一行的 4 个 k-blocks 是连续存储的
   - 但行与行之间有 stride 间隔
""")


def test_small_matrix():
    """用小矩阵测试，便于手动验证"""
    print("\n" + "=" * 70)
    print("Small Matrix Test (256 rows × 64 k-blocks)")
    print("=" * 70)

    M = 256  # 2 个 128-row tiles
    num_k_blocks = 64  # 16 个 4-k tiles

    # 创建测试数据：每个 scale 用其原始坐标编码
    # scale[r, k] = r * 1000 + k (便于追踪)
    scales = torch.zeros(M, num_k_blocks)
    for r in range(M):
        for k in range(num_k_blocks):
            scales[r, k] = r * 1000 + k

    print(f"\n原始 scales 形状: {scales.shape}")
    print(f"原始 scales[0, :4]: {scales[0, :4].tolist()}")
    print(f"原始 scales[1, :4]: {scales[1, :4].tolist()}")
    print(f"原始 scales[31, :4]: {scales[31, :4].tolist()}")
    print(f"原始 scales[32, :4]: {scales[32, :4].tolist()}")

    # 测试各种重排版本
    versions = [
        ("V1 (reshape+permute)", swizzle_scales_for_cutlass_v1),
        ("V2 (manual index)", swizzle_scales_for_cutlass_v2),
        ("V3 (simple)", swizzle_scales_for_cutlass_v3),
    ]

    for name, func in versions:
        print(f"\n--- {name} ---")
        try:
            reordered = func(scales, M, num_k_blocks)
            print(f"重排后大小: {reordered.shape}")

            # 显示前 16 个元素
            print(f"前 16 个元素:")
            for i in range(16):
                val = reordered[i].item()
                orig_r = int(val // 1000)
                orig_k = int(val % 1000)
                print(f"  [{i:3d}] = {val:8.0f} -> 原始 (row={orig_r:3d}, k={orig_k:2d})")

            # 分析模式
            print(f"\n模式分析 (第一个 tile 的前 128 元素):")
            tile_data = reordered[:128].view(32, 4)  # 假设 32x4 pattern
            print(f"  假设 32x4 排列:")
            for r in range(min(4, 32)):
                row_vals = tile_data[r].tolist()
                print(f"    row {r}: {[f'{v:.0f}' for v in row_vals]}")

        except Exception as e:
            print(f"错误: {e}")


def test_fp8_conversion():
    """测试 FP32 到 FP8 的转换"""
    print("\n" + "=" * 70)
    print("FP8 Conversion Test")
    print("=" * 70)

    # 典型的 scale factor 值范围
    test_values = torch.tensor([0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])

    print("\nFP32 -> FP8 E4M3 转换:")
    print(f"{'FP32':<12} | {'FP8 bytes':<12} | {'还原 FP32':<12} | {'相对误差':<12}")
    print("-" * 60)

    for val in test_values:
        # 转换为 FP8
        fp8 = val.to(torch.float8_e4m3fn)
        fp8_bytes = fp8.view(torch.uint8).item()

        # 还原
        restored = fp8.to(torch.float32).item()

        # 相对误差
        rel_error = abs(val.item() - restored) / (abs(val.item()) + 1e-12) * 100

        print(f"{val.item():<12.4f} | 0x{fp8_bytes:02X} ({fp8_bytes:3d})   | {restored:<12.4f} | {rel_error:<10.2f}%")


def test_precision_impact():
    """测试 NVFP4 量化对 GEMM 精度的影响"""
    print("\n" + "=" * 70)
    print("NVFP4 Quantization Precision Test")
    print("=" * 70)

    torch.manual_seed(42)

    M, K, N = 256, 2048, 16384
    block_size = 32

    # 随机输入和权重
    x = torch.randn(M, K, dtype=torch.float32)
    w = torch.randn(N, K, dtype=torch.float32)

    # 参考结果 (FP32)
    ref_output = torch.matmul(x, w.T)

    # 模拟 NVFP4 量化
    def quantize_nvfp4_sim(tensor, block_size=32):
        """模拟 NVFP4 量化"""
        shape = tensor.shape
        M, K = shape[0], shape[1]
        num_blocks = K // block_size

        # Reshape to blocks
        tensor_blocked = tensor.view(M, num_blocks, block_size)

        # Per-block scale
        block_max = tensor_blocked.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
        nvfp4_max = 6.0

        # Scale and quantize
        scaled = tensor_blocked / block_max * nvfp4_max

        # NVFP4 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
        nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

        signs = scaled.sign()
        abs_scaled = scaled.abs()

        # Find nearest
        distances = (abs_scaled.unsqueeze(-1) - nvfp4_values).abs()
        indices = distances.argmin(dim=-1)
        quantized_abs = nvfp4_values[indices]

        # Dequantize
        quantized = signs * quantized_abs
        dequantized = quantized * block_max / nvfp4_max

        return dequantized.view(shape)

    # 量化输入和权重
    x_q = quantize_nvfp4_sim(x, block_size)
    w_q = quantize_nvfp4_sim(w, block_size)

    # 量化后的结果
    q_output = torch.matmul(x_q, w_q.T)

    # 计算误差
    abs_error = (ref_output - q_output).abs()
    rel_error = abs_error / (ref_output.abs() + 1e-8)

    print(f"\n矩阵大小: M={M}, K={K}, N={N}, block_size={block_size}")
    print(f"\nFP32 输出统计:")
    print(f"  Mean: {ref_output.mean().item():.4f}")
    print(f"  Std:  {ref_output.std().item():.4f}")
    print(f"  Min:  {ref_output.min().item():.4f}")
    print(f"  Max:  {ref_output.max().item():.4f}")

    print(f"\nNVFP4 量化误差:")
    print(f"  Mean Abs Error: {abs_error.mean().item():.4f}")
    print(f"  Max Abs Error:  {abs_error.max().item():.4f}")
    print(f"  Mean Rel Error: {rel_error.mean().item() * 100:.2f}%")
    print(f"  Max Rel Error:  {rel_error.max().item() * 100:.2f}%")
    print(f"  <1% Error:      {(rel_error < 0.01).float().mean().item() * 100:.1f}%")
    print(f"  <5% Error:      {(rel_error < 0.05).float().mean().item() * 100:.1f}%")
    print(f"  <10% Error:     {(rel_error < 0.10).float().mean().item() * 100:.1f}%")

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_output.flatten().unsqueeze(0),
        q_output.flatten().unsqueeze(0)
    ).item()
    print(f"\n  Cosine Similarity: {cos_sim:.6f}")


def main():
    print("=" * 70)
    print("NVFP4 Scale Factor Reordering Verification")
    print("=" * 70)

    # 1. 布局分析
    analyze_cutlass_layout()

    # 2. 小矩阵测试
    test_small_matrix()

    # 3. FP8 转换测试
    test_fp8_conversion()

    # 4. 精度影响测试
    test_precision_impact()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
下一步:
1. 确定正确的 swizzle 版本 (需要与 CUTLASS binary 输出对比)
2. 将验证后的函数集成到 nvfp4_mlp.py
3. 更新 C++ extension 以使用重排后的 scales
""")


if __name__ == "__main__":
    main()
