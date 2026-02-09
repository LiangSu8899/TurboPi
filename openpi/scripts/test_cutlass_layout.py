#!/usr/bin/env python3
"""
精确测试 CUTLASS Scale Factor 布局

基于 sm100_blockscaled_layout.hpp:
  SfKMajorAtom = Layout<
    Shape<Shape<_32,_4>, Shape<Int<SFVecSize>, _4>>,
    Stride<Stride<_16,_4>, Stride<_0, _1>>
  >

解读:
- 外层: 32 rows × 4 k-blocks
- Stride: row*16 + k*4 (在 32×4 块内)
- 内层: SFVecSize × 4 用于向量化 (通常 SFVecSize=4)

关键: 每个 128-row × 4-k tile 由 4 个 32×4 块组成
"""

import torch
import numpy as np


def cutlass_scale_layout_exact(
    scales: torch.Tensor,
    M: int,
    num_k_blocks: int,
    sf_vec_size: int = 4
) -> torch.Tensor:
    """
    精确重现 CUTLASS SfKMajorAtom 布局

    CUTLASS 布局层次:
    1. 整个矩阵按 128-row × 4-k tiles 分块
    2. 每个 tile 内部按 32-row × 4-k 基本块组织
    3. 基本块内部 stride pattern: row*16 + k*4 (但需要紧凑存储)

    实际上由于是紧凑存储，每个 32×4 块的 128 个元素连续存储
    块内元素顺序需要根据 stride 重排
    """
    device = scales.device
    dtype = scales.dtype

    if scales.dim() == 1:
        scales = scales.view(M, num_k_blocks)

    # Tile 参数
    row_tile = 128  # Blk_MN
    k_tile = 4      # Blk_SF
    row_group = 32  # 基本块的行数

    # Padding
    M_padded = ((M + row_tile - 1) // row_tile) * row_tile
    K_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

    if M_padded != M or K_padded != num_k_blocks:
        scales_padded = torch.zeros(M_padded, K_padded, device=device, dtype=dtype)
        scales_padded[:M, :num_k_blocks] = scales
        scales = scales_padded

    num_row_tiles = M_padded // row_tile
    num_k_tiles = K_padded // k_tile

    # 输出
    output = torch.zeros(M_padded * K_padded, device=device, dtype=dtype)

    # CUTLASS tile_to_shape 使用 Step<_2, _1, _3>
    # 这意味着 tiles 按 (M, K, L) 顺序排列
    # 但内部 atom 是 K-major (4 个 k-blocks 连续)

    idx = 0
    for rt in range(num_row_tiles):
        for kt in range(num_k_tiles):
            # 处理一个 128×4 tile (由 4 个 32×4 块组成)
            for group in range(4):  # 4 个 32-row groups
                # 每个 32×4 块内的元素
                # 根据 Stride<_16, _4>: 物理位置 = row * 16 + k * 4
                # 但我们需要紧凑存储，所以需要重新排列

                # 方案 A: 直接按 row, k 顺序存储 (row-major within block)
                # 方案 B: 按 stride 暗示的顺序
                # 方案 C: K-major within block

                for row in range(32):
                    for k in range(4):
                        src_row = rt * row_tile + group * row_group + row
                        src_k = kt * k_tile + k
                        output[idx] = scales[src_row, src_k]
                        idx += 1

    return output


def cutlass_scale_layout_k_major(
    scales: torch.Tensor,
    M: int,
    num_k_blocks: int
) -> torch.Tensor:
    """
    K-major 布局: 每行的 k-blocks 连续存储
    这是 V2 测试中看起来正确的布局
    """
    device = scales.device
    dtype = scales.dtype

    if scales.dim() == 1:
        scales = scales.view(M, num_k_blocks)

    row_tile = 128
    k_tile = 4
    row_group = 32

    M_padded = ((M + row_tile - 1) // row_tile) * row_tile
    K_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

    if M_padded != M or K_padded != num_k_blocks:
        scales_padded = torch.zeros(M_padded, K_padded, device=device, dtype=dtype)
        scales_padded[:M, :num_k_blocks] = scales
        scales = scales_padded

    output = torch.zeros(M_padded * K_padded, device=device, dtype=dtype)

    idx = 0
    for rt in range(M_padded // row_tile):
        for kt in range(K_padded // k_tile):
            for group in range(4):
                for row in range(32):
                    for k in range(4):
                        src_row = rt * row_tile + group * row_group + row
                        src_k = kt * k_tile + k
                        output[idx] = scales[src_row, src_k]
                        idx += 1

    return output


def cutlass_scale_layout_interleaved(
    scales: torch.Tensor,
    M: int,
    num_k_blocks: int
) -> torch.Tensor:
    """
    Interleaved 布局: 基于 Stride<_16, _4> 的真正含义

    Stride<_16, _4> 意味着在 32×4 块内:
    - 元素 (row, k) 的 "逻辑位置" = row * 16 + k * 4
    - 但实际存储是紧凑的，需要按逻辑位置排序

    逻辑位置范围: 0, 4, 8, 12, 16, 20, ..., 496+12=508
    实际每 16 个位置只有 4 个被使用 (k=0,1,2,3)

    所以存储顺序应该是:
    - 位置 0-3: (row=0, k=0..3) 的 4 个元素
    - 位置 4-7: (row=1, k=0..3) 的 4 个元素
    等等...

    这实际上就是 row-major！
    """
    device = scales.device
    dtype = scales.dtype

    if scales.dim() == 1:
        scales = scales.view(M, num_k_blocks)

    row_tile = 128
    k_tile = 4
    row_group = 32

    M_padded = ((M + row_tile - 1) // row_tile) * row_tile
    K_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

    if M_padded != M or K_padded != num_k_blocks:
        scales_padded = torch.zeros(M_padded, K_padded, device=device, dtype=dtype)
        scales_padded[:M, :num_k_blocks] = scales
        scales = scales_padded

    # Stride 排序: 在每个 32×4 块内按 (row*16 + k*4) 排序
    # 创建索引表
    block_indices = []
    for row in range(32):
        for k in range(4):
            logical_pos = row * 16 + k * 4
            block_indices.append((logical_pos, row, k))

    # 按逻辑位置排序
    block_indices.sort(key=lambda x: x[0])

    output = torch.zeros(M_padded * K_padded, device=device, dtype=dtype)

    idx = 0
    for rt in range(M_padded // row_tile):
        for kt in range(K_padded // k_tile):
            for group in range(4):
                # 按 stride 顺序存储块内元素
                for _, row, k in block_indices:
                    src_row = rt * row_tile + group * row_group + row
                    src_k = kt * k_tile + k
                    output[idx] = scales[src_row, src_k]
                    idx += 1

    return output


def test_layouts():
    """测试不同布局版本"""
    print("=" * 70)
    print("CUTLASS Scale Factor Layout Comparison")
    print("=" * 70)

    M = 256
    num_k_blocks = 64

    # 创建测试数据
    scales = torch.zeros(M, num_k_blocks)
    for r in range(M):
        for k in range(num_k_blocks):
            scales[r, k] = r * 1000 + k

    layouts = [
        ("Exact (row-major blocks)", cutlass_scale_layout_exact),
        ("K-major within tiles", cutlass_scale_layout_k_major),
        ("Interleaved (stride sorted)", cutlass_scale_layout_interleaved),
    ]

    for name, func in layouts:
        print(f"\n--- {name} ---")
        reordered = func(scales, M, num_k_blocks)

        print("前 32 个元素 (第一个 32×4 块):")
        for i in range(32):
            val = reordered[i].item()
            orig_r = int(val // 1000)
            orig_k = int(val % 1000)
            print(f"  [{i:2d}] = {val:8.0f} -> (row={orig_r:3d}, k={orig_k:2d})")

        print("\n元素 128-159 (第二个 32×4 块，同一 tile):")
        for i in range(128, 144):
            val = reordered[i].item()
            orig_r = int(val // 1000)
            orig_k = int(val % 1000)
            print(f"  [{i:3d}] = {val:8.0f} -> (row={orig_r:3d}, k={orig_k:2d})")


def analyze_stride_pattern():
    """分析 Stride<_16, _4> 模式"""
    print("\n" + "=" * 70)
    print("Stride<_16, _4> Pattern Analysis for 32×4 Block")
    print("=" * 70)

    print("\n逻辑位置 = row * 16 + k * 4:")
    print("=" * 40)

    # 计算所有逻辑位置
    positions = []
    for row in range(32):
        for k in range(4):
            pos = row * 16 + k * 4
            positions.append((pos, row, k))

    # 按逻辑位置排序
    positions.sort()

    print("\n按逻辑位置排序后的前 16 个元素:")
    for i, (pos, row, k) in enumerate(positions[:16]):
        print(f"  存储位置 {i:2d}: 逻辑位置 {pos:3d} -> (row={row:2d}, k={k})")

    print("\n模式分析:")
    # 检查是否是简单的 row-major
    is_row_major = all(positions[i] == (i * 4, i // 4 * (i % 4 == 0), i % 4) for i in range(len(positions)))

    print("  逻辑位置序列的前 16 个:", [p[0] for p in positions[:16]])
    print("  逻辑位置序列步长:", positions[1][0] - positions[0][0])

    # 实际上 stride pattern 说明:
    # - k 增加 1，位置增加 4
    # - row 增加 1，位置增加 16
    # 所以位置序列是: 0, 4, 8, 12, 16, 20, 24, 28, ...
    # 这意味着: k=0,1,2,3 然后 row++ 重复
    # 等价于 K-major (每行的 4 个 k 连续，然后下一行)

    print("\n结论: Stride<_16, _4> 表示 K-major 存储")
    print("  - 每行的 4 个 k-blocks 连续存储")
    print("  - 然后是下一行的 4 个 k-blocks")


def test_simple_identity():
    """简单测试：如果输入已经是正确布局会怎样"""
    print("\n" + "=" * 70)
    print("Identity Test: 检查是否需要重排")
    print("=" * 70)

    M = 128  # 1 个 row tile
    num_k_blocks = 4  # 1 个 k tile

    # 简单的 row-major 输入
    scales_rm = torch.arange(M * num_k_blocks).float().view(M, num_k_blocks)
    print(f"\nRow-major 输入 scales[0:4, 0:4]:")
    print(scales_rm[:4, :4])

    # K-major 重排
    output = cutlass_scale_layout_k_major(scales_rm, M, num_k_blocks)
    output_view = output.view(4, 32, 4)  # 4 groups × 32 rows × 4 k

    print(f"\n重排后 (4 groups × 32 rows × 4 k):")
    print(f"Group 0, rows 0-3:")
    for g in range(1):
        for r in range(4):
            print(f"  Group {g}, row {r}: {output_view[g, r].tolist()}")

    # 验证: 如果是 K-major，则 scales_rm[row, :] 应该是连续的
    print("\n验证 K-major:")
    for r in range(4):
        expected = scales_rm[r, :].tolist()
        # 在输出中找到 row r 的元素
        actual = output_view[0, r, :].tolist()
        match = expected == actual
        print(f"  Row {r}: expected {expected}, got {actual}, match={match}")


def create_final_reorder_function():
    """创建最终的重排函数"""
    print("\n" + "=" * 70)
    print("Final Reorder Function")
    print("=" * 70)

    print("""
def swizzle_scales_for_cutlass(scales, M, num_k_blocks):
    '''
    将 row-major scales 重排为 CUTLASS K-major tile 布局

    布局层次:
    1. 矩阵按 128-row × 4-k tiles 分块
    2. 每个 tile 内部按 32-row groups 组织
    3. 每个 group 内，同一行的 4 个 k-blocks 连续 (K-major)

    输入: [M, num_k_blocks] row-major
    输出: flat tensor, 按 tile -> group -> row -> k 顺序
    '''
    device = scales.device
    dtype = scales.dtype

    row_tile = 128
    k_tile = 4
    row_group = 32

    # Padding
    M_padded = ((M + row_tile - 1) // row_tile) * row_tile
    K_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

    if M_padded != M or K_padded != num_k_blocks:
        scales_padded = torch.zeros(M_padded, K_padded, device=device, dtype=dtype)
        scales_padded[:M, :num_k_blocks] = scales
        scales = scales_padded

    # Reshape: [M, K] -> [num_row_tiles, 4_groups, 32_rows, num_k_tiles, 4_k]
    scales = scales.view(
        M_padded // row_tile,  # num_row_tiles
        4,                      # groups per row_tile (128/32)
        row_group,              # rows per group
        K_padded // k_tile,    # num_k_tiles
        k_tile                  # k per tile
    )

    # Permute to: [num_row_tiles, num_k_tiles, 4_groups, 32_rows, 4_k]
    # 这样 tiles 在最外层，然后 groups，最后 row 和 k
    scales = scales.permute(0, 3, 1, 2, 4)

    return scales.contiguous().flatten()
""")


if __name__ == "__main__":
    analyze_stride_pattern()
    test_layouts()
    test_simple_identity()
    create_final_reorder_function()
