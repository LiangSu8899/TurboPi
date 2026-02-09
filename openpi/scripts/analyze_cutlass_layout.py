#!/usr/bin/env python3
"""
分析 CUTLASS Scale Factor Layout

CUTLASS 的 scale factor 布局由 Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA 定义。
我们需要精确匹配这个布局。

从 sm100_blockscaled_layout.hpp:

SfAtomM = 128
SfAtomK = 4
SfVecSize = 4

SfKMajorAtom = Layout<
    Shape<Shape<_32,_4>, Shape<Int<SfVecSize>, _4>>,
    Stride<Stride<_16,_4>, Stride<_0, _1>>
>

这个布局的含义:
- Shape: [[32, 4], [4, 4]]
  - 外层: 32 rows × 4 k-blocks (一个 group)
  - 内层: 4 × 4 向量化 (SfVecSize)

- Stride: [[16, 4], [0, 1]]
  - 外层 stride: row * 16, k * 4
  - 内层 stride: 0 (broadcast), 1 (contiguous)

关键: 内层的 _0 stride 意味着向量化是 broadcast，不是真正的数据
所以实际上每个 32×4 group 只有 128 个 scale factors

tile_to_shape uses Step<_2, _1, _3> which is (M, K, L) order
"""

import torch
import numpy as np


def analyze_layout():
    """分析 CUTLASS 布局"""
    print("=" * 70)
    print("CUTLASS Scale Factor Layout Analysis")
    print("=" * 70)

    # 参数
    SfAtomM = 128  # 128 rows per tile
    SfAtomK = 4    # 4 k-blocks per tile
    SfVecSize = 4  # Vector size
    GroupRows = 32  # Rows per group

    print(f"\nParameters:")
    print(f"  SfAtomM = {SfAtomM}")
    print(f"  SfAtomK = {SfAtomK}")
    print(f"  SfVecSize = {SfVecSize}")
    print(f"  GroupRows = {GroupRows}")

    # 问题维度
    M = 256
    K_blocks = 64

    # Tile 计算
    num_m_tiles = (M + SfAtomM - 1) // SfAtomM
    num_k_tiles = (K_blocks + SfAtomK - 1) // SfAtomK
    num_groups = SfAtomM // GroupRows  # 4 groups per M tile

    print(f"\nProblem: M={M}, K_blocks={K_blocks}")
    print(f"  num_m_tiles = {num_m_tiles}")
    print(f"  num_k_tiles = {num_k_tiles}")
    print(f"  num_groups = {num_groups}")

    # 每个 tile 的 scale factors
    # 一个 128×4 tile = 4 groups × 32 rows × 4 k = 512 scales
    scales_per_tile = SfAtomM * SfAtomK
    total_scales = num_m_tiles * num_k_tiles * scales_per_tile

    print(f"\nScale factor counts:")
    print(f"  per tile = {scales_per_tile}")
    print(f"  total = {total_scales}")

    # 现在分析 stride 模式
    print("\n" + "-" * 50)
    print("Stride Analysis: Stride<Stride<_16,_4>, Stride<_0,_1>>")
    print("-" * 50)

    # 创建索引映射表
    # 对于一个 32×4 group:
    # position = outer_row * 16 + outer_k * 4 + inner_vec * 0 + inner_k * 1
    # = outer_row * 16 + outer_k * 4 + inner_k

    # 但 inner 的 shape 是 [4, 4]，stride 是 [0, 1]
    # 这意味着 inner 维度只有 4 个有效位置 (inner_k)

    print("\nFor a 32×4 group, computing positions:")
    positions = []
    for row in range(32):
        for k in range(4):
            # inner dimensions: vec (size 4, stride 0) × k (size 4, stride 1)
            # This is confusing - let me re-read the layout
            # Shape<Shape<_32,_4>, Shape<Int<SfVecSize>, _4>>
            # = Shape<[32, 4], [4, 4]>
            # Total elements = 32 * 4 * 4 * 4 = 2048?

            # Actually, looking at the Stride:
            # Stride<Stride<_16,_4>, Stride<_0, _1>>
            # The inner Stride<_0, _1> has a 0, which means broadcast
            # So the SfVecSize dimension is broadcast (stride 0)
            # Only the inner _4 dimension with stride 1 is real

            # This means for each (row, k) in outer [32, 4]:
            # pos = row * 16 + k * 4 + vec * 0 + inner_k * 1
            # = row * 16 + k * 4 + inner_k

            # But inner_k goes 0-3, so for each outer (row, k) we have 4 inner positions
            # Wait, that can't be right either - that would give 32*4*4 = 512 positions

            # Let me think about this differently...
            # A 32×4 group has 128 scale factors (32 rows × 4 k-blocks)
            # The stride pattern tells us how they're laid out in memory

            # If we ignore the inner dimensions (which seem to be for vectorized access),
            # the core layout is just:
            # pos = row * 16 + k * 4

            # But that gives positions 0,4,8,12,...,508
            # which spans [0, 508+3=511] but only 128 positions are used

            pos = row * 16 + k * 4
            positions.append((row, k, pos))

    # Sort by position
    positions.sort(key=lambda x: x[2])

    print("First 16 positions (sorted by memory position):")
    for row, k, pos in positions[:16]:
        print(f"  pos={pos:3d}: row={row:2d}, k={k}")

    # Check the pattern
    print("\nPosition sequence:", [p[2] for p in positions[:16]])

    # Hmm, the positions are 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, ...
    # which is: k=0 for all rows, then k=1 for all rows, etc.
    # That's K-major within the 32×4 block!

    print("\n" + "-" * 50)
    print("Deduced Layout Pattern")
    print("-" * 50)

    # Actually, let me reconsider:
    # pos = row * 16 + k * 4
    # For row=0: k=0->0, k=1->4, k=2->8, k=3->12
    # For row=1: k=0->16, k=1->20, k=2->24, k=3->28
    # For row=2: k=0->32, k=1->36, k=2->40, k=3->44
    # ...

    # Sorted by position:
    # 0 (r=0,k=0), 4 (r=0,k=1), 8 (r=0,k=2), 12 (r=0,k=3),
    # 16 (r=1,k=0), 20 (r=1,k=1), ...

    # This is ROW-MAJOR! Not K-major!
    # Each row's 4 k values are adjacent (0,4,8,12), then next row (16,20,24,28)

    print("The stride pattern row*16 + k*4 gives ROW-MAJOR order!")
    print("  - For each row (0-31):")
    print("    - k=0,1,2,3 give positions 0,4,8,12 relative to row")
    print("  - Row 0: [0, 4, 8, 12]")
    print("  - Row 1: [16, 20, 24, 28]")
    print("  - ...")

    # But wait, the positions are not contiguous (gaps of 4)
    # This is for vectorized access - real storage is contiguous

    print("\n" + "-" * 50)
    print("CUTLASS Tiled Layout Order")
    print("-" * 50)

    # Based on tile_to_shape with Step<_2, _1, _3>:
    # This means (M-tiles, K-tiles, L) are ordered as (first, second, third)
    # So tiles are ordered: M-major (M tiles iterate first), then K tiles

    print("Tile order: Step<_2, _1, _3> = (M, K, L)")
    print("  - Outer loop: L (batch, usually 1)")
    print("  - Middle loop: K tiles")
    print("  - Inner loop: M tiles")
    print("")
    print("So the layout is:")
    print("  for l in L:")
    print("    for k_tile in num_k_tiles:")
    print("      for m_tile in num_m_tiles:")
    print("        for group in num_groups:")  # 4 groups per M tile
    print("          for row in 32:")
    print("            for k in 4:")
    print("              store scale[m_tile*128 + group*32 + row, k_tile*4 + k]")


def generate_correct_reorder():
    """生成正确的重排函数"""
    print("\n" + "=" * 70)
    print("Correct Reorder Function")
    print("=" * 70)

    # 根据上面的分析，正确的重排顺序应该是:
    # K-tiles 在 M-tiles 之前 (Step<_2, _1> 表示 M first, then K)
    # 但我们的 Python 函数是 permute(0, 3, 1, 2, 4)
    # 原始: [num_row_tiles, num_groups, group_size, num_k_tiles, k_tile]
    # 变为: [num_row_tiles, num_k_tiles, num_groups, group_size, k_tile]
    # 这是 row_tile -> k_tile -> group -> row -> k

    # 但根据 CUTLASS Step<_2, _1, _3>:
    # _2 是 M-minor (内层), _1 是 K-middle, _3 是 L-major (外层)
    # 所以顺序应该是: L -> K -> M
    # 即: k_tile -> row_tile -> group -> row -> k

    print("Current Python permutation: (0, 3, 1, 2, 4)")
    print("  = row_tile -> k_tile -> group -> row -> k")
    print("")
    print("Based on Step<_2, _1, _3>:")
    print("  _1 = K is middle, _2 = M is minor (inner)")
    print("  Order should be: L -> K -> M")
    print("  = k_tile -> row_tile -> group -> row -> k")
    print("  = permute(3, 0, 1, 2, 4)")  # k_tile first, then row_tile


def test_new_reorder():
    """测试新的重排顺序"""
    print("\n" + "=" * 70)
    print("Testing New Reorder")
    print("=" * 70)

    M = 256
    num_k_blocks = 64
    block_size = 32

    row_tile = 128
    k_tile = 4
    row_group = 32

    # 创建测试 scales (每个位置一个唯一值)
    scales = torch.zeros(M, num_k_blocks)
    for r in range(M):
        for k in range(num_k_blocks):
            scales[r, k] = r * 1000 + k

    # 当前重排
    M_padded = ((M + row_tile - 1) // row_tile) * row_tile
    K_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

    num_row_tiles = M_padded // row_tile
    num_k_tiles = K_padded // k_tile
    num_groups = row_tile // row_group

    scales_reshaped = scales.view(
        num_row_tiles,
        num_groups,
        row_group,
        num_k_tiles,
        k_tile
    )

    # 当前 permutation
    current = scales_reshaped.permute(0, 3, 1, 2, 4).contiguous().flatten()

    # 新 permutation (k_tile first)
    new = scales_reshaped.permute(3, 0, 1, 2, 4).contiguous().flatten()

    print(f"Original scales shape: {scales.shape}")
    print(f"Reshaped: {scales_reshaped.shape}")

    print(f"\nCurrent permute(0,3,1,2,4) - first 16:")
    for i in range(16):
        val = current[i].item()
        r = int(val // 1000)
        k = int(val % 1000)
        print(f"  [{i:2d}] = row {r:3d}, k {k:2d}")

    print(f"\nNew permute(3,0,1,2,4) - first 16:")
    for i in range(16):
        val = new[i].item()
        r = int(val // 1000)
        k = int(val % 1000)
        print(f"  [{i:2d}] = row {r:3d}, k {k:2d}")


if __name__ == "__main__":
    analyze_layout()
    generate_correct_reorder()
    test_new_reorder()
