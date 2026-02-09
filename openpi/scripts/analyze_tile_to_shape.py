#!/usr/bin/env python3
"""
分析 CUTLASS tile_to_shape 布局

CUTLASS 使用:
  tile_to_shape(SfAtom{}, make_shape(M/N, K, L), Step<_2,_1,_3>{})

SfAtom = Layout<
    Shape<Shape<_32,_4>, Shape<SFVecSize, _4>>,
    Stride<Stride<_16,_4>, Stride<_0, _1>>
>

Step<_2,_1,_3> 表示维度排列:
  - _1 = K (fastest changing)
  - _2 = M/N (middle)
  - _3 = L (slowest, usually 1)
"""

import torch


def simulate_cutlass_layout(total_rows, num_k_blocks, row_tile=128, k_tile=4, row_group=32):
    """
    模拟 CUTLASS tile_to_shape 的布局

    SfAtom 形状: [[32, 4], [SFVecSize, 4]] = [[32, 4], [4, 4]]
    但实际上 SFVecSize 维度的 stride 是 0 (broadcast)，
    所以有效形状是 [32, 4] per group
    """
    # Padding
    M_padded = ((total_rows + row_tile - 1) // row_tile) * row_tile
    K_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

    num_row_tiles = M_padded // row_tile
    num_k_tiles = K_padded // k_tile
    num_groups = row_tile // row_group  # 4 groups per tile

    print(f"Rows: {total_rows} -> {M_padded} (padded)")
    print(f"K blocks: {num_k_blocks} -> {K_padded} (padded)")
    print(f"Tiles: {num_row_tiles} row tiles × {num_k_tiles} k tiles")
    print(f"Groups per tile: {num_groups}")

    # 创建位置映射
    # 每个位置存储 (row, k_block) 对
    positions = []

    # Step<_2, _1, _3> 意味着:
    # L (batch) 最慢, K tiles 次之, M/N tiles 最快
    # 但在每个 tile 内部，使用 SfAtom 的 stride 模式

    for l in range(1):  # L = 1
        for k_tile_idx in range(num_k_tiles):  # K tiles (Step _1 = middle)
            for m_tile_idx in range(num_row_tiles):  # M/N tiles (Step _2 = inner)
                for group_idx in range(num_groups):  # 4 groups per M tile
                    for row_in_group in range(row_group):  # 32 rows per group
                        for k_in_tile in range(k_tile):  # 4 k-blocks per tile
                            row = m_tile_idx * row_tile + group_idx * row_group + row_in_group
                            k = k_tile_idx * k_tile + k_in_tile
                            if row < total_rows and k < num_k_blocks:
                                positions.append((row, k))

    return positions


def compare_with_python_swizzle(total_rows, num_k_blocks, row_tile=128, k_tile=4, row_group=32):
    """比较 CUTLASS 布局与 Python swizzle"""

    print("\n" + "=" * 60)
    print("Comparing Layouts")
    print("=" * 60)

    # Python swizzle (当前实现)
    def python_swizzle(scales, M, num_k_blocks, row_tile=128, k_tile=4, row_group=32):
        M_padded = ((M + row_tile - 1) // row_tile) * row_tile
        K_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

        if M_padded != M or K_padded != num_k_blocks:
            scales_padded = torch.zeros(M_padded, K_padded)
            scales_padded[:M, :num_k_blocks] = scales
            scales = scales_padded

        num_row_tiles = M_padded // row_tile
        num_k_tiles = K_padded // k_tile
        num_groups = row_tile // row_group

        scales = scales.view(
            num_row_tiles,
            num_groups,
            row_group,
            num_k_tiles,
            k_tile
        )

        # Current permutation: (0, 3, 1, 2, 4)
        # = [row_tile, k_tile, group, row, k]
        scales = scales.permute(0, 3, 1, 2, 4)
        return scales.flatten()

    # 创建测试 scales: 每个位置一个唯一值
    scales = torch.zeros(total_rows, num_k_blocks)
    for r in range(total_rows):
        for k in range(num_k_blocks):
            scales[r, k] = r * 1000 + k  # Encode position

    # Python swizzle 结果
    py_result = python_swizzle(scales, total_rows, num_k_blocks)

    # CUTLASS 模拟结果
    cutlass_positions = simulate_cutlass_layout(total_rows, num_k_blocks)

    print(f"\nPython swizzle: {py_result.shape}")
    print(f"CUTLASS positions: {len(cutlass_positions)}")

    # 比较前 32 个位置
    print("\nFirst 32 positions comparison:")
    print("Index | Python (row, k) | CUTLASS (row, k) | Match")
    print("-" * 55)

    mismatches = 0
    for i in range(min(32, len(cutlass_positions))):
        py_val = py_result[i].item()
        py_r = int(py_val // 1000)
        py_k = int(py_val % 1000)

        cut_r, cut_k = cutlass_positions[i]

        match = "✓" if (py_r == cut_r and py_k == cut_k) else "✗"
        if match == "✗":
            mismatches += 1

        print(f"{i:5d} | ({py_r:3d}, {py_k:2d})      | ({cut_r:3d}, {cut_k:2d})        | {match}")

    print(f"\nMismatches in first 32: {mismatches}")

    # 检查 tile 边界附近
    print("\n\nAround tile boundary (index 512 = 128 rows × 4 k-blocks):")
    for i in [508, 509, 510, 511, 512, 513, 514, 515]:
        if i < len(cutlass_positions) and i < len(py_result):
            py_val = py_result[i].item()
            py_r = int(py_val // 1000)
            py_k = int(py_val % 1000)
            cut_r, cut_k = cutlass_positions[i]
            match = "✓" if (py_r == cut_r and py_k == cut_k) else "✗"
            print(f"{i:5d} | ({py_r:3d}, {py_k:2d})      | ({cut_r:3d}, {cut_k:2d})        | {match}")


def test_different_orderings():
    """测试不同的 permute 顺序"""
    print("\n" + "=" * 60)
    print("Testing Different Permute Orderings")
    print("=" * 60)

    total_rows = 256
    num_k_blocks = 64
    row_tile = 128
    k_tile = 4
    row_group = 32

    # CUTLASS 预期布局
    cutlass_positions = simulate_cutlass_layout(total_rows, num_k_blocks)

    # 创建测试 scales
    scales = torch.zeros(total_rows, num_k_blocks)
    for r in range(total_rows):
        for k in range(num_k_blocks):
            scales[r, k] = r * 1000 + k

    M_padded = 256
    K_padded = 64
    num_row_tiles = 2
    num_k_tiles = 16
    num_groups = 4

    scales_reshaped = scales.view(num_row_tiles, num_groups, row_group, num_k_tiles, k_tile)

    # 尝试不同的 permutations
    from itertools import permutations

    # 保持最后两个维度 (row_in_group, k_in_tile) 不变
    # 只交换前三个维度 (row_tile, group, k_tile)
    best_match = 0
    best_perm = None

    for perm in permutations([0, 1, 3]):
        full_perm = list(perm) + [2, 4]
        result = scales_reshaped.permute(*full_perm).flatten()

        matches = 0
        for i in range(min(len(cutlass_positions), len(result))):
            py_val = result[i].item()
            py_r = int(py_val // 1000)
            py_k = int(py_val % 1000)
            cut_r, cut_k = cutlass_positions[i]
            if py_r == cut_r and py_k == cut_k:
                matches += 1

        pct = matches / len(cutlass_positions) * 100
        print(f"permute{full_perm}: {matches}/{len(cutlass_positions)} ({pct:.1f}%)")

        if matches > best_match:
            best_match = matches
            best_perm = full_perm

    print(f"\nBest: permute{best_perm} with {best_match} matches")


def main():
    print("CUTLASS tile_to_shape Layout Analysis")
    print("=" * 60)

    print("\n--- Simulating CUTLASS layout for M=256, K_blocks=64 ---")
    positions = simulate_cutlass_layout(256, 64)
    print(f"Total positions: {len(positions)}")

    # Show first few positions
    print("\nFirst 16 positions (row, k):")
    for i, (r, k) in enumerate(positions[:16]):
        print(f"  [{i:2d}] = ({r:3d}, {k:2d})")

    compare_with_python_swizzle(256, 64)
    test_different_orderings()


if __name__ == "__main__":
    main()
