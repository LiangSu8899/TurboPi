#!/usr/bin/env python3
"""
比较不同的 scale factor layout 重排
"""

import torch


def compare_layouts():
    """比较不同的 permutation 结果"""
    print("=" * 70)
    print("Comparing Scale Factor Layouts")
    print("=" * 70)

    M = 256
    num_k_blocks = 64

    row_tile = 128
    k_tile = 4
    row_group = 32

    M_padded = ((M + row_tile - 1) // row_tile) * row_tile
    K_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

    num_row_tiles = M_padded // row_tile
    num_k_tiles = K_padded // k_tile
    num_groups = row_tile // row_group

    print(f"Dimensions: M={M}, K_blocks={num_k_blocks}")
    print(f"Tiles: {num_row_tiles} row tiles × {num_k_tiles} k tiles")
    print(f"Groups per tile: {num_groups}")

    # 创建测试 scales
    scales = torch.zeros(M, num_k_blocks)
    for r in range(M):
        for k in range(num_k_blocks):
            scales[r, k] = r * 1000 + k

    scales_reshaped = scales.view(
        num_row_tiles,
        num_groups,
        row_group,
        num_k_tiles,
        k_tile
    )

    # 不同的 permutations
    layouts = {
        'current (0,3,1,2,4)': scales_reshaped.permute(0, 3, 1, 2, 4).contiguous().flatten(),
        'new (3,0,1,2,4)': scales_reshaped.permute(3, 0, 1, 2, 4).contiguous().flatten(),
    }

    # 检查关键位置
    check_ranges = [
        (0, 16, "First group, first k-tile"),
        (128, 144, "Second group, first k-tile"),
        (512, 528, "First group, second k-tile (should differ)"),
        (4096, 4112, "After first 8 k-tiles"),
    ]

    for name, flat in layouts.items():
        print(f"\n--- Layout: {name} ---")
        for start, end, desc in check_ranges:
            print(f"\n{desc} (indices {start}-{end-1}):")
            for i in range(start, min(end, start+8)):
                val = flat[i].item()
                r = int(val // 1000)
                k = int(val % 1000)
                print(f"  [{i:4d}] = row {r:3d}, k {k:2d}")


def try_all_permutations():
    """尝试所有合理的 permutation"""
    print("\n" + "=" * 70)
    print("Testing All Reasonable Permutations")
    print("=" * 70)

    M = 256
    num_k_blocks = 64

    row_tile = 128
    k_tile = 4
    row_group = 32

    num_row_tiles = 2
    num_k_tiles = 16
    num_groups = 4

    # Shape: [num_row_tiles, num_groups, group_size, num_k_tiles, k_tile]
    #       = [2, 4, 32, 16, 4]

    # 创建测试 scales
    scales = torch.zeros(M, num_k_blocks)
    for r in range(M):
        for k in range(num_k_blocks):
            scales[r, k] = r * 1000 + k

    scales_reshaped = scales.view(2, 4, 32, 16, 4)

    # All permutations that keep (row_group=32, k_tile=4) as inner dimensions
    # i.e., last two positions are always 2 and 4
    # We can permute positions 0, 1, 3 (row_tile, group, k_tile)

    from itertools import permutations

    test_perms = []
    for perm in permutations([0, 1, 3]):
        full_perm = list(perm) + [2, 4]
        test_perms.append(tuple(full_perm))

    print(f"\nTesting {len(test_perms)} permutations:")

    for perm in test_perms:
        flat = scales_reshaped.permute(*perm).contiguous().flatten()
        print(f"\n{perm}:")
        # Print first 8 elements
        print("  First 8: ", end="")
        for i in range(8):
            val = flat[i].item()
            r = int(val // 1000)
            k = int(val % 1000)
            print(f"({r},{k}) ", end="")
        print()
        # Print element 512 (should be from k_tile=1)
        val512 = flat[512].item()
        r512 = int(val512 // 1000)
        k512 = int(val512 % 1000)
        print(f"  Elem 512: row={r512}, k={k512} (expected k>=4 for second k_tile)")


if __name__ == "__main__":
    compare_layouts()
    try_all_permutations()
