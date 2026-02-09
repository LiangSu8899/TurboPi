#!/usr/bin/env python3
"""
精确验证 swizzle 函数的正确性
"""

import torch
import sys

sys.path.insert(0, '/workspace/src')

from openpi.models_pytorch.nvfp4_mlp import (
    swizzle_scales_for_cutlass,
    CUTLASS_ROW_TILE,
    CUTLASS_K_TILE,
    CUTLASS_ROW_GROUP,
)


def simulate_cutlass_expected_layout(total_rows, num_k_blocks):
    """
    模拟 CUTLASS tile_to_shape(SfAtom{}, make_shape(M/N, K, L), Step<_2,_1,_3>{}) 的布局

    Step<_2,_1,_3> 意味着:
    - L (batch, _3) 最慢
    - K tiles (_1) 次之
    - M/N tiles (_2) 最快

    在每个 tile 内部 (128 rows × 4 k-blocks):
    - 4 个 groups，每个 32 rows
    - 每个 group 内: rows 变化，然后 k 变化
    """
    row_tile = CUTLASS_ROW_TILE  # 128
    k_tile = CUTLASS_K_TILE      # 4
    row_group = CUTLASS_ROW_GROUP  # 32

    M_padded = ((total_rows + row_tile - 1) // row_tile) * row_tile
    K_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

    num_row_tiles = M_padded // row_tile
    num_k_tiles = K_padded // k_tile
    num_groups = row_tile // row_group

    positions = []

    # Step<_2,_1,_3>: L -> K tiles -> M/N tiles
    for l in range(1):  # L = 1
        for k_tile_idx in range(num_k_tiles):
            for m_tile_idx in range(num_row_tiles):
                for group_idx in range(num_groups):
                    for row_in_group in range(row_group):
                        for k_in_tile in range(k_tile):
                            row = m_tile_idx * row_tile + group_idx * row_group + row_in_group
                            k = k_tile_idx * k_tile + k_in_tile
                            positions.append((row, k))

    return positions


def test_swizzle_correctness():
    """测试 swizzle 函数输出是否匹配 CUTLASS 预期"""
    print("=" * 60)
    print("Swizzle Correctness Test")
    print("=" * 60)

    total_rows = 256
    num_k_blocks = 64

    # 创建测试 scales
    scales = torch.zeros(total_rows, num_k_blocks)
    for r in range(total_rows):
        for k in range(num_k_blocks):
            scales[r, k] = r * 1000 + k  # Encode (row, k) as unique value

    # Python swizzle
    swizzled = swizzle_scales_for_cutlass(scales, total_rows, num_k_blocks)

    # CUTLASS 预期
    expected_positions = simulate_cutlass_expected_layout(total_rows, num_k_blocks)

    print(f"Swizzled size: {swizzled.numel()}")
    print(f"Expected positions: {len(expected_positions)}")

    # 比较
    matches = 0
    first_mismatch_idx = -1

    for i in range(min(len(expected_positions), len(swizzled))):
        py_val = swizzled[i].item()
        py_r = int(py_val // 1000)
        py_k = int(py_val % 1000)

        exp_r, exp_k = expected_positions[i]

        if py_r == exp_r and py_k == exp_k:
            matches += 1
        elif first_mismatch_idx == -1:
            first_mismatch_idx = i
            print(f"\nFirst mismatch at index {i}:")
            print(f"  Python: (row={py_r}, k={py_k})")
            print(f"  Expected: (row={exp_r}, k={exp_k})")

    total = len(expected_positions)
    pct = matches / total * 100
    print(f"\nMatches: {matches}/{total} ({pct:.1f}%)")

    if matches == total:
        print("✓ Swizzle function is CORRECT!")
        return True
    else:
        print("✗ Swizzle function has MISMATCHES")
        return False


def test_with_gemm():
    """使用正确 swizzle 的 GEMM 测试"""
    print("\n" + "=" * 60)
    print("GEMM with Correct Swizzle Test")
    print("=" * 60)

    import nvfp4_gemm
    from openpi.models_pytorch.nvfp4_mlp import (
        quantize_to_nvfp4_sim,
        prepare_scales_for_cutlass,
        pack_nvfp4_data,
        BLOCK_SIZE,
    )

    M, K, N = 256, 2048, 256  # 使用小 N 先测试
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # 创建统一数据
    x = torch.ones(M, K, device='cuda', dtype=torch.float32)
    w = torch.ones(N, K, device='cuda', dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    # 打包
    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # 准备 scales
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True)

    print(f"x_scales_fp8 size: {x_scales_fp8.numel()}")
    print(f"w_scales_fp8 size: {w_scales_fp8.numel()}")

    try:
        output = nvfp4_gemm.gemm_prepared(
            x_packed.cuda(),
            w_packed.cuda(),
            x_scales_fp8.cuda(),
            w_scales_fp8.cuda(),
            M, N, K
        )

        expected = K  # ones @ ones = K

        print(f"\nOutput shape: {output.shape}")
        print(f"Output[0, 0]: {output[0, 0].item():.1f}")
        print(f"Output[0, -1]: {output[0, -1].item():.1f}")
        print(f"Output mean: {output.mean().item():.1f}")
        print(f"Expected: {expected}")

        # 检查全部元素
        close_count = (torch.abs(output - expected) < 500).sum().item()
        total = output.numel()
        print(f"\nClose to expected: {close_count}/{total} ({close_count/total*100:.1f}%)")

        return close_count / total > 0.9

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_larger_n():
    """测试更大的 N"""
    print("\n" + "=" * 60)
    print("Larger N Test")
    print("=" * 60)

    import nvfp4_gemm
    from openpi.models_pytorch.nvfp4_mlp import (
        quantize_to_nvfp4_sim,
        prepare_scales_for_cutlass,
        pack_nvfp4_data,
        BLOCK_SIZE,
    )

    M, K = 256, 2048
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    for N in [256, 512, 1024, 2048, 4096, 8192]:
        print(f"\n--- N = {N} ---")

        x = torch.ones(M, K, device='cuda', dtype=torch.float32)
        w = torch.ones(N, K, device='cuda', dtype=torch.float32)

        x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
        w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

        x_packed = pack_nvfp4_data(x_q, block_size)
        w_packed = pack_nvfp4_data(w_q, block_size)

        x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True)
        w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True)

        try:
            output = nvfp4_gemm.gemm_prepared(
                x_packed.cuda(),
                w_packed.cuda(),
                x_scales_fp8.cuda(),
                w_scales_fp8.cuda(),
                M, N, K
            )

            expected = K
            out_first = output[0, 0].item()
            out_last = output[0, -1].item()
            out_mean = output.mean().item()

            status = "OK" if (abs(out_first - expected) < 500 and abs(out_last - expected) < 500) else "FAIL"
            print(f"  [0,0]={out_first:.0f}, [0,-1]={out_last:.0f}, mean={out_mean:.0f} [{status}]")

        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    print("Swizzle Correctness Verification")
    print("=" * 60)

    # 先验证 swizzle 函数本身
    swizzle_ok = test_swizzle_correctness()

    if swizzle_ok:
        # 然后测试 GEMM
        gemm_ok = test_with_gemm()
        test_larger_n()
    else:
        print("\nSwizzle function incorrect, skipping GEMM tests")


if __name__ == "__main__":
    main()
