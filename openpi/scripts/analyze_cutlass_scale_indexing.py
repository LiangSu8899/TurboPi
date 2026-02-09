#!/usr/bin/env python3
"""
分析 CUTLASS 的 scale 索引模式

CUTLASS 使用 tile_to_shape(SfAtom, make_shape(M/N, K, L), Step<_2,_1,_3>)
来生成 scale 的 layout。

SfAtom = Layout<
    Shape<Shape<_32, _4>, Shape<SFVecSize, _4>>,
    Stride<Stride<_16, _4>, Stride<_0, _1>>
>

这个 layout 定义了如何从逻辑坐标 (row, k_block) 映射到内存偏移。

关键：SFVecSize 维度有 stride=0（broadcast），所以实际存储的元素数 = 32 * 4 = 128 per atom。
但 CUTLASS 在读取时可能期望每 K 元素一个 scale（不是每 K/32 个 block 一个 scale）？

让我分析 tile_to_shape 的行为...
"""

import torch
import sys

sys.path.insert(0, '/workspace/src')


def simulate_cutlass_scale_access(M, K, block_size=32):
    """
    模拟 CUTLASS 如何访问 scale factors

    对于 SfAtom = Layout<Shape<Shape<_32,_4>,...>, Stride<Stride<_16,_4>,...>>:
    - 内层 shape [32, 4] 对应 32 rows × 4 k-positions
    - stride [16, 4] 意味着: offset = row * 16 + k * 4

    但这里的 k 是什么？是 k-blocks (每个 32 元素) 还是 k-tiles (每个 4 k-blocks)?
    """
    num_k_blocks = K // block_size

    # SfAtom 参数
    atom_rows = 32
    atom_k = 4  # 可能是 k-blocks 或 k-elements/32

    # 假设 1: k 是 k-blocks (每个 32 元素)
    print("=== Hypothesis 1: k is k-blocks ===")
    print(f"For M={M}, K={K}, block_size={block_size}")
    print(f"num_k_blocks = {num_k_blocks}")

    # 计算需要多少个 atoms
    num_row_atoms = (M + atom_rows - 1) // atom_rows  # 256 / 32 = 8
    num_k_atoms = (num_k_blocks + atom_k - 1) // atom_k  # 64 / 4 = 16

    print(f"num_row_atoms = {num_row_atoms}")
    print(f"num_k_atoms = {num_k_atoms}")

    # 每个 atom 的大小（根据 stride 计算）
    # 最大 offset = (32-1) * 16 + (4-1) * 4 = 31*16 + 3*4 = 496 + 12 = 508
    # 加上 1 = 509，但实际元素只有 32 * 4 = 128
    # 这说明 stride 不是连续的

    # 实际上，stride [16, 4] 表示：
    # row 0, k 0: offset 0
    # row 0, k 1: offset 4
    # row 0, k 2: offset 8
    # row 0, k 3: offset 12
    # row 1, k 0: offset 16
    # row 1, k 1: offset 20
    # ...

    # 这是 row-major 布局，每行 16 个位置（但只用了 4 个 k）
    # 总共 32 * 16 = 512 个位置，但只有 128 个实际使用

    # 让我重新理解...
    # SfAtom 形状: [[32, 4], [SFVecSize, 4]]
    # 这是一个嵌套的 shape
    # 外层 [32, 4] 的 stride 是 [16, 4]
    # 内层 [SFVecSize, 4] 的 stride 是 [0, 1]

    # 内层 stride [0, 1] 意味着 SFVecSize 维度是 broadcast
    # 所以内层只有 4 个有效元素 (stride 1 的那个维度)

    # 总的 stride 计算：
    # offset = outer_row * 16 + outer_k * 4 + inner_vec * 0 + inner_k * 1
    # = outer_row * 16 + outer_k * 4 + inner_k

    # 对于每个 (outer_row, outer_k)，有 4 个 inner_k 值 (0-3)
    # 所以每个 atom 有 32 * 4 * 4 = 512 个逻辑元素
    # 但由于 inner_vec 的 stride=0，实际只有 32 * 4 * 4 / 4 = 128 个物理元素

    print("\n=== Understanding the nested layout ===")
    print("SfAtom Shape: [[32, 4], [SFVecSize=4, 4]]")
    print("SfAtom Stride: [[16, 4], [0, 1]]")
    print("")
    print("For a given (row, k_block) in the scale matrix:")
    print("  - outer_row = row % 32")
    print("  - outer_k = k_block % 4")
    print("  - atom_index = (row // 32) + (k_block // 4) * num_row_atoms")
    print("  - offset within atom = outer_row * 16 + outer_k * 4 + inner_k")
    print("")
    print("But inner_k is 0-3, which adds 4 extra elements per (row, k_block)")
    print("This suggests CUTLASS expects 4 scale values per (row, k_block) pair")
    print("= 4 scales per 32 elements of input (SFVecSize = 4)")


def test_scale_indexing_pattern():
    """测试不同的 scale 索引模式"""
    print("\n" + "=" * 60)
    print("Testing Scale Indexing Patterns")
    print("=" * 60)

    import nvfp4_gemm
    from openpi.models_pytorch.nvfp4_mlp import (
        quantize_to_nvfp4_sim,
        dequantize_nvfp4_sim,
        pack_nvfp4_data,
        convert_scales_to_fp8,
        BLOCK_SIZE,
        CUTLASS_ROW_TILE,
    )
    import torch.nn.functional as F

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

    # 不同的 scale 布局方法
    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((K + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE

    # 方法 1: repeat_interleave block_size (当前方法)
    def method1(scales, rows, K):
        num_k_blocks = K // block_size
        scales = scales.cuda()
        scales_expanded = scales.repeat_interleave(block_size, dim=1)
        if scales_expanded.shape[1] < K_padded:
            extra = K_padded - scales_expanded.shape[1]
            scales_expanded = torch.cat([
                scales_expanded,
                torch.zeros(rows, extra, device='cuda', dtype=scales.dtype)
            ], dim=1)
        return convert_scales_to_fp8(scales_expanded.flatten())

    # 方法 2: 只按 k-blocks 排列（不扩展）
    def method2(scales, rows, K):
        scales = scales.cuda()
        K_blocks_padded = 64
        scales_padded = torch.zeros(rows, K_blocks_padded, device='cuda', dtype=scales.dtype)
        scales_padded[:scales.shape[0], :scales.shape[1]] = scales
        flat = scales_padded.flatten()
        # Pad to K_padded size
        if flat.numel() < rows * K_padded:
            flat = torch.cat([flat, torch.zeros(rows * K_padded - flat.numel(), device='cuda', dtype=flat.dtype)])
        return convert_scales_to_fp8(flat)

    methods = [
        ("repeat_interleave(32)", method1),
        ("k-blocks only", method2),
    ]

    for name, method in methods:
        print(f"\n--- {name} ---")

        x_scales_fp8 = method(x_scales, M, K)
        w_scales_fp8 = method(w_scales, N, K)

        print(f"  Scale sizes: x={x_scales_fp8.numel()}, w={w_scales_fp8.numel()}")

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

            print(f"  Cosine sim: {cos:.6f}")

        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    print("CUTLASS Scale Indexing Analysis")
    print("=" * 60)

    simulate_cutlass_scale_access(256, 2048, 32)
    test_scale_indexing_pattern()


if __name__ == "__main__":
    main()
