#!/usr/bin/env python3
"""
深入分析 CUTLASS Scale Layout

关键发现：
- SFVecSize = 16 for nv_float4_t (非稀疏)
- SfKMajorAtom = Layout<Shape<Shape<_32,_4>, Shape<_16,_4>>, Stride<Stride<_16,_4>, Stride<_0,_1>>>
- tile_to_shape(SfAtom, make_shape(M, K, L), Step<_2,_1,_3>)

问题：
- 我们用 repeat_interleave(32) 扩展 scales 到 M × K
- 但 CUTLASS 的 tile_to_shape 可能期望不同的 layout
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/workspace/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    pack_nvfp4_data,
    convert_scales_to_fp8,
    BLOCK_SIZE,
    CUTLASS_ROW_TILE,
    CUTLASS_K_TILE,
    CUTLASS_ROW_GROUP,
)


def analyze_sfatom_layout():
    """分析 SfAtom 的 layout 结构"""
    print("=" * 60)
    print("SfAtom Layout Analysis")
    print("=" * 60)

    # SfKMajorAtom for nv_float4_t:
    # Layout<Shape<Shape<_32,_4>, Shape<_16,_4>>, Stride<Stride<_16,_4>, Stride<_0,_1>>>
    #
    # Shape: [[32, 4], [16, 4]]
    #   - Outer: 32 rows × 4 k-blocks per atom
    #   - Inner: 16 × 4 (16 is broadcast with stride=0)
    #
    # Stride: [[16, 4], [0, 1]]
    #   - Outer: row_stride=16, k_stride=4
    #   - Inner: broadcast_stride=0, k_extra_stride=1

    print("""
SfAtom Layout for nv_float4_t:
  Shape:  [[32, 4], [16, 4]]
  Stride: [[16, 4], [0, 1]]

解读:
  - 每个 atom 有 32 行 × 4 个 k-blocks
  - 内层 [16, 4] 的 stride [0, 1] 表示:
    - 16 维度被 broadcast (stride=0)
    - 4 维度是实际存储 (stride=1)

  - 内存布局:
    offset = row * 16 + k_outer * 4 + k_inner * 1
    (k_outer in [0,4), k_inner in [0,4))

  - 每个 (row, k_block) 实际上需要 4 个连续的 scale 值?
    不对！k_inner 是用于索引 k_outer 内部的位置

  - 实际上每个 atom 存储 32 * 4 * 4 = 512 个 scales?
    不对！内层 16 被 broadcast，所以只需要 32 * 4 = 128 个 scales per atom
""")


def test_original_layout():
    """测试原始 layout (不使用 repeat_interleave)"""
    print("\n" + "=" * 60)
    print("Testing Original Layout (no repeat_interleave)")
    print("=" * 60)

    try:
        import nvfp4_gemm
    except ImportError:
        print("nvfp4_gemm not available")
        return

    device = torch.device('cuda')
    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # Python 参考
    x_deq = dequantize_nvfp4_sim(x_q, x_scales, block_size)
    w_deq = dequantize_nvfp4_sim(w_q, w_scales, block_size)
    ref = torch.matmul(x_deq, w_deq.T)

    # 尝试不同的 scale 准备方法
    methods = []

    # 方法 1: 当前方法 (repeat_interleave)
    def method1_repeat_interleave(scales, rows, K):
        num_k_blocks = K // block_size
        M_padded = ((rows + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
        K_blocks_padded = ((num_k_blocks + CUTLASS_K_TILE - 1) // CUTLASS_K_TILE) * CUTLASS_K_TILE

        scales_padded = torch.zeros(M_padded, K_blocks_padded, device=device, dtype=torch.float32)
        scales_padded[:rows, :num_k_blocks] = scales

        # 扩展 K 维度
        K_padded = ((K + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
        scales_expanded = scales_padded.repeat_interleave(block_size, dim=1)
        if scales_expanded.shape[1] < K_padded:
            extra = K_padded - scales_expanded.shape[1]
            scales_expanded = torch.cat([
                scales_expanded,
                torch.zeros(M_padded, extra, device=device, dtype=torch.float32)
            ], dim=1)

        return convert_scales_to_fp8(scales_expanded.flatten())

    methods.append(("repeat_interleave(32)", method1_repeat_interleave))

    # 方法 2: 直接使用 k-blocks (不扩展)
    def method2_kblocks_only(scales, rows, K):
        num_k_blocks = K // block_size
        M_padded = ((rows + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
        K_blocks_padded = ((num_k_blocks + CUTLASS_K_TILE - 1) // CUTLASS_K_TILE) * CUTLASS_K_TILE

        scales_padded = torch.zeros(M_padded, K_blocks_padded, device=device, dtype=torch.float32)
        scales_padded[:rows, :num_k_blocks] = scales

        # 计算 CUTLASS 期望的 scale tensor 大小
        # tile_to_shape(SfAtom, make_shape(M, K, L))
        # 这里 K 是元素数量，不是 block 数量
        # 但 SfAtom 每 32 行 × 4 k-blocks 为一个单位
        # 所以期望大小应该是 M × (K / block_size)?

        # Pad to expected size based on CUTLASS layout
        # Expected: M_padded * K_padded (使用 K 元素数量)
        K_padded = ((K + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
        expected_size = M_padded * K_padded

        flat = scales_padded.flatten()
        if flat.numel() < expected_size:
            flat = torch.cat([flat, torch.zeros(expected_size - flat.numel(), device=device)])

        return convert_scales_to_fp8(flat)

    methods.append(("k-blocks only (padded)", method2_kblocks_only))

    # 方法 3: K-major tiled layout (模拟 CUTLASS SfAtom)
    def method3_kmajor_tiled(scales, rows, K):
        num_k_blocks = K // block_size
        row_tile = CUTLASS_ROW_TILE  # 128
        k_tile = CUTLASS_K_TILE      # 4
        row_group = CUTLASS_ROW_GROUP  # 32

        M_padded = ((rows + row_tile - 1) // row_tile) * row_tile
        K_blocks_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

        scales_padded = torch.zeros(M_padded, K_blocks_padded, device=device, dtype=torch.float32)
        scales_padded[:rows, :num_k_blocks] = scales

        num_row_tiles = M_padded // row_tile
        num_k_tiles = K_blocks_padded // k_tile
        num_groups = row_tile // row_group

        # Reshape to tile structure
        # [num_row_tiles, num_groups, group_size, num_k_tiles, k_tile]
        scales_tiled = scales_padded.view(
            num_row_tiles, num_groups, row_group, num_k_tiles, k_tile
        )

        # K-major: K tiles vary slowest
        # Result: [num_k_tiles, num_row_tiles, num_groups, group_size, k_tile]
        scales_reordered = scales_tiled.permute(3, 0, 1, 2, 4)

        # Pad to expected size
        K_padded = ((K + row_tile - 1) // row_tile) * row_tile
        expected_size = M_padded * K_padded

        flat = scales_reordered.contiguous().flatten()
        if flat.numel() < expected_size:
            flat = torch.cat([flat, torch.zeros(expected_size - flat.numel(), device=device)])

        return convert_scales_to_fp8(flat)

    methods.append(("K-major tiled", method3_kmajor_tiled))

    # 方法 4: repeat_interleave + K-major tiled
    def method4_repeat_then_tile(scales, rows, K):
        num_k_blocks = K // block_size
        row_tile = CUTLASS_ROW_TILE  # 128
        k_tile = CUTLASS_K_TILE      # 4
        row_group = CUTLASS_ROW_GROUP  # 32

        M_padded = ((rows + row_tile - 1) // row_tile) * row_tile
        K_padded = ((K + row_tile - 1) // row_tile) * row_tile

        # Step 1: Expand scales to M × K
        scales_expanded = scales.repeat_interleave(block_size, dim=1)
        if scales_expanded.shape[1] < K_padded:
            extra = K_padded - scales_expanded.shape[1]
            scales_expanded = torch.cat([
                scales_expanded,
                torch.zeros(rows, extra, device=device, dtype=torch.float32)
            ], dim=1)

        if M_padded != rows:
            extra = torch.zeros(M_padded - rows, K_padded, device=device, dtype=torch.float32)
            scales_expanded = torch.cat([scales_expanded, extra], dim=0)

        # Step 2: Tile in K-major order
        # 将 K 维度按 block_size 分块，然后 K-major 排列
        k_blocks_expanded = K_padded // block_size
        num_row_tiles = M_padded // row_tile
        num_k_tiles_expanded = k_blocks_expanded // k_tile
        num_groups = row_tile // row_group

        # 这里尝试一种不同的 tiling 策略
        # 先按 block_size 分块，再按 tile 分块
        scales_blocked = scales_expanded.view(M_padded, k_blocks_expanded, block_size)

        # 重排为 K-major
        scales_tiled = scales_blocked.view(
            num_row_tiles, num_groups, row_group,
            num_k_tiles_expanded, k_tile, block_size
        )

        # Permute: K tiles vary slowest
        scales_reordered = scales_tiled.permute(3, 0, 1, 2, 4, 5)

        return convert_scales_to_fp8(scales_reordered.contiguous().flatten())

    methods.append(("repeat + K-major tile", method4_repeat_then_tile))

    # 测试每种方法
    for name, method in methods:
        print(f"\n--- {name} ---")
        try:
            x_scales_fp8 = method(x_scales.cuda(), M, K)
            w_scales_fp8 = method(w_scales.cuda(), N, K)

            print(f"  Scale sizes: x={x_scales_fp8.numel()}, w={w_scales_fp8.numel()}")

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


def test_sfvecsize_16():
    """测试 SFVecSize=16 的 layout"""
    print("\n" + "=" * 60)
    print("Testing SFVecSize=16 Layout")
    print("=" * 60)

    print("""
对于 nv_float4_t，SFVecSize = 16:
  - SfAtom Shape: [[32, 4], [16, 4]]
  - 这意味着 CUTLASS 期望每 16 个 K 元素有一个 scale
  - 我们的 block_size = 32 是硬编码的 (来自 CUTLASS)
  - 内层 [16, 4] 的 broadcast (stride=0) 意味着:
    每 4 个 k-blocks 共享相同的 scale 索引结构

实验: 尝试按 SFVecSize 而不是 block_size 扩展
""")

    try:
        import nvfp4_gemm
    except ImportError:
        print("nvfp4_gemm not available")
        return

    device = torch.device('cuda')
    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE  # 32
    sfvecsize = 16  # 根据 CUTLASS 源码
    num_k_blocks = K // block_size

    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)

    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    x_deq = dequantize_nvfp4_sim(x_q, x_scales, block_size)
    w_deq = dequantize_nvfp4_sim(w_q, w_scales, block_size)
    ref = torch.matmul(x_deq, w_deq.T)

    # 方法: 按 SFVecSize 扩展
    def prepare_scales_sfvec(scales, rows, K, sfvecsize=16):
        num_k_blocks = K // block_size
        row_tile = CUTLASS_ROW_TILE  # 128

        M_padded = ((rows + row_tile - 1) // row_tile) * row_tile
        K_padded = ((K + row_tile - 1) // row_tile) * row_tile

        # 每个 block (32 elements) 需要 32/16 = 2 个 scale slots?
        # 或者 broadcast 意味着 1 个 scale 对应 16 个元素?

        # 尝试: 每个 scale 重复 16 次 (而不是 32 次)
        scales_expanded = scales.repeat_interleave(sfvecsize, dim=1)

        # 但我们的 block_size 是 32，所以需要额外处理
        # 每个 block 有 32 个元素，每 16 个共享一个 scale
        # 所以每个 block 需要 2 个 scale slots

        # 重新思考: 实际上每个 block 只有 1 个 scale
        # sfvecsize 只是 CUTLASS 内部的 broadcast 粒度

        # 尝试原始 repeat_interleave(32)
        scales_expanded = scales.repeat_interleave(block_size, dim=1)

        if scales_expanded.shape[1] < K_padded:
            extra = K_padded - scales_expanded.shape[1]
            scales_expanded = torch.cat([
                scales_expanded,
                torch.zeros(rows, extra, device=device, dtype=torch.float32)
            ], dim=1)

        if M_padded != rows:
            extra = torch.zeros(M_padded - rows, K_padded, device=device, dtype=torch.float32)
            scales_expanded = torch.cat([scales_expanded, extra], dim=0)

        return convert_scales_to_fp8(scales_expanded.flatten())

    x_scales_fp8 = prepare_scales_sfvec(x_scales.cuda(), M, K)
    w_scales_fp8 = prepare_scales_sfvec(w_scales.cuda(), N, K)

    output = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(), w_packed.cuda(),
        x_scales_fp8.cuda(), w_scales_fp8.cuda(),
        M, N, K
    )

    cos = F.cosine_similarity(
        output.flatten().float().unsqueeze(0),
        ref.flatten().unsqueeze(0)
    ).item()

    print(f"Cosine sim (current method): {cos:.6f}")
    print(f"\n结论: 主要误差可能来自 layout，需要进一步分析 CUTLASS 的实际索引方式")


def main():
    print("CUTLASS Scale Layout V2 Analysis")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    analyze_sfatom_layout()
    test_original_layout()
    test_sfvecsize_16()


if __name__ == "__main__":
    main()
