#!/usr/bin/env python3
"""
测试 K-tile 结构

发现：只有 k_block % 4 == 0 时 scale 变化才生效
这说明 CUTLASS 的 SfAtom 有 4 个 k-positions 的结构

SfKMajorAtom = Layout<Shape<Shape<_32,_4>>, Stride<Stride<_16,_4>>>
- 32 rows
- 4 k-positions (per tile)

问题：我们的 repeat_interleave(32) 创建的 layout 可能不符合这个结构
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


def test_k_tile_alignment():
    """测试 K-tile 对齐"""
    print("=" * 60)
    print("Test: K-tile Alignment")
    print("=" * 60)

    try:
        import nvfp4_gemm
    except ImportError:
        print("nvfp4_gemm not available")
        return

    device = torch.device('cuda')
    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE  # 32
    num_k_blocks = K // block_size  # 64
    k_tile = CUTLASS_K_TILE  # 4

    # 使用全 1 输入
    x = torch.ones(M, K, device=device, dtype=torch.float32)
    w = torch.ones(N, K, device=device, dtype=torch.float32)

    x_q, _ = quantize_to_nvfp4_sim(x, block_size, use_mse_search=False)
    w_q, _ = quantize_to_nvfp4_sim(w, block_size, use_mse_search=False)

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((K + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE

    fp8_one = 0x38

    # 测试每个 k-tile (4 个 k-blocks 为一组)
    num_k_tiles = num_k_blocks // k_tile

    print(f"M={M}, K={K}, N={N}")
    print(f"num_k_blocks={num_k_blocks}, num_k_tiles={num_k_tiles}")
    print(f"block_size={block_size}, k_tile={k_tile}")

    print("\n测试每个 k-tile 的影响:")
    for k_tile_idx in [0, 1, 2, 15]:
        # 创建 scales: 指定 k_tile 的所有 k-blocks 使用 scale=2.0
        x_scales_blocks = torch.full((M, num_k_blocks), 0.5, dtype=torch.float32, device=device)

        start_k = k_tile_idx * k_tile
        end_k = start_k + k_tile
        if end_k <= num_k_blocks:
            x_scales_blocks[:, start_k:end_k] = 2.0

        # 使用 repeat_interleave 扩展
        x_scales_expanded = x_scales_blocks.repeat_interleave(block_size, dim=1)

        # Padding
        if x_scales_expanded.shape[0] < M_padded:
            extra = torch.full((M_padded - M, x_scales_expanded.shape[1]), 0.5, device=device)
            x_scales_expanded = torch.cat([x_scales_expanded, extra], dim=0)
        if x_scales_expanded.shape[1] < K_padded:
            extra = torch.full((M_padded, K_padded - x_scales_expanded.shape[1]), 0.5, device=device)
            x_scales_expanded = torch.cat([x_scales_expanded, extra], dim=1)

        x_scales_fp8 = convert_scales_to_fp8(x_scales_expanded.flatten())
        w_scales_fp8 = torch.full((N_padded * K_padded,), fp8_one, dtype=torch.uint8, device=device)

        output = nvfp4_gemm.gemm_prepared(
            x_packed, w_packed,
            x_scales_fp8, w_scales_fp8,
            M, N, K
        )

        # 期望值:
        # k-tile 内 4 * 32 = 128 个元素使用 scale=2.0
        # 其余 2048-128 = 1920 个元素使用 scale=0.5
        expected = 128 * 2.0 + (K - 128) * 0.5
        # 但因为量化，实际 FP4 值是 6 (来自原始 scale 0.167)
        # 所以实际是: 6 * (128 * 2.0 + 1920 * 0.5) = 6 * (256 + 960) = 6 * 1216 = 7296

        print(f"  k_tile={k_tile_idx} (k_blocks {start_k}-{end_k-1}): "
              f"output={output[0,0].item():.2f}")


def test_correct_scale_layout():
    """测试正确的 scale layout"""
    print("\n" + "=" * 60)
    print("Test: Correct Scale Layout")
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

    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size, use_mse_search=False)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size, use_mse_search=False)

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # Python 参考
    x_deq = dequantize_nvfp4_sim(x_q, x_scales, block_size)
    w_deq = dequantize_nvfp4_sim(w_q, w_scales, block_size)
    ref = torch.matmul(x_deq, w_deq.T)

    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((K + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE

    row_tile = CUTLASS_ROW_TILE  # 128
    k_tile = CUTLASS_K_TILE      # 4
    row_group = CUTLASS_ROW_GROUP  # 32

    # 尝试不同的 scale layout
    def prepare_scales_v1(scales, rows, K):
        """当前方法: repeat_interleave(32)"""
        num_k_blocks = K // block_size
        scales_padded = torch.zeros(M_padded, ((num_k_blocks + k_tile - 1) // k_tile) * k_tile,
                                    device=device, dtype=torch.float32)
        scales_padded[:rows, :num_k_blocks] = scales

        scales_expanded = scales_padded.repeat_interleave(block_size, dim=1)
        if scales_expanded.shape[1] < K_padded:
            extra = K_padded - scales_expanded.shape[1]
            scales_expanded = torch.cat([
                scales_expanded,
                torch.zeros(M_padded, extra, device=device)
            ], dim=1)

        return convert_scales_to_fp8(scales_expanded.flatten())

    def prepare_scales_v2(scales, rows, K):
        """尝试: 按 k-blocks 布局 (不扩展)"""
        num_k_blocks = K // block_size
        K_blocks_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

        scales_padded = torch.zeros(M_padded, K_blocks_padded, device=device, dtype=torch.float32)
        scales_padded[:rows, :num_k_blocks] = scales

        # 尝试 K-major reorder (Step<_2,_1,_3> means K tiles vary slowest)
        num_row_tiles = M_padded // row_tile
        num_k_tiles = K_blocks_padded // k_tile
        num_groups = row_tile // row_group

        # Reshape to tile structure
        scales_tiled = scales_padded.view(
            num_row_tiles, num_groups, row_group,
            num_k_tiles, k_tile
        )

        # K-major: [num_k_tiles, num_row_tiles, num_groups, row_group, k_tile]
        scales_reordered = scales_tiled.permute(3, 0, 1, 2, 4)

        # 然后每个 k-block 扩展为 block_size 个元素
        scales_flat = scales_reordered.flatten()

        # Pad to M_padded * K_padded
        target_size = M_padded * K_padded
        if scales_flat.numel() < target_size:
            scales_flat = scales_flat.repeat(target_size // scales_flat.numel() + 1)[:target_size]

        return convert_scales_to_fp8(scales_flat)

    def prepare_scales_v3(scales, rows, K):
        """尝试: 先 K-major reorder，再 repeat_interleave"""
        num_k_blocks = K // block_size
        K_blocks_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

        scales_padded = torch.zeros(M_padded, K_blocks_padded, device=device, dtype=torch.float32)
        scales_padded[:rows, :num_k_blocks] = scales

        num_row_tiles = M_padded // row_tile
        num_k_tiles = K_blocks_padded // k_tile
        num_groups = row_tile // row_group

        # Reshape and permute to K-major
        scales_tiled = scales_padded.view(
            num_row_tiles, num_groups, row_group,
            num_k_tiles, k_tile
        )
        scales_reordered = scales_tiled.permute(3, 0, 1, 2, 4)

        # 恢复为 [M_padded, K_blocks_padded] 但 K-major 顺序
        scales_kmajor = scales_reordered.permute(1, 2, 3, 0, 4).contiguous()
        scales_kmajor = scales_kmajor.view(M_padded, K_blocks_padded)

        # 扩展
        scales_expanded = scales_kmajor.repeat_interleave(block_size, dim=1)
        if scales_expanded.shape[1] < K_padded:
            extra = K_padded - scales_expanded.shape[1]
            scales_expanded = torch.cat([
                scales_expanded,
                torch.zeros(M_padded, extra, device=device)
            ], dim=1)

        return convert_scales_to_fp8(scales_expanded.flatten())

    methods = [
        ("v1: repeat_interleave(32)", prepare_scales_v1),
        ("v2: K-major blocks only", prepare_scales_v2),
        ("v3: K-major then repeat", prepare_scales_v3),
    ]

    for name, method in methods:
        print(f"\n{name}:")
        try:
            x_scales_fp8 = method(x_scales.cuda(), M, K)
            w_scales_fp8 = method(w_scales.cuda(), N, K)

            print(f"  Scale sizes: x={x_scales_fp8.numel()}, w={w_scales_fp8.numel()}")

            output = nvfp4_gemm.gemm_prepared(
                x_packed.cuda(), w_packed.cuda(),
                x_scales_fp8, w_scales_fp8,
                M, N, K
            )

            cos = F.cosine_similarity(
                output.flatten().float().unsqueeze(0),
                ref.flatten().unsqueeze(0)
            ).item()

            print(f"  Cosine sim vs Python ref: {cos:.6f}")

        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    print("K-tile Structure Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    test_k_tile_alignment()
    test_correct_scale_layout()


if __name__ == "__main__":
    main()
