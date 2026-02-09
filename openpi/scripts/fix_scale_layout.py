#!/usr/bin/env python3
"""
修复 Scale Layout 以匹配 CUTLASS 期望

CUTLASS Scale Layout (from sm100_blockscaled_layout.hpp):

tile_to_shape(SfAtom, make_shape(M, K, L), Step<_2,_1,_3>)

Step<_2,_1,_3> 意味着:
- K 维度 tiles 变化最慢 (位置 1)
- M 维度 tiles 变化次慢 (位置 2)
- L 变化最快 (位置 3)

SfAtom = Layout<Shape<Shape<_32,_4>>, Stride<Stride<_16,_4>>>
- 32 rows per atom
- 4 k-blocks per atom
- 内存布局: offset = row * 16 + k * 4 (K-major within atom)

所以完整的 layout 应该是:
for k_tile in range(num_k_tiles):           # K 最慢
    for m_tile in range(num_m_tiles):       # M 次慢
        for row_in_tile in range(32):       # SfAtom 内部
            for k_in_tile in range(4):
                # 存储 scale
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


def prepare_scales_cutlass_layout(scales, M, K, device):
    """
    将 [M, num_k_blocks] 的 scales 转换为 CUTLASS 期望的 layout。

    CUTLASS layout (based on tile_to_shape with Step<_2,_1,_3>):
    - K tiles 变化最慢
    - M tiles 变化次慢
    - 每个 tile 内: SfAtom 结构 (32 rows × 4 k-blocks, K-major)

    Memory layout:
    [k_tile0: [m_tile0: [SfAtom], m_tile1: [SfAtom], ...], k_tile1: ...]
    """
    block_size = BLOCK_SIZE  # 32
    row_tile = CUTLASS_ROW_TILE  # 128
    k_tile = CUTLASS_K_TILE  # 4
    row_group = CUTLASS_ROW_GROUP  # 32 (SfAtom row size)

    num_k_blocks = K // block_size
    num_groups_per_tile = row_tile // row_group  # 128 / 32 = 4

    # Padding
    M_padded = ((M + row_tile - 1) // row_tile) * row_tile
    K_blocks_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

    scales_padded = torch.zeros(M_padded, K_blocks_padded, device=device, dtype=torch.float32)
    scales_padded[:M, :num_k_blocks] = scales

    num_m_tiles = M_padded // row_tile
    num_k_tiles = K_blocks_padded // k_tile

    # Reshape to tile structure:
    # [num_m_tiles, num_groups_per_tile, row_group, num_k_tiles, k_tile]
    scales_tiled = scales_padded.view(
        num_m_tiles,          # M tiles
        num_groups_per_tile,  # Groups within M tile (4 × 32 = 128)
        row_group,            # Rows within group (32)
        num_k_tiles,          # K tiles
        k_tile                # K blocks within tile (4)
    )

    # Permute to CUTLASS layout (K tiles slowest, M tiles next):
    # [num_k_tiles, num_m_tiles, num_groups_per_tile, row_group, k_tile]
    scales_reordered = scales_tiled.permute(3, 0, 1, 2, 4)

    # SfAtom 内部是 K-major (stride [16, 4]):
    # 对于每个 (row, k_block)，offset = row * 16 + k * 4
    # 这意味着同一行的 4 个 k_blocks 是连续的

    # 但现在的 shape 是 [..., row_group, k_tile]
    # 需要调整为 K-major within atom：[..., k_tile, row_group] 然后交织

    # 实际上 SfAtom stride [16, 4] 意味着:
    # - Row stride = 16 (not 4, so not fully K-major)
    # - K stride = 4
    # 这是一种特殊的交织布局

    # 让我们直接实现 SfAtom 的内存布局:
    # offset = row * 16 + k * 4, 其中 row in [0,32), k in [0,4)
    # 最大 offset = 31 * 16 + 3 * 4 = 496 + 12 = 508
    # 每个 atom 使用 512 bytes (实际只填充 128 个位置)

    # 创建 SfAtom 内部索引
    atom_size = row_group * 16  # 32 * 16 = 512 (虽然只用 128 个位置)
    total_atoms = num_k_tiles * num_m_tiles * num_groups_per_tile

    # 构建输出 tensor
    output_size = total_atoms * atom_size

    # 为了简化，我们先创建一个足够大的输出 tensor
    output = torch.zeros(output_size, device=device, dtype=torch.float32)

    # 填充每个 atom
    atom_idx = 0
    for k_t in range(num_k_tiles):
        for m_t in range(num_m_tiles):
            for g in range(num_groups_per_tile):
                base_offset = atom_idx * atom_size
                for row in range(row_group):
                    for k in range(k_tile):
                        src_m = m_t * num_groups_per_tile * row_group + g * row_group + row
                        src_k = k_t * k_tile + k
                        if src_m < M and src_k < num_k_blocks:
                            scale_val = scales_padded[src_m, src_k]
                        else:
                            scale_val = 0.0

                        dst_offset = base_offset + row * 16 + k * 4
                        if dst_offset < output_size:
                            output[dst_offset] = scale_val

                atom_idx += 1

    # 实际上 CUTLASS 可能期望更紧凑的存储
    # 让我们也尝试不使用 stride 16，直接使用 [row, k] 顺序
    return convert_scales_to_fp8(output)


def prepare_scales_simple_kmajor(scales, M, K, device):
    """
    简化版 K-major layout，不考虑 SfAtom 的复杂 stride。

    Layout: [num_k_tiles, num_m_tiles, row_tile, k_tile]
    然后 repeat_interleave(block_size) 扩展 k_tile
    """
    block_size = BLOCK_SIZE
    row_tile = CUTLASS_ROW_TILE
    k_tile = CUTLASS_K_TILE

    num_k_blocks = K // block_size

    M_padded = ((M + row_tile - 1) // row_tile) * row_tile
    K_blocks_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile
    K_padded = K_blocks_padded * block_size

    scales_padded = torch.zeros(M_padded, K_blocks_padded, device=device, dtype=torch.float32)
    scales_padded[:M, :num_k_blocks] = scales

    num_m_tiles = M_padded // row_tile
    num_k_tiles = K_blocks_padded // k_tile

    # Reshape: [num_m_tiles, row_tile, num_k_tiles, k_tile]
    scales_tiled = scales_padded.view(num_m_tiles, row_tile, num_k_tiles, k_tile)

    # K-major permute: [num_k_tiles, num_m_tiles, row_tile, k_tile]
    scales_kmajor = scales_tiled.permute(2, 0, 1, 3)

    # 扩展每个 k_block 为 block_size 个元素
    # [num_k_tiles, num_m_tiles, row_tile, k_tile * block_size]
    scales_expanded = scales_kmajor.unsqueeze(-1).expand(-1, -1, -1, -1, block_size)
    scales_expanded = scales_expanded.reshape(num_k_tiles, num_m_tiles, row_tile, k_tile * block_size)

    # Flatten
    return convert_scales_to_fp8(scales_expanded.flatten())


def prepare_scales_row_major_expanded(scales, M, K, device):
    """
    当前方法: row-major 然后 repeat_interleave。
    作为 baseline 对比。
    """
    block_size = BLOCK_SIZE
    row_tile = CUTLASS_ROW_TILE

    num_k_blocks = K // block_size

    M_padded = ((M + row_tile - 1) // row_tile) * row_tile
    K_padded = ((K + row_tile - 1) // row_tile) * row_tile

    scales_padded = torch.zeros(M_padded, ((num_k_blocks + 4 - 1) // 4) * 4, device=device, dtype=torch.float32)
    scales_padded[:M, :num_k_blocks] = scales

    scales_expanded = scales_padded.repeat_interleave(block_size, dim=1)
    if scales_expanded.shape[1] < K_padded:
        extra = K_padded - scales_expanded.shape[1]
        scales_expanded = torch.cat([
            scales_expanded,
            torch.zeros(M_padded, extra, device=device)
        ], dim=1)

    return convert_scales_to_fp8(scales_expanded.flatten())


def test_layouts():
    """测试不同的 layout"""
    print("=" * 60)
    print("Testing Different Scale Layouts")
    print("=" * 60)

    try:
        import nvfp4_gemm
    except ImportError:
        print("nvfp4_gemm not available")
        return

    device = torch.device('cuda')
    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE

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

    # BF16 参考
    bf16_ref = torch.matmul(x, w.T)

    methods = [
        ("Row-major + repeat (current)", prepare_scales_row_major_expanded),
        ("Simple K-major", prepare_scales_simple_kmajor),
        ("Full CUTLASS layout", prepare_scales_cutlass_layout),
    ]

    for name, method in methods:
        print(f"\n{name}:")
        try:
            x_scales_fp8 = method(x_scales.cuda(), M, K, device)
            w_scales_fp8 = method(w_scales.cuda(), N, K, device)

            print(f"  Scale sizes: x={x_scales_fp8.numel()}, w={w_scales_fp8.numel()}")

            output = nvfp4_gemm.gemm_prepared(
                x_packed, w_packed,
                x_scales_fp8, w_scales_fp8,
                M, N, K
            )

            cos_ref = F.cosine_similarity(
                output.flatten().float().unsqueeze(0),
                ref.flatten().unsqueeze(0)
            ).item()

            cos_bf16 = F.cosine_similarity(
                output.flatten().float().unsqueeze(0),
                bf16_ref.flatten().unsqueeze(0)
            ).item()

            print(f"  Cosine vs Python sim: {cos_ref:.6f}")
            print(f"  Cosine vs BF16:       {cos_bf16:.6f}")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()


def main():
    print("Scale Layout Fix")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    test_layouts()


if __name__ == "__main__":
    main()
