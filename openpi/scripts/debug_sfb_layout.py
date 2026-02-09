#!/usr/bin/env python3
"""
调试 SFB (B 矩阵 scale factor) 布局问题

问题: N >= 8192 的输出为零
原因分析: B 矩阵是 ColumnMajor，其 scale 布局可能与 A 矩阵不同
"""

import torch
import sys

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    prepare_scales_for_cutlass,
    pack_nvfp4_data,
    swizzle_scales_for_cutlass,
    BLOCK_SIZE,
    CUTLASS_ROW_TILE,
    CUTLASS_K_TILE,
)


def analyze_scale_sizes():
    """分析不同维度的 scale 大小"""
    print("=" * 60)
    print("Scale Factor Size Analysis")
    print("=" * 60)

    M, K, N = 256, 2048, 16384
    num_k_blocks = K // BLOCK_SIZE  # 64

    print(f"\nProblem: M={M}, N={N}, K={K}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"K blocks: {num_k_blocks}")

    # A 矩阵 scales
    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((num_k_blocks + CUTLASS_K_TILE - 1) // CUTLASS_K_TILE) * CUTLASS_K_TILE

    print(f"\nA matrix scales:")
    print(f"  Original: {M} x {num_k_blocks} = {M * num_k_blocks}")
    print(f"  Padded: {M_padded} x {K_padded} = {M_padded * K_padded}")

    # B 矩阵 scales
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE

    print(f"\nB matrix scales:")
    print(f"  Original: {N} x {num_k_blocks} = {N * num_k_blocks}")
    print(f"  Padded: {N_padded} x {K_padded} = {N_padded * K_padded}")

    # Tile 数量
    num_m_tiles = M_padded // CUTLASS_ROW_TILE
    num_n_tiles = N_padded // CUTLASS_ROW_TILE
    num_k_tiles = K_padded // CUTLASS_K_TILE

    print(f"\nTile counts:")
    print(f"  M tiles: {num_m_tiles}")
    print(f"  N tiles: {num_n_tiles}")
    print(f"  K tiles: {num_k_tiles}")


def test_smaller_n():
    """测试较小的 N 值"""
    print("\n" + "=" * 60)
    print("Testing Smaller N Values")
    print("=" * 60)

    import nvfp4_gemm

    M, K = 256, 2048
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    test_n_values = [256, 512, 1024, 2048, 4096, 8192, 16384]

    for N in test_n_values:
        print(f"\n--- N = {N} ---")

        # 创建数据
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

        print(f"  w_scales_fp8 size: {w_scales_fp8.numel()}")

        try:
            output = nvfp4_gemm.gemm_prepared(
                x_packed.cuda(),
                w_packed.cuda(),
                x_scales_fp8.cuda(),
                w_scales_fp8.cuda(),
                M, N, K
            )

            # 检查输出
            out_first = output[0, 0].item()
            out_mid = output[0, N//2].item() if N > 1 else out_first
            out_last = output[0, -1].item()
            out_mean = output.mean().item()

            expected = K  # ones @ ones = K

            status = "OK" if abs(out_first - expected) < 10 and abs(out_last - expected) < 10 else "FAIL"

            print(f"  Output[0,0]: {out_first:.1f}, [0,{N//2}]: {out_mid:.1f}, [0,-1]: {out_last:.1f}")
            print(f"  Mean: {out_mean:.1f}, Expected: {expected} [{status}]")

        except Exception as e:
            print(f"  ERROR: {e}")


def test_scale_layout_correctness():
    """验证 scale 布局是否正确"""
    print("\n" + "=" * 60)
    print("Scale Layout Correctness Test")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 8192  # 使用 N=8192，刚好是问题出现的边界
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # 创建有规律的数据来追踪 scale 使用
    # 每个 k-block 使用不同的值
    x = torch.zeros(M, K, device='cuda', dtype=torch.float32)
    for k_block in range(num_k_blocks):
        x[:, k_block*block_size:(k_block+1)*block_size] = 1.0

    # W 矩阵: 对不同的 N 区域使用不同的值
    w = torch.zeros(N, K, device='cuda', dtype=torch.float32)
    for n in range(N):
        n_group = n // CUTLASS_ROW_TILE  # 按 128 分组
        w[n, :] = 1.0

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    # 检查 w_scales 的形状和值
    print(f"w_scales shape: {w_scales.shape}")
    print(f"w_scales unique: {w_scales.unique().numel()} values")

    # 打包
    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # 准备 scales
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True)

    # 检查 w_scales_fp8 的分布
    print(f"\nw_scales_fp8:")
    print(f"  Shape: {w_scales_fp8.shape}")
    print(f"  Unique values: {w_scales_fp8.unique().tolist()[:10]}...")

    # 找出第一个零值的位置
    zero_positions = (w_scales_fp8 == 0).nonzero(as_tuple=True)[0]
    if len(zero_positions) > 0:
        print(f"  First zero at index: {zero_positions[0].item()}")
        print(f"  Total zeros: {(w_scales_fp8 == 0).sum().item()}")
    else:
        print(f"  No zeros found")

    try:
        output = nvfp4_gemm.gemm_prepared(
            x_packed.cuda(),
            w_packed.cuda(),
            x_scales_fp8.cuda(),
            w_scales_fp8.cuda(),
            M, N, K
        )

        print(f"\nOutput analysis:")
        print(f"  Shape: {output.shape}")

        # 按 N 区域检查
        for n_start in range(0, N, N // 4):
            n_end = n_start + 10
            chunk = output[0, n_start:n_end]
            print(f"  Output[0, {n_start}:{n_end}]: {chunk.tolist()[:5]}...")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_manual_uniform_scales():
    """手动创建统一的 scales 测试"""
    print("\n" + "=" * 60)
    print("Manual Uniform Scales Test")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # 使用统一的 FP4 值 (1.0 = index 2)
    fp4_one_packed = 0x22  # 两个 1.0
    x_packed = torch.full((M, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')
    w_packed = torch.full((N, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')

    # 使用统一的 scale = 1.0 (FP8 E4M3: 0x38)
    fp8_one = 0x38

    # 计算正确的 scale 大小
    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((num_k_blocks + CUTLASS_K_TILE - 1) // CUTLASS_K_TILE) * CUTLASS_K_TILE

    x_scales_size = M_padded * K_padded
    w_scales_size = N_padded * K_padded

    print(f"Scale sizes: x={x_scales_size}, w={w_scales_size}")

    x_scales_fp8 = torch.full((x_scales_size,), fp8_one, dtype=torch.uint8, device='cuda')
    w_scales_fp8 = torch.full((w_scales_size,), fp8_one, dtype=torch.uint8, device='cuda')

    try:
        output = nvfp4_gemm.gemm_prepared(
            x_packed,
            w_packed,
            x_scales_fp8,
            w_scales_fp8,
            M, N, K
        )

        expected = K  # ones @ ones = K

        print(f"\nOutput analysis:")
        print(f"  Output[0, 0]: {output[0, 0].item():.1f}")
        print(f"  Output[0, 4096]: {output[0, 4096].item():.1f}")
        print(f"  Output[0, 8192]: {output[0, 8192].item():.1f}")
        print(f"  Output[0, -1]: {output[0, -1].item():.1f}")
        print(f"  Mean: {output.mean().item():.1f}")
        print(f"  Expected: {expected}")

        # 检查非零范围
        nonzero_mask = output[0] != 0
        if nonzero_mask.any():
            first_nonzero = nonzero_mask.nonzero()[0].item()
            last_nonzero = nonzero_mask.nonzero()[-1].item()
            print(f"\n  Non-zero range: [{first_nonzero}, {last_nonzero}]")

        zero_mask = output[0] == 0
        if zero_mask.any():
            first_zero = zero_mask.nonzero()[0].item()
            print(f"  First zero at: {first_zero}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("SFB Layout Debug")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    analyze_scale_sizes()
    test_smaller_n()
    test_manual_uniform_scales()
    test_scale_layout_correctness()


if __name__ == "__main__":
    main()
