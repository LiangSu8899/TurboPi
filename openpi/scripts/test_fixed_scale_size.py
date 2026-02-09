#!/usr/bin/env python3
"""
测试修复后的 scale tensor 大小

问题: CUTLASS 的 tile_atom_to_shape_SFB 使用 K（元素数量）计算 layout
      但 Python 使用 num_k_blocks 准备 scales

解决方案: 使用 K 维度（而不是 num_k_blocks）来计算 scale tensor 大小
         同时保持正确的 scale 值（每 32 个元素共享一个 scale）
"""

import torch
import sys

sys.path.insert(0, '/workspace/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    convert_scales_to_fp8,
    pack_nvfp4_data,
    BLOCK_SIZE,
    CUTLASS_ROW_TILE,
    CUTLASS_K_TILE,
)


def prepare_scales_cutlass_v2(
    scales: torch.Tensor,
    M: int,
    K: int,  # 使用 K（元素数量）而不是 num_k_blocks
    block_size: int = BLOCK_SIZE
) -> torch.Tensor:
    """
    准备 scale factors 以匹配 CUTLASS 的 layout 期望

    CUTLASS 使用 tile_atom_to_shape_SFB(make_shape(M, N, K, 1))
    其中 K 是元素数量，而不是 block 数量。

    SfAtom 形状: [[32, 4], [SFVecSize, 4]]
    - 32 rows × 4 k-positions
    - 每个 k-position 对应 block_size=32 个元素

    所以 layout 期望:
    - M 维度: 按 32-row groups 分组
    - K 维度: 按 4-k-blocks tiles 分组，但索引基于 K/32

    实际需要的 scale 数量: M × (K / block_size)
    但 layout 可能索引到更大的范围
    """
    device = scales.device
    dtype = scales.dtype

    num_k_blocks = K // block_size

    if scales.dim() == 1:
        scales = scales.view(M, num_k_blocks)

    # Padding 到 tile 边界
    row_tile = CUTLASS_ROW_TILE  # 128
    k_tile = CUTLASS_K_TILE      # 4

    M_padded = ((M + row_tile - 1) // row_tile) * row_tile
    K_blocks_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

    if M_padded != M or K_blocks_padded != num_k_blocks:
        scales_padded = torch.zeros(M_padded, K_blocks_padded, device=device, dtype=dtype)
        scales_padded[:M, :num_k_blocks] = scales
        scales = scales_padded

    # 关键: 需要创建一个大小为 M × K 的 layout
    # 但实际 scale 值每 block_size 个元素重复一次
    # 使用 repeat_interleave 来扩展 K 维度

    # scales 形状: [M_padded, K_blocks_padded]
    # 需要扩展为: [M_padded, K_padded] where K_padded = K_blocks_padded * block_size

    # 但这会产生太多数据...让我们用另一种方法
    # 直接使用较大的 tensor 大小，用正确的 scale 值填充

    # 实际上，CUTLASS 的 layout 可能只是期望 tensor 足够大来避免越界
    # 让我们尝试使用 M × K 大小的 tensor，但只填充有效的 scale 值

    K_padded = ((K + row_tile - 1) // row_tile) * row_tile  # 按 128 对齐 K

    # 创建扩展的 scale tensor
    # 每个 scale 值对应 block_size 个 K 元素
    scales_expanded = scales.repeat_interleave(block_size, dim=1)

    # 如果需要更多 padding
    if scales_expanded.shape[1] < K_padded:
        extra = K_padded - scales_expanded.shape[1]
        scales_expanded = torch.cat([
            scales_expanded,
            torch.zeros(M_padded, extra, device=device, dtype=dtype)
        ], dim=1)

    # Flatten 并转换为 FP8
    scales_flat = scales_expanded.flatten()
    scales_fp8 = convert_scales_to_fp8(scales_flat)

    return scales_fp8


def test_v2_scale_preparation():
    """测试 V2 scale 准备函数"""
    print("=" * 60)
    print("Testing V2 Scale Preparation")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    print(f"M={M}, K={K}, N={N}")
    print(f"block_size={block_size}, num_k_blocks={num_k_blocks}")

    # 创建统一数据
    x = torch.ones(M, K, device='cuda', dtype=torch.float32)
    w = torch.ones(N, K, device='cuda', dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    print(f"\nOriginal scales shape: x={x_scales.shape}, w={w_scales.shape}")

    # 打包
    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # 使用 V2 准备函数
    x_scales_fp8 = prepare_scales_cutlass_v2(x_scales.cuda(), M, K, block_size)
    w_scales_fp8 = prepare_scales_cutlass_v2(w_scales.cuda(), N, K, block_size)

    print(f"V2 scales size: x={x_scales_fp8.numel()}, w={w_scales_fp8.numel()}")

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

        # 检查正确的 N 数量
        correct_n = sum(1 for n in range(N) if abs(output[0, n].item() - expected) < 500)
        print(f"Correct N positions: {correct_n}/{N}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_v2_different_n():
    """测试不同 N 值的 V2 准备"""
    print("\n" + "=" * 60)
    print("Testing V2 with Different N Values")
    print("=" * 60)

    import nvfp4_gemm

    M, K = 256, 2048
    block_size = BLOCK_SIZE

    for N in [256, 512, 1024, 2048, 4096, 8192]:
        print(f"\n--- N = {N} ---")

        x = torch.ones(M, K, device='cuda', dtype=torch.float32)
        w = torch.ones(N, K, device='cuda', dtype=torch.float32)

        x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
        w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

        x_packed = pack_nvfp4_data(x_q, block_size)
        w_packed = pack_nvfp4_data(w_q, block_size)

        x_scales_fp8 = prepare_scales_cutlass_v2(x_scales.cuda(), M, K, block_size)
        w_scales_fp8 = prepare_scales_cutlass_v2(w_scales.cuda(), N, K, block_size)

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
    print("Fixed Scale Size Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    test_v2_scale_preparation()
    test_v2_different_n()


if __name__ == "__main__":
    main()
