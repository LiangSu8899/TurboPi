#!/usr/bin/env python3
"""
测试不进行 swizzle 的 scale factors

假设: CUTLASS 期望 row-major scales，内部使用 layout 描述符来读取
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


def prepare_scales_no_swizzle(scales, M, num_k_blocks):
    """
    准备 scales 不进行 swizzle - 只 padding 和转换为 FP8
    """
    row_tile = CUTLASS_ROW_TILE  # 128
    k_tile = CUTLASS_K_TILE      # 4

    device = scales.device
    dtype = scales.dtype

    if scales.dim() == 1:
        scales = scales.view(M, num_k_blocks)

    # Padding 到 tile 边界
    M_padded = ((M + row_tile - 1) // row_tile) * row_tile
    K_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile

    if M_padded != M or K_padded != num_k_blocks:
        scales_padded = torch.zeros(M_padded, K_padded, device=device, dtype=dtype)
        scales_padded[:M, :num_k_blocks] = scales
        scales = scales_padded

    # Flatten 成 row-major 并转换为 FP8
    scales_flat = scales.flatten()
    scales_fp8 = convert_scales_to_fp8(scales_flat)

    return scales_fp8


def test_no_swizzle():
    """测试不 swizzle 的 scales"""
    print("=" * 60)
    print("Test: No Swizzle (Row-Major Scales)")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # 创建统一数据
    x = torch.ones(M, K, device='cuda', dtype=torch.float32)
    w = torch.ones(N, K, device='cuda', dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    print(f"x_scales shape: {x_scales.shape}")
    print(f"w_scales shape: {w_scales.shape}")

    # 打包
    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # 准备 scales - 不 swizzle
    x_scales_fp8 = prepare_scales_no_swizzle(x_scales.cuda(), M, num_k_blocks)
    w_scales_fp8 = prepare_scales_no_swizzle(w_scales.cuda(), N, num_k_blocks)

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

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_larger_n_no_swizzle():
    """测试更大的 N（不 swizzle）"""
    print("\n" + "=" * 60)
    print("Test: Larger N (No Swizzle)")
    print("=" * 60)

    import nvfp4_gemm

    M, K = 256, 2048
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    for N in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        print(f"\n--- N = {N} ---")

        x = torch.ones(M, K, device='cuda', dtype=torch.float32)
        w = torch.ones(N, K, device='cuda', dtype=torch.float32)

        x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
        w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

        x_packed = pack_nvfp4_data(x_q, block_size)
        w_packed = pack_nvfp4_data(w_q, block_size)

        x_scales_fp8 = prepare_scales_no_swizzle(x_scales.cuda(), M, num_k_blocks)
        w_scales_fp8 = prepare_scales_no_swizzle(w_scales.cuda(), N, num_k_blocks)

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
    print("Testing Row-Major Scales (No Swizzle)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    test_no_swizzle()
    test_larger_n_no_swizzle()


if __name__ == "__main__":
    main()
