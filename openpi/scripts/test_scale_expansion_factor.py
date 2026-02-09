#!/usr/bin/env python3
"""
测试不同的 scale 扩展因子
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
)


def test_expansion_factors():
    """测试不同的扩展因子"""
    print("=" * 60)
    print("Testing Scale Expansion Factors")
    print("=" * 60)

    import nvfp4_gemm

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

    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((K + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE

    # 测试不同的扩展因子
    expansion_factors = [4, 8, 16, 32, 64]

    for factor in expansion_factors:
        print(f"\n--- Expansion factor = {factor} ---")

        def expand_scales(scales, rows):
            scales = scales.cuda()
            scales_expanded = scales.repeat_interleave(factor, dim=1)
            target_cols = K_padded
            if scales_expanded.shape[1] < target_cols:
                extra = target_cols - scales_expanded.shape[1]
                scales_expanded = torch.cat([
                    scales_expanded,
                    torch.zeros(rows, extra, device='cuda', dtype=scales.dtype)
                ], dim=1)
            elif scales_expanded.shape[1] > target_cols:
                scales_expanded = scales_expanded[:, :target_cols]
            return convert_scales_to_fp8(scales_expanded.flatten())

        try:
            x_scales_fp8 = expand_scales(x_scales, M)
            w_scales_fp8 = expand_scales(w_scales, N)

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

            # 也计算平均相对误差
            rel_err = (output.float() - ref).abs() / (ref.abs() + 1e-8)
            mean_rel_err = rel_err.mean().item() * 100

            print(f"  Cosine sim: {cos:.6f}")
            print(f"  Mean rel error: {mean_rel_err:.2f}%")

        except Exception as e:
            print(f"  ERROR: {e}")


def test_with_sfvecsize():
    """基于 SFVecSize 的测试"""
    print("\n" + "=" * 60)
    print("Testing with SFVecSize consideration")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE  # 32
    sfvecsize = 4  # CUTLASS 的 SFVecSize

    # 每个 k-block (32 elements) 可能需要 32/sfvecsize = 8 个 scale 值
    # 或者每个 k-block 只需要 1 个 scale 值（sfvecsize 是 broadcast）

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

    M_padded = 256
    N_padded = 256
    K_padded = 2048

    # 假设 CUTLASS 需要 block_size 扩展但 SFVecSize=4 意味着连续 4 个 K 位置共享内存
    # 实际上每个 k-block 对应一个 scale，但 CUTLASS layout 期望更大的 tensor

    # 让我尝试一个假设：
    # CUTLASS 使用 K 元素数量来计算 layout，但实际只读取每 block_size 个元素的一个 scale
    # 所以 repeat_interleave(block_size) 是正确的

    # 当前精度是 0.93，可能就是 FP8 转换的精度损失
    # 让我检查不使用 FP8 转换的结果

    print("Testing precision breakdown:")

    # 1. 使用 FP32 scales (不转换为 FP8)
    # 这不直接支持，但可以估算 FP8 误差

    # 2. 检查 FP8 转换前后的 scale 差异
    x_scales_cuda = x_scales.cuda()
    x_scales_expanded = x_scales_cuda.repeat_interleave(block_size, dim=1)

    x_scales_fp8 = convert_scales_to_fp8(x_scales_expanded.flatten())
    x_scales_back = x_scales_fp8.view(torch.float8_e4m3fn).to(torch.float32)

    fp8_error = (x_scales_back.view_as(x_scales_expanded.flatten()) - x_scales_expanded.flatten()).abs()
    rel_fp8_error = fp8_error / (x_scales_expanded.flatten().abs() + 1e-8)

    print(f"FP8 conversion error (mean): {rel_fp8_error.mean().item() * 100:.2f}%")
    print(f"FP8 conversion error (max): {rel_fp8_error.max().item() * 100:.2f}%")

    # 3. 使用扩展后的 FP32 scales 计算参考
    # 这模拟如果 CUTLASS 使用完美的 scales 会得到什么结果
    # 但 CUTLASS 需要 FP8 scales，所以这只是上界

    # 当前 cosine sim 0.93 可能已经是最佳结果（考虑 FP8 误差）


def main():
    print("Scale Expansion Factor Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    test_expansion_factors()
    test_with_sfvecsize()


if __name__ == "__main__":
    main()
