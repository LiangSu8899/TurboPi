#!/usr/bin/env python3
"""
精确测试 Scale 索引问题

策略：
1. 使用 uniform scale = 1.0 验证基础正确性
2. 使用特定模式的 scale 来测试索引
3. 逐步定位 layout 问题
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


def test_uniform_scale():
    """测试 uniform scale = 1.0"""
    print("=" * 60)
    print("Test 1: Uniform Scale = 1.0")
    print("=" * 60)

    try:
        import nvfp4_gemm
    except ImportError:
        print("nvfp4_gemm not available")
        return

    device = torch.device('cuda')
    M, K, N = 256, 2048, 256

    # 使用全 1 输入
    x = torch.ones(M, K, device=device, dtype=torch.float32)
    w = torch.ones(N, K, device=device, dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, BLOCK_SIZE, use_mse_search=False)
    w_q, w_scales = quantize_to_nvfp4_sim(w, BLOCK_SIZE, use_mse_search=False)

    print(f"x_scales unique: {x_scales.unique().tolist()}")
    print(f"w_scales unique: {w_scales.unique().tolist()}")

    # Pack data
    x_packed = pack_nvfp4_data(x_q, BLOCK_SIZE)
    w_packed = pack_nvfp4_data(w_q, BLOCK_SIZE)

    # 计算 padding 后的尺寸
    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((K + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE

    # 使用真正的 uniform scale = 1.0 (FP8: 0x38)
    fp8_one = 0x38
    x_scales_fp8 = torch.full((M_padded * K_padded,), fp8_one, dtype=torch.uint8, device=device)
    w_scales_fp8 = torch.full((N_padded * K_padded,), fp8_one, dtype=torch.uint8, device=device)

    output = nvfp4_gemm.gemm_prepared(
        x_packed, w_packed,
        x_scales_fp8, w_scales_fp8,
        M, N, K
    )

    # 期望值: K * 1 * 1 = K = 2048 (全 1 输入，scale=1)
    expected = K
    print(f"\nExpected: {expected}")
    print(f"Output[0,0]: {output[0,0].item():.2f}")
    print(f"Output mean: {output.mean().item():.2f}")
    print(f"Output unique count: {output.unique().numel()}")

    # 检查是否所有元素相同
    if output.unique().numel() == 1:
        print("All outputs are the same (GOOD)")
    else:
        print("Outputs vary (may indicate issues)")


def test_scale_indexing_pattern():
    """测试 scale 的索引模式"""
    print("\n" + "=" * 60)
    print("Test 2: Scale Indexing Pattern")
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

    # 创建有模式的 x_scales
    # 只有第一个 k-block 的 scale 是 1.0，其他是 0.5
    fp8_one = 0x38  # 1.0
    fp8_half = 0x30  # 0.5

    # 方法 A: row-major layout [M, K]
    x_scales_2d = torch.full((M_padded, K_padded), fp8_half, dtype=torch.uint8, device=device)
    # 设置第一列 (k=0:32) 的 scale 为 1.0
    x_scales_2d[:, :32] = fp8_one

    w_scales_fp8 = torch.full((N_padded * K_padded,), fp8_one, dtype=torch.uint8, device=device)

    output_A = nvfp4_gemm.gemm_prepared(
        x_packed, w_packed,
        x_scales_2d.flatten(), w_scales_fp8,
        M, N, K
    )

    # 方法 B: 使用 repeat_interleave 模拟
    x_scales_blocks = torch.full((M, num_k_blocks), 0.5, dtype=torch.float32, device=device)
    x_scales_blocks[:, 0] = 1.0  # 第一个 k-block scale = 1.0

    x_scales_expanded = x_scales_blocks.repeat_interleave(block_size, dim=1)
    if x_scales_expanded.shape[0] < M_padded:
        extra = torch.full((M_padded - M, x_scales_expanded.shape[1]), 0.5, device=device)
        x_scales_expanded = torch.cat([x_scales_expanded, extra], dim=0)
    if x_scales_expanded.shape[1] < K_padded:
        extra = torch.full((M_padded, K_padded - x_scales_expanded.shape[1]), 0.5, device=device)
        x_scales_expanded = torch.cat([x_scales_expanded, extra], dim=1)

    x_scales_B = convert_scales_to_fp8(x_scales_expanded.flatten())

    output_B = nvfp4_gemm.gemm_prepared(
        x_packed, w_packed,
        x_scales_B, w_scales_fp8,
        M, N, K
    )

    # 期望值分析
    # 第一个 k-block (32 elements): scale = 1.0, 贡献 = 32 * 1 = 32
    # 其余 k-blocks (2016 elements): scale = 0.5, 贡献 = 2016 * 0.5 = 1008
    # 总和 = 32 + 1008 = 1040
    expected = 32 * 1.0 + (K - 32) * 0.5

    print(f"Expected (first k-block=1.0, rest=0.5): {expected}")
    print(f"Method A (row-major): {output_A[0,0].item():.2f}")
    print(f"Method B (repeat_interleave): {output_B[0,0].item():.2f}")

    # 测试不同的 k-block
    print("\n测试不同 k-block 的 scale:")
    for test_k_block in [0, 1, 31, 32, 63]:
        x_scales_test = torch.full((M, num_k_blocks), 0.5, dtype=torch.float32, device=device)
        if test_k_block < num_k_blocks:
            x_scales_test[:, test_k_block] = 2.0  # 使用 2.0 使效果更明显

        x_scales_expanded = x_scales_test.repeat_interleave(block_size, dim=1)
        if x_scales_expanded.shape[0] < M_padded:
            extra = torch.full((M_padded - M, x_scales_expanded.shape[1]), 0.5, device=device)
            x_scales_expanded = torch.cat([x_scales_expanded, extra], dim=0)
        if x_scales_expanded.shape[1] < K_padded:
            extra = torch.full((M_padded, K_padded - x_scales_expanded.shape[1]), 0.5, device=device)
            x_scales_expanded = torch.cat([x_scales_expanded, extra], dim=1)

        x_scales_fp8 = convert_scales_to_fp8(x_scales_expanded.flatten())

        output_test = nvfp4_gemm.gemm_prepared(
            x_packed, w_packed,
            x_scales_fp8, w_scales_fp8,
            M, N, K
        )

        # 期望: 32 * 2.0 + (K - 32) * 0.5 = 64 + 1008 = 1072
        expected_test = 32 * 2.0 + (K - 32) * 0.5
        print(f"  k_block={test_k_block}: output={output_test[0,0].item():.2f}, expected={expected_test:.2f}")


def test_row_indexing():
    """测试 row 索引"""
    print("\n" + "=" * 60)
    print("Test 3: Row Indexing")
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

    # 只有第一行的 scale 不同
    print("Testing row-specific scales:")
    for test_row in [0, 1, 31, 32, 127, 128, 255]:
        x_scales_2d = torch.full((M, num_k_blocks), 0.5, dtype=torch.float32, device=device)
        if test_row < M:
            x_scales_2d[test_row, :] = 2.0  # 该行 scale = 2.0

        x_scales_expanded = x_scales_2d.repeat_interleave(block_size, dim=1)
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

        # 期望: test_row 行输出 = K * 2.0 = 4096, 其他行 = K * 0.5 = 1024
        expected_test_row = K * 2.0
        expected_other = K * 0.5

        print(f"  row={test_row}: output[row,0]={output[test_row,0].item():.2f}, "
              f"expected={expected_test_row:.2f}, "
              f"other_row={output[(test_row+1)%M,0].item():.2f}")


def main():
    print("Scale Indexing Debug")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    test_uniform_scale()
    test_scale_indexing_pattern()
    test_row_indexing()


if __name__ == "__main__":
    main()
