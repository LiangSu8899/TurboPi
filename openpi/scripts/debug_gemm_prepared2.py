#!/usr/bin/env python3
"""
调试 NVFP4 GEMM - 第二轮: 测试非零数据
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    prepare_scales_for_cutlass,
    pack_nvfp4_data,
    BLOCK_SIZE,
    CUTLASS_ROW_TILE,
    CUTLASS_K_TILE,
)


def test_ones_scales():
    """使用 scale=1.0 测试"""
    print("=" * 60)
    print("Test: All ones scales, zeros data")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # 全零 packed 数据
    x_packed = torch.zeros(M, K // 2, dtype=torch.uint8, device='cuda')
    w_packed = torch.zeros(N, K // 2, dtype=torch.uint8, device='cuda')

    # Scale = 1.0 in FP8 E4M3
    # FP8 E4M3: 1.0 = 0b0_0111_000 = 0x38
    fp8_one = 0x38

    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((num_k_blocks + CUTLASS_K_TILE - 1) // CUTLASS_K_TILE) * CUTLASS_K_TILE

    x_scales_fp8 = torch.full((M_padded * K_padded,), fp8_one, dtype=torch.uint8, device='cuda')
    w_scales_fp8 = torch.full((N_padded * K_padded,), fp8_one, dtype=torch.uint8, device='cuda')

    print(f"FP8 scale value: 0x{fp8_one:02X}")
    print(f"x_scales_fp8: {x_scales_fp8.shape}")
    print(f"w_scales_fp8: {w_scales_fp8.shape}")

    try:
        output = nvfp4_gemm.gemm_prepared(
            x_packed, w_packed,
            x_scales_fp8, w_scales_fp8,
            M, N, K
        )
        print(f"Output: {output.shape}")
        print(f"Output sum: {output.sum().item()}")  # Should be 0
        print(f"Output has NaN: {output.isnan().any().item()}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_simple_values():
    """测试简单非零值"""
    print("\n" + "=" * 60)
    print("Test: Simple non-zero values")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # 创建简单的 packed 数据
    # FP4 值 1.0 = index 2, 编码 0x2
    # 两个 1.0 打包: (0x2 << 4) | 0x2 = 0x22
    fp4_one_packed = 0x22

    x_packed = torch.full((M, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')
    w_packed = torch.full((N, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')

    # Scale = 1.0
    fp8_one = 0x38
    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((num_k_blocks + CUTLASS_K_TILE - 1) // CUTLASS_K_TILE) * CUTLASS_K_TILE

    x_scales_fp8 = torch.full((M_padded * K_padded,), fp8_one, dtype=torch.uint8, device='cuda')
    w_scales_fp8 = torch.full((N_padded * K_padded,), fp8_one, dtype=torch.uint8, device='cuda')

    print(f"FP4 packed value: 0x{fp4_one_packed:02X} (two 1.0 values)")
    print(f"FP8 scale value: 0x{fp8_one:02X}")

    # 期望结果: 每个输出元素 = K * 1.0 * 1.0 * 1.0 * 1.0 = K = 2048
    expected_value = K

    try:
        output = nvfp4_gemm.gemm_prepared(
            x_packed, w_packed,
            x_scales_fp8, w_scales_fp8,
            M, N, K
        )
        print(f"Output: {output.shape}")
        print(f"Output[0,0]: {output[0,0].item():.2f}")
        print(f"Expected: {expected_value}")
        print(f"Output has NaN: {output.isnan().any().item()}")
        print(f"Output has Inf: {output.isinf().any().item()}")

        # 检查是否接近预期
        mean_val = output.mean().item()
        print(f"Output mean: {mean_val:.2f}")

        return not output.isnan().any().item()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_python_quantize():
    """使用 Python 量化但正确的 FP8 转换"""
    print("\n" + "=" * 60)
    print("Test: Python quantize with proper FP8 conversion")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE

    # 简单输入: 全 1
    x = torch.ones(M, K, device='cuda', dtype=torch.float32)
    w = torch.ones(N, K, device='cuda', dtype=torch.float32)

    # Python 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    print(f"x_q stats: min={x_q.min():.2f}, max={x_q.max():.2f}")
    print(f"x_scales stats: min={x_scales.min():.4f}, max={x_scales.max():.4f}")

    # 打包
    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # 准备 scales (FP8 + reorder)
    num_k_blocks = K // block_size
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True)

    print(f"\nPrepared data:")
    print(f"  x_packed: {x_packed.shape}")
    print(f"  x_scales_fp8: {x_scales_fp8.shape}")

    # 计算期望结果
    # 输入全 1，量化后应该还是 1，scale = 1/6 ≈ 0.167
    # 输出 = K * 1 * scale_x * 1 * scale_w ≈ K * 0.167 * 0.167 ≈ K * 0.028
    expected_approx = K * (1.0 / 6.0) * (1.0 / 6.0)  # ~57

    try:
        output = nvfp4_gemm.gemm_prepared(
            x_packed.cuda(),
            w_packed.cuda(),
            x_scales_fp8.cuda(),
            w_scales_fp8.cuda(),
            M, N, K
        )

        print(f"\nOutput: {output.shape}")
        print(f"Output[0,0]: {output[0,0].item():.2f}")
        print(f"Output mean: {output.mean().item():.2f}")
        print(f"Expected approx: {expected_approx:.2f}")
        print(f"Output has NaN: {output.isnan().any().item()}")

        # BF16 参考
        ref = torch.matmul(x, w.T)
        print(f"\nBF16 reference[0,0]: {ref[0,0].item():.2f}")

        return not output.isnan().any().item()

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fp8_conversion():
    """验证 FP8 转换是否正确"""
    print("\n" + "=" * 60)
    print("Test: FP8 E4M3 conversion verification")
    print("=" * 60)

    # FP8 E4M3 格式: 1 sign + 4 exponent + 3 mantissa
    # 正数格式: 0_eeee_mmm
    # 值 = 2^(e-7) * (1 + m/8)  for 1 <= e <= 14
    # 值 = 2^(-6) * (m/8)       for e = 0 (subnormal)

    test_values = [0.5, 1.0, 2.0, 4.0, 6.0, 0.167]
    scales = torch.tensor(test_values, device='cuda', dtype=torch.float32)

    # PyTorch FP8 转换
    scales_fp8 = scales.to(torch.float8_e4m3fn)
    scales_fp8_uint8 = scales_fp8.view(torch.uint8)

    # 转回 float 验证
    scales_back = scales_fp8.to(torch.float32)

    print("Value -> FP8 (hex) -> Back to float")
    for i, v in enumerate(test_values):
        fp8_val = scales_fp8_uint8[i].item()
        back_val = scales_back[i].item()
        print(f"  {v:6.3f} -> 0x{fp8_val:02X} -> {back_val:6.3f}")

    return True


def main():
    print("NVFP4 GEMM Debug Tests - Round 2")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    results = {}

    results['fp8_conv'] = test_fp8_conversion()
    results['ones_scales'] = test_ones_scales()
    results['simple_values'] = test_simple_values()
    results['python_quantize'] = test_with_python_quantize()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
