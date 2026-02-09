#!/usr/bin/env python3
"""
调试 scale 值的处理
"""

import torch
import sys

sys.path.insert(0, '/workspace/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    prepare_scales_for_cutlass,
    pack_nvfp4_data,
    convert_scales_to_fp8,
    BLOCK_SIZE,
)


def analyze_scale_conversion():
    """分析 scale 值的 FP8 转换"""
    print("=" * 60)
    print("Scale FP8 Conversion Analysis")
    print("=" * 60)

    # 测试不同的 scale 值
    test_scales = [0.167, 0.5, 1.0, 2.0, 0.1, 0.25]

    print("\nScale value -> FP8 -> Back to FP32:")
    for s in test_scales:
        scale_tensor = torch.tensor([s], dtype=torch.float32, device='cuda')
        scale_fp8 = convert_scales_to_fp8(scale_tensor)
        scale_back = scale_fp8.view(torch.float8_e4m3fn).to(torch.float32)

        error = abs(scale_back.item() - s) / s * 100
        print(f"  {s:.4f} -> 0x{scale_fp8.item():02X} -> {scale_back.item():.4f} (error: {error:.2f}%)")


def test_simple_scale():
    """测试简单的 scale 值"""
    print("\n" + "=" * 60)
    print("Simple Scale Test")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # 使用已知的 FP4 和 scale 值
    # FP4 value 1.0 = index 2, packed as 0x22
    fp4_one_packed = 0x22

    x_packed = torch.full((M, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')
    w_packed = torch.full((N, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')

    # 使用 scale = 1.0 (FP8: 0x38)
    fp8_one = 0x38

    # 计算正确的 scale tensor 大小
    from openpi.models_pytorch.nvfp4_mlp import CUTLASS_ROW_TILE
    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((K + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE

    x_scales_size = M_padded * K_padded
    w_scales_size = N_padded * K_padded

    print(f"Scale sizes: x={x_scales_size}, w={w_scales_size}")

    x_scales_fp8 = torch.full((x_scales_size,), fp8_one, dtype=torch.uint8, device='cuda')
    w_scales_fp8 = torch.full((w_scales_size,), fp8_one, dtype=torch.uint8, device='cuda')

    output = nvfp4_gemm.gemm_prepared(
        x_packed, w_packed,
        x_scales_fp8, w_scales_fp8,
        M, N, K
    )

    # 期望: K * 1.0 * 1.0 * 1.0 * 1.0 = K = 2048
    expected = K
    actual = output[0, 0].item()

    print(f"\nExpected: {expected}")
    print(f"Actual: {actual}")
    print(f"Ratio: {actual / expected:.4f}")

    # 检查是否所有元素都相同
    unique = output.unique()
    print(f"Unique output values: {unique.numel()}")
    if unique.numel() == 1:
        print(f"All outputs are: {unique[0].item():.2f}")


def compare_scale_preparation_methods():
    """比较不同的 scale 准备方法"""
    print("\n" + "=" * 60)
    print("Scale Preparation Methods Comparison")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # 使用统一输入
    x = torch.ones(M, K, device='cuda', dtype=torch.float32)
    w = torch.ones(N, K, device='cuda', dtype=torch.float32)

    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    print(f"Original x_scales: shape={x_scales.shape}, unique={x_scales.unique().tolist()}")
    print(f"Original w_scales: shape={w_scales.shape}, unique={w_scales.unique().tolist()}")

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # 方法 1: 使用新的 prepare_scales_for_cutlass (with K)
    x_scales_v1 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=K)
    w_scales_v1 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=K)

    print(f"\nMethod 1 (with K): x={x_scales_v1.numel()}, w={w_scales_v1.numel()}")
    print(f"  x_scales unique FP8 values: {x_scales_v1.unique().tolist()}")

    output_v1 = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(), w_packed.cuda(),
        x_scales_v1.cuda(), w_scales_v1.cuda(),
        M, N, K
    )

    print(f"  Output[0,0]: {output_v1[0,0].item():.2f}, [0,-1]: {output_v1[0,-1].item():.2f}")

    # 方法 2: 手动创建统一 scale = 1.0
    from openpi.models_pytorch.nvfp4_mlp import CUTLASS_ROW_TILE
    fp8_one = 0x38
    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((K + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE

    x_scales_v2 = torch.full((M_padded * K_padded,), fp8_one, dtype=torch.uint8, device='cuda')
    w_scales_v2 = torch.full((N_padded * K_padded,), fp8_one, dtype=torch.uint8, device='cuda')

    print(f"\nMethod 2 (manual scale=1.0): x={x_scales_v2.numel()}, w={w_scales_v2.numel()}")

    output_v2 = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(), w_packed.cuda(),
        x_scales_v2.cuda(), w_scales_v2.cuda(),
        M, N, K
    )

    print(f"  Output[0,0]: {output_v2[0,0].item():.2f}, [0,-1]: {output_v2[0,-1].item():.2f}")

    # 分析差异
    print(f"\nComparison:")
    print(f"  Method 1 uses quantized scale ≈ 0.167")
    print(f"  Method 2 uses scale = 1.0")
    print(f"  Ratio: {output_v2[0,0].item() / output_v1[0,0].item():.2f}")


def main():
    print("Scale Values Debug")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    analyze_scale_conversion()
    test_simple_scale()
    compare_scale_preparation_methods()


if __name__ == "__main__":
    main()
