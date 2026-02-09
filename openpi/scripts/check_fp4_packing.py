#!/usr/bin/env python3
"""
检查 FP4 数据的 packing 格式

NVFP4 (e2m1) 值: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
编码: sign_bit(1) + magnitude_index(3) = 4 bits

Packing: 两个 FP4 值打包成一个 byte
- Low nibble: even index value
- High nibble: odd index value
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
    prepare_scales_for_cutlass,
    BLOCK_SIZE,
    CUTLASS_ROW_TILE,
)


def analyze_fp4_encoding():
    """分析 FP4 编码"""
    print("=" * 60)
    print("FP4 Encoding Analysis")
    print("=" * 60)

    # NVFP4 值表
    nvfp4_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    print("NVFP4 值表:")
    for i, v in enumerate(nvfp4_values):
        print(f"  Index {i}: {v}")

    print("\n编码格式: sign_bit(1) + magnitude_index(3)")
    print("  正数: 0xxx (0-7)")
    print("  负数: 1xxx (8-15)")

    print("\n所有可能的 FP4 编码:")
    for sign in [0, 1]:
        sign_str = "+" if sign == 0 else "-"
        for idx in range(8):
            code = (sign << 3) | idx
            val = nvfp4_values[idx] * (1 if sign == 0 else -1)
            print(f"  0x{code:X}: {sign_str}{nvfp4_values[idx]} = {val}")


def test_packing_correctness():
    """测试 packing 是否正确"""
    print("\n" + "=" * 60)
    print("Packing Correctness Test")
    print("=" * 60)

    device = torch.device('cuda')

    # 测试已知值
    print("\n测试已知值:")

    # 创建一个简单的输入: [1.0, 1.0, 1.0, ...]
    test_cases = [
        ("All 1.0", torch.ones(256, 64, device=device)),
        ("All 0.5", torch.full((256, 64), 0.5, device=device)),
        ("All 6.0", torch.full((256, 64), 6.0, device=device)),
        ("Alternating 1/2", torch.tensor([[1.0, 2.0] * 32] * 256, device=device)),
    ]

    for name, test_data in test_cases:
        # 量化
        x_q, x_scales = quantize_to_nvfp4_sim(test_data, BLOCK_SIZE, use_mse_search=False)

        # 打包
        x_packed = pack_nvfp4_data(x_q, BLOCK_SIZE)

        # 检查打包结果
        print(f"\n{name}:")
        print(f"  Input unique: {test_data.unique().tolist()}")
        print(f"  Quantized unique: {x_q.unique().tolist()}")
        print(f"  Packed first 8 bytes: {[f'0x{b:02X}' for b in x_packed[0, :8].tolist()]}")

        # 解析打包值
        packed_byte = x_packed[0, 0].item()
        low_nibble = packed_byte & 0x0F
        high_nibble = (packed_byte >> 4) & 0x0F

        low_sign = (low_nibble >> 3) & 1
        low_mag = low_nibble & 7
        high_sign = (high_nibble >> 3) & 1
        high_mag = high_nibble & 7

        nvfp4_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        low_val = nvfp4_values[low_mag] * (-1 if low_sign else 1)
        high_val = nvfp4_values[high_mag] * (-1 if high_sign else 1)

        print(f"  First packed byte decoded:")
        print(f"    Low nibble (idx 0): sign={low_sign}, mag={low_mag} -> {low_val}")
        print(f"    High nibble (idx 1): sign={high_sign}, mag={high_mag} -> {high_val}")


def test_gemm_with_known_values():
    """使用已知值测试 GEMM"""
    print("\n" + "=" * 60)
    print("GEMM with Known Values")
    print("=" * 60)

    try:
        import nvfp4_gemm
    except ImportError:
        print("nvfp4_gemm not available")
        return

    device = torch.device('cuda')
    M, K, N = 256, 64, 256  # 小尺寸便于调试

    # 测试 1: 全 1 输入
    print("\nTest 1: All ones")
    x = torch.ones(M, K, device=device, dtype=torch.float32)
    w = torch.ones(N, K, device=device, dtype=torch.float32)

    x_q, x_scales = quantize_to_nvfp4_sim(x, BLOCK_SIZE, use_mse_search=False)
    w_q, w_scales = quantize_to_nvfp4_sim(w, BLOCK_SIZE, use_mse_search=False)

    x_packed = pack_nvfp4_data(x_q, BLOCK_SIZE)
    w_packed = pack_nvfp4_data(w_q, BLOCK_SIZE)

    # Python 参考
    x_deq = dequantize_nvfp4_sim(x_q, x_scales, BLOCK_SIZE)
    w_deq = dequantize_nvfp4_sim(w_q, w_scales, BLOCK_SIZE)
    ref = torch.matmul(x_deq, w_deq.T)

    print(f"  x_q unique: {x_q.unique().tolist()}")
    print(f"  x_scales unique: {x_scales.unique().tolist()}")
    print(f"  x_deq unique: {x_deq.unique().tolist()}")
    print(f"  Python ref [0,0]: {ref[0,0].item():.4f}")

    # CUTLASS
    num_k_blocks = K // BLOCK_SIZE
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=K)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=K)

    output = nvfp4_gemm.gemm_prepared(
        x_packed, w_packed,
        x_scales_fp8, w_scales_fp8,
        M, N, K
    )

    print(f"  CUTLASS output [0,0]: {output[0,0].item():.4f}")
    print(f"  Ratio (CUTLASS/Python): {output[0,0].item() / ref[0,0].item():.4f}")

    # 测试 2: 单位输入 (使输出接近预期)
    print("\nTest 2: Input that should give known output")

    # 设计: x = scale_factor, 使得量化后 x_q = 6 (最大值)
    # 然后 CUTLASS 应该输出 K * scale_x * scale_w * 6 * 6

    # 实际上更简单的测试: 直接用 6.0
    x = torch.full((M, K), 6.0, device=device, dtype=torch.float32)
    w = torch.ones(N, K, device=device, dtype=torch.float32)

    x_q, x_scales = quantize_to_nvfp4_sim(x, BLOCK_SIZE, use_mse_search=False)
    w_q, w_scales = quantize_to_nvfp4_sim(w, BLOCK_SIZE, use_mse_search=False)

    x_packed = pack_nvfp4_data(x_q, BLOCK_SIZE)
    w_packed = pack_nvfp4_data(w_q, BLOCK_SIZE)

    x_deq = dequantize_nvfp4_sim(x_q, x_scales, BLOCK_SIZE)
    w_deq = dequantize_nvfp4_sim(w_q, w_scales, BLOCK_SIZE)
    ref = torch.matmul(x_deq, w_deq.T)

    print(f"  x_q unique: {x_q.unique().tolist()}")
    print(f"  x_scales unique: {x_scales.unique().tolist()}")
    print(f"  Python ref [0,0]: {ref[0,0].item():.4f}")

    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=K)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=K)

    output = nvfp4_gemm.gemm_prepared(
        x_packed, w_packed,
        x_scales_fp8, w_scales_fp8,
        M, N, K
    )

    print(f"  CUTLASS output [0,0]: {output[0,0].item():.4f}")
    print(f"  Ratio (CUTLASS/Python): {output[0,0].item() / ref[0,0].item():.4f}")


def main():
    print("FP4 Packing Check")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    analyze_fp4_encoding()
    test_packing_correctness()
    test_gemm_with_known_values()


if __name__ == "__main__":
    main()
