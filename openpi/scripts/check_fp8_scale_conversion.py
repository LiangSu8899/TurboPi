#!/usr/bin/env python3
"""
检查 FP8 Scale 转换的精度

发现: CUTLASS 输出比 Python 参考高 3-6%
假设: FP8 E4M3 表示的 scale 偏大
"""

import torch
import sys

sys.path.insert(0, '/workspace/src')


def analyze_fp8_representation():
    """分析 FP8 E4M3 的表示能力"""
    print("=" * 60)
    print("FP8 E4M3 Representation Analysis")
    print("=" * 60)

    # 测试常见 scale 值的 FP8 转换
    test_scales = [
        0.1666666716337204,  # 1/6 (全 1 输入的 scale)
        0.083333,  # 0.5/6
        1.0,
        0.5,
        0.25,
        0.125,
        2.0,
    ]

    print("\nScale value -> FP8 -> Back to FP32:")
    for scale in test_scales:
        scale_tensor = torch.tensor([scale], dtype=torch.float32, device='cuda')

        # 转换为 FP8
        scale_fp8 = scale_tensor.to(torch.float8_e4m3fn)

        # 转回 FP32
        scale_back = scale_fp8.to(torch.float32)

        # 计算误差
        abs_error = abs(scale_back.item() - scale)
        rel_error = abs_error / scale * 100
        ratio = scale_back.item() / scale

        print(f"  {scale:.6f} -> {scale_fp8.view(torch.uint8).item():02X} -> {scale_back.item():.6f} "
              f"(error: {rel_error:.2f}%, ratio: {ratio:.4f})")


def test_scale_product_error():
    """测试 scale 乘积的误差累积"""
    print("\n" + "=" * 60)
    print("Scale Product Error Analysis")
    print("=" * 60)

    # 在 GEMM 中，每个输出元素是 sum(x_i * w_i * scale_x * scale_w)
    # 如果 scale 转换偏大，乘积会放大

    scale_x = 0.1666666716337204  # 1/6
    scale_w = 0.1666666716337204

    # FP32 乘积
    fp32_product = scale_x * scale_w

    # FP8 乘积
    scale_x_fp8 = torch.tensor([scale_x], dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    scale_w_fp8 = torch.tensor([scale_w], dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)

    scale_x_back = scale_x_fp8.to(torch.float32).item()
    scale_w_back = scale_w_fp8.to(torch.float32).item()
    fp8_product = scale_x_back * scale_w_back

    print(f"\nOriginal scales: x={scale_x:.6f}, w={scale_w:.6f}")
    print(f"FP32 product: {fp32_product:.6f}")
    print(f"FP8 scales: x={scale_x_back:.6f}, w={scale_w_back:.6f}")
    print(f"FP8 product: {fp8_product:.6f}")
    print(f"Product ratio: {fp8_product / fp32_product:.4f}")

    # 对于每个元素的输出
    # output = sum(fp4_x * fp4_w) * scale_x * scale_w
    # 如果 scale 偏大，输出也会偏大

    fp4_product = 6 * 6  # 假设 FP4 值都是 6
    K = 64

    fp32_output = K * fp4_product * fp32_product
    fp8_output = K * fp4_product * fp8_product

    print(f"\nFor K={K}, FP4 values = 6:")
    print(f"  FP32 expected output: {fp32_output:.4f}")
    print(f"  FP8 expected output: {fp8_output:.4f}")
    print(f"  Ratio: {fp8_output / fp32_output:.4f}")


def test_unsigned_vs_signed_fp8():
    """测试 unsigned vs signed FP8"""
    print("\n" + "=" * 60)
    print("Unsigned vs Signed FP8")
    print("=" * 60)

    # CUTLASS 使用 float_ue4m3_t (unsigned E4M3)
    # PyTorch 使用 float8_e4m3fn (signed E4M3)

    # 对于正的 scale 值，它们应该一样
    # 但让我验证一下

    test_scales = [0.1666666716337204, 0.5, 1.0, 2.0]

    print("\nScale -> PyTorch FP8 (signed) -> bits:")
    for scale in test_scales:
        scale_tensor = torch.tensor([scale], dtype=torch.float32, device='cuda')
        scale_fp8 = scale_tensor.to(torch.float8_e4m3fn)
        bits = scale_fp8.view(torch.uint8).item()
        scale_back = scale_fp8.to(torch.float32).item()

        # 分析 bits
        sign_bit = (bits >> 7) & 1
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7

        print(f"  {scale:.6f} -> 0x{bits:02X} (sign={sign_bit}, exp={exponent}, mant={mantissa}) -> {scale_back:.6f}")

    print("\nNote: CUTLASS float_ue4m3_t is unsigned, so sign bit should always be 0")
    print("      But FP8 E4M3 has bias=7, so exponent 0xF represents very large values")


def suggest_fix():
    """建议修复方案"""
    print("\n" + "=" * 60)
    print("Suggested Fix")
    print("=" * 60)

    print("""
分析结果:
- FP8 E4M3 转换导致 scale 轻微偏大
- 这导致 CUTLASS 输出系统性偏高 3-6%

可能的修复方案:

1. **调整 scale 计算**:
   - 在量化时考虑 FP8 表示，选择更精确的 scale
   - 使用 FP8 能精确表示的值

2. **使用 MSE 优化时考虑 FP8**:
   - 在 MSE 搜索中，先转换为 FP8 再计算误差
   - 这样选择的 scale 会对 FP8 更友好

3. **检查 CUTLASS 的 scale 处理**:
   - 验证 CUTLASS 如何应用 scale
   - 可能需要调整我们传入的 scale 值

4. **实验性: 预补偿**:
   - 如果误差是系统性的，可以预先缩小 scale
   - 例如 scale *= 0.94 来补偿
""")


def main():
    print("FP8 Scale Conversion Check")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    analyze_fp8_representation()
    test_scale_product_error()
    test_unsigned_vs_signed_fp8()
    suggest_fix()


if __name__ == "__main__":
    main()
