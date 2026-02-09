#!/usr/bin/env python3
"""
调试 NVFP4 大输入范围问题

layer16 down_proj 输入范围 [-5248, 430] 导致 NaN
"""

import torch
import torch.nn.functional as F
import sys
import json
import pathlib

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    NVFP4Linear,
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    pack_nvfp4_data,
    prepare_scales_for_cutlass,
    BLOCK_SIZE,
)


def test_nvfp4_with_large_input():
    """测试 NVFP4 对大输入范围的处理"""
    print("=" * 70)
    print("测试 NVFP4 对大输入范围的处理")
    print("=" * 70)

    device = torch.device('cuda')

    # 模拟 layer16 down_proj 的输入
    # Shape: [1, 968, 16384] -> flatten to [968, 16384]
    batch_size = 968
    in_features = 16384
    out_features = 2048

    # 创建 NVFP4 Linear
    linear = torch.nn.Linear(in_features, out_features, bias=False).cuda()
    nvfp4_linear = NVFP4Linear.from_linear(linear, BLOCK_SIZE, use_cutlass=True)
    nvfp4_linear = nvfp4_linear.cuda()
    nvfp4_linear.prepare_for_cutlass()

    print(f"\n配置: batch={batch_size}, in={in_features}, out={out_features}")
    print("-" * 70)

    # 测试不同的输入范围
    test_ranges = [
        (-10, 10, "Normal [-10, 10]"),
        (-100, 100, "Medium [-100, 100]"),
        (-1000, 500, "Large [-1000, 500]"),
        (-5000, 500, "Very Large [-5000, 500]"),  # 接近实际情况
        (-5248, 430, "Actual [-5248, 430]"),
    ]

    for min_val, max_val, name in test_ranges:
        # 生成测试输入
        x = torch.rand(batch_size, in_features, device=device, dtype=torch.float32)
        x = x * (max_val - min_val) + min_val

        # 检查 quantize 步骤
        x_q, x_scales = quantize_to_nvfp4_sim(x, BLOCK_SIZE, use_mse_search=False)

        # 检查 scales 范围
        scale_min = x_scales.min().item()
        scale_max = x_scales.max().item()

        # 检查 FP8 转换后的 scales
        x_scales_fp8 = x_scales.to(torch.float8_e4m3fn)
        x_scales_fp8_back = x_scales_fp8.to(torch.float32)
        scale_fp8_error = (x_scales - x_scales_fp8_back).abs().max().item()

        # CUTLASS forward
        nvfp4_linear.use_cutlass = True
        with torch.no_grad():
            try:
                out_cutlass = nvfp4_linear(x)
                has_nan_cutlass = torch.isnan(out_cutlass).any().item()
            except Exception as e:
                has_nan_cutlass = True
                out_cutlass = None

        # Simulation forward
        nvfp4_linear.use_cutlass = False
        with torch.no_grad():
            out_sim = nvfp4_linear(x)
            has_nan_sim = torch.isnan(out_sim).any().item()

        # Compare
        if out_cutlass is not None and not has_nan_cutlass and not has_nan_sim:
            cos_sim = F.cosine_similarity(
                out_cutlass.flatten().float().unsqueeze(0),
                out_sim.flatten().float().unsqueeze(0)
            ).item()
        else:
            cos_sim = float('nan')

        # Print results
        cutlass_status = "NaN!" if has_nan_cutlass else "OK"
        sim_status = "NaN!" if has_nan_sim else "OK"

        print(f"\n{name}:")
        print(f"  Input range:    [{min_val}, {max_val}]")
        print(f"  Scale range:    [{scale_min:.4f}, {scale_max:.4f}]")
        print(f"  FP8 scale err:  {scale_fp8_error:.6f}")
        print(f"  Simulation:     {sim_status}")
        print(f"  CUTLASS:        {cutlass_status}")
        print(f"  Cosine sim:     {cos_sim:.6f}")

    # 测试 scale overflow
    print("\n" + "=" * 70)
    print("测试 Scale Overflow")
    print("=" * 70)

    # FP8 E4M3 的最大值约为 448
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    print(f"\nFP8 E4M3 max value: {fp8_max}")

    # 计算导致 overflow 的输入值
    # scale = max(abs(input)) / 6.0  (NVFP4 max = 6)
    # 如果 scale > 448, 就会 overflow
    overflow_threshold = fp8_max * 6.0
    print(f"Scale overflow threshold: input max > {overflow_threshold:.0f}")

    # 实际看到的最大输入
    actual_max = 5248
    actual_scale = actual_max / 6.0
    print(f"\nActual max input: {actual_max}")
    print(f"Actual scale: {actual_scale:.1f}")
    print(f"FP8 representable: {'Yes' if actual_scale <= fp8_max else 'NO - OVERFLOW!'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_nvfp4_with_large_input()
