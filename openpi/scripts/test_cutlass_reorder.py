#!/usr/bin/env python3
"""
测试 CUTLASS C++ reorder_scales 函数

验证使用 CUTLASS layout 迭代器实现的 scale reordering 是否正确。
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    pack_nvfp4_data,
    convert_scales_to_fp8,
    BLOCK_SIZE,
)


def test_reorder_scales():
    """测试 C++ reorder_scales 函数"""
    print("=" * 70)
    print("Testing CUTLASS C++ reorder_scales function")
    print("=" * 70)

    try:
        import nvfp4_gemm
        print("[OK] CUTLASS extension loaded")
        print(f"Available functions: {dir(nvfp4_gemm)}")
    except ImportError as e:
        print(f"[ERROR] nvfp4_gemm not available: {e}")
        return

    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # 测试配置
    M, K, N = 256, 2048, 256
    num_k_blocks = K // BLOCK_SIZE

    print(f"Test config: M={M}, K={K}, N={N}")
    print(f"num_k_blocks={num_k_blocks}, BLOCK_SIZE={BLOCK_SIZE}\n")

    # 检查 layout size
    try:
        size_a = nvfp4_gemm.get_scale_layout_size(M, K, False)
        size_b = nvfp4_gemm.get_scale_layout_size(N, K, True)
        print(f"Scale layout size for A (M={M}, K={K}): {size_a}")
        print(f"Scale layout size for B (N={N}, K={K}): {size_b}")
        print(f"Expected (simple): M*num_k_blocks = {M * num_k_blocks}")
        print()
    except Exception as e:
        print(f"get_scale_layout_size failed: {e}")
        return

    # 准备测试数据
    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, BLOCK_SIZE, use_mse_search=False)
    w_q, w_scales = quantize_to_nvfp4_sim(w, BLOCK_SIZE, use_mse_search=False)

    x_packed = pack_nvfp4_data(x_q, BLOCK_SIZE)
    w_packed = pack_nvfp4_data(w_q, BLOCK_SIZE)

    # Python 参考 (使用 FP8 scales)
    x_deq = dequantize_nvfp4_sim(x_q, x_scales, BLOCK_SIZE, use_fp8_scales=True)
    w_deq = dequantize_nvfp4_sim(w_q, w_scales, BLOCK_SIZE, use_fp8_scales=True)
    ref = torch.matmul(x_deq, w_deq.T)

    # BF16 参考
    bf16_ref = torch.matmul(x, w.T)

    # 转换 scales 为 FP8
    x_scales_fp8 = convert_scales_to_fp8(x_scales.flatten())
    w_scales_fp8 = convert_scales_to_fp8(w_scales.flatten())

    print(f"x_scales_fp8 shape: {x_scales_fp8.shape}")
    print(f"w_scales_fp8 shape: {w_scales_fp8.shape}")
    print()

    # 测试 1: 使用 C++ reorder_scales
    print("=" * 70)
    print("Test 1: Using C++ reorder_scales function")
    print("=" * 70)

    try:
        x_scales_reordered = nvfp4_gemm.reorder_scales(x_scales_fp8, M, K, False)
        w_scales_reordered = nvfp4_gemm.reorder_scales(w_scales_fp8, N, K, True)

        print(f"x_scales_reordered shape: {x_scales_reordered.shape}")
        print(f"w_scales_reordered shape: {w_scales_reordered.shape}")

        # Padding if needed
        expected_size_x = size_a
        expected_size_w = size_b

        if x_scales_reordered.numel() < expected_size_x:
            pad = torch.zeros(expected_size_x - x_scales_reordered.numel(),
                            dtype=torch.uint8, device=device)
            x_scales_reordered = torch.cat([x_scales_reordered, pad])

        if w_scales_reordered.numel() < expected_size_w:
            pad = torch.zeros(expected_size_w - w_scales_reordered.numel(),
                            dtype=torch.uint8, device=device)
            w_scales_reordered = torch.cat([w_scales_reordered, pad])

        # Run CUTLASS GEMM
        output = nvfp4_gemm.gemm_prepared(
            x_packed, w_packed,
            x_scales_reordered, w_scales_reordered,
            M, N, K
        )

        # 计算 cosine similarity
        cos_ref = F.cosine_similarity(
            output.flatten().float().unsqueeze(0),
            ref.flatten().unsqueeze(0)
        ).item()

        cos_bf16 = F.cosine_similarity(
            output.flatten().float().unsqueeze(0),
            bf16_ref.flatten().unsqueeze(0)
        ).item()

        print(f"\nResults with C++ reorder_scales:")
        print(f"  Cosine sim vs Python ref:  {cos_ref:.6f}")
        print(f"  Cosine sim vs BF16:        {cos_bf16:.6f}")

        if cos_ref > 0.98:
            print("\n  SUCCESS! >0.98 cosine similarity!")
        elif cos_ref > 0.95:
            print("\n  Good progress. >0.95 cosine similarity.")
        else:
            print(f"\n  Still low ({cos_ref:.4f}). Need to fix the layout mapping.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # 测试 2: 对比原始方法 (baseline)
    print()
    print("=" * 70)
    print("Test 2: Baseline (original Python scales, no reorder)")
    print("=" * 70)

    try:
        # 直接使用 flatten 的 scales (原始方法)
        x_scales_flat = x_scales_fp8
        w_scales_flat = w_scales_fp8

        # Expand to match CUTLASS expected size
        if x_scales_flat.numel() < size_a:
            # Repeat to fill
            x_scales_flat = x_scales_flat.repeat((size_a // x_scales_flat.numel()) + 1)[:size_a]
        if w_scales_flat.numel() < size_b:
            w_scales_flat = w_scales_flat.repeat((size_b // w_scales_flat.numel()) + 1)[:size_b]

        output_baseline = nvfp4_gemm.gemm_prepared(
            x_packed, w_packed,
            x_scales_flat, w_scales_flat,
            M, N, K
        )

        cos_baseline = F.cosine_similarity(
            output_baseline.flatten().float().unsqueeze(0),
            ref.flatten().unsqueeze(0)
        ).item()

        print(f"Baseline cosine sim: {cos_baseline:.6f}")

    except Exception as e:
        print(f"Baseline error: {e}")

    # 测试 3: Debug print layout
    print()
    print("=" * 70)
    print("Test 3: Debug print layout mapping (first 32 entries)")
    print("=" * 70)

    try:
        nvfp4_gemm.debug_print_layout(256, 128, 32)
    except Exception as e:
        print(f"Debug print error: {e}")


if __name__ == "__main__":
    test_reorder_scales()
