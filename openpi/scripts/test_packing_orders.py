#!/usr/bin/env python3
"""
测试不同的 FP4 打包顺序

CUTLASS 可能使用不同的 nibble 顺序
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    prepare_scales_for_cutlass,
    BLOCK_SIZE,
    CUTLASS_ROW_TILE,
    CUTLASS_K_TILE,
)


def pack_nvfp4_v1(quantized, block_size=32):
    """原始打包: low=even, high=odd"""
    M, K = quantized.shape

    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 device=quantized.device)

    signs = (quantized < 0).to(torch.uint8)
    abs_vals = quantized.abs()

    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    encoded = (signs << 3) | indices

    low = encoded[:, 0::2]
    high = encoded[:, 1::2]
    packed = low | (high << 4)

    return packed


def pack_nvfp4_v2(quantized, block_size=32):
    """交换打包: low=odd, high=even"""
    M, K = quantized.shape

    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 device=quantized.device)

    signs = (quantized < 0).to(torch.uint8)
    abs_vals = quantized.abs()

    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    encoded = (signs << 3) | indices

    # Swap
    low = encoded[:, 1::2]
    high = encoded[:, 0::2]
    packed = low | (high << 4)

    return packed


def pack_nvfp4_v3(quantized, block_size=32):
    """逐行交替: 每 32 元素内的打包方式可能不同"""
    M, K = quantized.shape

    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 device=quantized.device)

    signs = (quantized < 0).to(torch.uint8)
    abs_vals = quantized.abs()

    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    encoded = (signs << 3) | indices

    # 重新排列: 每个 block 内的顺序可能不同
    # 尝试 K-major packing 在 block 内
    packed = torch.zeros(M, K // 2, dtype=torch.uint8, device=quantized.device)

    for m in range(M):
        for b in range(K // block_size):
            for i in range(block_size // 2):
                idx = b * block_size + i * 2
                packed[m, b * (block_size // 2) + i] = encoded[m, idx] | (encoded[m, idx + 1] << 4)

    return packed


def test_packing_order(pack_func, name):
    """测试特定的打包函数"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE

    # 统一数据测试
    x = torch.ones(M, K, device='cuda', dtype=torch.float32)
    w = torch.ones(N, K, device='cuda', dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    # 使用指定的打包函数
    x_packed = pack_func(x_q, block_size)
    w_packed = pack_func(w_q, block_size)

    # 准备 scales
    num_k_blocks = K // block_size
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True)

    # 参考
    ref_val = K  # ones @ ones = K

    try:
        output = nvfp4_gemm.gemm_prepared(
            x_packed.cuda(),
            w_packed.cuda(),
            x_scales_fp8.cuda(),
            w_scales_fp8.cuda(),
            M, N, K
        )

        out_mean = output.mean().item()
        out_0_0 = output[0, 0].item()
        ratio = out_0_0 / ref_val if ref_val != 0 else 0

        print(f"Output[0,0]: {out_0_0:.2f}")
        print(f"Output mean: {out_mean:.2f}")
        print(f"Expected: {ref_val}")
        print(f"Ratio: {ratio:.4f}")

        # 判断
        if 0.9 < ratio < 1.1:
            print("GOOD - Close to expected!")
            return True
        else:
            print("BAD - Far from expected")
            return False

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_random_data():
    """用随机数据测试 cosine similarity"""
    print(f"\n{'='*60}")
    print("Random Data Test")
    print('='*60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE

    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w = torch.randn(N, K, device='cuda', dtype=torch.float32)

    # 参考
    ref = torch.matmul(x, w.T)

    # 量化和打包
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    # 尝试不同打包
    for pack_func, name in [(pack_nvfp4_v1, "v1 (low=even)"),
                             (pack_nvfp4_v2, "v2 (low=odd)"),
                             (pack_nvfp4_v3, "v3 (block order)")]:
        print(f"\n--- {name} ---")

        x_packed = pack_func(x_q, block_size)
        w_packed = pack_func(w_q, block_size)

        num_k_blocks = K // block_size
        x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True)
        w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True)

        try:
            output = nvfp4_gemm.gemm_prepared(
                x_packed.cuda(),
                w_packed.cuda(),
                x_scales_fp8.cuda(),
                w_scales_fp8.cuda(),
                M, N, K
            )

            cos_sim = F.cosine_similarity(
                output.flatten().float().unsqueeze(0),
                ref.flatten().unsqueeze(0)
            ).item()

            print(f"Cosine similarity: {cos_sim:.6f}")

            if cos_sim > 0.9:
                print("GOOD!")
            elif cos_sim > 0.5:
                print("Partial match")
            else:
                print("Poor match")

        except Exception as e:
            print(f"FAILED: {e}")


def main():
    print("Testing FP4 Packing Orders")

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    # 测试统一数据
    results = {}
    results['v1'] = test_packing_order(pack_nvfp4_v1, "v1 (low=even, high=odd)")
    results['v2'] = test_packing_order(pack_nvfp4_v2, "v2 (low=odd, high=even)")
    results['v3'] = test_packing_order(pack_nvfp4_v3, "v3 (block reorder)")

    # 随机数据测试
    test_random_data()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
