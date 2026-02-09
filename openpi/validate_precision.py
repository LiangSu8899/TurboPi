#!/usr/bin/env python3
"""
验证修复后的 NVFP4 GEMM 精度
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/workspace/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    prepare_scales_for_cutlass,
    pack_nvfp4_data,
    BLOCK_SIZE,
)


def test_uniform_data():
    """测试统一数据"""
    print("=" * 60)
    print("Test 1: Uniform Data (ones)")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    x = torch.ones(M, K, device='cuda', dtype=torch.float32)
    w = torch.ones(N, K, device='cuda', dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    # 打包
    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # 准备 scales (使用新的 K 参数)
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=K)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=K)

    print(f"x_scales_fp8 size: {x_scales_fp8.numel()}")
    print(f"w_scales_fp8 size: {w_scales_fp8.numel()}")

    # CUTLASS GEMM
    output = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(),
        w_packed.cuda(),
        x_scales_fp8.cuda(),
        w_scales_fp8.cuda(),
        M, N, K
    )

    # BF16 参考
    ref = torch.matmul(x, w.T)

    # Python 模拟
    x_dequant = dequantize_nvfp4_sim(x_q, x_scales, block_size)
    w_dequant = dequantize_nvfp4_sim(w_q, w_scales, block_size)
    sim_output = torch.matmul(x_dequant.cuda(), w_dequant.cuda().T)

    print(f"\nBF16 ref[0,0]: {ref[0,0].item():.2f}")
    print(f"Python sim[0,0]: {sim_output[0,0].item():.2f}")
    print(f"CUTLASS[0,0]: {output[0,0].item():.2f}")
    print(f"CUTLASS[0,-1]: {output[0,-1].item():.2f}")
    print(f"CUTLASS mean: {output.mean().item():.2f}")

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        output.flatten().float().unsqueeze(0),
        sim_output.flatten().unsqueeze(0)
    ).item()
    print(f"\nCosine similarity (CUTLASS vs Python sim): {cos_sim:.6f}")


def test_random_data():
    """测试随机数据精度"""
    print("\n" + "=" * 60)
    print("Test 2: Random Data")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384  # 使用实际的 MLP 尺寸
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w = torch.randn(N, K, device='cuda', dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    # 打包
    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    # 准备 scales
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=K)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=K)

    print(f"x_scales_fp8 size: {x_scales_fp8.numel()}")
    print(f"w_scales_fp8 size: {w_scales_fp8.numel()}")

    # CUTLASS GEMM
    output = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(),
        w_packed.cuda(),
        x_scales_fp8.cuda(),
        w_scales_fp8.cuda(),
        M, N, K
    )

    # BF16 参考
    ref = torch.matmul(x, w.T)

    # Python 模拟
    x_dequant = dequantize_nvfp4_sim(x_q, x_scales, block_size)
    w_dequant = dequantize_nvfp4_sim(w_q, w_scales, block_size)
    sim_output = torch.matmul(x_dequant.cuda(), w_dequant.cuda().T)

    print(f"\nOutput shapes: ref={ref.shape}, sim={sim_output.shape}, cutlass={output.shape}")

    # Cosine similarities
    cos_sim_cutlass_sim = F.cosine_similarity(
        output.flatten().float().unsqueeze(0),
        sim_output.flatten().unsqueeze(0)
    ).item()

    cos_sim_sim_ref = F.cosine_similarity(
        sim_output.flatten().unsqueeze(0),
        ref.flatten().unsqueeze(0)
    ).item()

    cos_sim_cutlass_ref = F.cosine_similarity(
        output.flatten().float().unsqueeze(0),
        ref.flatten().unsqueeze(0)
    ).item()

    print(f"Cosine similarity (CUTLASS vs Python sim): {cos_sim_cutlass_sim:.6f}")
    print(f"Cosine similarity (Python sim vs BF16 ref): {cos_sim_sim_ref:.6f}")
    print(f"Cosine similarity (CUTLASS vs BF16 ref): {cos_sim_cutlass_ref:.6f}")

    # 相对误差
    rel_error_cutlass_sim = (output.float() - sim_output).abs() / (sim_output.abs() + 1e-8)
    print(f"Mean relative error (CUTLASS vs sim): {rel_error_cutlass_sim.mean().item() * 100:.2f}%")

    return cos_sim_cutlass_sim


def test_different_sizes():
    """测试不同尺寸"""
    print("\n" + "=" * 60)
    print("Test 3: Different Sizes")
    print("=" * 60)

    import nvfp4_gemm

    test_cases = [
        (256, 2048, 256),
        (256, 2048, 512),
        (256, 2048, 1024),
        (256, 2048, 2048),
        (256, 2048, 4096),
        (256, 2048, 8192),
        (256, 2048, 16384),
    ]

    block_size = BLOCK_SIZE

    for M, K, N in test_cases:
        print(f"\n--- M={M}, K={K}, N={N} ---")

        num_k_blocks = K // block_size

        torch.manual_seed(42)
        x = torch.randn(M, K, device='cuda', dtype=torch.float32)
        w = torch.randn(N, K, device='cuda', dtype=torch.float32)

        x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
        w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

        x_packed = pack_nvfp4_data(x_q, block_size)
        w_packed = pack_nvfp4_data(w_q, block_size)

        x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=K)
        w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=K)

        try:
            output = nvfp4_gemm.gemm_prepared(
                x_packed.cuda(),
                w_packed.cuda(),
                x_scales_fp8.cuda(),
                w_scales_fp8.cuda(),
                M, N, K
            )

            # Python 模拟
            x_dequant = dequantize_nvfp4_sim(x_q, x_scales, block_size)
            w_dequant = dequantize_nvfp4_sim(w_q, w_scales, block_size)
            sim_output = torch.matmul(x_dequant.cuda(), w_dequant.cuda().T)

            cos_sim = F.cosine_similarity(
                output.flatten().float().unsqueeze(0),
                sim_output.flatten().unsqueeze(0)
            ).item()

            print(f"  Cosine sim: {cos_sim:.6f} {'✓' if cos_sim > 0.99 else '✗'}")

        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    print("NVFP4 GEMM Precision Validation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    test_uniform_data()
    cos_sim = test_random_data()
    test_different_sizes()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if cos_sim > 0.99:
        print("✓ NVFP4 GEMM precision is GOOD (cosine sim > 0.99)")
    else:
        print(f"✗ NVFP4 GEMM precision needs improvement (cosine sim = {cos_sim:.4f})")


if __name__ == "__main__":
    main()
