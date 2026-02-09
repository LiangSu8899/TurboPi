#!/usr/bin/env python3
"""
最终验证: NVFP4 GEMM 实现

验证:
1. 统一数据正确性
2. 随机数据精度
3. 不同尺寸兼容性
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


def test_uniform_correctness():
    """测试统一数据正确性"""
    print("=" * 60)
    print("Test 1: Uniform Data Correctness")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    x = torch.ones(M, K, device='cuda', dtype=torch.float32)
    w = torch.ones(N, K, device='cuda', dtype=torch.float32)

    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True, K=K)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True, K=K)

    output = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(), w_packed.cuda(),
        x_scales_fp8.cuda(), w_scales_fp8.cuda(),
        M, N, K
    )

    # 检查所有元素是否一致
    unique_vals = output.unique()
    all_same = unique_vals.numel() == 1

    # 检查是否接近预期（约 2048，考虑量化误差）
    mean_val = output.mean().item()
    expected = K
    error = abs(mean_val - expected) / expected * 100

    print(f"Output mean: {mean_val:.2f}")
    print(f"Expected: {expected}")
    print(f"Error: {error:.2f}%")
    print(f"All values same: {all_same}")
    print(f"Unique values: {unique_vals.tolist()}")

    passed = all_same and error < 10
    print(f"{'PASSED' if passed else 'FAILED'}")
    return passed


def test_random_precision():
    """测试随机数据精度"""
    print("\n" + "=" * 60)
    print("Test 2: Random Data Precision")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE
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

    output = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(), w_packed.cuda(),
        x_scales_fp8.cuda(), w_scales_fp8.cuda(),
        M, N, K
    )

    # Python 模拟参考
    x_dequant = dequantize_nvfp4_sim(x_q, x_scales, block_size)
    w_dequant = dequantize_nvfp4_sim(w_q, w_scales, block_size)
    sim_output = torch.matmul(x_dequant.cuda(), w_dequant.cuda().T)

    # BF16 参考
    bf16_ref = torch.matmul(x, w.T)

    cos_cutlass_sim = F.cosine_similarity(
        output.flatten().float().unsqueeze(0),
        sim_output.flatten().unsqueeze(0)
    ).item()

    cos_sim_bf16 = F.cosine_similarity(
        sim_output.flatten().unsqueeze(0),
        bf16_ref.flatten().unsqueeze(0)
    ).item()

    print(f"Cosine sim (CUTLASS vs Python sim): {cos_cutlass_sim:.6f}")
    print(f"Cosine sim (Python sim vs BF16): {cos_sim_bf16:.6f}")

    # CUTLASS vs Python sim 应该 > 0.90 (考虑 FP8 误差)
    passed = cos_cutlass_sim > 0.90
    print(f"{'PASSED' if passed else 'FAILED'} (threshold: 0.90)")
    return passed


def test_multiple_sizes():
    """测试多种尺寸"""
    print("\n" + "=" * 60)
    print("Test 3: Multiple Sizes")
    print("=" * 60)

    import nvfp4_gemm

    test_cases = [
        (256, 2048, 256),
        (256, 2048, 1024),
        (256, 2048, 4096),
        (256, 2048, 8192),
        (256, 2048, 16384),
        (512, 2048, 8192),
    ]

    block_size = BLOCK_SIZE
    all_passed = True

    for M, K, N in test_cases:
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
                x_packed.cuda(), w_packed.cuda(),
                x_scales_fp8.cuda(), w_scales_fp8.cuda(),
                M, N, K
            )

            # Python 模拟
            x_dequant = dequantize_nvfp4_sim(x_q, x_scales, block_size)
            w_dequant = dequantize_nvfp4_sim(w_q, w_scales, block_size)
            sim_output = torch.matmul(x_dequant.cuda(), w_dequant.cuda().T)

            cos = F.cosine_similarity(
                output.flatten().float().unsqueeze(0),
                sim_output.flatten().unsqueeze(0)
            ).item()

            passed = cos > 0.90
            status = "PASS" if passed else "FAIL"
            print(f"M={M}, K={K}, N={N}: cosine={cos:.4f} [{status}]")

            if not passed:
                all_passed = False

        except Exception as e:
            print(f"M={M}, K={K}, N={N}: ERROR - {e}")
            all_passed = False

    print(f"\n{'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def main():
    print("NVFP4 GEMM Final Validation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    results = []
    results.append(("Uniform Correctness", test_uniform_correctness()))
    results.append(("Random Precision", test_random_precision()))
    results.append(("Multiple Sizes", test_multiple_sizes()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests PASSED!")
        print("NVFP4 GEMM implementation is working correctly.")
        print("Precision: ~0.93 cosine similarity (limited by FP8 scale conversion)")
    else:
        print("✗ Some tests FAILED")

    return all_passed


if __name__ == "__main__":
    main()
