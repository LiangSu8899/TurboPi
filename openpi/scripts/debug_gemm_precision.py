#!/usr/bin/env python3
"""
精确验证 NVFP4 GEMM 精度

问题: Python quantize 的输出均值偏离预期
可能原因:
1. Scale factor layout 与 CUTLASS 不匹配
2. FP4 packing 格式不正确
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    prepare_scales_for_cutlass,
    pack_nvfp4_data,
    BLOCK_SIZE,
)


def test_python_simulation_accuracy():
    """先验证 Python 模拟模式的精度"""
    print("=" * 60)
    print("Test 1: Python simulation accuracy (no CUTLASS)")
    print("=" * 60)

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE

    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w = torch.randn(N, K, device='cuda', dtype=torch.float32)

    # BF16 参考
    ref = torch.matmul(x, w.T)

    # Python 量化和反量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    x_dequant = dequantize_nvfp4_sim(x_q, x_scales, block_size)
    w_dequant = dequantize_nvfp4_sim(w_q, w_scales, block_size)

    # 模拟 GEMM
    sim_output = torch.matmul(x_dequant, w_dequant.T)

    # 精度指标
    cos_sim = F.cosine_similarity(
        sim_output.flatten().unsqueeze(0),
        ref.flatten().unsqueeze(0)
    ).item()

    rel_error = (sim_output - ref).abs() / (ref.abs() + 1e-8)
    mean_rel_error = rel_error.mean().item() * 100

    print(f"Reference: mean={ref.mean().item():.4f}, std={ref.std().item():.4f}")
    print(f"Simulated: mean={sim_output.mean().item():.4f}, std={sim_output.std().item():.4f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Mean relative error: {mean_rel_error:.2f}%")

    return cos_sim


def test_uniform_data():
    """使用统一数据测试 - 这样 layout 错误会显现"""
    print("\n" + "=" * 60)
    print("Test 2: Uniform data to detect layout issues")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE

    # 使用统一数据
    x = torch.ones(M, K, device='cuda', dtype=torch.float32)
    w = torch.ones(N, K, device='cuda', dtype=torch.float32)

    # Python 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    print(f"x_q: unique values = {x_q.unique().tolist()}")
    print(f"x_scales: unique values = {x_scales.unique().tolist()[:5]}...")

    # 反量化验证
    x_dequant = dequantize_nvfp4_sim(x_q, x_scales, block_size)
    print(f"x_dequant: unique values = {x_dequant.unique().tolist()[:5]}...")
    print(f"x_dequant mean: {x_dequant.mean().item():.6f}")  # Should be ~1.0

    # BF16 参考
    ref = torch.matmul(x, w.T)  # = K = 2048 for each element
    print(f"\nBF16 reference[0,0]: {ref[0,0].item():.2f}")

    # Python 模拟
    w_dequant = dequantize_nvfp4_sim(w_q, w_scales, block_size)
    sim_output = torch.matmul(x_dequant, w_dequant.T)
    print(f"Python sim[0,0]: {sim_output[0,0].item():.2f}")

    # CUTLASS GEMM
    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    num_k_blocks = K // block_size
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True)

    output = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(),
        w_packed.cuda(),
        x_scales_fp8.cuda(),
        w_scales_fp8.cuda(),
        M, N, K
    )

    print(f"CUTLASS[0,0]: {output[0,0].item():.2f}")
    print(f"CUTLASS mean: {output.mean().item():.2f}")

    # 比较
    cutlass_ratio = output[0,0].item() / ref[0,0].item()
    print(f"\nCUTLASS / BF16 ratio: {cutlass_ratio:.4f}")

    return cutlass_ratio


def test_varying_scales():
    """使用不同的 scale 值测试每个 block"""
    print("\n" + "=" * 60)
    print("Test 3: Varying scales per block")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # 创建有规律的数据
    # 每个 block 的值不同，用于检测 layout
    x = torch.zeros(M, K, device='cuda', dtype=torch.float32)
    for k_block in range(num_k_blocks):
        k_start = k_block * block_size
        k_end = k_start + block_size
        # 设置值使得 scale 可以检测
        x[:, k_start:k_end] = (k_block + 1) * 0.1

    w = torch.ones(N, K, device='cuda', dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size)

    print(f"x_scales shape: {x_scales.shape}")
    print(f"x_scales [0, :8]: {x_scales[0, :8].tolist()}")

    # 反量化
    x_dequant = dequantize_nvfp4_sim(x_q, x_scales, block_size)

    # 检查第一行的 k-blocks 是否正确恢复
    print("\nDequantized x row 0, first 8 blocks (first value each):")
    for k_block in range(8):
        k_start = k_block * block_size
        original = x[0, k_start].item()
        dequant = x_dequant[0, k_start].item()
        print(f"  Block {k_block}: original={original:.4f}, dequant={dequant:.4f}")

    # BF16 参考
    ref = torch.matmul(x, w.T)

    # Python 模拟
    w_dequant = dequantize_nvfp4_sim(w_q, w_scales, block_size)
    sim_output = torch.matmul(x_dequant, w_dequant.T)

    print(f"\nBF16 ref[0,0]: {ref[0,0].item():.4f}")
    print(f"Python sim[0,0]: {sim_output[0,0].item():.4f}")

    # CUTLASS GEMM
    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    x_scales_fp8 = prepare_scales_for_cutlass(x_scales.cuda(), M, num_k_blocks, convert_to_fp8=True)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales.cuda(), N, num_k_blocks, convert_to_fp8=True)

    output = nvfp4_gemm.gemm_prepared(
        x_packed.cuda(),
        w_packed.cuda(),
        x_scales_fp8.cuda(),
        w_scales_fp8.cuda(),
        M, N, K
    )

    print(f"CUTLASS[0,0]: {output[0,0].item():.4f}")

    cos_sim = F.cosine_similarity(
        output.flatten().float().unsqueeze(0),
        sim_output.flatten().unsqueeze(0)
    ).item()
    print(f"\nCosine similarity (CUTLASS vs Python sim): {cos_sim:.6f}")

    return cos_sim


def test_fp4_packing():
    """验证 FP4 packing 是否与 CUTLASS 兼容"""
    print("\n" + "=" * 60)
    print("Test 4: FP4 packing verification")
    print("=" * 60)

    # NVFP4 编码:
    # 正数: 0 + 3-bit index (0-7)
    # 负数: 8 + 3-bit index (8-15)

    # 值 -> 索引映射
    # 0 -> 0, 0.5 -> 1, 1.0 -> 2, 1.5 -> 3, 2.0 -> 4, 3.0 -> 5, 4.0 -> 6, 6.0 -> 7

    print("NVFP4 encoding:")
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    for i, v in enumerate(values):
        pos_code = i
        neg_code = i + 8
        print(f"  {v:4.1f} -> 0x{pos_code:X}, -{v:4.1f} -> 0x{neg_code:X}")

    # 测试打包
    test_data = torch.tensor([
        1.0, 2.0,  # Should pack as: (4 << 4) | 2 = 0x42
        -1.0, -2.0,  # Should pack as: (12 << 4) | 10 = 0xCA
    ], device='cuda').view(1, 4)

    # 需要扩展到 32 元素 (block_size)
    test_data_full = torch.zeros(1, 32, device='cuda')
    test_data_full[0, :4] = test_data

    from openpi.models_pytorch.nvfp4_mlp import quantize_to_nvfp4_sim, pack_nvfp4_data
    q, s = quantize_to_nvfp4_sim(test_data_full, 32)
    packed = pack_nvfp4_data(q, 32)

    print(f"\nTest packing:")
    print(f"  Input: {test_data_full[0, :4].tolist()}")
    print(f"  Quantized: {q[0, :4].tolist()}")
    print(f"  Packed (hex): ", end="")
    for b in packed[0, :2].cpu().numpy():
        print(f"0x{b:02X} ", end="")
    print()

    return True


def main():
    print("NVFP4 GEMM Precision Debug")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    results = {}

    # Test 1: Python simulation accuracy
    results['python_sim'] = test_python_simulation_accuracy()

    # Test 2: FP4 packing
    results['fp4_pack'] = test_fp4_packing()

    # Test 3: Uniform data
    results['uniform'] = test_uniform_data()

    # Test 4: Varying scales
    results['varying'] = test_varying_scales()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Python simulation cosine sim: {results['python_sim']:.6f}")
    print(f"Uniform data CUTLASS/BF16 ratio: {results['uniform']:.4f}")
    print(f"Varying scales cosine sim: {results['varying']:.6f}")


if __name__ == "__main__":
    main()
