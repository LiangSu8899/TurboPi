#!/usr/bin/env python3
"""
Nibble Order 验证脚本

NVFP4 精度问题的最后希望 - 验证 4-bit 数据的高低位顺序是否正确。

测试方案:
1. 原始: packed = (high << 4) | low  (当前)
2. 交换: packed = (low << 4) | high
3. 交换取样: low = odd, high = even
4. 更复杂的 swizzle

如果某个方案能让 cosine similarity 从 0.93 跳到 0.98+，那就找到了答案！
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    convert_scales_to_fp8,
    prepare_scales_for_cutlass,
    BLOCK_SIZE,
    CUTLASS_ROW_TILE,
    CUTLASS_K_TILE,
)


def pack_nvfp4_original(quantized: torch.Tensor) -> torch.Tensor:
    """原始 packing: low nibble = even, high nibble = odd"""
    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 device=quantized.device)

    signs = (quantized < 0).to(torch.uint8)
    abs_vals = quantized.abs()

    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    encoded = (signs << 3) | indices

    low = encoded[:, 0::2]
    high = encoded[:, 1::2]
    packed = low | (high << 4)  # 当前方式

    return packed


def pack_nvfp4_swap_nibbles(quantized: torch.Tensor) -> torch.Tensor:
    """交换 nibbles: high nibble = even, low nibble = odd"""
    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 device=quantized.device)

    signs = (quantized < 0).to(torch.uint8)
    abs_vals = quantized.abs()

    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    encoded = (signs << 3) | indices

    low = encoded[:, 0::2]
    high = encoded[:, 1::2]
    packed = high | (low << 4)  # 交换!

    return packed


def pack_nvfp4_swap_sampling(quantized: torch.Tensor) -> torch.Tensor:
    """交换取样: low = odd indices, high = even indices"""
    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 device=quantized.device)

    signs = (quantized < 0).to(torch.uint8)
    abs_vals = quantized.abs()

    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    encoded = (signs << 3) | indices

    # 交换取样顺序
    low = encoded[:, 1::2]   # odd indices
    high = encoded[:, 0::2]  # even indices
    packed = low | (high << 4)

    return packed


def pack_nvfp4_both_swap(quantized: torch.Tensor) -> torch.Tensor:
    """同时交换 nibbles 和取样"""
    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 device=quantized.device)

    signs = (quantized < 0).to(torch.uint8)
    abs_vals = quantized.abs()

    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    encoded = (signs << 3) | indices

    low = encoded[:, 1::2]   # odd indices
    high = encoded[:, 0::2]  # even indices
    packed = high | (low << 4)  # 也交换 nibbles

    return packed


def pack_nvfp4_interleave_4(quantized: torch.Tensor) -> torch.Tensor:
    """每 4 个元素交织"""
    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 device=quantized.device)

    signs = (quantized < 0).to(torch.uint8)
    abs_vals = quantized.abs()

    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    encoded = (signs << 3) | indices

    M, K = encoded.shape

    # 每 4 个元素一组重排
    # [e0, e1, e2, e3] -> [e0, e2, e1, e3]
    encoded = encoded.view(M, K // 4, 4)
    encoded = encoded[:, :, [0, 2, 1, 3]].contiguous().view(M, K)

    low = encoded[:, 0::2]
    high = encoded[:, 1::2]
    packed = low | (high << 4)

    return packed


def pack_nvfp4_interleave_8(quantized: torch.Tensor) -> torch.Tensor:
    """每 8 个元素交织 (Tensor Core friendly)"""
    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 device=quantized.device)

    signs = (quantized < 0).to(torch.uint8)
    abs_vals = quantized.abs()

    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    encoded = (signs << 3) | indices

    M, K = encoded.shape

    # 每 8 个元素一组重排
    # [e0, e1, e2, e3, e4, e5, e6, e7] -> [e0, e4, e1, e5, e2, e6, e3, e7]
    encoded = encoded.view(M, K // 8, 8)
    encoded = encoded[:, :, [0, 4, 1, 5, 2, 6, 3, 7]].contiguous().view(M, K)

    low = encoded[:, 0::2]
    high = encoded[:, 1::2]
    packed = low | (high << 4)

    return packed


def pack_nvfp4_reverse_8(quantized: torch.Tensor) -> torch.Tensor:
    """每 8 个元素反向交织"""
    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 device=quantized.device)

    signs = (quantized < 0).to(torch.uint8)
    abs_vals = quantized.abs()

    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    encoded = (signs << 3) | indices

    M, K = encoded.shape

    # 每 8 个元素一组重排 - 另一种模式
    # [e0, e1, e2, e3, e4, e5, e6, e7] -> [e4, e0, e5, e1, e6, e2, e7, e3]
    encoded = encoded.view(M, K // 8, 8)
    encoded = encoded[:, :, [4, 0, 5, 1, 6, 2, 7, 3]].contiguous().view(M, K)

    low = encoded[:, 0::2]
    high = encoded[:, 1::2]
    packed = low | (high << 4)

    return packed


def pack_nvfp4_block_32(quantized: torch.Tensor) -> torch.Tensor:
    """按 32 元素 block 重排 (CUTLASS block size)"""
    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                 device=quantized.device)

    signs = (quantized < 0).to(torch.uint8)
    abs_vals = quantized.abs()

    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    encoded = (signs << 3) | indices

    M, K = encoded.shape

    # 每 32 个元素一组，前后 16 个交换
    # [e0..e15, e16..e31] -> [e16..e31, e0..e15]
    encoded = encoded.view(M, K // 32, 2, 16)
    encoded = encoded[:, :, [1, 0], :].contiguous().view(M, K)

    low = encoded[:, 0::2]
    high = encoded[:, 1::2]
    packed = low | (high << 4)

    return packed


def test_nibble_order():
    """测试不同的 Nibble Order 方案"""
    print("=" * 70)
    print("NVFP4 Nibble Order 验证")
    print("=" * 70)

    try:
        import nvfp4_gemm
        print("[OK] CUTLASS extension loaded")
    except ImportError:
        print("[ERROR] nvfp4_gemm not available")
        print("请先构建 CUTLASS extension")
        return

    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # 测试配置
    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # Padding
    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_blocks_padded = ((num_k_blocks + CUTLASS_K_TILE - 1) // CUTLASS_K_TILE) * CUTLASS_K_TILE

    print(f"Test config: M={M}, K={K}, N={N}")
    print(f"Padded: M_pad={M_padded}, K_blocks_pad={K_blocks_padded}\n")

    # 准备测试数据
    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)

    # 量化
    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size, use_mse_search=False)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size, use_mse_search=False)

    # Python 参考 (使用 FP8 scales)
    x_deq = dequantize_nvfp4_sim(x_q, x_scales, block_size, use_fp8_scales=True)
    w_deq = dequantize_nvfp4_sim(w_q, w_scales, block_size, use_fp8_scales=True)
    ref = torch.matmul(x_deq, w_deq.T)

    # BF16 参考
    bf16_ref = torch.matmul(x, w.T)

    # 准备 scales
    x_scales_fp8 = prepare_scales_for_cutlass(x_scales, M, num_k_blocks, convert_to_fp8=True, K=K)
    w_scales_fp8 = prepare_scales_for_cutlass(w_scales, N, num_k_blocks, convert_to_fp8=True, K=K)

    # 测试所有 packing 方案
    pack_methods = [
        ("原始 (low=even, high=odd)", pack_nvfp4_original),
        ("交换 nibbles (high=even, low=odd)", pack_nvfp4_swap_nibbles),
        ("交换取样 (low=odd, high=even)", pack_nvfp4_swap_sampling),
        ("同时交换", pack_nvfp4_both_swap),
        ("每4元素交织 [0,2,1,3]", pack_nvfp4_interleave_4),
        ("每8元素交织 [0,4,1,5,2,6,3,7]", pack_nvfp4_interleave_8),
        ("每8元素反向 [4,0,5,1,6,2,7,3]", pack_nvfp4_reverse_8),
        ("每32元素块交换", pack_nvfp4_block_32),
    ]

    print("=" * 70)
    print("Testing Nibble Order variants...")
    print("=" * 70)
    print()

    best_name = None
    best_cos = -1
    results = []

    for name, pack_func in pack_methods:
        try:
            # Pack with this method
            x_packed = pack_func(x_q)
            w_packed = pack_func(w_q)

            # Run CUTLASS
            output = nvfp4_gemm.gemm_prepared(
                x_packed, w_packed,
                x_scales_fp8, w_scales_fp8,
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

            results.append((name, cos_ref, cos_bf16, True, ""))

            if cos_ref > best_cos:
                best_cos = cos_ref
                best_name = name

            status = "***WINNER***" if cos_ref > 0.98 else ""
            print(f"  {name:40s}: cos_sim={cos_ref:.6f}  vs_bf16={cos_bf16:.6f}  {status}")

        except Exception as e:
            results.append((name, 0, 0, False, str(e)[:50]))
            print(f"  {name:40s}: ERROR - {str(e)[:50]}")

    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print()
    print(f"Best method: {best_name}")
    print(f"Best cosine: {best_cos:.6f}")
    print()

    if best_cos > 0.98:
        print("SUCCESS! Found nibble order with >0.98 cosine similarity!")
        print(f"Use packing method: {best_name}")
    elif best_cos > 0.95:
        print("Good progress. Found method with >0.95 cosine similarity.")
    else:
        print("No improvement found.")
        print("Issue is NOT nibble order - need to investigate:")
        print("  1. SfAtom stride pattern")
        print("  2. Data + Scale alignment")
        print("  3. Something else in CUTLASS layout")


if __name__ == "__main__":
    test_nibble_order()
