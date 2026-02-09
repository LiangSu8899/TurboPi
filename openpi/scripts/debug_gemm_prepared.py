#!/usr/bin/env python3
"""
调试 NVFP4 GEMM 数据格式问题

问题: gemm_prepared 返回 CUDA 内存错误
原因可能:
1. FP4 数据打包格式不正确
2. Scale factor 布局不正确
3. 尺寸/对齐问题
"""

import torch
import sys

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    prepare_scales_for_cutlass,
    pack_nvfp4_data,
    BLOCK_SIZE,
    CUTLASS_ROW_TILE,
    CUTLASS_K_TILE,
)


def test_minimal():
    """最小测试 - 使用全零数据"""
    print("=" * 60)
    print("Test 1: Minimal test with zeros")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # 全零 packed 数据
    x_packed = torch.zeros(M, K // 2, dtype=torch.uint8, device='cuda')
    w_packed = torch.zeros(N, K // 2, dtype=torch.uint8, device='cuda')

    # 计算需要的 scale factor 数量 (包含 padding)
    # CUTLASS 需要 padding 到 tile 边界
    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_padded = ((num_k_blocks + CUTLASS_K_TILE - 1) // CUTLASS_K_TILE) * CUTLASS_K_TILE

    print(f"M={M} -> M_padded={M_padded}")
    print(f"N={N} -> N_padded={N_padded}")
    print(f"num_k_blocks={num_k_blocks} -> K_padded={K_padded}")

    # Scale factor 数量
    x_scales_size = M_padded * K_padded
    w_scales_size = N_padded * K_padded

    print(f"x_scales_size = {x_scales_size}")
    print(f"w_scales_size = {w_scales_size}")

    # 全零 FP8 scale factors
    x_scales_fp8 = torch.zeros(x_scales_size, dtype=torch.uint8, device='cuda')
    w_scales_fp8 = torch.zeros(w_scales_size, dtype=torch.uint8, device='cuda')

    print(f"\nTensors:")
    print(f"  x_packed: {x_packed.shape}")
    print(f"  w_packed: {w_packed.shape}")
    print(f"  x_scales_fp8: {x_scales_fp8.shape}")
    print(f"  w_scales_fp8: {w_scales_fp8.shape}")

    print("\nCalling gemm_prepared...")
    try:
        output = nvfp4_gemm.gemm_prepared(
            x_packed, w_packed,
            x_scales_fp8, w_scales_fp8,
            M, N, K
        )
        print(f"SUCCESS! Output: {output.shape}")
        print(f"Output sum: {output.sum().item()}")  # Should be 0
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_with_cxx_quantize():
    """使用 C++ quantize 函数 (但 scales 还是 FP32)"""
    print("\n" + "=" * 60)
    print("Test 2: Using C++ quantize_to_nvfp4")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 16384

    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w = torch.randn(N, K, device='cuda', dtype=torch.float32)

    # C++ 量化
    x_packed, x_scales = nvfp4_gemm.quantize_to_nvfp4(x)
    w_packed, w_scales = nvfp4_gemm.quantize_to_nvfp4(w)

    print(f"C++ quantize results:")
    print(f"  x_packed: {x_packed.shape}, dtype={x_packed.dtype}")
    print(f"  x_scales: {x_scales.shape}, dtype={x_scales.dtype}")
    print(f"  w_packed: {w_packed.shape}, dtype={w_packed.dtype}")
    print(f"  w_scales: {w_scales.shape}, dtype={w_scales.dtype}")

    # x_scales 是 FP32，需要转换
    print(f"\nScale stats:")
    print(f"  x_scales: min={x_scales.min():.6f}, max={x_scales.max():.6f}")
    print(f"  w_scales: min={w_scales.min():.6f}, max={w_scales.max():.6f}")

    return True


def test_scale_factor_sizes():
    """验证 scale factor 尺寸是否正确"""
    print("\n" + "=" * 60)
    print("Test 3: Scale factor size verification")
    print("=" * 60)

    M, K, N = 256, 2048, 16384
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size  # 64

    print(f"Problem: M={M}, K={K}, N={N}")
    print(f"Block size: {block_size}")
    print(f"Num K blocks: {num_k_blocks}")

    # 按 CUTLASS 的 Sm1xxBlkScaledConfig 分析
    # 从 sm100_blockscaled_layout.hpp:
    # - SfAtomM/N = 128
    # - SfAtomK = 4
    # - SF layout: 128 rows × 4 k-blocks per tile

    sf_m = 128
    sf_k = 4

    # Tile 数量
    num_m_tiles = (M + sf_m - 1) // sf_m
    num_n_tiles = (N + sf_m - 1) // sf_m
    num_k_tiles = (num_k_blocks + sf_k - 1) // sf_k

    print(f"\nCUTLASS tile structure:")
    print(f"  SF tile: {sf_m} rows × {sf_k} k-blocks")
    print(f"  Num M tiles: {num_m_tiles}")
    print(f"  Num N tiles: {num_n_tiles}")
    print(f"  Num K tiles: {num_k_tiles}")

    # 每个 tile 的 scale factor 数量
    scales_per_tile = sf_m * sf_k  # 128 * 4 = 512

    # 总 scale factor 数量
    total_sfa = num_m_tiles * num_k_tiles * scales_per_tile
    total_sfb = num_n_tiles * num_k_tiles * scales_per_tile

    print(f"\nScale factor counts:")
    print(f"  Scales per tile: {scales_per_tile}")
    print(f"  Total SFA (input): {total_sfa}")
    print(f"  Total SFB (weight): {total_sfb}")

    # 与 Python prepare 函数对比
    M_padded = num_m_tiles * sf_m
    N_padded = num_n_tiles * sf_m
    K_padded = num_k_tiles * sf_k

    python_sfa = M_padded * K_padded
    python_sfb = N_padded * K_padded

    print(f"\nPython prepare_scales_for_cutlass output:")
    print(f"  M_padded={M_padded}, K_padded={K_padded}")
    print(f"  Expected SFA size: {python_sfa}")
    print(f"  Expected SFB size: {python_sfb}")

    match_a = total_sfa == python_sfa
    match_b = total_sfb == python_sfb
    print(f"\n  SFA size match: {match_a}")
    print(f"  SFB size match: {match_b}")

    return match_a and match_b


def test_packed_data_format():
    """检查 packed FP4 数据格式"""
    print("\n" + "=" * 60)
    print("Test 4: Packed FP4 data format check")
    print("=" * 60)

    # NVFP4 值: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    # 编码: 4-bit, sign + 3-bit magnitude index

    # 创建一个简单的测试向量
    test_vals = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,  # 正值
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # 负值
    ], device='cuda', dtype=torch.float32)

    # 扩展到 32 元素 (1 个 block)
    test_vals = test_vals.repeat(2)  # 32 elements
    test_input = test_vals.view(1, 32)  # [1, 32]

    print(f"Test input shape: {test_input.shape}")
    print(f"Test values: {test_vals.tolist()}")

    # 使用 Python 量化和打包
    from openpi.models_pytorch.nvfp4_mlp import quantize_to_nvfp4_sim, pack_nvfp4_data

    q, scales = quantize_to_nvfp4_sim(test_input, block_size=32)
    packed = pack_nvfp4_data(q, block_size=32)

    print(f"\nQuantized: {q.shape}")
    print(f"Scales: {scales.shape}, value={scales.item():.6f}")
    print(f"Packed: {packed.shape}")

    # 打印 packed 字节
    print(f"\nPacked bytes (hex):")
    packed_cpu = packed.cpu().numpy()
    for i, byte in enumerate(packed_cpu[0]):
        lo = byte & 0x0F
        hi = (byte >> 4) & 0x0F
        print(f"  Byte {i:2d}: 0x{byte:02X} -> lo=0x{lo:X} hi=0x{hi:X}")

    return True


def main():
    print("NVFP4 GEMM Debug Tests")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    results = {}

    # Test 1: Scale factor sizes
    results['sizes'] = test_scale_factor_sizes()

    # Test 2: Packed data format
    results['packed'] = test_packed_data_format()

    # Test 3: C++ quantize
    results['cxx_quantize'] = test_with_cxx_quantize()

    # Test 4: Minimal zeros test
    results['minimal'] = test_minimal()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
