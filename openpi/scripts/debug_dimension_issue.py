#!/usr/bin/env python3
"""
调试维度问题

观察: N=2048 (16 tiles) 工作正确，其他 N 值有问题
假设: CUTLASS 可能期望特定的 scale tensor 形状
"""

import torch
import sys

sys.path.insert(0, '/workspace/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    convert_scales_to_fp8,
    pack_nvfp4_data,
    BLOCK_SIZE,
    CUTLASS_ROW_TILE,
    CUTLASS_K_TILE,
)


def test_scale_size_requirements():
    """测试不同 scale size 的要求"""
    print("=" * 60)
    print("Scale Size Requirements Test")
    print("=" * 60)

    import nvfp4_gemm

    M, K = 256, 2048
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size  # 64
    num_k_tiles = num_k_blocks // CUTLASS_K_TILE  # 16

    # FP4 value 1.0 = index 2
    fp4_one_packed = 0x22  # Two 1.0 values

    # Scale = 1.0 in FP8 E4M3
    fp8_one = 0x38

    print(f"M={M}, K={K}")
    print(f"num_k_blocks={num_k_blocks}, num_k_tiles={num_k_tiles}")

    for N in [128, 256, 512, 1024, 2048, 4096]:
        print(f"\n--- N = {N} ---")

        N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
        num_n_tiles = N_padded // CUTLASS_ROW_TILE
        M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
        K_padded = ((num_k_blocks + CUTLASS_K_TILE - 1) // CUTLASS_K_TILE) * CUTLASS_K_TILE

        print(f"  N_padded={N_padded}, num_n_tiles={num_n_tiles}")
        print(f"  M_padded={M_padded}, K_padded={K_padded}")

        # 创建 packed 数据
        x_packed = torch.full((M, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')
        w_packed = torch.full((N, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')

        # 创建统一 scales
        x_scale_size = M_padded * K_padded
        w_scale_size = N_padded * K_padded

        x_scales_fp8 = torch.full((x_scale_size,), fp8_one, dtype=torch.uint8, device='cuda')
        w_scales_fp8 = torch.full((w_scale_size,), fp8_one, dtype=torch.uint8, device='cuda')

        print(f"  x_scale_size={x_scale_size}, w_scale_size={w_scale_size}")

        try:
            output = nvfp4_gemm.gemm_prepared(
                x_packed,
                w_packed,
                x_scales_fp8,
                w_scales_fp8,
                M, N, K
            )

            expected = K  # ones @ ones = K

            out_first = output[0, 0].item()
            out_last = output[0, -1].item()
            out_mean = output.mean().item()

            # 找出第一个不正确的 N 索引
            first_bad = -1
            for n in range(N):
                if abs(output[0, n].item() - expected) > 100:
                    first_bad = n
                    break

            status = "OK" if first_bad == -1 else f"BAD@{first_bad}"
            print(f"  [0,0]={out_first:.0f}, [0,-1]={out_last:.0f}, mean={out_mean:.0f} [{status}]")

        except Exception as e:
            print(f"  ERROR: {e}")


def test_with_known_scale_values():
    """使用已知 scale 值测试"""
    print("\n" + "=" * 60)
    print("Known Scale Values Test")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    # FP4 value 1.0 = index 2
    fp4_one_packed = 0x22

    x_packed = torch.full((M, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')
    w_packed = torch.full((N, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')

    # 使用不同的 scale 值
    # FP8 E4M3: 1.0 = 0x38, 0.5 = 0x30, 2.0 = 0x40
    test_scales = [0x38, 0x30, 0x40]

    M_padded = 256
    K_padded = 64
    N_padded = 256

    for scale_val in test_scales:
        print(f"\n--- Scale FP8 = 0x{scale_val:02X} ---")

        x_scales_fp8 = torch.full((M_padded * K_padded,), scale_val, dtype=torch.uint8, device='cuda')
        w_scales_fp8 = torch.full((N_padded * K_padded,), scale_val, dtype=torch.uint8, device='cuda')

        try:
            output = nvfp4_gemm.gemm_prepared(
                x_packed,
                w_packed,
                x_scales_fp8,
                w_scales_fp8,
                M, N, K
            )

            # 对于 scale = 1.0: expected = K * 1.0 * 1.0 * 1.0 * 1.0 = K = 2048
            # 对于 scale = 0.5: expected = K * 1.0 * 0.5 * 1.0 * 0.5 = K * 0.25 = 512
            # 对于 scale = 2.0: expected = K * 1.0 * 2.0 * 1.0 * 2.0 = K * 4 = 8192

            print(f"  [0,0]={output[0,0].item():.0f}, [0,-1]={output[0,-1].item():.0f}")
            print(f"  mean={output.mean().item():.0f}")

        except Exception as e:
            print(f"  ERROR: {e}")


def analyze_output_pattern():
    """分析输出模式"""
    print("\n" + "=" * 60)
    print("Output Pattern Analysis")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE

    fp4_one_packed = 0x22
    fp8_one = 0x38

    x_packed = torch.full((M, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')
    w_packed = torch.full((N, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')

    M_padded = 256
    K_padded = 64
    N_padded = 256

    x_scales_fp8 = torch.full((M_padded * K_padded,), fp8_one, dtype=torch.uint8, device='cuda')
    w_scales_fp8 = torch.full((N_padded * K_padded,), fp8_one, dtype=torch.uint8, device='cuda')

    output = nvfp4_gemm.gemm_prepared(
        x_packed, w_packed,
        x_scales_fp8, w_scales_fp8,
        M, N, K
    )

    expected = K

    print(f"Output shape: {output.shape}")
    print(f"Expected: {expected}")

    # 按 N 区域分析
    print("\nOutput by N region:")
    for n_start in range(0, N, 32):
        n_end = min(n_start + 32, N)
        chunk = output[0, n_start:n_end]
        mean = chunk.mean().item()
        nonzero = (chunk != 0).sum().item()
        print(f"  N[{n_start:3d}:{n_end:3d}]: mean={mean:8.1f}, nonzero={nonzero}/32")

    # 按 M 区域分析
    print("\nOutput by M region (at N=0):")
    for m_start in range(0, M, 32):
        m_end = min(m_start + 32, M)
        chunk = output[m_start:m_end, 0]
        mean = chunk.mean().item()
        nonzero = (chunk != 0).sum().item()
        print(f"  M[{m_start:3d}:{m_end:3d}]: mean={mean:8.1f}, nonzero={nonzero}/32")


def main():
    print("Dimension Issue Debug")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    test_scale_size_requirements()
    test_with_known_scale_values()
    analyze_output_pattern()


if __name__ == "__main__":
    main()
