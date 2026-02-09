#!/usr/bin/env python3
"""
检查 CUTLASS 期望的 scale tensor 大小

关键: tile_atom_to_shape_SFA/SFB 使用 (M, K) 或 (N, K) 的元素数量
而 SfAtom 内部处理 block size
"""

import torch
import sys

sys.path.insert(0, '/workspace/src')


def analyze_cutlass_layout():
    """
    分析 CUTLASS scale layout

    CUTLASS 使用:
    tile_to_shape(SfAtom{}, make_shape(N, K, L), Step<_2,_1,_3>{})

    SfAtom 形状: [[32, 4], [SFVecSize, 4]]
    - 32 rows × 4 k-blocks = 128 scale factors per atom
    - SFVecSize 维度有 stride=0 (broadcast)，不增加实际元素数

    tile_to_shape 会计算:
    - num_N_atoms = ceil(N / 32)  # 32 是 SfAtom 的 row 数
    - num_K_atoms = ceil(K / 4)   # 4 是 SfAtom 的 k-block 数（但 K 是元素数，不是 blocks！）

    这里有问题：K=2048 会被当作 2048/4 = 512 个 K atoms
    但实际上应该是 K/32/4 = 2048/32/4 = 16 个 K atoms
    """
    print("=" * 60)
    print("CUTLASS Scale Layout Analysis")
    print("=" * 60)

    M, K, N = 256, 2048, 256
    block_size = 32
    num_k_blocks = K // block_size  # 64

    # SfAtom 参数
    atom_rows = 32
    atom_k_blocks = 4  # 但 CUTLASS 可能把这当作 K 元素数！

    print(f"Problem: M={M}, N={N}, K={K}")
    print(f"Block size: {block_size}")
    print(f"num_k_blocks: {num_k_blocks}")

    print("\n--- 正确理解 (基于 k-blocks) ---")
    # 正确的理解：SfAtom 覆盖 32 rows × 4 k-blocks
    num_N_atoms = (N + atom_rows - 1) // atom_rows
    num_K_atoms = (num_k_blocks + atom_k_blocks - 1) // atom_k_blocks
    scale_per_atom = atom_rows * atom_k_blocks  # 128
    correct_total = num_N_atoms * num_K_atoms * scale_per_atom

    print(f"N atoms: {num_N_atoms}")
    print(f"K atoms (from k-blocks): {num_K_atoms}")
    print(f"Scales per atom: {scale_per_atom}")
    print(f"Correct total: {correct_total}")
    print(f"This matches: N_padded * K_padded = {256 * 64}")

    print("\n--- 错误理解 (如果 CUTLASS 把 K 当作元素数) ---")
    # 如果 CUTLASS 把 K=2048 当作元素数而不是 block 数
    # tile_to_shape 可能计算: K / atom_k_size
    # 但 atom_k_size 可能是 4*32 = 128（每个 k-block 32 个元素）
    atom_k_elements = atom_k_blocks * block_size  # 4 * 32 = 128
    num_K_atoms_wrong = (K + atom_k_elements - 1) // atom_k_elements  # 2048/128 = 16
    wrong_total = num_N_atoms * num_K_atoms_wrong * scale_per_atom

    print(f"K atoms (from K elements / 128): {num_K_atoms_wrong}")
    print(f"Wrong total: {wrong_total}")

    print("\n--- 另一种可能 ---")
    # SfAtom 的实际大小可能不是 128
    # 因为 stride<_0, _1> 意味着 SFVecSize 维度是 broadcast
    # 实际有效元素数 = 32 * 4 = 128 (不包含 broadcast 维度)
    # 但 filter_zeros 可能进一步减少

    # filter_zeros 会移除 stride=0 的维度
    # 所以实际大小是 32 * 4 = 128 per atom
    print("With filter_zeros: same as above (128 per atom)")


def test_different_scale_sizes():
    """测试不同的 scale tensor 大小"""
    print("\n" + "=" * 60)
    print("Testing Different Scale Tensor Sizes")
    print("=" * 60)

    import nvfp4_gemm

    M, K, N = 256, 2048, 256
    block_size = 32
    num_k_blocks = K // block_size

    fp4_one_packed = 0x22
    fp8_one = 0x38

    x_packed = torch.full((M, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')
    w_packed = torch.full((N, K // 2), fp4_one_packed, dtype=torch.uint8, device='cuda')

    # 尝试不同的 scale 大小
    # 正常大小: N_padded * K_padded = 256 * 64 = 16384
    # 如果 CUTLASS 期望 K 元素维度的布局: 可能不同

    M_padded = 256
    K_padded = 64
    N_padded = 256

    test_sizes = [
        ("Normal (N*Kb)", N_padded * K_padded),
        ("Large (N*K)", N * K),  # 如果 CUTLASS 使用 K 而不是 num_k_blocks
        ("Very large", N * K * 2),
        ("Half", N_padded * K_padded // 2),
    ]

    x_scales_fp8 = torch.full((M_padded * K_padded,), fp8_one, dtype=torch.uint8, device='cuda')

    for name, size in test_sizes:
        print(f"\n--- w_scale size = {size} ({name}) ---")

        w_scales_fp8 = torch.full((size,), fp8_one, dtype=torch.uint8, device='cuda')

        try:
            output = nvfp4_gemm.gemm_prepared(
                x_packed, w_packed,
                x_scales_fp8, w_scales_fp8,
                M, N, K
            )

            out_first = output[0, 0].item()
            out_last = output[0, -1].item()
            out_mean = output.mean().item()

            # 检查哪些 N 值正确
            correct_n = 0
            for n in range(N):
                if abs(output[0, n].item() - K) < 100:
                    correct_n += 1

            print(f"  [0,0]={out_first:.0f}, [0,-1]={out_last:.0f}, correct_n={correct_n}/{N}")

        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    print("CUTLASS Scale Size Investigation")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    analyze_cutlass_layout()
    test_different_scale_sizes()


if __name__ == "__main__":
    main()
