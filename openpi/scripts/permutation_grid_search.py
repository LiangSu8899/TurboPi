#!/usr/bin/env python3
"""
Permutation Grid Search for CUTLASS Scale Layout

暴力逆向工程 CUTLASS 的 Scale Layout。

核心思路:
- 不推导 CuTe Layout 公式（太复杂）
- 让 GPU "告诉" 我们正确的排列方式
- 遍历所有可能的 reshape + permute 组合
- 找到使 cosine similarity 最高的组合

关键维度 (from sm100_blockscaled_layout.hpp):
- SfKMajorAtom = Layout<Shape<Shape<_32,_4>, Shape<_16,_4>>, Stride<...>>
- 关键数字: 32, 4, 16, 128
"""

import torch
import torch.nn.functional as F
import itertools
import sys
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    pack_nvfp4_data,
    convert_scales_to_fp8,
    BLOCK_SIZE,
    CUTLASS_ROW_TILE,
    CUTLASS_K_TILE,
    CUTLASS_ROW_GROUP,
)


@dataclass
class PermResult:
    """排列组合搜索结果"""
    name: str
    cos_sim: float
    cos_bf16: float
    success: bool
    error: str = ""


def generate_all_permutations() -> List[Tuple[str, Callable]]:
    """
    生成所有可能的 Scale 重排函数 - 暴力穷举版。

    关键维度: 32, 4, 16
    SfKMajorAtom = Shape<Shape<_32,_4>, Shape<_16,_4>>
    """
    permutations = []

    # ========================================================================
    # 基础方法
    # ========================================================================

    def v0_original(scales, M, K_blocks, M_padded, K_padded):
        """原始: repeat_interleave + flatten"""
        s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
        s[:M, :K_blocks] = scales
        return s.repeat_interleave(BLOCK_SIZE, dim=1).flatten()
    permutations.append(("v0_original", v0_original))

    def v0b_flatten_only(scales, M, K_blocks, M_padded, K_padded):
        """只 flatten，不扩展"""
        s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
        s[:M, :K_blocks] = scales
        return s.flatten()
    permutations.append(("v0b_flatten_only", v0b_flatten_only))

    def v0c_transpose(scales, M, K_blocks, M_padded, K_padded):
        """转置后 flatten"""
        s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
        s[:M, :K_blocks] = scales
        return s.T.contiguous().flatten()
    permutations.append(("v0c_transpose", v0c_transpose))

    # ========================================================================
    # (32, 4) 系列 - SfKMajorAtom 的内层
    # ========================================================================

    # 对于每个 (32, K) 的 scale slice，按 4 为单位重排
    def gen_32_4_permute(dim_order):
        def fn(scales, M, K_blocks, M_padded, K_padded):
            s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
            s[:M, :K_blocks] = scales

            # 确保维度可整除
            if M_padded % 32 != 0 or K_padded % 4 != 0:
                return s.flatten()

            num_m_groups = M_padded // 32
            num_k_groups = K_padded // 4

            # [num_m_groups, 32, num_k_groups, 4]
            s = s.view(num_m_groups, 32, num_k_groups, 4)
            s = s.permute(*dim_order).contiguous()
            return s.flatten()
        return fn

    # 所有 4D permute 组合
    for perm in itertools.permutations([0, 1, 2, 3]):
        name = f"v1_32x4_perm{perm}"
        permutations.append((name, gen_32_4_permute(perm)))

    # ========================================================================
    # (32, 16) 系列 - SfKMajorAtom 的另一种解释
    # ========================================================================

    def gen_32_16_permute(dim_order):
        def fn(scales, M, K_blocks, M_padded, K_padded):
            s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
            s[:M, :K_blocks] = scales

            if M_padded % 32 != 0 or K_padded % 16 != 0:
                # 如果不能整除，padding K
                K_padded_16 = ((K_padded + 15) // 16) * 16
                s_new = torch.zeros(M_padded, K_padded_16, device=scales.device, dtype=scales.dtype)
                s_new[:, :K_padded] = s
                s = s_new
                K_padded = K_padded_16

            num_m_groups = M_padded // 32
            num_k_groups = K_padded // 16

            # [num_m_groups, 32, num_k_groups, 16]
            s = s.view(num_m_groups, 32, num_k_groups, 16)
            s = s.permute(*dim_order).contiguous()
            return s.flatten()
        return fn

    for perm in itertools.permutations([0, 1, 2, 3]):
        name = f"v2_32x16_perm{perm}"
        permutations.append((name, gen_32_16_permute(perm)))

    # ========================================================================
    # (128, 4) 系列 - CUTLASS row tile
    # ========================================================================

    def gen_128_4_permute(dim_order):
        def fn(scales, M, K_blocks, M_padded, K_padded):
            s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
            s[:M, :K_blocks] = scales

            if M_padded % 128 != 0 or K_padded % 4 != 0:
                return s.flatten()

            num_m_tiles = M_padded // 128
            num_k_tiles = K_padded // 4

            # [num_m_tiles, 128, num_k_tiles, 4]
            s = s.view(num_m_tiles, 128, num_k_tiles, 4)
            s = s.permute(*dim_order).contiguous()
            return s.flatten()
        return fn

    for perm in itertools.permutations([0, 1, 2, 3]):
        name = f"v3_128x4_perm{perm}"
        permutations.append((name, gen_128_4_permute(perm)))

    # ========================================================================
    # 嵌套 Shape<Shape<32,4>, Shape<16,4>> - 6D 重排
    # ========================================================================

    def gen_nested_32_4_16_4(outer_order, inner_order):
        """
        SfKMajorAtom = Shape<Shape<_32,_4>, Shape<_16,_4>>

        解释 1: (32, 4) 是 M 方向的嵌套，(16, 4) 是 K 方向的嵌套
        M -> (num_m32, 32) -> (num_m32, 8, 4)
        K -> (num_k4, 4)   -> ...

        这里我们尝试 6D reshape + permute
        """
        def fn(scales, M, K_blocks, M_padded, K_padded):
            s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
            s[:M, :K_blocks] = scales

            # 需要 M % 32 == 0 且 K % 4 == 0
            if M_padded % 32 != 0 or K_padded % 4 != 0:
                return s.flatten()

            # 进一步分解: 32 = 8 * 4, 但这可能不是正确的分解
            # 先尝试: [num_m32, 32, num_k4, 4] -> 内部再分
            num_m32 = M_padded // 32
            num_k4 = K_padded // 4

            # [num_m32, 32, num_k4, 4]
            s = s.view(num_m32, 32, num_k4, 4)
            # 先做外层 permute
            s = s.permute(*outer_order).contiguous()
            # 然后 flatten
            return s.flatten()
        return fn

    # 只选择一些有代表性的组合（不然太多了）
    key_perms = [
        (0, 1, 2, 3),  # 原始
        (2, 0, 1, 3),  # K major
        (0, 2, 1, 3),  # 交错
        (2, 0, 3, 1),  # K major + 内部交错
        (0, 2, 3, 1),  # 另一种交错
        (2, 3, 0, 1),  # 完全反转
    ]

    for outer in key_perms:
        name = f"v4_nested_outer{outer}"
        permutations.append((name, gen_nested_32_4_16_4(outer, None)))

    # ========================================================================
    # (32, 4, 16) 3D 系列 - 穷举所有 3D 排列
    # ========================================================================

    def gen_3d_reshape_permute(shape, perm):
        """
        将 [M, K] reshape 成 3D 后 permute。
        shape: (a, b, c) where M = num_a * a, K = num_bc * b * c
        """
        a, b, c = shape
        def fn(scales, M, K_blocks, M_padded, K_padded):
            s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
            s[:M, :K_blocks] = scales

            # 计算维度
            if M_padded % a != 0:
                return s.flatten()

            num_a = M_padded // a
            total_k = K_padded

            # 尝试将 K reshape 成 (b, c) 或 (c, b)
            if total_k % (b * c) == 0:
                num_bc = total_k // (b * c)
                # [num_a, a, num_bc, b, c]
                try:
                    s = s.view(num_a, a, num_bc, b, c)
                    s = s.permute(*perm).contiguous()
                except:
                    pass

            return s.flatten()
        return fn

    # 核心 3D 形状 (32, 4, 16) 的各种排列
    shapes_3d = [
        (32, 4, 16),  # SfKMajorAtom 提示
        (32, 16, 4),
        (128, 4, 4),  # row_tile x k_tile 变体
        (128, 1, 4),
        (32, 1, 4),
    ]

    # 对于 5D tensor，选择一些 key permutations
    key_5d_perms = [
        (0, 1, 2, 3, 4),  # 原始
        (2, 0, 1, 3, 4),  # K major
        (0, 2, 1, 3, 4),
        (2, 0, 3, 1, 4),
        (0, 2, 3, 4, 1),
        (2, 3, 0, 1, 4),
        (2, 3, 0, 4, 1),
        (4, 3, 2, 1, 0),  # 完全反转
    ]

    for shape in shapes_3d:
        for perm in key_5d_perms:
            name = f"v5_3d_{shape[0]}x{shape[1]}x{shape[2]}_perm{perm}"
            permutations.append((name, gen_3d_reshape_permute(shape, perm)))

    # ========================================================================
    # Stride 模拟系列
    # ========================================================================

    def gen_stride_pattern(stride_m, stride_k):
        """模拟 CUTLASS stride 模式"""
        def fn(scales, M, K_blocks, M_padded, K_padded):
            s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
            s[:M, :K_blocks] = scales

            # 计算输出大小
            out_size = M_padded * K_padded
            output = torch.zeros(out_size, device=scales.device, dtype=scales.dtype)

            # 按 stride 填充
            for m in range(M_padded):
                for k in range(K_padded):
                    idx = (m * stride_m + k * stride_k) % out_size
                    output[idx] = s[m, k]

            return output
        return fn

    # SfKMajorAtom Stride 线索: Stride<Stride<_16, _4>, Stride<_64, _1>>
    stride_patterns = [
        (16, 4),    # Stride<_16, _4>
        (4, 16),
        (1, 64),    # Stride<_1, _64>
        (64, 1),
        (32, 4),    # 基于 32 的变体
        (4, 32),
        (128, 1),   # 基于 128 的变体
        (1, 128),
    ]

    for stride_m, stride_k in stride_patterns:
        name = f"v6_stride_m{stride_m}_k{stride_k}"
        permutations.append((name, gen_stride_pattern(stride_m, stride_k)))

    # ========================================================================
    # K-expansion 变体
    # ========================================================================

    def gen_k_expand_method(method):
        """不同的 K 扩展方式"""
        def fn(scales, M, K_blocks, M_padded, K_padded):
            s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
            s[:M, :K_blocks] = scales

            if method == "repeat_last":
                # repeat_interleave on dim=1
                s = s.repeat_interleave(BLOCK_SIZE, dim=1)
            elif method == "repeat_first":
                # 先 transpose，repeat，再 transpose
                s = s.T.repeat_interleave(BLOCK_SIZE, dim=1).T.contiguous()
            elif method == "tile":
                # tile instead of repeat
                s = s.unsqueeze(-1).expand(-1, -1, BLOCK_SIZE).reshape(M_padded, -1)
            elif method == "interleave":
                # 交错扩展
                s = s.unsqueeze(-1).expand(-1, -1, BLOCK_SIZE)
                s = s.permute(0, 2, 1).contiguous().reshape(M_padded, -1)
            else:
                pass

            return s.flatten()
        return fn

    for method in ["repeat_last", "repeat_first", "tile", "interleave"]:
        name = f"v7_kexpand_{method}"
        permutations.append((name, gen_k_expand_method(method)))

    # ========================================================================
    # 按 block 重排
    # ========================================================================

    def gen_block_reorder(block_m, block_k, perm_4d):
        """按 block 重排后 permute"""
        def fn(scales, M, K_blocks, M_padded, K_padded):
            s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
            s[:M, :K_blocks] = scales

            if M_padded % block_m != 0 or K_padded % block_k != 0:
                return s.flatten()

            num_m = M_padded // block_m
            num_k = K_padded // block_k

            # [num_m, block_m, num_k, block_k]
            s = s.view(num_m, block_m, num_k, block_k)
            s = s.permute(*perm_4d).contiguous()
            return s.flatten()
        return fn

    # 关键 block 大小
    block_sizes = [
        (32, 4),
        (32, 16),
        (128, 4),
        (128, 16),
    ]

    # 所有 4D permute
    for bm, bk in block_sizes:
        for perm in itertools.permutations([0, 1, 2, 3]):
            name = f"v8_block_{bm}x{bk}_perm{perm}"
            permutations.append((name, gen_block_reorder(bm, bk, perm)))

    # ========================================================================
    # 特殊组合: 先 expand K，再 tile permute
    # ========================================================================

    def gen_expand_then_tile(tile_m, tile_k, perm_4d):
        """先扩展 K (×32)，再按 tile 重排"""
        def fn(scales, M, K_blocks, M_padded, K_padded):
            s = torch.zeros(M_padded, K_padded, device=scales.device, dtype=scales.dtype)
            s[:M, :K_blocks] = scales

            # 扩展 K
            s = s.repeat_interleave(BLOCK_SIZE, dim=1)  # [M, K]
            K_full = s.shape[1]

            if M_padded % tile_m != 0 or K_full % tile_k != 0:
                return s.flatten()

            num_m = M_padded // tile_m
            num_k = K_full // tile_k

            s = s.view(num_m, tile_m, num_k, tile_k)
            s = s.permute(*perm_4d).contiguous()
            return s.flatten()
        return fn

    tile_sizes_full = [
        (128, 128),  # full K block
        (128, 32),
        (32, 128),
        (32, 32),
    ]

    key_4d_perms = [
        (0, 1, 2, 3),
        (2, 0, 1, 3),
        (0, 2, 1, 3),
        (2, 0, 3, 1),
        (2, 3, 0, 1),
    ]

    for tm, tk in tile_sizes_full:
        for perm in key_4d_perms:
            name = f"v9_expand_tile_{tm}x{tk}_perm{perm}"
            permutations.append((name, gen_expand_then_tile(tm, tk, perm)))

    return permutations


def run_grid_search(verbose: bool = True):
    """运行 Permutation Grid Search"""
    print("=" * 70)
    print("CUTLASS Scale Layout - Permutation Grid Search (暴力版)")
    print("=" * 70)

    try:
        import nvfp4_gemm
        print("[OK] CUTLASS extension loaded")
    except ImportError:
        print("[ERROR] nvfp4_gemm not available")
        print("Please build the extension first")
        return None

    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # 测试配置 - 使用和实际模型接近的尺寸
    test_configs = [
        # (M, K, N) - 先用小尺寸快速测试
        (256, 128, 256),   # 小尺寸，快速
        (256, 2048, 256),  # 中尺寸
    ]

    all_results = []

    for M, K, N in test_configs:
        print(f"\n{'='*70}")
        print(f"Testing: M={M}, K={K}, N={N}")
        print(f"{'='*70}")

        block_size = BLOCK_SIZE
        num_k_blocks = K // block_size

        # Padding
        row_tile = CUTLASS_ROW_TILE
        k_tile = CUTLASS_K_TILE
        M_padded = ((M + row_tile - 1) // row_tile) * row_tile
        N_padded = ((N + row_tile - 1) // row_tile) * row_tile
        K_blocks_padded = ((num_k_blocks + k_tile - 1) // k_tile) * k_tile
        K_padded = K_blocks_padded

        print(f"Padded: M_pad={M_padded}, K_blocks_pad={K_blocks_padded}")

        # 准备测试数据
        torch.manual_seed(42)
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        w = torch.randn(N, K, device=device, dtype=torch.float32)

        # 量化
        x_q, x_scales = quantize_to_nvfp4_sim(x, block_size, use_mse_search=False)
        w_q, w_scales = quantize_to_nvfp4_sim(w, block_size, use_mse_search=False)

        x_packed = pack_nvfp4_data(x_q, block_size)
        w_packed = pack_nvfp4_data(w_q, block_size)

        # Python 参考 (使用 FP8 scales)
        x_deq = dequantize_nvfp4_sim(x_q, x_scales, block_size, use_fp8_scales=True)
        w_deq = dequantize_nvfp4_sim(w_q, w_scales, block_size, use_fp8_scales=True)
        ref = torch.matmul(x_deq, w_deq.T)

        # BF16 参考
        bf16_ref = torch.matmul(x, w.T)

        # 获取所有 permutation 函数
        permutations = generate_all_permutations()
        print(f"\nTesting {len(permutations)} permutations...")

        best_name = None
        best_cosine = -1
        results = []

        for i, (name, perm_func) in enumerate(permutations):
            try:
                # 准备 scales
                x_scales_perm = perm_func(x_scales.clone(), M, num_k_blocks, M_padded, K_blocks_padded)
                w_scales_perm = perm_func(w_scales.clone(), N, num_k_blocks, N_padded, K_blocks_padded)

                # 转换为 FP8
                x_scales_fp8 = convert_scales_to_fp8(x_scales_perm)
                w_scales_fp8 = convert_scales_to_fp8(w_scales_perm)

                # 确保大小正确 (CUTLASS 期望 M * K 个 scales)
                expected_x = M_padded * K_blocks_padded * BLOCK_SIZE
                expected_w = N_padded * K_blocks_padded * BLOCK_SIZE

                if x_scales_fp8.numel() < expected_x:
                    pad = torch.zeros(expected_x - x_scales_fp8.numel(), dtype=torch.uint8, device=device)
                    x_scales_fp8 = torch.cat([x_scales_fp8, pad])
                elif x_scales_fp8.numel() > expected_x:
                    x_scales_fp8 = x_scales_fp8[:expected_x]

                if w_scales_fp8.numel() < expected_w:
                    pad = torch.zeros(expected_w - w_scales_fp8.numel(), dtype=torch.uint8, device=device)
                    w_scales_fp8 = torch.cat([w_scales_fp8, pad])
                elif w_scales_fp8.numel() > expected_w:
                    w_scales_fp8 = w_scales_fp8[:expected_w]

                # 运行 CUTLASS
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

                results.append(PermResult(name, cos_ref, cos_bf16, True))

                if cos_ref > best_cosine:
                    best_cosine = cos_ref
                    best_name = name

                # 只打印好的结果
                if verbose and cos_ref > 0.95:
                    print(f"  [{i+1}/{len(permutations)}] {name:45s}: {cos_ref:.6f} ***")
                elif verbose and i % 50 == 0:
                    print(f"  [{i+1}/{len(permutations)}] Progress... best so far: {best_cosine:.6f}")

            except Exception as e:
                results.append(PermResult(name, 0, 0, False, str(e)[:50]))
                if verbose and i % 100 == 0:
                    pass  # 静默错误

        all_results.extend(results)

        # 打印这个配置的最佳结果
        print(f"\n--- Config M={M}, K={K}, N={N} ---")
        print(f"Best: {best_name} with cos_sim={best_cosine:.6f}")

        # Top 10
        sorted_results = sorted(results, key=lambda x: x.cos_sim, reverse=True)
        print("\nTop 10:")
        for r in sorted_results[:10]:
            if r.success:
                print(f"  {r.name:45s}: {r.cos_sim:.6f}")

    # 全局最佳
    print("\n" + "=" * 70)
    print("GLOBAL RESULTS")
    print("=" * 70)

    sorted_all = sorted(all_results, key=lambda x: x.cos_sim, reverse=True)
    best = sorted_all[0]

    print(f"\nBest permutation: {best.name}")
    print(f"Best cosine sim:  {best.cos_sim:.6f}")

    if best.cos_sim > 0.99:
        print("\n🎉 SUCCESS! Found permutation with >0.99 cosine similarity!")
    elif best.cos_sim > 0.98:
        print("\n✓ GOOD! Found permutation with >0.98 cosine similarity!")
    elif best.cos_sim > 0.95:
        print("\n→ Progress. Found permutation with >0.95 cosine similarity.")
    else:
        print("\n✗ No significant improvement. Issue may be:")
        print("  1. SfAtom internal stride pattern more complex")
        print("  2. Data packing format mismatch")
        print("  3. Need more permutation combinations")

    return sorted_all


def quick_test_permutation(perm_name: str):
    """快速测试指定的 permutation"""
    print(f"Quick test for: {perm_name}")

    permutations = generate_all_permutations()
    perm_dict = dict(permutations)

    if perm_name not in perm_dict:
        print(f"Permutation '{perm_name}' not found")
        print(f"Available: {list(perm_dict.keys())[:10]}...")
        return

    perm_func = perm_dict[perm_name]

    try:
        import nvfp4_gemm
    except ImportError:
        print("nvfp4_gemm not available")
        return

    device = torch.device('cuda')
    M, K, N = 256, 2048, 256
    block_size = BLOCK_SIZE
    num_k_blocks = K // block_size

    M_padded = ((M + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    N_padded = ((N + CUTLASS_ROW_TILE - 1) // CUTLASS_ROW_TILE) * CUTLASS_ROW_TILE
    K_blocks_padded = ((num_k_blocks + CUTLASS_K_TILE - 1) // CUTLASS_K_TILE) * CUTLASS_K_TILE

    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)

    x_q, x_scales = quantize_to_nvfp4_sim(x, block_size, use_mse_search=False)
    w_q, w_scales = quantize_to_nvfp4_sim(w, block_size, use_mse_search=False)

    x_packed = pack_nvfp4_data(x_q, block_size)
    w_packed = pack_nvfp4_data(w_q, block_size)

    x_deq = dequantize_nvfp4_sim(x_q, x_scales, block_size, use_fp8_scales=True)
    w_deq = dequantize_nvfp4_sim(w_q, w_scales, block_size, use_fp8_scales=True)
    ref = torch.matmul(x_deq, w_deq.T)

    # Apply permutation
    x_scales_perm = perm_func(x_scales.clone(), M, num_k_blocks, M_padded, K_blocks_padded)
    w_scales_perm = perm_func(w_scales.clone(), N, num_k_blocks, N_padded, K_blocks_padded)

    x_scales_fp8 = convert_scales_to_fp8(x_scales_perm)
    w_scales_fp8 = convert_scales_to_fp8(w_scales_perm)

    # Padding
    expected_x = M_padded * K_blocks_padded * BLOCK_SIZE
    expected_w = N_padded * K_blocks_padded * BLOCK_SIZE

    if x_scales_fp8.numel() < expected_x:
        x_scales_fp8 = torch.cat([x_scales_fp8, torch.zeros(expected_x - x_scales_fp8.numel(), dtype=torch.uint8, device=device)])
    if w_scales_fp8.numel() < expected_w:
        w_scales_fp8 = torch.cat([w_scales_fp8, torch.zeros(expected_w - w_scales_fp8.numel(), dtype=torch.uint8, device=device)])

    output = nvfp4_gemm.gemm_prepared(
        x_packed, w_packed,
        x_scales_fp8[:expected_x], w_scales_fp8[:expected_w],
        M, N, K
    )

    cos_sim = F.cosine_similarity(
        output.flatten().float().unsqueeze(0),
        ref.flatten().unsqueeze(0)
    ).item()

    print(f"Cosine similarity: {cos_sim:.6f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 测试特定 permutation
        quick_test_permutation(sys.argv[1])
    else:
        # 完整 grid search
        run_grid_search()
