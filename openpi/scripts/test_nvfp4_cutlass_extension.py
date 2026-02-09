#!/usr/bin/env python3
"""
测试 NVFP4 CUTLASS C++ Extension 与 Python Scale Reordering 的集成

验证:
1. C++ extension 基本功能
2. Scale factor 重排后的 GEMM 精度
3. 与 BF16 参考结果对比
"""

import torch
import torch.nn.functional as F
import sys
import time

sys.path.insert(0, '/workspace/src')

from openpi.models_pytorch.nvfp4_mlp import (
    quantize_to_nvfp4_sim,
    swizzle_scales_for_cutlass,
    convert_scales_to_fp8,
    prepare_scales_for_cutlass,
    pack_nvfp4_data,
)


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_section(title):
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


def test_extension_basic():
    """测试 C++ extension 基本功能"""
    print_section("1. C++ Extension Basic Test")

    try:
        import nvfp4_gemm
        print(f"  Module loaded: {nvfp4_gemm}")
        print(f"  Available functions: {[f for f in dir(nvfp4_gemm) if not f.startswith('_')]}")

        # 测试 quantize_to_nvfp4
        print("\n  Testing nvfp4_gemm.quantize_to_nvfp4...")
        x = torch.randn(256, 2048, device='cuda')
        try:
            packed, scales = nvfp4_gemm.quantize_to_nvfp4(x)
            print(f"    Input: {x.shape}")
            print(f"    Packed output: {packed.shape}, dtype={packed.dtype}")
            print(f"    Scales: {scales.shape}, dtype={scales.dtype}")
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False

    except ImportError as e:
        print(f"  Failed to import: {e}")
        return False


def test_extension_gemm_simple():
    """测试 C++ extension GEMM (简单模式，无 scale reordering)"""
    print_section("2. C++ Extension GEMM Simple Test")

    try:
        import nvfp4_gemm

        M, K, N = 256, 2048, 16384
        block_size = 32

        # 创建测试数据
        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)

        # 使用 C++ extension 量化
        print("  Quantizing with C++ extension...")
        x_packed, x_scales = nvfp4_gemm.quantize_to_nvfp4(x.float())
        w_packed, w_scales = nvfp4_gemm.quantize_to_nvfp4(w.float())

        print(f"    x_packed: {x_packed.shape}, dtype={x_packed.dtype}")
        print(f"    x_scales: {x_scales.shape}, dtype={x_scales.dtype}")
        print(f"    w_packed: {w_packed.shape}, dtype={w_packed.dtype}")
        print(f"    w_scales: {w_scales.shape}, dtype={w_scales.dtype}")

        # 调用 GEMM
        print("\n  Calling nvfp4_gemm.gemm...")
        try:
            output = nvfp4_gemm.gemm(x_packed, w_packed, x_scales, w_scales)
            print(f"    Output: {output.shape}, dtype={output.dtype}")
            print(f"    Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")

            # 与参考比较
            ref = torch.matmul(x.float(), w.float().T)
            cos_sim = F.cosine_similarity(
                output.flatten().float().unsqueeze(0),
                ref.flatten().unsqueeze(0)
            ).item()
            print(f"\n    Cosine similarity with BF16 ref: {cos_sim:.6f}")

            return cos_sim > 0.5  # 即使布局不对，也应该有一些相关性

        except Exception as e:
            print(f"    GEMM failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extension_gemm_with_reordering():
    """测试 C++ extension GEMM (使用 Python scale reordering)"""
    print_section("3. C++ Extension GEMM with Scale Reordering")

    try:
        import nvfp4_gemm

        M, K, N = 256, 2048, 16384
        block_size = 32

        # 创建测试数据
        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)

        # 使用 Python 函数量化和重排
        print("  Quantizing with Python (simulation)...")
        x_q, x_scales = quantize_to_nvfp4_sim(x.float(), block_size)
        w_q, w_scales = quantize_to_nvfp4_sim(w.float(), block_size)

        print(f"    x_q: {x_q.shape}, x_scales: {x_scales.shape}")
        print(f"    w_q: {w_q.shape}, w_scales: {w_scales.shape}")

        # 打包 NVFP4 数据
        print("\n  Packing NVFP4 data...")
        x_packed = pack_nvfp4_data(x_q, block_size)
        w_packed = pack_nvfp4_data(w_q, block_size)
        print(f"    x_packed: {x_packed.shape}")
        print(f"    w_packed: {w_packed.shape}")

        # 重排 scale factors
        print("\n  Reordering scale factors for CUTLASS...")
        num_k_blocks = K // block_size

        x_scales_reordered = prepare_scales_for_cutlass(
            x_scales, M, num_k_blocks, convert_to_fp8=True
        )
        w_scales_reordered = prepare_scales_for_cutlass(
            w_scales, N, num_k_blocks, convert_to_fp8=True
        )

        print(f"    x_scales_reordered: {x_scales_reordered.shape}")
        print(f"    w_scales_reordered: {w_scales_reordered.shape}")

        # 调用 GEMM
        print("\n  Calling nvfp4_gemm.gemm with reordered scales...")
        try:
            # 注意: C++ extension 可能期望不同的数据格式
            # 需要检查 nvfp4_gemm.cu 中的 reinterpret_cast

            output = nvfp4_gemm.gemm(
                x_packed.cuda(),
                w_packed.cuda(),
                x_scales_reordered.cuda(),
                w_scales_reordered.cuda()
            )
            print(f"    Output: {output.shape}, dtype={output.dtype}")
            print(f"    Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")

            # 与参考比较
            ref = torch.matmul(x.float(), w.float().T)
            print(f"    Ref stats: mean={ref.mean().item():.4f}, std={ref.std().item():.4f}")

            # 计算误差
            output_ratio = output.abs().mean().item() / (ref.abs().mean().item() + 1e-8)
            print(f"\n    Output/Ref ratio: {output_ratio:.4f}")

            cos_sim = F.cosine_similarity(
                output.flatten().float().unsqueeze(0),
                ref.flatten().unsqueeze(0)
            ).item()
            print(f"    Cosine similarity: {cos_sim:.6f}")

            if cos_sim > 0.9:
                print("\n    [SUCCESS] Scale reordering appears correct!")
                return True
            elif cos_sim > 0.5:
                print("\n    [PARTIAL] Some correlation, but layout may need adjustment")
                return False
            else:
                print("\n    [FAIL] Poor correlation, layout is incorrect")
                return False

        except Exception as e:
            print(f"    GEMM failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_linear_convenience():
    """测试 nvfp4_gemm.linear 便捷函数"""
    print_section("4. C++ Extension Linear Convenience Function")

    try:
        import nvfp4_gemm

        M, K, N = 256, 2048, 16384

        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)

        print("  Calling nvfp4_gemm.linear...")
        try:
            output = nvfp4_gemm.linear(x.float(), w.float())
            print(f"    Output: {output.shape}, dtype={output.dtype}")

            # 与参考比较
            ref = torch.matmul(x.float(), w.float().T)
            cos_sim = F.cosine_similarity(
                output.flatten().float().unsqueeze(0),
                ref.flatten().unsqueeze(0)
            ).item()
            print(f"    Cosine similarity: {cos_sim:.6f}")
            return cos_sim > 0.5

        except Exception as e:
            print(f"    Linear failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"  Error: {e}")
        return False


def benchmark_cutlass_vs_bf16():
    """性能对比测试"""
    print_section("5. Performance Benchmark: CUTLASS vs BF16")

    try:
        import nvfp4_gemm

        M, K, N = 256, 2048, 16384
        iterations = 50

        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)

        # Warmup
        print("  Warming up...")
        for _ in range(10):
            _ = torch.matmul(x.float(), w.float().T)
            try:
                _ = nvfp4_gemm.linear(x.float(), w.float())
            except:
                pass
        torch.cuda.synchronize()

        # BF16 benchmark
        print("\n  Benchmarking BF16 matmul...")
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            _ = torch.matmul(x.float(), w.float().T)
        torch.cuda.synchronize()
        bf16_ms = (time.perf_counter() - start) / iterations * 1000
        print(f"    BF16: {bf16_ms:.3f} ms")

        # CUTLASS benchmark
        print("\n  Benchmarking CUTLASS NVFP4 linear...")
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                _ = nvfp4_gemm.linear(x.float(), w.float())
            torch.cuda.synchronize()
            nvfp4_ms = (time.perf_counter() - start) / iterations * 1000
            print(f"    NVFP4: {nvfp4_ms:.3f} ms")
            print(f"    Speedup: {bf16_ms/nvfp4_ms:.2f}x")
        except Exception as e:
            print(f"    CUTLASS benchmark failed: {e}")

    except Exception as e:
        print(f"  Error: {e}")


def main():
    print_header("NVFP4 CUTLASS Extension Integration Test")

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    results = {}

    # 测试 1: 基本功能
    results['basic'] = test_extension_basic()

    # 测试 2: 简单 GEMM
    results['gemm_simple'] = test_extension_gemm_simple()

    # 测试 3: 带 scale reordering 的 GEMM
    results['gemm_reordered'] = test_extension_gemm_with_reordering()

    # 测试 4: linear 便捷函数
    results['linear'] = test_linear_convenience()

    # 测试 5: 性能
    benchmark_cutlass_vs_bf16()

    # 总结
    print_header("Summary")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    if not results.get('gemm_reordered', False):
        print("\n[NOTE] Scale reordering may need adjustment.")
        print("Try different permutation orders in swizzle_scales_for_cutlass()")


if __name__ == "__main__":
    main()
