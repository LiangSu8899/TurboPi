#!/usr/bin/env python3
"""
测试 CUTLASS NVFP4 在 Thor SM110 上的性能

根据 NVIDIA Forum 的成功案例:
- 需要使用 sm_110a 架构标志
- 需要 CUDA 13.0+
- 可达到 ~878 TFLOP/s dense FP4

运行前需要安装 CUTLASS:
    pip install nvidia-cutlass

如果安装失败，可能需要从源码编译:
    git clone https://github.com/NVIDIA/cutlass.git
    cd cutlass
    pip install .
"""

import os
import sys
import time
import subprocess


def check_cuda_version():
    """检查 CUDA 版本"""
    print("=" * 60)
    print("CUDA 版本检查")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True
        )
        print(result.stdout)

        # 解析版本
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                parts = line.split(',')
                for part in parts:
                    if 'release' in part.lower():
                        version = part.strip().split()[-1]
                        major, minor = version.split('.')[:2]
                        print(f"CUDA 版本: {major}.{minor}")

                        if int(major) >= 13:
                            print("✅ CUDA 13.0+ 满足要求")
                            return True
                        else:
                            print("❌ 需要 CUDA 13.0+")
                            return False
    except Exception as e:
        print(f"❌ 无法检查 CUDA 版本: {e}")
        return False

    return False


def check_gpu_arch():
    """检查 GPU 架构"""
    print("\n" + "=" * 60)
    print("GPU 架构检查")
    print("=" * 60)

    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA 不可用")
            return None

        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Compute Capability: SM {props.major}.{props.minor}")

        # Thor 是 SM 11.0 (Blackwell)
        if props.major == 11 and props.minor == 0:
            print("✅ 检测到 Thor (SM 11.0 / Blackwell)")
            print("   需要使用架构标志: sm_110a 或 sm_110f")
            return "110a"
        elif props.major >= 10:
            print(f"✅ 检测到 Blackwell 架构 SM {props.major}.{props.minor}")
            return f"{props.major}{props.minor}a"
        else:
            print(f"⚠️ 非 Blackwell GPU: SM {props.major}.{props.minor}")
            print("   NVFP4 需要 Blackwell 架构 (SM 10.0+)")
            return None

    except ImportError:
        print("❌ PyTorch 未安装")
        return None


def check_cutlass_installation():
    """检查 CUTLASS 安装"""
    print("\n" + "=" * 60)
    print("CUTLASS 安装检查")
    print("=" * 60)

    # 检查 Python 包
    try:
        import cutlass
        print(f"✅ CUTLASS Python 包已安装")
        if hasattr(cutlass, '__version__'):
            print(f"   版本: {cutlass.__version__}")
        return True
    except ImportError:
        print("❌ CUTLASS Python 包未安装")

    # 检查 nvidia-cutlass
    try:
        import nvidia.cutlass as cutlass
        print(f"✅ nvidia-cutlass 包已安装")
        return True
    except ImportError:
        print("❌ nvidia-cutlass 包未安装")

    print("\n安装建议:")
    print("  pip install nvidia-cutlass")
    print("  或从源码:")
    print("  git clone https://github.com/NVIDIA/cutlass.git")
    print("  cd cutlass && pip install .")

    return False


def test_cutlass_basic():
    """测试基本 CUTLASS 功能"""
    print("\n" + "=" * 60)
    print("CUTLASS 基本功能测试")
    print("=" * 60)

    try:
        # 尝试导入 cutlass
        try:
            import cutlass
        except ImportError:
            import nvidia.cutlass as cutlass

        import torch

        # 检查是否有 Blackwell 支持
        if hasattr(cutlass, 'DataType'):
            print(f"✅ CUTLASS DataType 可用")

            # 检查 FP4 支持
            if hasattr(cutlass.DataType, 'e2m1'):
                print(f"✅ FP4 (e2m1) 数据类型支持")
            elif hasattr(cutlass.DataType, 'f4'):
                print(f"✅ FP4 数据类型支持")
            else:
                print("⚠️ 未找到 FP4 数据类型 (可能需要更新 CUTLASS)")

        return True

    except Exception as e:
        print(f"❌ CUTLASS 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cutlass_dsl_nvfp4():
    """测试 CUTLASS DSL NVFP4 GEMM"""
    print("\n" + "=" * 60)
    print("CUTLASS DSL NVFP4 GEMM 测试")
    print("=" * 60)

    try:
        # 检查 CuTe DSL 是否可用
        try:
            from cutlass.cute.dsl import kernel, gemm
            print("✅ CuTe DSL 可用")
        except ImportError:
            print("❌ CuTe DSL 不可用")
            print("   可能需要更新到 CUTLASS 4.x")
            return False

        # 尝试运行 NVFP4 示例
        # 这需要具体的 CUTLASS 版本和配置
        print("⚠️ NVFP4 GEMM 测试需要完整的 CUTLASS 4.x 环境")
        print("   请参考: examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py")

        return True

    except Exception as e:
        print(f"❌ CUTLASS DSL 测试失败: {e}")
        return False


def benchmark_baseline():
    """测试 cuBLAS baseline 性能"""
    print("\n" + "=" * 60)
    print("cuBLAS BF16 Baseline 性能")
    print("=" * 60)

    import torch

    # Pi0.5 MLP 尺寸
    M, K, N = 712, 2048, 16384

    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)

    # Warmup
    for _ in range(20):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 100 * 1000

    # 计算 TFLOPS
    flops = 2 * M * K * N
    tflops = flops / (elapsed / 1000) / 1e12

    print(f"Shape: ({M}, {K}) @ ({K}, {N})")
    print(f"Latency: {elapsed:.3f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")

    # MLP 完整计算 (gate + up + SiLU + mul + down)
    print(f"\n完整 MLP 估算 (3 个 GEMM):")
    print(f"  单层 MLP: {elapsed * 3:.2f} ms")
    print(f"  18 层 MLP: {elapsed * 3 * 18:.1f} ms")

    return elapsed


def suggest_next_steps():
    """建议下一步操作"""
    print("\n" + "=" * 60)
    print("下一步建议")
    print("=" * 60)

    print("""
根据 NVIDIA Forum 的成功案例，要在 Thor 上使用 NVFP4:

1. 安装 CUTLASS 4.x:
   pip install nvidia-cutlass

   或从源码编译 (推荐):
   git clone https://github.com/NVIDIA/cutlass.git
   cd cutlass
   mkdir build && cd build
   cmake .. -DCUTLASS_NVCC_ARCHS="110a" \\
            -DCUTLASS_LIBRARY_KERNELS=all \\
            -DCUTLASS_UNITY_BUILD_ENABLED=ON
   make -j$(nproc)
   cd ../python && pip install .

2. 测试 NVFP4 GEMM 示例:
   cd cutlass/examples/python/CuTeDSL/blackwell
   python dense_blockscaled_gemm_persistent.py

3. 如果成功，将 MLP 层替换为 CUTLASS NVFP4 kernel

预期性能:
- cuBLAS BF16: ~0.47 ms per GEMM
- CUTLASS NVFP4: ~0.12-0.15 ms per GEMM (3-4x 加速)
- 总延迟从 ~173 ms 降到 ~70-80 ms
- 频率从 ~5.7 Hz 提升到 ~13-14 Hz
""")


def main():
    print("=" * 60)
    print("Thor SM110 CUTLASS NVFP4 测试")
    print("=" * 60)

    # 1. 检查 CUDA 版本
    cuda_ok = check_cuda_version()

    # 2. 检查 GPU 架构
    arch = check_gpu_arch()

    # 3. 检查 CUTLASS 安装
    cutlass_ok = check_cutlass_installation()

    # 4. Baseline 性能
    baseline = benchmark_baseline()

    if cutlass_ok:
        # 5. 测试 CUTLASS 基本功能
        test_cutlass_basic()

        # 6. 测试 CUTLASS DSL
        test_cutlass_dsl_nvfp4()

    # 7. 建议下一步
    suggest_next_steps()

    # Summary
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    print(f"CUDA 13.0+: {'✅' if cuda_ok else '❌'}")
    print(f"Thor SM110: {'✅' if arch else '❌'}")
    print(f"CUTLASS: {'✅' if cutlass_ok else '❌ 需要安装'}")
    print(f"Baseline 延迟: {baseline:.3f} ms")

    if not cutlass_ok:
        print("\n⚠️ 需要先安装 CUTLASS 才能测试 NVFP4")
        print("   pip install nvidia-cutlass")


if __name__ == "__main__":
    main()
