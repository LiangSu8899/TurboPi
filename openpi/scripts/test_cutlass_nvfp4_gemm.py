#!/usr/bin/env python3
"""
测试 CUTLASS NVFP4 GEMM 在 Thor SM110 上的性能

使用 CUTLASS 4.x 的 Python API (cutlass_library, cutlass_cppgen)
"""

import os
import sys
import time
import torch
import numpy as np


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_cutlass_nvfp4_support():
    """检查 CUTLASS NVFP4 支持"""
    print_header("CUTLASS NVFP4 支持检查")

    try:
        import cutlass_library
        print("✅ cutlass_library 可用")

        # 检查数据类型
        dt = cutlass_library.DataType
        print("\n可用的低精度数据类型:")

        fp4_types = []
        for attr in dir(dt):
            if any(x in attr.lower() for x in ['fp4', 'e2m1', 'e4m3', 'e5m2', 'f8', 'f4', 'bf8']):
                val = getattr(dt, attr)
                print(f"  {attr}: {val}")
                if 'e2m1' in attr.lower() or 'fp4' in attr.lower():
                    fp4_types.append(attr)

        if fp4_types:
            print(f"\n✅ NVFP4 数据类型可用: {fp4_types}")
            return True
        else:
            print("\n⚠️ 未找到 NVFP4 数据类型")
            return False

    except ImportError as e:
        print(f"❌ cutlass_library 导入失败: {e}")
        return False


def check_cutlass_cppgen():
    """检查 CUTLASS C++ 代码生成器"""
    print_header("CUTLASS C++ 代码生成器检查")

    try:
        import cutlass_cppgen as cutlass
        print("✅ cutlass_cppgen 可用")

        # 检查 Blackwell 支持
        if hasattr(cutlass, 'sm'):
            print(f"  SM 支持: {cutlass.sm}")

        # 检查 GEMM 配置
        if hasattr(cutlass, 'GemmDescription'):
            print("  GemmDescription 可用")

        return True

    except ImportError as e:
        print(f"❌ cutlass_cppgen 导入失败: {e}")
        return False


def benchmark_cublas_baseline():
    """cuBLAS BF16 baseline"""
    print_header("cuBLAS BF16 Baseline")

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
    iterations = 100
    for _ in range(iterations):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations * 1000

    # TFLOPS
    flops = 2 * M * K * N
    tflops = flops / (elapsed / 1000) / 1e12

    print(f"Shape: ({M}, {K}) @ ({K}, {N})")
    print(f"Latency: {elapsed:.3f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")

    return elapsed, tflops


def try_cutlass_profiler():
    """尝试使用 CUTLASS profiler 运行 NVFP4 GEMM"""
    print_header("CUTLASS Profiler 测试 (如果可用)")

    import subprocess
    import shutil

    # 检查 cutlass_profiler 是否可用
    profiler = shutil.which('cutlass_profiler')
    if not profiler:
        # 检查常见位置
        possible_paths = [
            '/usr/local/cutlass/build/tools/profiler/cutlass_profiler',
            '/opt/cutlass/build/tools/profiler/cutlass_profiler',
            os.path.expanduser('~/cutlass/build/tools/profiler/cutlass_profiler'),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                profiler = p
                break

    if not profiler:
        print("⚠️ cutlass_profiler 未找到")
        print("   需要从源码编译 CUTLASS 才能使用 profiler")
        print("\n   编译步骤:")
        print("   git clone https://github.com/NVIDIA/cutlass.git")
        print("   cd cutlass && mkdir build && cd build")
        print('   cmake .. -DCUTLASS_NVCC_ARCHS="110a" -DCUTLASS_LIBRARY_KERNELS=all')
        print("   make cutlass_profiler -j$(nproc)")
        return None

    print(f"✅ 找到 cutlass_profiler: {profiler}")

    # 运行 NVFP4 GEMM 测试
    M, K, N = 712, 2048, 16384

    cmd = [
        profiler,
        '--operation=gemm',
        f'--m={M}',
        f'--n={N}',
        f'--k={K}',
        '--A=e2m1',  # NVFP4
        '--B=e2m1',  # NVFP4
        '--C=bf16',
        '--D=bf16',
        '--warmup-iterations=10',
        '--profiling-iterations=100',
    ]

    print(f"\n运行: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.returncode != 0:
            print(f"stderr: {result.stderr}")
        return result.stdout
    except Exception as e:
        print(f"❌ Profiler 运行失败: {e}")
        return None


def check_cute_dsl():
    """检查 CuTe DSL 是否可用"""
    print_header("CuTe DSL 检查")

    try:
        from cutlass.cute.dsl import kernel, gemm
        print("✅ CuTe DSL 可用")
        return True
    except ImportError:
        pass

    try:
        from cutlass_cppgen.cute import dsl
        print("✅ CuTe DSL (via cutlass_cppgen) 可用")
        return True
    except ImportError:
        pass

    print("⚠️ CuTe DSL 不可用")
    print("   CuTe DSL 在 CUTLASS 4.x 中提供 Python NVFP4 GEMM 示例")
    print("   可能需要从源码安装完整版 CUTLASS")

    return False


def search_nvfp4_examples():
    """搜索系统中的 NVFP4 示例"""
    print_header("搜索 NVFP4 示例")

    import subprocess

    # 搜索可能的位置
    search_paths = [
        '/usr/local',
        '/opt',
        os.path.expanduser('~'),
        '/workspace',
    ]

    for base in search_paths:
        if not os.path.exists(base):
            continue

        try:
            result = subprocess.run(
                ['find', base, '-name', '*nvfp4*', '-o', '-name', '*blockscaled*gemm*'],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                print(f"在 {base} 中找到:")
                for line in result.stdout.strip().split('\n')[:10]:
                    print(f"  {line}")
        except Exception:
            pass


def summary():
    """总结和建议"""
    print_header("总结与建议")

    print("""
当前状态:
- CUTLASS 4.2.0 已安装 (cutlass_library, cutlass_cppgen)
- NVFP4 (e2m1) 数据类型已支持
- 但完整的 NVFP4 GEMM 需要:
  1. 从源码编译 CUTLASS 以获取 profiler 和示例
  2. 或使用 CuTe DSL Python API

下一步行动:

方案 A: 从源码编译 CUTLASS (推荐)
   git clone https://github.com/NVIDIA/cutlass.git
   cd cutlass
   mkdir build && cd build
   cmake .. -DCUTLASS_NVCC_ARCHS="110a" \\
            -DCUTLASS_LIBRARY_KERNELS=all \\
            -DCUTLASS_UNITY_BUILD_ENABLED=ON
   make cutlass_profiler -j$(nproc)

方案 B: 使用 sgl-kernel (如果可用)
   pip install sgl-kernel
   # sgl-kernel 包含预编译的 NVFP4 kernel

方案 C: 直接使用 PyTorch FP8 (降级方案)
   # 虽然不是 FP4，但 FP8 可能在 Thor 上有加速
   # 需要 PyTorch 2.5+ with torch.float8_e4m3fn

预期性能收益 (如果 NVFP4 工作):
- cuBLAS BF16:  ~0.45 ms / GEMM
- CUTLASS NVFP4: ~0.12 ms / GEMM (3-4x 加速)
- 总体频率: 5.7 Hz -> 13-17 Hz
""")


def main():
    print("=" * 60)
    print("Thor SM110 CUTLASS NVFP4 GEMM 详细测试")
    print("=" * 60)

    # 1. GPU 信息
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}")
        print(f"Compute Capability: SM {props.major}.{props.minor}")
        print(f"Memory: {props.total_memory / 1024**3:.1f} GB")

    # 2. 检查 CUTLASS NVFP4 支持
    nvfp4_ok = check_cutlass_nvfp4_support()

    # 3. 检查 C++ 代码生成器
    cppgen_ok = check_cutlass_cppgen()

    # 4. 检查 CuTe DSL
    dsl_ok = check_cute_dsl()

    # 5. cuBLAS baseline
    baseline_ms, baseline_tflops = benchmark_cublas_baseline()

    # 6. 尝试 CUTLASS profiler
    try_cutlass_profiler()

    # 7. 搜索现有示例
    search_nvfp4_examples()

    # 8. 总结
    summary()

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print(f"NVFP4 数据类型支持: {'✅' if nvfp4_ok else '❌'}")
    print(f"C++ 代码生成器: {'✅' if cppgen_ok else '❌'}")
    print(f"CuTe DSL: {'✅' if dsl_ok else '❌'}")
    print(f"cuBLAS Baseline: {baseline_ms:.3f} ms ({baseline_tflops:.1f} TFLOPS)")
    print()
    print("需要从源码编译 CUTLASS 才能运行 NVFP4 GEMM benchmark")


if __name__ == "__main__":
    main()
