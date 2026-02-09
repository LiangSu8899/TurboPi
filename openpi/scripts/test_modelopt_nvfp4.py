#!/usr/bin/env python3
"""
测试 ModelOpt NVFP4 GEMM 在 Thor SM110 上的性能
"""

import torch
import time

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def benchmark_cublas_baseline():
    """cuBLAS BF16 baseline"""
    print_header("cuBLAS BF16 Baseline")

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

    flops = 2 * M * K * N
    tflops = flops / (elapsed / 1000) / 1e12

    print(f"Shape: ({M}, {K}) @ ({K}, {N})")
    print(f"Latency: {elapsed:.3f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")

    return elapsed


def explore_modelopt():
    """探索 ModelOpt NVFP4 接口"""
    print_header("ModelOpt NVFP4 探索")

    try:
        import modelopt.torch.quantization as mtq
        print("✅ ModelOpt quantization 可用")

        # 检查 backends
        from modelopt.torch.quantization import backends
        print("\n可用 backends:")
        for attr in dir(backends):
            if not attr.startswith('_'):
                obj = getattr(backends, attr)
                print(f"  {attr}: {type(obj).__name__}")

        # 检查 nvfp4_gemm
        from modelopt.torch.quantization.backends import nvfp4_gemm
        print(f"\nnvfp4_gemm type: {type(nvfp4_gemm)}")

        import inspect
        try:
            sig = inspect.signature(nvfp4_gemm)
            print(f"nvfp4_gemm signature: {sig}")
        except Exception as e:
            print(f"Cannot get signature: {e}")

        # 查看源码位置
        try:
            source_file = inspect.getfile(nvfp4_gemm)
            print(f"Source file: {source_file}")
        except Exception as e:
            print(f"Cannot get source file: {e}")

        return True

    except Exception as e:
        print(f"❌ ModelOpt 探索失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_modelopt_quantize():
    """测试 ModelOpt 量化功能"""
    print_header("ModelOpt 量化测试")

    try:
        import modelopt.torch.quantization as mtq

        # 检查可用的量化配置
        print("可用的量化配置:")
        for attr in dir(mtq):
            if 'FP4' in attr.upper() or 'NVFP4' in attr.upper() or 'QUANT' in attr.upper():
                print(f"  {attr}")

        # 尝试 FP4 量化配置
        if hasattr(mtq, 'FP4_DEFAULT_CFG'):
            print("\n✅ FP4_DEFAULT_CFG 可用")
            print(f"   {mtq.FP4_DEFAULT_CFG}")
        elif hasattr(mtq, 'NVFP4_DEFAULT_CFG'):
            print("\n✅ NVFP4_DEFAULT_CFG 可用")
            print(f"   {mtq.NVFP4_DEFAULT_CFG}")

        # 检查量化配置
        configs = ['FP4_DEFAULT_CFG', 'NVFP4_DEFAULT_CFG', 'INT4_AWQ_CFG', 'FP8_DEFAULT_CFG']
        for cfg_name in configs:
            if hasattr(mtq, cfg_name):
                print(f"\n{cfg_name}:")
                cfg = getattr(mtq, cfg_name)
                print(f"  {cfg}")

        return True

    except Exception as e:
        print(f"❌ ModelOpt 量化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nvfp4_gemm_direct():
    """直接测试 nvfp4_gemm"""
    print_header("NVFP4 GEMM 直接测试")

    try:
        from modelopt.torch.quantization.backends import nvfp4_gemm

        M, K, N = 712, 2048, 16384

        # 创建输入
        input_tensor = torch.randn(M, K, device='cuda', dtype=torch.float16)
        weight = torch.randn(N, K, device='cuda', dtype=torch.float16)

        # 尝试调用
        print("尝试调用 nvfp4_gemm...")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Weight shape: {weight.shape}")

        # 检查函数参数
        import inspect
        try:
            sig = inspect.signature(nvfp4_gemm)
            print(f"Function signature: {sig}")

            # 尝试获取源码
            source = inspect.getsource(nvfp4_gemm)
            print(f"\nSource code (first 500 chars):")
            print(source[:500])
        except Exception as e:
            print(f"Cannot inspect function: {e}")

        return True

    except Exception as e:
        print(f"❌ NVFP4 GEMM 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def search_nvfp4_in_modelopt():
    """搜索 ModelOpt 中的 NVFP4 实现"""
    print_header("搜索 ModelOpt NVFP4 实现")

    import subprocess
    import os

    try:
        # 获取 modelopt 安装路径
        import modelopt
        modelopt_path = os.path.dirname(modelopt.__file__)
        print(f"ModelOpt 路径: {modelopt_path}")

        # 搜索 nvfp4 相关文件
        result = subprocess.run(
            ['grep', '-r', 'nvfp4\|NVFP4\|fp4_gemm', modelopt_path],
            capture_output=True, text=True, timeout=30
        )

        if result.stdout:
            print("\n找到 NVFP4 相关代码:")
            lines = result.stdout.strip().split('\n')
            for line in lines[:30]:  # 只显示前30行
                print(f"  {line[:100]}...")

        return True

    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        return False


def main():
    print("=" * 60)
    print("Thor SM110 ModelOpt NVFP4 测试")
    print("=" * 60)

    # GPU 信息
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Compute Capability: SM {props.major}.{props.minor}")

    # 1. Baseline
    baseline_ms = benchmark_cublas_baseline()

    # 2. 探索 ModelOpt
    explore_modelopt()

    # 3. 测试量化配置
    test_modelopt_quantize()

    # 4. 直接测试 NVFP4 GEMM
    test_nvfp4_gemm_direct()

    # 5. 搜索 NVFP4 实现
    search_nvfp4_in_modelopt()

    # Summary
    print_header("总结")
    print(f"cuBLAS BF16 Baseline: {baseline_ms:.3f} ms")


if __name__ == "__main__":
    main()
