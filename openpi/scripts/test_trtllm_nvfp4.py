#!/usr/bin/env python3
"""
测试 TensorRT-LLM 的 NVFP4 GEMM 在 Thor SM110 上的性能

发现系统中已有的 NVFP4 实现:
1. torch.ops.trtllm.nvfp4_gemm (via tensorrt_llm._torch)
2. nvfuser_direct.nvf_cutlass.nvfp4_scaled_mm (via pytorch fuser)
3. modelopt.torch.quantization.backends.nvfp4_gemm

这些都是预编译的 CUTLASS NVFP4 kernel!
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


def test_trtllm_nvfp4():
    """测试 TensorRT-LLM NVFP4 GEMM"""
    print_header("TensorRT-LLM NVFP4 GEMM")

    try:
        import tensorrt_llm._torch
        print("✅ tensorrt_llm._torch 可用")

        # 检查 ops
        if hasattr(torch.ops, 'trtllm'):
            trtllm_ops = [x for x in dir(torch.ops.trtllm) if not x.startswith('_')]
            print(f"   可用 ops: {len(trtllm_ops)} 个")

            # 查找 NVFP4 相关
            nvfp4_ops = [x for x in trtllm_ops if 'fp4' in x.lower() or 'nvfp' in x.lower()]
            print(f"   NVFP4 ops: {nvfp4_ops}")

            if 'nvfp4_gemm' in trtllm_ops:
                print("✅ torch.ops.trtllm.nvfp4_gemm 可用")
                return True
            elif 'fp4_quantize' in trtllm_ops:
                print("✅ torch.ops.trtllm.fp4_quantize 可用")
                return True
        else:
            print("⚠️ torch.ops.trtllm 不可用")

    except ImportError as e:
        print(f"❌ tensorrt_llm 导入失败: {e}")

    return False


def test_nvfuser_nvfp4():
    """测试 nvfuser NVFP4 GEMM"""
    print_header("NVFuser NVFP4 GEMM")

    try:
        from nvfuser_direct import nvf_cutlass
        print("✅ nvfuser_direct.nvf_cutlass 可用")

        if hasattr(nvf_cutlass, 'nvfp4_scaled_mm'):
            print("✅ nvf_cutlass.nvfp4_scaled_mm 可用")
            return True
        else:
            print("⚠️ nvfp4_scaled_mm 不可用")

    except ImportError as e:
        print(f"⚠️ nvfuser_direct 导入失败: {e}")

    return False


def test_modelopt_nvfp4():
    """测试 ModelOpt NVFP4"""
    print_header("ModelOpt NVFP4")

    try:
        import modelopt.torch.quantization as mtq
        print("✅ modelopt.torch.quantization 可用")

        # 检查 NVFP4 支持
        from modelopt.torch.quantization.backends import nvfp4_gemm
        print("✅ modelopt nvfp4_gemm 模块可用")

        return True

    except ImportError as e:
        print(f"⚠️ modelopt 导入失败: {e}")

    return False


def test_transformer_engine_nvfp4():
    """测试 TransformerEngine NVFP4"""
    print_header("TransformerEngine NVFP4")

    try:
        import transformer_engine.pytorch as te
        print(f"✅ transformer_engine 可用")

        # 检查 NVFP4 支持
        try:
            from transformer_engine.pytorch.custom_recipes import quantization_nvfp4
            print("✅ TransformerEngine quantization_nvfp4 可用")
            return True
        except ImportError:
            pass

        # 检查 FP8 作为备选
        if hasattr(te, 'fp8_autocast'):
            print("✅ TransformerEngine FP8 可用")

    except ImportError as e:
        print(f"⚠️ transformer_engine 导入失败: {e}")

    return False


def benchmark_trtllm_nvfp4():
    """Benchmark TensorRT-LLM NVFP4 GEMM"""
    print_header("TensorRT-LLM NVFP4 GEMM Benchmark")

    try:
        import tensorrt_llm._torch  # noqa: F401

        M, K, N = 712, 2048, 16384

        # 创建输入
        input_tensor = torch.randn(M, K, device='cuda', dtype=torch.float16)
        weight = torch.randn(N, K, device='cuda', dtype=torch.float16)

        # 计算 global scale
        FLOAT8_E4M3_MAX = 448.0
        FLOAT4_E2M1_MAX = 6.0
        input_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(input_tensor.flatten())
        weight_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(weight.flatten())

        # FP4 量化
        x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(input_tensor, input_global_scale, 16, False)
        w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(weight, weight_global_scale, 16, False)

        alpha = 1.0 / (input_global_scale * weight_global_scale)

        print(f"x_fp4 shape: {x_fp4.shape}, dtype: {x_fp4.dtype}")
        print(f"w_fp4 shape: {w_fp4.shape}, dtype: {w_fp4.dtype}")
        print(f"alpha: {alpha.item():.6f}")

        # Warmup
        for _ in range(20):
            out = torch.ops.trtllm.nvfp4_gemm(
                x_fp4, w_fp4, x_sf, w_sf, alpha, input_tensor.dtype
            )
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            out = torch.ops.trtllm.nvfp4_gemm(
                x_fp4, w_fp4, x_sf, w_sf, alpha, input_tensor.dtype
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 100 * 1000

        flops = 2 * M * K * N
        tflops = flops / (elapsed / 1000) / 1e12

        print(f"\nNVFP4 GEMM:")
        print(f"  Shape: ({M}, {K}) @ ({K}, {N})")
        print(f"  Latency: {elapsed:.3f} ms")
        print(f"  Throughput: {tflops:.2f} TFLOPS")

        return elapsed

    except Exception as e:
        print(f"❌ TRT-LLM NVFP4 benchmark 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_nvfuser_nvfp4():
    """Benchmark NVFuser NVFP4 GEMM"""
    print_header("NVFuser NVFP4 GEMM Benchmark")

    try:
        from nvfuser_direct import nvf_cutlass
        from python.direct_utils import (
            pytorch_nvfp4_quantize,
            linear_to_swizzled_128_4,
        )

        M, K, N = 712, 2048, 16384

        # 创建输入
        a_dtype = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b_dtype = torch.randn(N, K, device='cuda', dtype=torch.float16)  # Note: N x K for transpose

        # 计算 global scale
        FLOAT8_E4M3_MAX = 448.0
        FLOAT4_E2M1_MAX = 6.0
        a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten())
        b_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten())
        alpha = 1.0 / (a_global_scale * b_global_scale)

        # FP4 量化
        a_fp4, a_scale_linear = pytorch_nvfp4_quantize(a_dtype, a_global_scale)
        b_fp4, b_scale_linear = pytorch_nvfp4_quantize(b_dtype, b_global_scale)
        a_scale = linear_to_swizzled_128_4(a_scale_linear)
        b_scale = linear_to_swizzled_128_4(b_scale_linear)

        # Warmup
        for _ in range(20):
            out = nvf_cutlass.nvfp4_scaled_mm(
                a_fp4, b_fp4, a_scale, b_scale, alpha, torch.float16
            )
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            out = nvf_cutlass.nvfp4_scaled_mm(
                a_fp4, b_fp4, a_scale, b_scale, alpha, torch.float16
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 100 * 1000

        flops = 2 * M * K * N
        tflops = flops / (elapsed / 1000) / 1e12

        print(f"\nNVFP4 GEMM:")
        print(f"  Shape: ({M}, {K}) @ ({K}, {N})")
        print(f"  Latency: {elapsed:.3f} ms")
        print(f"  Throughput: {tflops:.2f} TFLOPS")

        return elapsed

    except Exception as e:
        print(f"❌ NVFuser NVFP4 benchmark 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("Thor SM110 NVFP4 GEMM 测试")
    print("=" * 60)

    # GPU 信息
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Compute Capability: SM {props.major}.{props.minor}")

    # 1. Baseline
    baseline_ms = benchmark_cublas_baseline()

    # 2. 检查可用实现
    trtllm_ok = test_trtllm_nvfp4()
    nvfuser_ok = test_nvfuser_nvfp4()
    modelopt_ok = test_modelopt_nvfp4()
    te_ok = test_transformer_engine_nvfp4()

    # 3. Benchmark
    nvfp4_ms = None

    if trtllm_ok:
        nvfp4_ms = benchmark_trtllm_nvfp4()

    # if nvfuser_ok and nvfp4_ms is None:
    #     nvfp4_ms = benchmark_nvfuser_nvfp4()

    # 4. Summary
    print_header("总结")

    print(f"cuBLAS BF16 Baseline: {baseline_ms:.3f} ms")

    if nvfp4_ms:
        speedup = baseline_ms / nvfp4_ms
        print(f"NVFP4 GEMM: {nvfp4_ms:.3f} ms")
        print(f"加速比: {speedup:.2f}x")

        # 估算完整模型性能
        print(f"\n完整 MLP 估算 (3 GEMM × 18 层):")
        print(f"  BF16: {baseline_ms * 3 * 18:.1f} ms")
        print(f"  NVFP4: {nvfp4_ms * 3 * 18:.1f} ms")
        print(f"  节省: {(baseline_ms - nvfp4_ms) * 3 * 18:.1f} ms")
    else:
        print("⚠️ 无法运行 NVFP4 GEMM benchmark")
        print("\n可用的实现:")
        print(f"  TensorRT-LLM: {'✅' if trtllm_ok else '❌'}")
        print(f"  NVFuser: {'✅' if nvfuser_ok else '❌'}")
        print(f"  ModelOpt: {'✅' if modelopt_ok else '❌'}")
        print(f"  TransformerEngine: {'✅' if te_ok else '❌'}")


if __name__ == "__main__":
    main()
