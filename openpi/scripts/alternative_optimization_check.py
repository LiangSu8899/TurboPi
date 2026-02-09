#!/usr/bin/env python3
"""
替代优化方案评估

由于 Triton 在 Thor 上性能差 (0.41x cuBLAS)，
需要评估其他优化方案。
"""

import torch
import time
import sys


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_torch_compile():
    """测试 torch.compile 在 Thor 上的效果"""
    print_header("torch.compile 优化测试")

    M, K, N = 712, 2048, 16384

    # Create linear layer
    linear = torch.nn.Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)
    x = torch.randn(1, M, K, device='cuda', dtype=torch.bfloat16)

    # Baseline
    for _ in range(20):
        _ = linear(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = linear(x)
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) / 100 * 1000

    print(f"Baseline (eager): {baseline_time:.3f} ms")

    # torch.compile with different modes
    modes = ["default", "reduce-overhead", "max-autotune"]

    for mode in modes:
        try:
            compiled_linear = torch.compile(linear, mode=mode)

            # Warmup (compile happens here)
            for _ in range(5):
                _ = compiled_linear(x)
            torch.cuda.synchronize()

            # More warmup
            for _ in range(20):
                _ = compiled_linear(x)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                _ = compiled_linear(x)
            torch.cuda.synchronize()
            compiled_time = (time.perf_counter() - start) / 100 * 1000

            speedup = baseline_time / compiled_time
            print(f"torch.compile(mode='{mode}'): {compiled_time:.3f} ms ({speedup:.2f}x)")

        except Exception as e:
            print(f"torch.compile(mode='{mode}'): FAILED - {e}")


def check_cuda_graph():
    """测试 CUDA Graph 的效果"""
    print_header("CUDA Graph 优化测试")

    M, K, N = 712, 2048, 16384

    linear = torch.nn.Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)
    x = torch.randn(1, M, K, device='cuda', dtype=torch.bfloat16)

    # Baseline
    for _ in range(20):
        _ = linear(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = linear(x)
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) / 100 * 1000

    print(f"Baseline: {baseline_time:.3f} ms")

    # CUDA Graph
    try:
        # Static input buffer
        static_x = torch.zeros(1, M, K, device='cuda', dtype=torch.bfloat16)

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_out = linear(static_x)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = linear(static_x)

        # Copy input and replay
        static_x.copy_(x)

        # Warmup replay
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            g.replay()
        torch.cuda.synchronize()
        graph_time = (time.perf_counter() - start) / 100 * 1000

        speedup = baseline_time / graph_time
        print(f"CUDA Graph: {graph_time:.3f} ms ({speedup:.2f}x)")

        # Verify correctness
        expected = linear(x)
        actual = static_out
        cos_sim = torch.nn.functional.cosine_similarity(
            expected.flatten().float(),
            actual.flatten().float(),
            dim=0,
        ).item()
        print(f"  Correctness (cosine): {cos_sim:.6f}")

    except Exception as e:
        print(f"CUDA Graph: FAILED - {e}")


def check_tensorrt_int8():
    """检查 TensorRT INT8 量化"""
    print_header("TensorRT INT8 量化检查")

    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")

        # Check if INT8 is supported
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        if builder.platform_has_fast_int8:
            print("✅ Thor 支持 INT8 加速")
        else:
            print("⚠️ Thor 不支持快速 INT8")

    except ImportError:
        print("❌ TensorRT not installed")


def check_torch_ao_quantization():
    """检查 PyTorch AO 量化"""
    print_header("PyTorch AO 量化检查")

    try:
        import torchao
        print(f"✅ torchao installed")

        # 检查可用的量化方法
        if hasattr(torchao, 'quantize_'):
            print("  quantize_ available")
        if hasattr(torchao.quantization, 'int8_dynamic_activation_int8_weight'):
            print("  int8_dynamic_activation_int8_weight available")
        if hasattr(torchao.quantization, 'int4_weight_only'):
            print("  int4_weight_only available")

    except ImportError:
        print("❌ torchao not installed")
        print("   Install: pip install torchao")


def check_bitsandbytes():
    """检查 bitsandbytes"""
    print_header("bitsandbytes 检查")

    try:
        import bitsandbytes as bnb
        print(f"✅ bitsandbytes version: {bnb.__version__}")

        # 测试 INT8 Linear
        M, K, N = 712, 2048, 16384

        linear_fp16 = torch.nn.Linear(K, N, bias=False, device='cuda', dtype=torch.float16)
        x = torch.randn(1, M, K, device='cuda', dtype=torch.float16)

        # 创建 INT8 Linear
        linear_int8 = bnb.nn.Linear8bitLt(K, N, bias=False)
        linear_int8.weight = bnb.nn.Int8Params(linear_fp16.weight.data.clone())
        linear_int8 = linear_int8.to('cuda')

        # Test
        try:
            out = linear_int8(x)
            print(f"  INT8 Linear output shape: {out.shape}")

            # Benchmark
            for _ in range(20):
                _ = linear_int8(x)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(100):
                _ = linear_int8(x)
            torch.cuda.synchronize()
            int8_time = (time.perf_counter() - start) / 100 * 1000

            # FP16 baseline
            for _ in range(20):
                _ = linear_fp16(x)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(100):
                _ = linear_fp16(x)
            torch.cuda.synchronize()
            fp16_time = (time.perf_counter() - start) / 100 * 1000

            print(f"  FP16: {fp16_time:.3f} ms")
            print(f"  INT8 (bitsandbytes): {int8_time:.3f} ms")
            print(f"  Speedup: {fp16_time/int8_time:.2f}x")

        except Exception as e:
            print(f"  ⚠️ INT8 Linear failed: {e}")

    except ImportError:
        print("❌ bitsandbytes not installed")
        print("   Install: pip install bitsandbytes")


def check_gptq():
    """检查 GPTQ/AWQ"""
    print_header("GPTQ/AWQ 量化库检查")

    libs = [
        ("auto_gptq", "AutoGPTQ"),
        ("awq", "AWQ"),
        ("optimum", "Hugging Face Optimum"),
    ]

    for lib, name in libs:
        try:
            __import__(lib)
            print(f"✅ {name} ({lib}) installed")
        except ImportError:
            print(f"❌ {name} ({lib}) not installed")


def check_kernel_fusion_opportunities():
    """分析 kernel fusion 机会"""
    print_header("Kernel Fusion 分析")

    M, K, N = 712, 2048, 16384

    # MLP with SiLU
    gate_proj = torch.nn.Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)
    up_proj = torch.nn.Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)
    down_proj = torch.nn.Linear(N, K, bias=False, device='cuda', dtype=torch.bfloat16)

    x = torch.randn(1, M, K, device='cuda', dtype=torch.bfloat16)

    def mlp_forward(x):
        gate = torch.nn.functional.silu(gate_proj(x))
        up = up_proj(x)
        return down_proj(gate * up)

    # Baseline
    for _ in range(20):
        _ = mlp_forward(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = mlp_forward(x)
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) / 100 * 1000

    print(f"MLP baseline (gate + up + SiLU + mul + down): {baseline_time:.3f} ms")

    # Try torch.compile
    try:
        compiled_mlp = torch.compile(mlp_forward, mode="reduce-overhead")

        for _ in range(5):
            _ = compiled_mlp(x)
        torch.cuda.synchronize()

        for _ in range(20):
            _ = compiled_mlp(x)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(100):
            _ = compiled_mlp(x)
        torch.cuda.synchronize()
        compiled_time = (time.perf_counter() - start) / 100 * 1000

        print(f"MLP compiled: {compiled_time:.3f} ms ({baseline_time/compiled_time:.2f}x)")

    except Exception as e:
        print(f"MLP compiled: FAILED - {e}")


def summary():
    """总结可用的优化方案"""
    print_header("SUMMARY: 可用的优化方案")

    print("""
基于测试结果，以下是 Thor 平台上的可行优化方案:

1. ✅ CUDA Graph (可用)
   - 消除 CPU launch overhead
   - 预期节省: 2-5 ms

2. ⚠️ torch.compile (需要测试)
   - Kernel fusion 机会有限 (因为主要是 GEMM)
   - 预期节省: 1-3 ms

3. ❌ Triton (不推荐)
   - 在 Thor 上性能只有 cuBLAS 的 41%
   - INT4 kernel 会更慢

4. ⚠️ TensorRT INT8 (需要测试)
   - 如果 scale 问题只影响 FP8/FP4，INT8 可能可用
   - 预期加速: 1.5-2x

5. ⚠️ bitsandbytes INT8 (需要安装测试)
   - 依赖 cuBLAS INT8
   - 需要验证 Thor 支持情况

结论:
- 如果 TensorRT/bitsandbytes INT8 可用: 目标 ~7-8 Hz 可行
- 如果只有 CUDA Graph + compile: 目标 ~6.5 Hz
- 如果都不行: 维持 ~5.7 Hz，需要模型级改动
""")


def main():
    print("=" * 60)
    print("Thor 替代优化方案评估")
    print("=" * 60)

    check_torch_compile()
    check_cuda_graph()
    check_tensorrt_int8()
    check_torch_ao_quantization()
    check_bitsandbytes()
    check_gptq()
    check_kernel_fusion_opportunities()
    summary()


if __name__ == "__main__":
    main()
