#!/usr/bin/env python3
"""
测试 TensorRT INT8 量化

Thor 声称支持 INT8 加速，测试是否真的有效。
"""

import torch
import time
import os


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_cublas_int8():
    """直接测试 cuBLAS INT8 GEMM"""
    print_header("cuBLAS INT8 GEMM 测试")

    M, K, N = 712, 2048, 16384

    # FP16 baseline
    a_fp16 = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b_fp16 = torch.randn(K, N, device='cuda', dtype=torch.float16)

    for _ in range(20):
        _ = torch.matmul(a_fp16, b_fp16)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(a_fp16, b_fp16)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / 100 * 1000

    print(f"FP16 GEMM: {fp16_time:.3f} ms")

    # INT8 GEMM
    a_int8 = torch.randint(-128, 127, (M, K), device='cuda', dtype=torch.int8)
    b_int8 = torch.randint(-128, 127, (K, N), device='cuda', dtype=torch.int8)

    # 检查 _int_mm 是否可用
    try:
        out = torch._int_mm(a_int8, b_int8)
        print(f"  torch._int_mm output shape: {out.shape}, dtype: {out.dtype}")

        for _ in range(20):
            _ = torch._int_mm(a_int8, b_int8)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(100):
            _ = torch._int_mm(a_int8, b_int8)
        torch.cuda.synchronize()
        int8_time = (time.perf_counter() - start) / 100 * 1000

        print(f"INT8 GEMM (torch._int_mm): {int8_time:.3f} ms")
        print(f"Speedup vs FP16: {fp16_time/int8_time:.2f}x")

    except Exception as e:
        print(f"torch._int_mm 不可用: {e}")


def test_torch_backends():
    """检查 torch 后端配置"""
    print_header("PyTorch 后端配置检查")

    # cuDNN 配置
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

    # cuBLAS 配置
    if hasattr(torch.backends, 'cuda'):
        print(f"matmul precision: {torch.backends.cuda.matmul.allow_tf32}")

    # 检查 INT8 matmul 是否有专门的配置
    if hasattr(torch.backends, 'cudnn'):
        print(f"allow_tf32: {torch.backends.cudnn.allow_tf32}")


def test_tensorrt_linear_engine():
    """尝试用 TensorRT 构建 INT8 Linear engine"""
    print_header("TensorRT INT8 Linear 测试")

    try:
        import tensorrt as trt
        import numpy as np

        print(f"TensorRT version: {trt.__version__}")

        M, K, N = 712, 2048, 16384

        # 创建 logger
        logger = trt.Logger(trt.Logger.WARNING)

        # 创建 builder
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()

        # 启用 INT8
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("✅ INT8 mode enabled")
        else:
            print("⚠️ Fast INT8 not available")

        # 构建一个简单的 MatMul 网络
        input_tensor = network.add_input("input", trt.float16, (M, K))

        # 权重
        weights = np.random.randn(K, N).astype(np.float16)
        weights_trt = trt.Weights(weights)

        # 添加 MatMul 操作 (通过 Constant + MatMul)
        weight_const = network.add_constant((K, N), weights_trt)
        matmul = network.add_matrix_multiply(
            input_tensor, trt.MatrixOperation.NONE,
            weight_const.get_output(0), trt.MatrixOperation.NONE
        )

        matmul.get_output(0).name = "output"
        network.mark_output(matmul.get_output(0))

        # 设置 INT8 校准器 (使用简单的 min-max)
        # 注意: 实际应用需要真实数据校准
        if config.get_flag(trt.BuilderFlag.INT8):
            # 设置动态范围
            input_tensor.set_dynamic_range(-1.0, 1.0)
            weight_const.get_output(0).set_dynamic_range(-1.0, 1.0)
            matmul.get_output(0).set_dynamic_range(-1.0, 1.0)

        # 构建 engine
        print("Building TensorRT engine...")
        serialized = builder.build_serialized_network(network, config)

        if serialized is None:
            print("❌ Failed to build TensorRT engine")
            return

        # 创建 runtime 和 engine
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)

        if engine is None:
            print("❌ Failed to deserialize engine")
            return

        # 创建执行上下文
        context = engine.create_execution_context()

        # 分配缓冲区
        input_buf = torch.randn(M, K, device='cuda', dtype=torch.float16)
        output_buf = torch.empty(M, N, device='cuda', dtype=torch.float16)

        # 执行
        bindings = [input_buf.data_ptr(), output_buf.data_ptr()]

        # Warmup
        for _ in range(20):
            context.execute_v2(bindings)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            context.execute_v2(bindings)
        torch.cuda.synchronize()
        trt_time = (time.perf_counter() - start) / 100 * 1000

        # Compare with PyTorch
        linear = torch.nn.Linear(K, N, bias=False, device='cuda', dtype=torch.float16)
        linear.weight.data = torch.from_numpy(weights.T).to('cuda')

        for _ in range(20):
            _ = linear(input_buf)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(100):
            _ = linear(input_buf)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / 100 * 1000

        print(f"\nResults:")
        print(f"PyTorch FP16: {pytorch_time:.3f} ms")
        print(f"TensorRT INT8: {trt_time:.3f} ms")
        print(f"Speedup: {pytorch_time/trt_time:.2f}x")

        # 检查输出
        pytorch_out = linear(input_buf)
        cos_sim = torch.nn.functional.cosine_similarity(
            pytorch_out.flatten().float(),
            output_buf.flatten().float(),
            dim=0,
        ).item()
        print(f"Cosine similarity: {cos_sim:.6f}")

    except Exception as e:
        print(f"TensorRT INT8 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_current_best_performance():
    """测试当前最佳性能 (BF16 + CUDA Graph)"""
    print_header("当前最佳性能测试 (BF16 + CUDA Graph)")

    M, K, N = 712, 2048, 16384

    # MLP
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(K, N, bias=False)
            self.up_proj = torch.nn.Linear(K, N, bias=False)
            self.down_proj = torch.nn.Linear(N, K, bias=False)

        def forward(self, x):
            gate = torch.nn.functional.silu(self.gate_proj(x))
            up = self.up_proj(x)
            return self.down_proj(gate * up)

    mlp = MLP().to(device='cuda', dtype=torch.bfloat16)
    x = torch.randn(1, M, K, device='cuda', dtype=torch.bfloat16)

    # Baseline
    for _ in range(20):
        _ = mlp(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = mlp(x)
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) / 100 * 1000

    print(f"MLP baseline: {baseline_time:.3f} ms")
    print(f"18 层 MLP: {baseline_time * 18:.1f} ms")

    # CUDA Graph
    static_x = torch.zeros(1, M, K, device='cuda', dtype=torch.bfloat16)

    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_out = mlp(static_x)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = mlp(static_x)

    # Replay
    static_x.copy_(x)

    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    graph_time = (time.perf_counter() - start) / 100 * 1000

    print(f"\nMLP with CUDA Graph: {graph_time:.3f} ms")
    print(f"18 层 MLP: {graph_time * 18:.1f} ms")
    print(f"Speedup: {baseline_time/graph_time:.2f}x")


def summary():
    print_header("结论")
    print("""
测试总结:

1. ❌ Triton: 在 Thor 上性能只有 cuBLAS 的 41%
2. ❌ torchao INT8: 比 BF16 慢 11 倍
3. ❌ torchao INT4: 需要 fbgemm-gpu-genai
4. ⚠️ TensorRT INT8: 需要进一步测试
5. ✅ CUDA Graph: 微小提升 (~1-2%)

当前结论:
- Thor 平台上的量化加速基本不可用
- Triton 和 torchao 的 kernel 都没有针对 Thor SM 11.0 优化
- 只能依赖 cuBLAS BF16，维持当前 ~5.7 Hz

下一步建议:
1. 等待 NVIDIA 发布 Thor 优化的量化库
2. 使用 TensorRT 构建完整的 INT8 pipeline
3. 考虑模型级改动 (蒸馏、剪枝、减少层数)
""")


def main():
    test_torch_backends()
    test_cublas_int8()
    test_current_best_performance()
    # test_tensorrt_linear_engine()  # 可能会很慢，按需启用
    summary()


if __name__ == "__main__":
    main()
