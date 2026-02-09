#!/usr/bin/env python3
"""
测试 torchao 量化方案

torchao 是 PyTorch 官方的量化库，支持多种量化方案。
"""

import torch
import time
import torchao
from torchao.quantization import (
    int8_dynamic_activation_int8_weight,
    int4_weight_only,
    quantize_,
)


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_int8_quantization():
    """测试 INT8 动态激活 + INT8 权重"""
    print_header("torchao INT8 量化测试")

    M, K, N = 712, 2048, 16384

    # Create FP16 baseline
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
    baseline_out = linear(x).clone()

    print(f"BF16 baseline: {baseline_time:.3f} ms")

    # INT8 量化
    try:
        linear_int8 = torch.nn.Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)
        linear_int8.load_state_dict(linear.state_dict())

        # Apply INT8 quantization
        quantize_(linear_int8, int8_dynamic_activation_int8_weight())

        # Warmup
        for _ in range(20):
            _ = linear_int8(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = linear_int8(x)
        torch.cuda.synchronize()
        int8_time = (time.perf_counter() - start) / 100 * 1000

        # Accuracy
        int8_out = linear_int8(x)
        cos_sim = torch.nn.functional.cosine_similarity(
            baseline_out.flatten().float(),
            int8_out.flatten().float(),
            dim=0,
        ).item()

        print(f"INT8 (torchao): {int8_time:.3f} ms")
        print(f"Speedup: {baseline_time/int8_time:.2f}x")
        print(f"Cosine similarity: {cos_sim:.6f}")

        return {
            "baseline_ms": baseline_time,
            "int8_ms": int8_time,
            "cosine": cos_sim,
        }

    except Exception as e:
        print(f"INT8 量化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_int4_quantization():
    """测试 INT4 weight-only 量化"""
    print_header("torchao INT4 量化测试")

    M, K, N = 712, 2048, 16384

    # Create FP16 baseline
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
    baseline_out = linear(x).clone()

    print(f"BF16 baseline: {baseline_time:.3f} ms")

    # INT4 量化
    try:
        linear_int4 = torch.nn.Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)
        linear_int4.load_state_dict(linear.state_dict())

        # Apply INT4 weight-only quantization
        quantize_(linear_int4, int4_weight_only())

        # Warmup
        for _ in range(20):
            _ = linear_int4(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = linear_int4(x)
        torch.cuda.synchronize()
        int4_time = (time.perf_counter() - start) / 100 * 1000

        # Accuracy
        int4_out = linear_int4(x)
        cos_sim = torch.nn.functional.cosine_similarity(
            baseline_out.flatten().float(),
            int4_out.flatten().float(),
            dim=0,
        ).item()

        print(f"INT4 (torchao): {int4_time:.3f} ms")
        print(f"Speedup: {baseline_time/int4_time:.2f}x")
        print(f"Cosine similarity: {cos_sim:.6f}")

        return {
            "baseline_ms": baseline_time,
            "int4_ms": int4_time,
            "cosine": cos_sim,
        }

    except Exception as e:
        print(f"INT4 量化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_mlp_quantization():
    """测试完整 MLP 的量化"""
    print_header("MLP 量化测试 (gate + up + SiLU + mul + down)")

    M, K, N = 712, 2048, 16384

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
    baseline_out = mlp(x).clone()

    print(f"BF16 baseline: {baseline_time:.3f} ms")
    print(f"  (这对应 18 层 MLP: {baseline_time * 18:.1f} ms)")

    # INT8 量化
    print("\n--- INT8 量化 ---")
    try:
        mlp_int8 = MLP().to(device='cuda', dtype=torch.bfloat16)
        mlp_int8.load_state_dict(mlp.state_dict())

        quantize_(mlp_int8, int8_dynamic_activation_int8_weight())

        for _ in range(20):
            _ = mlp_int8(x)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(100):
            _ = mlp_int8(x)
        torch.cuda.synchronize()
        int8_time = (time.perf_counter() - start) / 100 * 1000

        int8_out = mlp_int8(x)
        cos_sim = torch.nn.functional.cosine_similarity(
            baseline_out.flatten().float(),
            int8_out.flatten().float(),
            dim=0,
        ).item()

        print(f"INT8 MLP: {int8_time:.3f} ms (18层: {int8_time * 18:.1f} ms)")
        print(f"Speedup: {baseline_time/int8_time:.2f}x")
        print(f"Cosine: {cos_sim:.6f}")

    except Exception as e:
        print(f"INT8 MLP 失败: {e}")

    # INT4 量化
    print("\n--- INT4 量化 ---")
    try:
        mlp_int4 = MLP().to(device='cuda', dtype=torch.bfloat16)
        mlp_int4.load_state_dict(mlp.state_dict())

        quantize_(mlp_int4, int4_weight_only())

        for _ in range(20):
            _ = mlp_int4(x)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(100):
            _ = mlp_int4(x)
        torch.cuda.synchronize()
        int4_time = (time.perf_counter() - start) / 100 * 1000

        int4_out = mlp_int4(x)
        cos_sim = torch.nn.functional.cosine_similarity(
            baseline_out.flatten().float(),
            int4_out.flatten().float(),
            dim=0,
        ).item()

        print(f"INT4 MLP: {int4_time:.3f} ms (18层: {int4_time * 18:.1f} ms)")
        print(f"Speedup: {baseline_time/int4_time:.2f}x")
        print(f"Cosine: {cos_sim:.6f}")

    except Exception as e:
        print(f"INT4 MLP 失败: {e}")


def test_different_group_sizes():
    """测试不同 group size 对 INT4 的影响"""
    print_header("INT4 Group Size 测试")

    M, K, N = 712, 2048, 16384
    linear = torch.nn.Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)
    x = torch.randn(1, M, K, device='cuda', dtype=torch.bfloat16)

    baseline_out = linear(x).clone()

    group_sizes = [32, 64, 128, 256]

    for gs in group_sizes:
        try:
            linear_int4 = torch.nn.Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)
            linear_int4.load_state_dict(linear.state_dict())

            quantize_(linear_int4, int4_weight_only(group_size=gs))

            # Warmup
            for _ in range(20):
                _ = linear_int4(x)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                _ = linear_int4(x)
            torch.cuda.synchronize()
            time_ms = (time.perf_counter() - start) / 100 * 1000

            # Accuracy
            out = linear_int4(x)
            cos_sim = torch.nn.functional.cosine_similarity(
                baseline_out.flatten().float(),
                out.flatten().float(),
                dim=0,
            ).item()

            print(f"Group size {gs:3d}: {time_ms:.3f} ms, cosine: {cos_sim:.6f}")

        except Exception as e:
            print(f"Group size {gs:3d}: FAILED - {e}")


def main():
    print("=" * 60)
    print("torchao 量化方案评估")
    print("=" * 60)

    # 单层测试
    test_int8_quantization()
    test_int4_quantization()

    # 完整 MLP 测试
    test_mlp_quantization()

    # Group size 测试
    test_different_group_sizes()

    # Summary
    print_header("SUMMARY")
    print("""
测试完成。关键结论:

1. 如果 INT4 有显著加速 (>1.5x) 且精度可接受 (cosine > 0.95):
   → 可以应用到 KV Cache MLP，目标 7-8 Hz

2. 如果只有 INT8 有效:
   → 目标 6.5-7 Hz

3. 如果都没有加速:
   → 保持当前 5.7 Hz，需要模型级改动
""")


if __name__ == "__main__":
    main()
