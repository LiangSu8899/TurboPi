#!/usr/bin/env python3
"""
测试 FP8 MLP 加速效果

使用 PyTorch FP8 _scaled_mm 替代标准 matmul
验证在 Thor SM110 上的实际加速效果
"""

import torch
import torch.nn as nn
import time


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


class FP8Linear(nn.Module):
    """FP8 量化的 Linear 层 - 优化版本"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 原始权重 (FP16)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)

        # FP8 量化权重 (缓存)
        self.register_buffer('weight_fp8', None)
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_fp8_t', None)  # 预转置

    def quantize_weight(self):
        """预量化权重到 FP8"""
        with torch.no_grad():
            # 计算 scale
            amax = torch.amax(self.weight.abs()).clamp(min=1e-12)
            # FP8 e4m3 max = 448
            self.weight_scale = torch.tensor(448.0 / amax.item(), device=self.weight.device, dtype=torch.float32)

            # 量化到 FP8
            w_scaled = self.weight.float() * self.weight_scale.item()
            w_scaled = w_scaled.clamp(-448, 448)
            self.weight_fp8 = w_scaled.to(torch.float8_e4m3fn)
            # 预转置并确保连续
            self.weight_fp8_t = self.weight_fp8.T.contiguous()

    def forward(self, x):
        if self.weight_fp8 is None:
            self.quantize_weight()

        # 量化输入 - 使用更稳定的方法
        x_amax = torch.amax(x.abs()).clamp(min=1e-12)
        x_scale = torch.tensor(448.0 / x_amax.item(), device=x.device, dtype=torch.float32)
        x_scaled = x.float() * x_scale.item()
        x_scaled = x_scaled.clamp(-448, 448)
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)

        # FP8 GEMM
        # _scaled_mm: out = (A @ B.T) * scale_a * scale_b
        scale_a = 1.0 / x_scale
        scale_b = 1.0 / self.weight_scale

        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8_t,  # 使用预转置
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.float16
        )

        if self.bias is not None:
            out = out + self.bias

        return out


class FP8LinearFast(nn.Module):
    """FP8 Linear - 跳过输入量化的快速版本 (用于 benchmark)"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # FP8 权重
        self.register_buffer('weight_fp8_t', None)
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('input_scale', torch.tensor(1.0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)

    def set_weights(self, weight):
        """设置权重并量化"""
        with torch.no_grad():
            amax = torch.amax(weight.abs()).clamp(min=1e-12)
            self.weight_scale = torch.tensor(448.0 / amax.item(), device=weight.device, dtype=torch.float32)
            w_scaled = weight.float() * self.weight_scale.item()
            w_scaled = w_scaled.clamp(-448, 448)
            weight_fp8 = w_scaled.to(torch.float8_e4m3fn)
            self.weight_fp8_t = weight_fp8.T.contiguous()

    def set_input_scale(self, x):
        """基于典型输入设置输入 scale (推理时固定)"""
        with torch.no_grad():
            amax = torch.amax(x.abs()).clamp(min=1e-12)
            self.input_scale = torch.tensor(448.0 / amax.item(), device=x.device, dtype=torch.float32)

    def forward(self, x):
        # 快速量化 (使用预计算的 scale)
        x_scaled = x.float() * self.input_scale.item()
        x_fp8 = x_scaled.clamp(-448, 448).to(torch.float8_e4m3fn)

        scale_a = 1.0 / self.input_scale
        scale_b = 1.0 / self.weight_scale

        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8_t,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.float16
        )

        if self.bias is not None:
            out = out + self.bias

        return out


class StandardMLP(nn.Module):
    """标准 MLP (BF16)"""

    def __init__(self, hidden_dim=2048, intermediate_dim=16384):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = torch.nn.functional.silu(gate) * up
        return self.down_proj(hidden)


class FP8MLP(nn.Module):
    """FP8 量化 MLP"""

    def __init__(self, hidden_dim=2048, intermediate_dim=16384):
        super().__init__()
        self.gate_proj = FP8Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = FP8Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = FP8Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = torch.nn.functional.silu(gate) * up
        return self.down_proj(hidden)


def benchmark_mlp(model, x, name, warmup=20, iterations=100):
    """Benchmark MLP performance"""

    # Warmup
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations * 1000

    return elapsed


def main():
    print("=" * 60)
    print("Thor SM110 FP8 MLP Acceleration Test")
    print("=" * 60)

    # GPU info
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Compute Capability: SM {props.major}.{props.minor}")

    # Pi0.5 dimensions
    batch_size = 712  # Typical batch size
    hidden_dim = 2048
    intermediate_dim = 16384

    print(f"\nMLP Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Intermediate dim: {intermediate_dim}")

    # Create input
    x = torch.randn(batch_size, hidden_dim, device='cuda', dtype=torch.float16)

    # 1. Standard BF16 MLP
    print_header("Standard BF16 MLP")

    mlp_bf16 = StandardMLP(hidden_dim, intermediate_dim).cuda().bfloat16()
    x_bf16 = x.bfloat16()

    bf16_ms = benchmark_mlp(mlp_bf16, x_bf16, "BF16")
    print(f"Latency: {bf16_ms:.3f} ms")

    # 2. FP8 MLP
    print_header("FP8 MLP")

    mlp_fp8 = FP8MLP(hidden_dim, intermediate_dim).cuda().half()

    # Pre-quantize weights
    mlp_fp8.gate_proj.quantize_weight()
    mlp_fp8.up_proj.quantize_weight()
    mlp_fp8.down_proj.quantize_weight()

    fp8_ms = benchmark_mlp(mlp_fp8, x, "FP8")
    print(f"Latency: {fp8_ms:.3f} ms")

    # Summary
    print_header("Summary")

    speedup = bf16_ms / fp8_ms
    print(f"BF16 MLP: {bf16_ms:.3f} ms")
    print(f"FP8 MLP:  {fp8_ms:.3f} ms")
    print(f"Speedup:  {speedup:.2f}x")

    # Project to full model
    num_layers = 18
    print(f"\nFull Model Estimate ({num_layers} layers):")
    print(f"  BF16: {bf16_ms * num_layers:.1f} ms")
    print(f"  FP8:  {fp8_ms * num_layers:.1f} ms")
    print(f"  Saved: {(bf16_ms - fp8_ms) * num_layers:.1f} ms")

    # Accuracy check
    print_header("Accuracy Check")

    with torch.no_grad():
        # Use same input
        test_x = torch.randn(10, hidden_dim, device='cuda', dtype=torch.float16)

        # BF16 reference
        out_bf16 = mlp_bf16(test_x.bfloat16()).float()

        # FP8
        out_fp8 = mlp_fp8(test_x).float()

        # Compare
        diff = (out_bf16 - out_fp8).abs()
        print(f"Max absolute diff: {diff.max().item():.6f}")
        print(f"Mean absolute diff: {diff.mean().item():.6f}")
        print(f"Relative error: {(diff / (out_bf16.abs() + 1e-6)).mean().item() * 100:.2f}%")


if __name__ == "__main__":
    main()
