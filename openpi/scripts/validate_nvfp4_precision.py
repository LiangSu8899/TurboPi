#!/usr/bin/env python3
"""
验证 NVFP4 精度正确性 - 跳过 LIBERO 环境

测试项：
1. NVFP4Linear 单层精度
2. NVFP4MLP 模块精度
3. 完整模型输出一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    NVFP4Linear,
    NVFP4MLP,
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
    BLOCK_SIZE,
)


def test_nvfp4_linear_precision():
    """测试 NVFP4Linear 单层精度"""
    print("=" * 70)
    print("Test 1: NVFP4Linear 单层精度")
    print("=" * 70)

    device = torch.device('cuda')
    in_features = 2048
    out_features = 8192
    batch_size = 16

    # 创建 BF16 Linear
    linear_bf16 = nn.Linear(in_features, out_features, bias=False).to(device, dtype=torch.bfloat16)

    # 创建 NVFP4 Linear (使用 simulation mode - 慢但精确)
    nvfp4_linear = NVFP4Linear.from_linear(linear_bf16, BLOCK_SIZE, use_cutlass=True)
    nvfp4_linear = nvfp4_linear.to(device)

    # 测试输入
    x = torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16)

    # BF16 输出
    with torch.no_grad():
        out_bf16 = linear_bf16(x)

    # NVFP4 模拟输出 (使用 simulation mode)
    nvfp4_linear.use_cutlass = False
    with torch.no_grad():
        out_nvfp4_sim = nvfp4_linear(x)

    # NVFP4 CUTLASS 输出
    nvfp4_linear.use_cutlass = True
    with torch.no_grad():
        out_nvfp4_cutlass = nvfp4_linear(x)

    # 计算 cosine similarity
    cos_sim_vs_bf16 = F.cosine_similarity(
        out_nvfp4_sim.flatten().float().unsqueeze(0),
        out_bf16.flatten().float().unsqueeze(0)
    ).item()

    cos_cutlass_vs_sim = F.cosine_similarity(
        out_nvfp4_cutlass.flatten().float().unsqueeze(0),
        out_nvfp4_sim.flatten().float().unsqueeze(0)
    ).item()

    cos_cutlass_vs_bf16 = F.cosine_similarity(
        out_nvfp4_cutlass.flatten().float().unsqueeze(0),
        out_bf16.flatten().float().unsqueeze(0)
    ).item()

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out_bf16.shape}")
    print()
    print(f"  NVFP4 Sim vs BF16:     {cos_sim_vs_bf16:.6f}")
    print(f"  CUTLASS vs NVFP4 Sim:  {cos_cutlass_vs_sim:.6f}")
    print(f"  CUTLASS vs BF16:       {cos_cutlass_vs_bf16:.6f}")
    print()

    if cos_cutlass_vs_sim > 0.99:
        print("  ✅ CUTLASS 与 Python 模拟一致 (>0.99)")
    else:
        print("  ❌ CUTLASS 与 Python 模拟不一致")

    return cos_cutlass_vs_sim > 0.99


def test_nvfp4_mlp_precision():
    """测试 NVFP4MLP 模块精度"""
    print()
    print("=" * 70)
    print("Test 2: NVFP4MLP 模块精度 (GemmaMLP 结构)")
    print("=" * 70)

    device = torch.device('cuda')
    hidden_size = 2048
    intermediate_size = 8192
    batch_size = 16

    # 创建模拟的 GemmaMLP
    class DummyGemmaMLP(nn.Module):
        def __init__(self, hidden_size, intermediate_size):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            gate = F.gelu(self.gate_proj(x), approximate='tanh')
            up = self.up_proj(x)
            return self.down_proj(gate * up)

    # 创建 BF16 MLP
    mlp_bf16 = DummyGemmaMLP(hidden_size, intermediate_size).to(device, dtype=torch.bfloat16)

    # 创建 NVFP4 MLP
    nvfp4_mlp = NVFP4MLP.from_gemma_mlp(mlp_bf16, BLOCK_SIZE, use_cutlass=True)
    nvfp4_mlp = nvfp4_mlp.to(device)

    # 测试输入
    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    # BF16 输出
    with torch.no_grad():
        out_bf16 = mlp_bf16(x)

    # NVFP4 CUTLASS 输出
    with torch.no_grad():
        out_nvfp4 = nvfp4_mlp(x)

    # 计算 cosine similarity
    cos_sim = F.cosine_similarity(
        out_nvfp4.flatten().float().unsqueeze(0),
        out_bf16.flatten().float().unsqueeze(0)
    ).item()

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out_bf16.shape}")
    print()
    print(f"  NVFP4 MLP vs BF16 MLP: {cos_sim:.6f}")
    print()

    if cos_sim > 0.95:
        print("  ✅ NVFP4 MLP 精度合格 (>0.95)")
    elif cos_sim > 0.90:
        print("  ⚠️  NVFP4 MLP 精度可用 (>0.90)")
    else:
        print("  ❌ NVFP4 MLP 精度不足")

    return cos_sim > 0.90


def test_timing_breakdown():
    """测试时间分解 - 确认瓶颈"""
    print()
    print("=" * 70)
    print("Test 3: 时间分解 (确认在线量化瓶颈)")
    print("=" * 70)

    device = torch.device('cuda')
    in_features = 2048
    out_features = 8192
    batch_size = 256

    linear = nn.Linear(in_features, out_features, bias=False).to(device)
    nvfp4_linear = NVFP4Linear.from_linear(linear, BLOCK_SIZE, use_cutlass=True)
    nvfp4_linear = nvfp4_linear.to(device)

    x = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(3):
        _ = nvfp4_linear(x)
    torch.cuda.synchronize()

    # 测量 quantize_to_nvfp4_sim (这是瓶颈)
    x_2d = x.view(-1, in_features)

    torch.cuda.synchronize()
    start = time.time()
    n_iters = 10
    for _ in range(n_iters):
        x_q, x_scales = quantize_to_nvfp4_sim(x_2d, BLOCK_SIZE, use_mse_search=False)
    torch.cuda.synchronize()
    quant_time = (time.time() - start) / n_iters * 1000

    # 测量完整 forward
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = nvfp4_linear(x)
    torch.cuda.synchronize()
    forward_time = (time.time() - start) / n_iters * 1000

    print(f"  Batch size: {batch_size}")
    print(f"  Matrix:     [{batch_size}x{in_features}] x [{out_features}x{in_features}]")
    print()
    print(f"  激活量化时间 (quantize_to_nvfp4_sim): {quant_time:.2f} ms")
    print(f"  完整 forward 时间:                    {forward_time:.2f} ms")
    print(f"  量化占比:                             {quant_time/forward_time*100:.1f}%")
    print()

    if quant_time / forward_time > 0.7:
        print("  ⚠️  在线量化是主要瓶颈！")
        print("     解决方案:")
        print("     1. W4A16: 只量化权重，激活保持 BF16")
        print("     2. W4A4:  写 CUDA kernel 做快速激活量化")
    else:
        print("  ✅ 在线量化开销可接受")


def main():
    print("=" * 70)
    print("NVFP4 精度验证测试")
    print("=" * 70)
    print()

    try:
        import nvfp4_gemm
        print(f"CUTLASS extension: ✅ Loaded")
        print(f"  Available functions: {[f for f in dir(nvfp4_gemm) if not f.startswith('_')]}")
    except ImportError as e:
        print(f"CUTLASS extension: ❌ Not available ({e})")
        print("  Will use simulation mode only")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    # 运行测试
    test1_passed = test_nvfp4_linear_precision()
    test2_passed = test_nvfp4_mlp_precision()
    test_timing_breakdown()

    print()
    print("=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"  NVFP4Linear 精度: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  NVFP4MLP 精度:    {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print()

    if test1_passed and test2_passed:
        print("  🎉 NVFP4 精度验证通过！")
        print("     Scale Layout 修复成功。")
        print()
        print("  下一步：优化在线量化速度")
        print("     - 选项 A: W4A16 (只量化权重，激活保持 BF16)")
        print("     - 选项 B: 写 CUDA kernel 做快速激活量化")
    else:
        print("  ❌ NVFP4 精度验证失败，需要检查 Scale Layout 逻辑")


if __name__ == "__main__":
    main()
