#!/usr/bin/env python3
"""
Debug NVFP4 timing breakdown.

Measure where time is spent in NVFP4Linear._forward_cutlass.
"""

import torch
import torch.nn as nn
import time
import sys

sys.path.insert(0, '/home/heima-thor/suliang/Turbo-Pi/openpi/src')

from openpi.models_pytorch.nvfp4_mlp import (
    NVFP4Linear,
    quantize_to_nvfp4_sim,
    pack_nvfp4_data,
    prepare_scales_for_cutlass,
    BLOCK_SIZE,
)


def benchmark_timing_breakdown():
    """Measure where time is spent in NVFP4 forward."""
    print("=" * 70)
    print("NVFP4 Timing Breakdown")
    print("=" * 70)

    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # Typical MLP dimensions
    in_features = 2048
    out_features = 16384
    batch_size = 256

    print(f"Config: batch={batch_size}, in={in_features}, out={out_features}\n")

    # Create dummy linear and NVFP4Linear
    linear = nn.Linear(in_features, out_features, bias=False).cuda()
    nvfp4_linear = NVFP4Linear.from_linear(linear, BLOCK_SIZE, use_cutlass=True)
    nvfp4_linear = nvfp4_linear.cuda()

    # Warmup
    x = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)
    for _ in range(3):
        _ = nvfp4_linear(x)
    torch.cuda.synchronize()

    # Prepare test input
    x = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)
    x_2d = x.view(-1, in_features)

    # Time each step
    n_iters = 10

    print("Timing each step (averaged over {} iters):".format(n_iters))
    print("-" * 50)

    # 1. Quantize input
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        x_q, x_scales = quantize_to_nvfp4_sim(x_2d, BLOCK_SIZE)
    torch.cuda.synchronize()
    quant_time = (time.time() - start) / n_iters * 1000
    print(f"  1. quantize_to_nvfp4_sim:    {quant_time:8.2f} ms")

    # 2. Pack input
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        x_packed = pack_nvfp4_data(x_q, BLOCK_SIZE)
    torch.cuda.synchronize()
    pack_time = (time.time() - start) / n_iters * 1000
    print(f"  2. pack_nvfp4_data:          {pack_time:8.2f} ms")

    # 3. Prepare scales (C++ reorder)
    M = x_2d.shape[0]
    K = in_features
    num_k_blocks = K // BLOCK_SIZE

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        x_scales_cutlass = prepare_scales_for_cutlass(
            x_scales, M, num_k_blocks, convert_to_fp8=True, K=K, is_weight=False
        )
    torch.cuda.synchronize()
    scale_time = (time.time() - start) / n_iters * 1000
    print(f"  3. prepare_scales_for_cutlass: {scale_time:8.2f} ms")

    # 4. CUTLASS GEMM
    try:
        import nvfp4_gemm

        x_scales_cutlass = prepare_scales_for_cutlass(
            x_scales, M, num_k_blocks, convert_to_fp8=True, K=K, is_weight=False
        )

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iters):
            out = nvfp4_gemm.gemm(
                x_packed,
                nvfp4_linear.weight_packed,
                x_scales_cutlass,
                nvfp4_linear.weight_scales_cutlass,
                None  # No bias
            )
        torch.cuda.synchronize()
        gemm_time = (time.time() - start) / n_iters * 1000
        print(f"  4. nvfp4_gemm.gemm:          {gemm_time:8.2f} ms")

    except ImportError as e:
        print(f"  4. CUTLASS GEMM: NOT AVAILABLE ({e})")
        gemm_time = 0

    print("-" * 50)
    total = quant_time + pack_time + scale_time + gemm_time
    print(f"  TOTAL:                       {total:8.2f} ms")
    print()

    # Compare with full forward
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = nvfp4_linear(x)
    torch.cuda.synchronize()
    full_time = (time.time() - start) / n_iters * 1000
    print(f"  Full forward (measured):     {full_time:8.2f} ms")

    # Compare with BF16
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = linear(x)
    torch.cuda.synchronize()
    bf16_time = (time.time() - start) / n_iters * 1000
    print(f"  BF16 forward:                {bf16_time:8.2f} ms")

    print()
    print("=" * 70)
    print("Analysis:")
    print("=" * 70)

    bottleneck = max(
        ("quantize_to_nvfp4_sim", quant_time),
        ("pack_nvfp4_data", pack_time),
        ("prepare_scales_for_cutlass", scale_time),
        ("nvfp4_gemm.gemm", gemm_time),
        key=lambda x: x[1]
    )
    print(f"  Main bottleneck: {bottleneck[0]} ({bottleneck[1]:.2f} ms)")
    print(f"  Expected GEMM time: ~0.2-0.5 ms (from CUTLASS benchmark)")
    print()

    if bottleneck[1] > 10:
        print("  Recommendation: Move bottleneck operation to C++/CUDA")
    else:
        print("  Performance looks reasonable")


if __name__ == "__main__":
    benchmark_timing_breakdown()
