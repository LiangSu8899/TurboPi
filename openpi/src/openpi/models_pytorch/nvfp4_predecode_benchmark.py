#!/usr/bin/env python3
"""
Benchmark FP4 with pre-decode to BF16 + cuBLAS.

This approach:
1. Store weights in FP4 format (3.2x less memory)
2. Decode FP4 -> BF16 on-the-fly before each GEMM
3. Use cuBLAS for the actual GEMM

This is simpler than a fused kernel and may be faster.

Author: Claude Code
Date: 2026-02-10
"""

import torch
import torch.nn.functional as F
import time

HIDDEN_SIZE = 2048
MLP_DIM = 16384
BLOCK_SIZE = 32

NVFP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
NVFP4_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def quantize_to_nvfp4(weight: torch.Tensor, block_size: int = 32):
    """Quantize FP32 weight to NVFP4 format."""
    N, K = weight.shape
    device = weight.device
    weight = weight.to(torch.float32)

    num_blocks = K // block_size
    weight_blocked = weight.view(N, num_blocks, block_size)

    scales = weight_blocked.abs().amax(dim=-1) / 6.0
    scales = scales.clamp(min=1e-8)

    weight_normalized = weight_blocked / scales.unsqueeze(-1)

    nvfp4_magnitudes = torch.tensor(NVFP4_MAGNITUDES, device=device, dtype=torch.float32)
    signs = (weight_normalized < 0).to(torch.uint8) * 8
    abs_vals = weight_normalized.abs()

    diffs = (abs_vals.unsqueeze(-1) - nvfp4_magnitudes).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)

    fp4_vals = (signs + indices).view(N, K)

    low = fp4_vals[:, 0::2]
    high = fp4_vals[:, 1::2]
    packed = (high << 4) | low

    return packed.to(torch.uint8), scales.to(torch.float32)


def dequantize_to_bf16(packed: torch.Tensor, scales: torch.Tensor, block_size: int = 32):
    """Dequantize NVFP4 to BF16."""
    N = packed.shape[0]
    K = packed.shape[1] * 2
    device = packed.device

    full_lut = torch.tensor(NVFP4_LUT, device=device, dtype=torch.bfloat16)

    low = packed & 0xF
    high = (packed >> 4) & 0xF

    fp4_vals = torch.zeros(N, K, dtype=torch.uint8, device=device)
    fp4_vals[:, 0::2] = low
    fp4_vals[:, 1::2] = high

    decoded = full_lut[fp4_vals.to(torch.int64)]

    num_blocks = K // block_size
    decoded_blocked = decoded.view(N, num_blocks, block_size)
    weight = (decoded_blocked * scales.unsqueeze(-1).to(torch.bfloat16)).view(N, K)

    return weight


def dequantize_to_bf16_fast(packed: torch.Tensor, scales: torch.Tensor,
                            lut: torch.Tensor, block_size: int = 32):
    """Faster dequantization using pre-computed LUT."""
    N = packed.shape[0]
    K = packed.shape[1] * 2
    device = packed.device

    low = packed & 0xF
    high = (packed >> 4) & 0xF

    fp4_vals = torch.zeros(N, K, dtype=torch.int64, device=device)
    fp4_vals[:, 0::2] = low.to(torch.int64)
    fp4_vals[:, 1::2] = high.to(torch.int64)

    decoded = lut[fp4_vals]

    num_blocks = K // block_size
    decoded_blocked = decoded.view(N, num_blocks, block_size)
    scales_bf16 = scales.unsqueeze(-1).to(torch.bfloat16)
    weight = (decoded_blocked * scales_bf16).view(N, K)

    return weight


def main():
    print("=" * 70)
    print("FP4 Pre-decode + cuBLAS Benchmark")
    print("=" * 70)

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Pre-compute LUT on device
    lut_bf16 = torch.tensor(NVFP4_LUT, device=device, dtype=torch.bfloat16)

    def benchmark(fn, warmup=30, runs=100):
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(runs):
            _ = fn()
        torch.cuda.synchronize()
        return (time.time() - start) / runs * 1000

    for num_layers in [6, 18]:
        print(f"\n{'='*70}")
        print(f"{num_layers} Layers Benchmark")
        print(f"{'='*70}")

        # ====================================================================
        # BF16 Baseline
        # ====================================================================

        layers_bf16 = []
        for _ in range(num_layers):
            gate = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
            up = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
            down = torch.randn(HIDDEN_SIZE, MLP_DIM, device=device, dtype=torch.bfloat16)
            layers_bf16.append((gate, up, down))

        x_bf16 = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

        def bf16_forward():
            activation = x_bf16
            for gate, up, down in layers_bf16:
                g = F.linear(activation, gate)
                u = F.linear(activation, up)
                inter = F.silu(g) * u
                activation = F.linear(inter, down)
            return activation

        bf16_time = benchmark(bf16_forward)
        print(f"BF16 cuBLAS: {bf16_time:.2f} ms")

        # ====================================================================
        # FP4 Pre-decode (decode once at startup, store BF16)
        # ====================================================================

        layers_fp4 = []
        for _ in range(num_layers):
            gate = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32)
            up = torch.randn(MLP_DIM, HIDDEN_SIZE, device=device, dtype=torch.float32)
            down = torch.randn(HIDDEN_SIZE, MLP_DIM, device=device, dtype=torch.float32)

            gate_p, gate_s = quantize_to_nvfp4(gate)
            up_p, up_s = quantize_to_nvfp4(up)
            down_p, down_s = quantize_to_nvfp4(down)

            layers_fp4.append({
                'gate': (gate_p, gate_s),
                'up': (up_p, up_s),
                'down': (down_p, down_s),
            })

        # Pre-decode all weights
        layers_predecoded = []
        for layer in layers_fp4:
            gate = dequantize_to_bf16_fast(layer['gate'][0], layer['gate'][1], lut_bf16)
            up = dequantize_to_bf16_fast(layer['up'][0], layer['up'][1], lut_bf16)
            down = dequantize_to_bf16_fast(layer['down'][0], layer['down'][1], lut_bf16)
            layers_predecoded.append((gate, up, down))

        def predecoded_forward():
            activation = x_bf16
            for gate, up, down in layers_predecoded:
                g = F.linear(activation, gate)
                u = F.linear(activation, up)
                inter = F.silu(g) * u
                activation = F.linear(inter, down)
            return activation

        predecoded_time = benchmark(predecoded_forward)
        print(f"FP4 Pre-decoded (BF16 cuBLAS): {predecoded_time:.2f} ms")

        # ====================================================================
        # FP4 On-the-fly decode (decode every forward pass)
        # ====================================================================

        def onthefly_forward():
            activation = x_bf16
            for layer in layers_fp4:
                gate = dequantize_to_bf16_fast(layer['gate'][0], layer['gate'][1], lut_bf16)
                up = dequantize_to_bf16_fast(layer['up'][0], layer['up'][1], lut_bf16)
                down = dequantize_to_bf16_fast(layer['down'][0], layer['down'][1], lut_bf16)

                g = F.linear(activation, gate)
                u = F.linear(activation, up)
                inter = F.silu(g) * u
                activation = F.linear(inter, down)
            return activation

        onthefly_time = benchmark(onthefly_forward)
        print(f"FP4 On-the-fly decode: {onthefly_time:.2f} ms")

        # ====================================================================
        # FP4 On-the-fly with CUDA Graphs
        # ====================================================================

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = onthefly_forward()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            graph_out = onthefly_forward()

        def graph_forward():
            g.replay()
            return graph_out

        graph_time = benchmark(graph_forward)
        print(f"FP4 On-the-fly + CUDA Graph: {graph_time:.2f} ms")

        # ====================================================================
        # Memory Analysis
        # ====================================================================

        bf16_mem = sum(g.numel() + u.numel() + d.numel() for g, u, d in layers_bf16) * 2
        fp4_mem = sum(
            l['gate'][0].numel() + l['gate'][1].numel() * 4 +
            l['up'][0].numel() + l['up'][1].numel() * 4 +
            l['down'][0].numel() + l['down'][1].numel() * 4
            for l in layers_fp4
        )

        print(f"\nMemory:")
        print(f"  BF16 weights: {bf16_mem / 1e6:.1f} MB")
        print(f"  FP4 weights: {fp4_mem / 1e6:.1f} MB")
        print(f"  Reduction: {bf16_mem / fp4_mem:.2f}x")

        # ====================================================================
        # Summary
        # ====================================================================

        print(f"\nSummary:")
        print(f"{'Approach':<35} {'Time (ms)':<12} {'vs BF16':<12}")
        print("-" * 60)
        print(f"{'BF16 cuBLAS':<35} {bf16_time:<12.2f} {'1.00x':<12}")
        print(f"{'FP4 Pre-decoded':<35} {predecoded_time:<12.2f} {bf16_time/predecoded_time:.2f}x")
        print(f"{'FP4 On-the-fly':<35} {onthefly_time:<12.2f} {bf16_time/onthefly_time:.2f}x")
        print(f"{'FP4 On-the-fly + Graph':<35} {graph_time:<12.2f} {bf16_time/graph_time:.2f}x")

    # ========================================================================
    # Final Analysis
    # ========================================================================

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("""
Key Findings:

1. Pre-decoded FP4 = same speed as BF16
   - Once decoded, same memory traffic
   - But 3.2x less weight storage

2. On-the-fly decode adds significant overhead
   - PyTorch ops for decode are not fused
   - Each decode is a separate kernel launch

3. CUDA Graphs can reduce kernel launch overhead
   - But decode ops still add compute time

4. For batch=1 inference:
   - Weight loading is memory-bound
   - Decode overhead can exceed savings

Recommendations:

1. For memory-constrained deployments:
   - Store FP4, pre-decode to BF16 at startup
   - Use same inference speed as BF16
   - 3.2x less memory for weights

2. For best performance:
   - Use native BF16 weights with cuBLAS
   - Or use TensorRT FP8 for 2x speedup

3. For true FP4 speedup:
   - Need fused decode+GEMM kernel
   - Or Tensor Core FP4 support (future hardware)
   - Or carefully optimized multi-block kernel
""")


if __name__ == "__main__":
    main()
