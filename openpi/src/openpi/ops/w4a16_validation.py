#!/usr/bin/env python3
"""
W4A16 Validation and CUDA Graphs Benchmark Script.

This script validates the W4A16 implementation and demonstrates:
1. Correctness verification against FP16 reference
2. Performance with and without CUDA Graphs
3. End-to-end latency with zero Python overhead

Author: Claude Code
Date: 2026-02-11
"""

import sys
import os
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add module path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_ops_dir = _script_dir  # openpi/ops
_openpi_dir = os.path.dirname(_ops_dir)  # openpi
_src_dir = os.path.dirname(_openpi_dir)  # src
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from openpi.ops.w4a16_gemv import w4a16_gemv, precompile_kernels, QUANT_BLOCK
from openpi.utils.w4a16_packer import W4A16Packer, W4A16PackerFast, PackedWeight
from openpi.modules.w4a16_linear import W4A16Linear


# ============================================================================
# Test Configuration
# ============================================================================

# MLP dimensions (Qwen2.5-3B style)
IN_FEATURES = 2048
HIDDEN_FEATURES = 16384
BATCH_SIZE = 1

# Benchmark settings
WARMUP_ITERS = 100
BENCHMARK_ITERS = 500


# ============================================================================
# Test Functions
# ============================================================================

def test_correctness():
    """Test W4A16 correctness against FP16 reference."""
    print("\n" + "=" * 60)
    print("1. CORRECTNESS VERIFICATION")
    print("=" * 60)

    device = torch.device("cuda")

    # Create reference linear layer
    linear = nn.Linear(IN_FEATURES, HIDDEN_FEATURES, bias=True).to(device)
    linear.weight.data = torch.randn_like(linear.weight)
    linear.bias.data = torch.randn_like(linear.bias)

    # Convert to W4A16
    w4a16_layer = W4A16Linear.from_linear(linear)

    # Test input
    x = torch.randn(BATCH_SIZE, IN_FEATURES, dtype=torch.float16, device=device)

    # Forward pass
    with torch.no_grad():
        y_ref = linear(x.float()).half()
        y_w4a16 = w4a16_layer(x)

    # Compute metrics
    cos_sim = F.cosine_similarity(
        y_ref.flatten().unsqueeze(0),
        y_w4a16.flatten().unsqueeze(0)
    ).item()

    mse = F.mse_loss(y_ref, y_w4a16).item()
    max_diff = (y_ref - y_w4a16).abs().max().item()

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y_w4a16.shape}")
    print(f"\nAccuracy metrics:")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  MSE:               {mse:.6f}")
    print(f"  Max diff:          {max_diff:.6f}")
    print(f"\nVerification: {'PASS' if cos_sim > 0.99 else 'FAIL'}")

    return cos_sim > 0.99


def test_raw_kernel_performance():
    """Test raw TVM kernel performance."""
    print("\n" + "=" * 60)
    print("2. RAW KERNEL PERFORMANCE")
    print("=" * 60)

    device = torch.device("cuda")

    # Precompile kernel
    print("Precompiling TVM kernel...")
    precompile_kernels([(HIDDEN_FEATURES, IN_FEATURES)])

    # Create test data
    x = torch.randn(1, IN_FEATURES, dtype=torch.float16, device=device)

    num_scale_blocks = IN_FEATURES // QUANT_BLOCK
    weight_packed = torch.randint(
        0, 2**31, (num_scale_blocks, HIDDEN_FEATURES, 4),
        dtype=torch.int32, device=device
    )
    scales = torch.randn(
        num_scale_blocks, HIDDEN_FEATURES,
        dtype=torch.float16, device=device
    )

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = w4a16_gemv(x, weight_packed, scales)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(BENCHMARK_ITERS):
        _ = w4a16_gemv(x, weight_packed, scales)
    torch.cuda.synchronize()

    avg_ms = (time.time() - start) / BENCHMARK_ITERS * 1000

    print(f"Dimensions: N={HIDDEN_FEATURES}, K={IN_FEATURES}")
    print(f"Average latency: {avg_ms:.4f} ms")
    print(f"Target (< 0.2ms): {'ACHIEVED!' if avg_ms < 0.2 else f'{avg_ms/0.2:.1f}x slower'}")

    return avg_ms


def test_module_performance():
    """Test W4A16Linear module performance."""
    print("\n" + "=" * 60)
    print("3. W4A16Linear MODULE PERFORMANCE")
    print("=" * 60)

    device = torch.device("cuda")

    # Create layer
    layer = W4A16Linear(IN_FEATURES, HIDDEN_FEATURES, bias=True, device=device)

    # Initialize weights
    weight = torch.randn(HIDDEN_FEATURES, IN_FEATURES, dtype=torch.bfloat16)
    layer.pack_weights(weight)

    # Test input
    x = torch.randn(BATCH_SIZE, IN_FEATURES, dtype=torch.float16, device=device)

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = layer(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(BENCHMARK_ITERS):
        _ = layer(x)
    torch.cuda.synchronize()

    avg_ms = (time.time() - start) / BENCHMARK_ITERS * 1000

    print(f"W4A16Linear({IN_FEATURES}, {HIDDEN_FEATURES})")
    print(f"Average latency: {avg_ms:.4f} ms")
    print(f"Target (< 0.2ms): {'ACHIEVED!' if avg_ms < 0.2 else f'{avg_ms/0.2:.1f}x slower'}")

    return avg_ms


def test_cuda_graphs():
    """Test CUDA Graphs performance improvement."""
    print("\n" + "=" * 60)
    print("4. CUDA GRAPHS BENCHMARK")
    print("=" * 60)

    device = torch.device("cuda")

    # Create layer
    layer = W4A16Linear(IN_FEATURES, HIDDEN_FEATURES, bias=True, device=device)
    weight = torch.randn(HIDDEN_FEATURES, IN_FEATURES, dtype=torch.bfloat16)
    layer.pack_weights(weight)

    # Fixed input (required for CUDA Graphs)
    x = torch.randn(BATCH_SIZE, IN_FEATURES, dtype=torch.float16, device=device)

    # ==================== Without CUDA Graphs ====================
    print("\n--- Without CUDA Graphs ---")

    for _ in range(WARMUP_ITERS):
        _ = layer(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(BENCHMARK_ITERS):
        y = layer(x)
    torch.cuda.synchronize()

    no_graph_ms = (time.time() - start) / BENCHMARK_ITERS * 1000
    print(f"Average latency: {no_graph_ms:.4f} ms")

    # ==================== With CUDA Graphs ====================
    print("\n--- With CUDA Graphs ---")

    # Capture graph
    print("Capturing CUDA Graph...")

    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y = layer(x)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_graph = layer(x)

    print("Graph captured!")

    # Warmup graph replay
    for _ in range(WARMUP_ITERS):
        g.replay()
    torch.cuda.synchronize()

    # Benchmark graph replay
    start = time.time()
    for _ in range(BENCHMARK_ITERS):
        g.replay()
    torch.cuda.synchronize()

    with_graph_ms = (time.time() - start) / BENCHMARK_ITERS * 1000
    print(f"Average latency: {with_graph_ms:.4f} ms")

    # ==================== Comparison ====================
    print("\n--- Comparison ---")
    print(f"Without CUDA Graphs: {no_graph_ms:.4f} ms")
    print(f"With CUDA Graphs:    {with_graph_ms:.4f} ms")
    print(f"Speedup:             {no_graph_ms / with_graph_ms:.2f}x")
    print(f"Overhead eliminated: {no_graph_ms - with_graph_ms:.4f} ms")

    return no_graph_ms, with_graph_ms


def test_mlp_simulation():
    """Simulate full MLP layer (gate + up + silu*mul + down)."""
    print("\n" + "=" * 60)
    print("5. MLP SIMULATION (gate_proj + up_proj + silu*mul + down_proj)")
    print("=" * 60)

    device = torch.device("cuda")

    # Create MLP layers
    gate_proj = W4A16Linear(IN_FEATURES, HIDDEN_FEATURES, bias=False, device=device)
    up_proj = W4A16Linear(IN_FEATURES, HIDDEN_FEATURES, bias=False, device=device)
    down_proj = W4A16Linear(HIDDEN_FEATURES, IN_FEATURES, bias=False, device=device)

    # Initialize weights
    for layer in [gate_proj, up_proj, down_proj]:
        weight = torch.randn(layer.out_features, layer.in_features, dtype=torch.bfloat16)
        layer.pack_weights(weight)

    # Input
    x = torch.randn(BATCH_SIZE, IN_FEATURES, dtype=torch.float16, device=device)

    def mlp_forward(x):
        gate = gate_proj(x)
        up = up_proj(x)
        hidden = F.silu(gate) * up
        out = down_proj(hidden)
        return out

    # ==================== Without CUDA Graphs ====================
    print("\n--- Without CUDA Graphs ---")

    for _ in range(WARMUP_ITERS):
        _ = mlp_forward(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(BENCHMARK_ITERS):
        y = mlp_forward(x)
    torch.cuda.synchronize()

    no_graph_ms = (time.time() - start) / BENCHMARK_ITERS * 1000
    print(f"MLP latency: {no_graph_ms:.4f} ms")

    # ==================== With CUDA Graphs ====================
    print("\n--- With CUDA Graphs ---")

    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y = mlp_forward(x)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_graph = mlp_forward(x)

    # Benchmark
    for _ in range(WARMUP_ITERS):
        g.replay()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(BENCHMARK_ITERS):
        g.replay()
    torch.cuda.synchronize()

    with_graph_ms = (time.time() - start) / BENCHMARK_ITERS * 1000
    print(f"MLP latency: {with_graph_ms:.4f} ms")

    # ==================== Comparison ====================
    print("\n--- MLP Comparison ---")
    print(f"Without CUDA Graphs: {no_graph_ms:.4f} ms")
    print(f"With CUDA Graphs:    {with_graph_ms:.4f} ms")
    print(f"Speedup:             {no_graph_ms / with_graph_ms:.2f}x")

    # Estimate 18-layer overhead
    print(f"\n--- 18-Layer Projection ---")
    print(f"18x MLP (no graph):   {18 * no_graph_ms:.2f} ms")
    print(f"18x MLP (with graph): {18 * with_graph_ms:.2f} ms")

    return no_graph_ms, with_graph_ms


def test_torch_compile():
    """Test torch.compile compatibility."""
    print("\n" + "=" * 60)
    print("6. torch.compile COMPATIBILITY")
    print("=" * 60)

    device = torch.device("cuda")

    # Create layer
    layer = W4A16Linear(IN_FEATURES, HIDDEN_FEATURES, bias=True, device=device)
    weight = torch.randn(HIDDEN_FEATURES, IN_FEATURES, dtype=torch.bfloat16)
    layer.pack_weights(weight)

    x = torch.randn(BATCH_SIZE, IN_FEATURES, dtype=torch.float16, device=device)

    # Try to compile
    print("Attempting torch.compile...")

    try:
        compiled_layer = torch.compile(layer, mode="reduce-overhead")

        # Warmup
        for _ in range(10):
            _ = compiled_layer(x)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(100):
            _ = compiled_layer(x)
        torch.cuda.synchronize()

        compile_ms = (time.time() - start) / 100 * 1000
        print(f"torch.compile latency: {compile_ms:.4f} ms")
        print("torch.compile: COMPATIBLE")
        return True

    except Exception as e:
        print(f"torch.compile failed: {e}")
        print("torch.compile: NOT COMPATIBLE (expected for custom TVM op)")
        return False


def test_memory_comparison():
    """Compare memory usage between FP16 and W4A16."""
    print("\n" + "=" * 60)
    print("7. MEMORY COMPARISON")
    print("=" * 60)

    # FP16 weight size
    fp16_bytes = HIDDEN_FEATURES * IN_FEATURES * 2
    fp16_mb = fp16_bytes / 1e6

    # W4A16 weight size
    num_scale_blocks = IN_FEATURES // QUANT_BLOCK
    packed_bytes = num_scale_blocks * HIDDEN_FEATURES * 4 * 4  # int32
    scale_bytes = num_scale_blocks * HIDDEN_FEATURES * 2  # float16
    w4a16_bytes = packed_bytes + scale_bytes
    w4a16_mb = w4a16_bytes / 1e6

    # Theoretical INT4 (just weights)
    int4_bytes = HIDDEN_FEATURES * IN_FEATURES // 2
    int4_mb = int4_bytes / 1e6

    print(f"Single Layer ({HIDDEN_FEATURES}x{IN_FEATURES}):")
    print(f"  FP16/BF16:   {fp16_mb:.2f} MB")
    print(f"  W4A16:       {w4a16_mb:.2f} MB (includes scales)")
    print(f"  Pure INT4:   {int4_mb:.2f} MB (theoretical)")
    print(f"  Compression: {fp16_bytes / w4a16_bytes:.1f}x")

    # 18-layer MLP (3 matrices each)
    print(f"\n18-Layer MLP (gate+up+down per layer):")
    mlp_fp16_mb = 18 * 3 * fp16_mb
    mlp_w4a16_mb = 18 * 3 * w4a16_mb
    print(f"  FP16/BF16:   {mlp_fp16_mb:.0f} MB ({mlp_fp16_mb/1024:.2f} GB)")
    print(f"  W4A16:       {mlp_w4a16_mb:.0f} MB ({mlp_w4a16_mb/1024:.2f} GB)")
    print(f"  Savings:     {mlp_fp16_mb - mlp_w4a16_mb:.0f} MB")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("W4A16 COMPREHENSIVE VALIDATION")
    print("PyTorch Custom Op + CUDA Graphs Benchmark")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  IN_FEATURES:     {IN_FEATURES}")
    print(f"  HIDDEN_FEATURES: {HIDDEN_FEATURES}")
    print(f"  BATCH_SIZE:      {BATCH_SIZE}")
    print(f"  WARMUP_ITERS:    {WARMUP_ITERS}")
    print(f"  BENCHMARK_ITERS: {BENCHMARK_ITERS}")

    # Run all tests
    results = {}

    # 1. Correctness
    results['correctness'] = test_correctness()

    # 2. Raw kernel performance
    results['raw_kernel_ms'] = test_raw_kernel_performance()

    # 3. Module performance
    results['module_ms'] = test_module_performance()

    # 4. CUDA Graphs
    no_graph, with_graph = test_cuda_graphs()
    results['no_graph_ms'] = no_graph
    results['with_graph_ms'] = with_graph

    # 5. MLP simulation
    mlp_no_graph, mlp_with_graph = test_mlp_simulation()
    results['mlp_no_graph_ms'] = mlp_no_graph
    results['mlp_with_graph_ms'] = mlp_with_graph

    # 6. torch.compile
    results['torch_compile'] = test_torch_compile()

    # 7. Memory comparison
    test_memory_comparison()

    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nCorrectness: {'PASS' if results['correctness'] else 'FAIL'}")
    print(f"\nSingle Layer Latency (16384x2048):")
    print(f"  Raw TVM kernel:        {results['raw_kernel_ms']:.4f} ms")
    print(f"  W4A16Linear (no graph):{results['no_graph_ms']:.4f} ms")
    print(f"  W4A16Linear (graph):   {results['with_graph_ms']:.4f} ms")

    print(f"\nMLP Latency (gate+up+silu*mul+down):")
    print(f"  Without CUDA Graphs:   {results['mlp_no_graph_ms']:.4f} ms")
    print(f"  With CUDA Graphs:      {results['mlp_with_graph_ms']:.4f} ms")

    print(f"\n18-Layer Projection:")
    print(f"  Without CUDA Graphs:   {18 * results['mlp_no_graph_ms']:.2f} ms")
    print(f"  With CUDA Graphs:      {18 * results['mlp_with_graph_ms']:.2f} ms")

    # Target check
    print(f"\n" + "=" * 70)
    print("TARGET CHECK")
    print("=" * 70)

    target_met = results['with_graph_ms'] < 0.2
    print(f"Single layer target (< 0.2ms): {'ACHIEVED!' if target_met else 'NOT MET'}")

    mlp_target = 0.6  # 3 layers * 0.2ms
    mlp_target_met = results['mlp_with_graph_ms'] < mlp_target
    print(f"MLP target (< {mlp_target}ms):     {'ACHIEVED!' if mlp_target_met else 'NOT MET'}")


if __name__ == "__main__":
    main()
