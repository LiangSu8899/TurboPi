#!/usr/bin/env python3
"""
Manually Scheduled nvFP4 GEMM Kernels using TVM 0.24 TIR Schedule.

TVM 0.24 uses te.create_prim_func() to convert TE to TensorIR,
then uses tvm.tir.Schedule for scheduling.

Author: Claude Code
Date: 2026-02-10
"""

import sys
import os

# Add TVM to path
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import te, tir
from tvm.script import tir as T
import numpy as np
import time

print(f"TVM Version: {tvm.__version__}")
print(f"CUDA Available: {tvm.cuda().exist}")

# Constants
BLOCK_SIZE = 32
M = 1
N = 3072
K = 3072


def create_w4a16_te(M, N, K, block_size=32):
    """Create W4A16 GEMM using Tensor Expression."""
    num_blocks_k = (K + block_size - 1) // block_size

    A = te.placeholder((M, K), name="A", dtype="float32")
    W = te.placeholder((N, K), name="W", dtype="float32")
    scale_W = te.placeholder((N, num_blocks_k), name="scale_W", dtype="float32")

    k = te.reduce_axis((0, K), name="k")

    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k] * W[j, k] * scale_W[j, k // block_size],
            axis=k
        ),
        name="C"
    )

    return [A, W, scale_W, C]


def create_w4a4_te(M, N, K, block_size=32):
    """Create W4A4 GEMM using Tensor Expression."""
    num_blocks_k = (K + block_size - 1) // block_size

    A = te.placeholder((M, K), name="A", dtype="float32")
    W = te.placeholder((N, K), name="W", dtype="float32")
    scale_A = te.placeholder((M, num_blocks_k), name="scale_A", dtype="float32")
    scale_W = te.placeholder((N, num_blocks_k), name="scale_W", dtype="float32")

    k = te.reduce_axis((0, K), name="k")

    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k] * scale_A[i, k // block_size] * W[j, k] * scale_W[j, k // block_size],
            axis=k
        ),
        name="C"
    )

    return [A, W, scale_A, scale_W, C]


def schedule_basic(sch: tir.Schedule):
    """Basic GPU schedule: bind to blocks and threads."""
    block = sch.get_block("C")

    # Get loops
    loops = sch.get_loops(block)

    if len(loops) == 3:  # i, j, k
        i, j, k = loops
        # Fuse i, j
        ij = sch.fuse(i, j)
        # Split into blocks and threads
        bx, tx = sch.split(ij, factors=[None, 256])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
    elif len(loops) == 2:  # j, k (M=1 case, i already fused)
        j, k = loops
        bx, tx = sch.split(j, factors=[None, 256])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

    return sch


def schedule_unroll(sch: tir.Schedule, unroll_factor=8):
    """GPU schedule with K loop unrolling."""
    block = sch.get_block("C")

    # Get loops
    loops = sch.get_loops(block)

    if len(loops) == 3:  # i, j, k
        i, j, k = loops
        # Fuse i, j
        ij = sch.fuse(i, j)
        # Split into blocks and threads
        bx, tx = sch.split(ij, factors=[None, 256])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        # Unroll k
        ko, ki = sch.split(k, factors=[None, unroll_factor])
        sch.unroll(ki)
    elif len(loops) == 2:  # j, k
        j, k = loops
        bx, tx = sch.split(j, factors=[None, 256])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        ko, ki = sch.split(k, factors=[None, unroll_factor])
        sch.unroll(ki)

    return sch


def build_and_benchmark(name, te_args, schedule_func, warmup=50, runs=200):
    """Build and benchmark a scheduled kernel."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    num_blocks_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Convert TE to TensorIR
    func = te.create_prim_func(te_args)
    mod = tvm.IRModule({"main": func})

    # Create schedule
    sch = tir.Schedule(mod)

    # Apply schedule
    try:
        sch = schedule_func(sch)
    except Exception as e:
        print(f"  Schedule failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Build
    target = tvm.target.Target("cuda -arch=sm_110")
    print("  Building...")
    build_start = time.time()
    try:
        with tvm.transform.PassContext(opt_level=3):
            built_mod = tvm.build(sch.mod, target=target)
        build_time = (time.time() - build_start) * 1000
        print(f"  Build time: {build_time:.2f} ms")
    except Exception as e:
        print(f"  Build failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Prepare data
    device = tvm.runtime.cuda(0)

    # Determine number of inputs based on te_args length
    if len(te_args) == 4:  # W4A16: A, W, scale_W, C
        a_np = np.random.randn(M, K).astype("float32")
        w_np = np.random.randn(N, K).astype("float32")
        scale_w_np = (np.random.rand(N, num_blocks_k) * 0.1 + 0.01).astype("float32")

        A_tvm = tvm.runtime.empty((M, K), dtype="float32", device=device)
        A_tvm.copyfrom(a_np)

        W_tvm = tvm.runtime.empty((N, K), dtype="float32", device=device)
        W_tvm.copyfrom(w_np)

        scale_W_tvm = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
        scale_W_tvm.copyfrom(scale_w_np)

        C_tvm = tvm.runtime.empty((M, N), dtype="float32", device=device)

        inputs = [A_tvm, W_tvm, scale_W_tvm, C_tvm]
    else:  # W4A4: A, W, scale_A, scale_W, C
        a_np = np.random.randn(M, K).astype("float32")
        w_np = np.random.randn(N, K).astype("float32")
        scale_a_np = (np.random.rand(M, num_blocks_k) * 0.1 + 0.01).astype("float32")
        scale_w_np = (np.random.rand(N, num_blocks_k) * 0.1 + 0.01).astype("float32")

        A_tvm = tvm.runtime.empty((M, K), dtype="float32", device=device)
        A_tvm.copyfrom(a_np)

        W_tvm = tvm.runtime.empty((N, K), dtype="float32", device=device)
        W_tvm.copyfrom(w_np)

        scale_A_tvm = tvm.runtime.empty((M, num_blocks_k), dtype="float32", device=device)
        scale_A_tvm.copyfrom(scale_a_np)

        scale_W_tvm = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
        scale_W_tvm.copyfrom(scale_w_np)

        C_tvm = tvm.runtime.empty((M, N), dtype="float32", device=device)

        inputs = [A_tvm, W_tvm, scale_A_tvm, scale_W_tvm, C_tvm]

    # Warmup
    for _ in range(warmup):
        built_mod(*inputs)
    tvm.runtime.cuda(0).sync()

    # Benchmark
    start = time.time()
    for _ in range(runs):
        built_mod(*inputs)
    tvm.runtime.cuda(0).sync()

    elapsed = (time.time() - start) / runs * 1000
    print(f"  Avg time: {elapsed:.4f} ms")

    return elapsed


def main():
    print("="*70)
    print("Manually Scheduled nvFP4 GEMM Kernels (TVM 0.24)")
    print("="*70)
    print(f"\nMatrix: M={M}, N={N}, K={K}")
    print(f"Block size: {BLOCK_SIZE}")

    TRT_FP8_PER_GEMM = 47.4 / 90  # ~0.527 ms
    print(f"Target: Beat TRT FP8 = {TRT_FP8_PER_GEMM:.4f} ms")

    results = {}

    # W4A16 Basic
    te_args = create_w4a16_te(M, N, K)
    t = build_and_benchmark("W4A16 Basic", te_args, schedule_basic)
    if t:
        results["W4A16 Basic"] = t

    # W4A16 Unroll 8x
    te_args = create_w4a16_te(M, N, K)
    t = build_and_benchmark("W4A16 Unroll 8x", te_args,
                            lambda sch: schedule_unroll(sch, 8))
    if t:
        results["W4A16 Unroll 8x"] = t

    # W4A16 Unroll 16x
    te_args = create_w4a16_te(M, N, K)
    t = build_and_benchmark("W4A16 Unroll 16x", te_args,
                            lambda sch: schedule_unroll(sch, 16))
    if t:
        results["W4A16 Unroll 16x"] = t

    # W4A16 Unroll 32x
    te_args = create_w4a16_te(M, N, K)
    t = build_and_benchmark("W4A16 Unroll 32x", te_args,
                            lambda sch: schedule_unroll(sch, 32))
    if t:
        results["W4A16 Unroll 32x"] = t

    # W4A4 Basic
    te_args = create_w4a4_te(M, N, K)
    t = build_and_benchmark("W4A4 Basic", te_args, schedule_basic)
    if t:
        results["W4A4 Basic"] = t

    # W4A4 Unroll 8x
    te_args = create_w4a4_te(M, N, K)
    t = build_and_benchmark("W4A4 Unroll 8x", te_args,
                            lambda sch: schedule_unroll(sch, 8))
    if t:
        results["W4A4 Unroll 8x"] = t

    # W4A4 Unroll 16x
    te_args = create_w4a4_te(M, N, K)
    t = build_and_benchmark("W4A4 Unroll 16x", te_args,
                            lambda sch: schedule_unroll(sch, 16))
    if t:
        results["W4A4 Unroll 16x"] = t

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Kernel':<25} {'Time (ms)':<12} {'vs TRT FP8':<12} {'Status'}")
    print("-"*60)
    print(f"{'TRT FP8 Baseline':<25} {TRT_FP8_PER_GEMM:<12.4f} {'1.00x':<12} baseline")

    for name, time_ms in sorted(results.items(), key=lambda x: x[1]):
        ratio = TRT_FP8_PER_GEMM / time_ms
        status = "FASTER" if time_ms < TRT_FP8_PER_GEMM else "SLOWER"
        print(f"{name:<25} {time_ms:<12.4f} {ratio:.2f}x{' ':<8} {status}")

    print("="*70)

    # Best result
    if results:
        best = min(results.items(), key=lambda x: x[1])
        print(f"\nBest: {best[0]} @ {best[1]:.4f} ms")
        if best[1] < TRT_FP8_PER_GEMM:
            print("SUCCESS! Beat TRT FP8 baseline!")
        else:
            speedup_needed = best[1] / TRT_FP8_PER_GEMM
            print(f"Need {speedup_needed:.1f}x more speedup to match TRT FP8")


if __name__ == "__main__":
    main()
