#!/usr/bin/env python3
"""
TVM Meta Schedule (Ansor) Auto-Tuning for nvFP4 GEMM.

Uses TVM's auto-scheduler to find optimal kernel configurations.
This can achieve significant speedups over naive TensorIR implementations.

Run with TVM environment:
    source /home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/venv/bin/activate
    python nvfp4_auto_schedule.py

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
from tvm import te, auto_scheduler
import numpy as np
import time

print(f"TVM Version: {tvm.__version__}")
print(f"CUDA Available: {tvm.cuda().exist}")

# Constants
BLOCK_SIZE = 32
M = 1
N = 3072
K = 3072


def create_w4a4_gemm_te(M, N, K, block_size=32):
    """
    Create Tensor Expression for W4A4 GEMM.
    Using TE (not TensorIR) for auto_scheduler compatibility.
    """
    num_blocks_k = (K + block_size - 1) // block_size

    # Placeholders
    A = te.placeholder((M, K), name="A", dtype="float32")
    W = te.placeholder((N, K), name="W", dtype="float32")
    scale_A = te.placeholder((M, num_blocks_k), name="scale_A", dtype="float32")
    scale_W = te.placeholder((N, num_blocks_k), name="scale_W", dtype="float32")

    # Reduce axis
    k = te.reduce_axis((0, K), name="k")

    # GEMM with fused dequantization
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k] * scale_A[i, k // block_size] *
            W[j, k] * scale_W[j, k // block_size],
            axis=k
        ),
        name="C"
    )

    return A, W, scale_A, scale_W, C


def create_w4a16_gemm_te(M, N, K, block_size=32):
    """
    Create Tensor Expression for W4A16 GEMM.
    Only weight has scale.
    """
    num_blocks_k = (K + block_size - 1) // block_size

    # Placeholders
    A = te.placeholder((M, K), name="A", dtype="float32")
    W = te.placeholder((N, K), name="W", dtype="float32")
    scale_W = te.placeholder((N, num_blocks_k), name="scale_W", dtype="float32")

    # Reduce axis
    k = te.reduce_axis((0, K), name="k")

    # GEMM with weight dequantization only
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k] * W[j, k] * scale_W[j, k // block_size],
            axis=k
        ),
        name="C"
    )

    return A, W, scale_W, C


@auto_scheduler.register_workload
def w4a4_gemm_auto():
    """Auto-scheduler workload for W4A4 GEMM."""
    A, W, scale_A, scale_W, C = create_w4a4_gemm_te(M, N, K)
    return [A, W, scale_A, scale_W, C]


@auto_scheduler.register_workload
def w4a16_gemm_auto():
    """Auto-scheduler workload for W4A16 GEMM."""
    A, W, scale_W, C = create_w4a16_gemm_te(M, N, K)
    return [A, W, scale_W, C]


def auto_schedule_w4a4(num_trials=200, log_file="w4a4_gemm_auto.json"):
    """Run auto-scheduler for W4A4 GEMM."""
    print(f"\n{'='*60}")
    print(f"Auto-Scheduling W4A4 GEMM ({num_trials} trials)")
    print(f"{'='*60}")

    target = tvm.target.Target("cuda -arch=sm_110")

    # Create task
    task = auto_scheduler.SearchTask(
        func=w4a4_gemm_auto,
        args=[],
        target=target,
    )

    print(f"  Compute DAG:\n{task.compute_dag}")

    # Tuning options
    tune_options = auto_scheduler.TuningOptions(
        num_measure_trials=num_trials,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
    )

    # Run tuning
    print("\n  Starting auto-scheduler tuning...")
    start = time.time()
    task.tune(tune_options)
    tune_time = time.time() - start
    print(f"  Tuning completed in {tune_time:.1f}s")

    # Apply best schedule
    sch, args = task.apply_best(log_file)

    # Build
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(sch, args, target)

    return mod


def auto_schedule_w4a16(num_trials=200, log_file="w4a16_gemm_auto.json"):
    """Run auto-scheduler for W4A16 GEMM."""
    print(f"\n{'='*60}")
    print(f"Auto-Scheduling W4A16 GEMM ({num_trials} trials)")
    print(f"{'='*60}")

    target = tvm.target.Target("cuda -arch=sm_110")

    # Create task
    task = auto_scheduler.SearchTask(
        func=w4a16_gemm_auto,
        args=[],
        target=target,
    )

    print(f"  Compute DAG:\n{task.compute_dag}")

    # Tuning options
    tune_options = auto_scheduler.TuningOptions(
        num_measure_trials=num_trials,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
    )

    # Run tuning
    print("\n  Starting auto-scheduler tuning...")
    start = time.time()
    task.tune(tune_options)
    tune_time = time.time() - start
    print(f"  Tuning completed in {tune_time:.1f}s")

    # Apply best schedule
    sch, args = task.apply_best(log_file)

    # Build
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(sch, args, target)

    return mod


def benchmark_auto_scheduled_kernel(mod, name, num_inputs, warmup=50, runs=200):
    """Benchmark an auto-scheduled kernel."""
    print(f"\n  Benchmarking {name}...")

    num_blocks_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    device = tvm.runtime.cuda(0)

    # Prepare inputs
    A = tvm.runtime.empty((M, K), dtype="float32", device=device)
    A.copyfrom(np.random.randn(M, K).astype("float32"))

    W = tvm.runtime.empty((N, K), dtype="float32", device=device)
    W.copyfrom(np.random.randn(N, K).astype("float32"))

    scale_W = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
    scale_W.copyfrom((np.random.rand(N, num_blocks_k) * 0.1 + 0.01).astype("float32"))

    C = tvm.runtime.empty((M, N), dtype="float32", device=device)

    if num_inputs == 5:  # W4A4
        scale_A = tvm.runtime.empty((M, num_blocks_k), dtype="float32", device=device)
        scale_A.copyfrom((np.random.rand(M, num_blocks_k) * 0.1 + 0.01).astype("float32"))

        # Get default function
        func = mod

        # Warmup
        for _ in range(warmup):
            func(A, W, scale_A, scale_W, C)
        tvm.runtime.cuda(0).sync()

        # Benchmark
        start = time.time()
        for _ in range(runs):
            func(A, W, scale_A, scale_W, C)
        tvm.runtime.cuda(0).sync()
    else:  # W4A16 (4 inputs)
        func = mod

        # Warmup
        for _ in range(warmup):
            func(A, W, scale_W, C)
        tvm.runtime.cuda(0).sync()

        # Benchmark
        start = time.time()
        for _ in range(runs):
            func(A, W, scale_W, C)
        tvm.runtime.cuda(0).sync()

    elapsed = (time.time() - start) / runs * 1000
    print(f"  {name}: {elapsed:.4f} ms")
    return elapsed


def quick_benchmark_without_tuning():
    """Quick benchmark using existing logs if available."""
    print("="*70)
    print("Quick Benchmark (using existing logs if available)")
    print("="*70)

    target = tvm.target.Target("cuda -arch=sm_110")
    results = {}

    # Try W4A4
    log_file = "w4a4_gemm_auto.json"
    if os.path.exists(log_file):
        print(f"\n  Loading W4A4 from {log_file}...")
        task = auto_scheduler.SearchTask(func=w4a4_gemm_auto, args=[], target=target)
        sch, args = task.apply_best(log_file)
        with tvm.transform.PassContext(opt_level=3):
            mod = tvm.build(sch, args, target)
        results["W4A4 Auto"] = benchmark_auto_scheduled_kernel(mod, "W4A4 Auto", 5)
    else:
        print(f"\n  W4A4 log not found. Run tuning first.")

    # Try W4A16
    log_file = "w4a16_gemm_auto.json"
    if os.path.exists(log_file):
        print(f"\n  Loading W4A16 from {log_file}...")
        task = auto_scheduler.SearchTask(func=w4a16_gemm_auto, args=[], target=target)
        sch, args = task.apply_best(log_file)
        with tvm.transform.PassContext(opt_level=3):
            mod = tvm.build(sch, args, target)
        results["W4A16 Auto"] = benchmark_auto_scheduled_kernel(mod, "W4A16 Auto", 4)
    else:
        print(f"\n  W4A16 log not found. Run tuning first.")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TVM Auto-Scheduler for nvFP4 GEMM")
    parser.add_argument("--trials", type=int, default=200, help="Number of tuning trials")
    parser.add_argument("--kernel", type=str, default="all",
                        choices=["w4a4", "w4a16", "all", "benchmark"],
                        help="Which kernel to tune")
    args = parser.parse_args()

    print("="*70)
    print("TVM Auto-Scheduler for nvFP4 GEMM")
    print("="*70)
    print(f"\nMatrix: M={M}, N={N}, K={K}")
    print(f"Block size: {BLOCK_SIZE}")

    TRT_FP8_PER_GEMM = 47.4 / 90  # ~0.527 ms
    print(f"Target: Beat TRT FP8 = {TRT_FP8_PER_GEMM:.4f} ms")

    results = {}

    if args.kernel == "benchmark":
        results = quick_benchmark_without_tuning()
    else:
        if args.kernel in ["w4a4", "all"]:
            try:
                mod = auto_schedule_w4a4(args.trials, "w4a4_gemm_auto.json")
                results["W4A4 Auto"] = benchmark_auto_scheduled_kernel(mod, "W4A4 Auto", 5)
            except Exception as e:
                print(f"  W4A4 auto-schedule failed: {e}")

        if args.kernel in ["w4a16", "all"]:
            try:
                mod = auto_schedule_w4a16(args.trials, "w4a16_gemm_auto.json")
                results["W4A16 Auto"] = benchmark_auto_scheduled_kernel(mod, "W4A16 Auto", 4)
            except Exception as e:
                print(f"  W4A16 auto-schedule failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Kernel':<25} {'Time (ms)':<12} {'vs TRT FP8':<12} {'Status'}")
    print("-"*60)
    print(f"{'TRT FP8 Baseline':<25} {TRT_FP8_PER_GEMM:<12.4f} {'1.00x':<12} baseline")

    for name, time_ms in results.items():
        ratio = TRT_FP8_PER_GEMM / time_ms
        status = "FASTER" if time_ms < TRT_FP8_PER_GEMM else "SLOWER"
        print(f"{name:<25} {time_ms:<12.4f} {ratio:.2f}x{' ':<8} {status}")

    print("="*70)


if __name__ == "__main__":
    main()
