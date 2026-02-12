#!/usr/bin/env python3
"""
TVM Meta Schedule (TVM 0.24) Auto-Tuning for nvFP4 GEMM.

Uses TVM's meta_schedule to find optimal kernel configurations.
This replaces the old auto_scheduler API.

Key optimizations enabled by Meta-Schedule:
1. Automatic tiling and loop ordering
2. Shared memory usage optimization
3. Vectorized memory access
4. Thread coarsening
5. Unroll hints

Run with TVM environment:
    source /home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/venv/bin/activate
    python nvfp4_meta_schedule.py --trials 500 --kernel all

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
from tvm import meta_schedule as ms
import numpy as np
import time
import tempfile

print(f"TVM Version: {tvm.__version__}")
print(f"CUDA Available: {tvm.cuda().exist}")

# Constants
BLOCK_SIZE = 32
M = 1
N = 3072
K = 3072


def create_w4a16_gemm_tir(M: int, N: int, K: int, block_size: int = 32):
    """
    Create TensorIR for W4A16 GEMM with GPU thread binding.
    """
    num_blocks_k = (K + block_size - 1) // block_size
    THREADS = 256
    total = M * N
    num_blocks = (total + THREADS - 1) // THREADS

    @T.prim_func
    def func(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N
                    C[i, j] = T.float32(0)
                    for k in T.serial(K):
                        block_idx = k // block_size
                        C[i, j] = C[i, j] + A[i, k] * W[j, k] * scale_W[j, block_idx]

    return func


def create_w4a4_gemm_tir(M: int, N: int, K: int, block_size: int = 32):
    """
    Create TensorIR for W4A4 GEMM with GPU thread binding.
    Both A and W have block scales.
    """
    num_blocks_k = (K + block_size - 1) // block_size
    THREADS = 256
    total = M * N
    num_blocks = (total + THREADS - 1) // THREADS

    @T.prim_func
    def func(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N
                    C[i, j] = T.float32(0)
                    for k in T.serial(K):
                        block_idx = k // block_size
                        a_val = A[i, k] * scale_A[i, block_idx]
                        w_val = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + a_val * w_val

    return func


def tune_with_meta_schedule(func, target, work_dir, max_trials=200):
    """Run meta_schedule tuning."""
    print(f"\n  Starting meta_schedule tuning (max {max_trials} trials)...")
    print(f"  Work directory: {work_dir}")

    # Create IRModule
    mod = tvm.IRModule({"main": func})

    # Create database
    database = ms.database.JSONDatabase(
        path_workload=os.path.join(work_dir, "workload.json"),
        path_tuning_record=os.path.join(work_dir, "tuning_records.json"),
    )

    # Tune
    start = time.time()
    with ms.Profiler() as profiler:
        sch = ms.tune_tir(
            mod=mod,
            target=target,
            work_dir=work_dir,
            max_trials_global=max_trials,
            num_trials_per_iter=64,
            database=database,
        )
    tune_time = time.time() - start
    print(f"  Tuning completed in {tune_time:.1f}s")

    if sch is None:
        print("  WARNING: Tuning failed, using default schedule")
        return None

    # Build optimized module
    with tvm.transform.PassContext(opt_level=3):
        built_mod = tvm.build(sch.mod, target=target)

    return built_mod


def benchmark_kernel(mod, name, num_inputs, warmup=50, runs=200):
    """Benchmark a kernel."""
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

    func = mod

    if num_inputs == 5:  # W4A4
        scale_A = tvm.runtime.empty((M, num_blocks_k), dtype="float32", device=device)
        scale_A.copyfrom((np.random.rand(M, num_blocks_k) * 0.1 + 0.01).astype("float32"))

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


def benchmark_naive():
    """Benchmark naive kernel without tuning for comparison."""
    print("\n" + "="*60)
    print("Naive Kernel (No Tuning)")
    print("="*60)

    target = tvm.target.Target("cuda -arch=sm_110")

    # W4A16
    func = create_w4a16_gemm_tir(M, N, K)
    mod = tvm.IRModule({"main": func})
    with tvm.transform.PassContext(opt_level=3):
        built_mod = tvm.build(mod, target=target)

    naive_time = benchmark_kernel(built_mod, "W4A16 Naive", 4)
    return naive_time


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TVM Meta Schedule for nvFP4 GEMM")
    parser.add_argument("--trials", type=int, default=100, help="Number of tuning trials")
    parser.add_argument("--kernel", type=str, default="w4a16",
                        choices=["w4a4", "w4a16", "all", "naive"],
                        help="Which kernel to tune")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Work directory for tuning logs")
    args = parser.parse_args()

    print("="*70)
    print("TVM Meta Schedule for nvFP4 GEMM")
    print("="*70)
    print(f"\nMatrix: M={M}, N={N}, K={K}")
    print(f"Block size: {BLOCK_SIZE}")

    TRT_FP8_PER_GEMM = 47.4 / 90  # ~0.527 ms
    print(f"Target: Beat TRT FP8 = {TRT_FP8_PER_GEMM:.4f} ms")

    # Thor SM110 has 1024 max threads per block
    target = tvm.target.Target(
        "cuda -arch=sm_110 -max_threads_per_block=1024 -max_shared_memory_per_block=49152"
    )
    results = {}

    if args.kernel == "naive":
        results["W4A16 Naive"] = benchmark_naive()
    else:
        # Set up work directory
        if args.work_dir:
            work_dir = args.work_dir
        else:
            work_dir = tempfile.mkdtemp(prefix="tvm_tune_")

        if args.kernel in ["w4a16", "all"]:
            print(f"\n{'='*60}")
            print("Tuning W4A16 GEMM")
            print("="*60)

            func = create_w4a16_gemm_tir(M, N, K)
            w4a16_dir = os.path.join(work_dir, "w4a16")
            os.makedirs(w4a16_dir, exist_ok=True)

            try:
                mod = tune_with_meta_schedule(func, target, w4a16_dir, args.trials)
                if mod:
                    results["W4A16 Tuned"] = benchmark_kernel(mod, "W4A16 Tuned", 4)
            except Exception as e:
                print(f"  W4A16 tuning failed: {e}")
                import traceback
                traceback.print_exc()

        if args.kernel in ["w4a4", "all"]:
            print(f"\n{'='*60}")
            print("Tuning W4A4 GEMM")
            print("="*60)

            func = create_w4a4_gemm_tir(M, N, K)
            w4a4_dir = os.path.join(work_dir, "w4a4")
            os.makedirs(w4a4_dir, exist_ok=True)

            try:
                mod = tune_with_meta_schedule(func, target, w4a4_dir, args.trials)
                if mod:
                    results["W4A4 Tuned"] = benchmark_kernel(mod, "W4A4 Tuned", 5)
            except Exception as e:
                print(f"  W4A4 tuning failed: {e}")
                import traceback
                traceback.print_exc()

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
