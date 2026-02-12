#!/usr/bin/env python3
"""
Optimized nvFP4 GEMM kernel with TVM TensorIR.

Optimizations:
1. Shared memory tiling
2. Vectorized memory access
3. Register blocking
4. Auto-scheduler compatible

Target: SM110 (Jetson Thor)
"""

import os
import sys
from typing import Tuple, Optional


def check_tvm():
    """Check TVM environment."""
    try:
        import tvm
        print(f"TVM version: {tvm.__version__}")
        return True
    except ImportError:
        print("TVM not found. Set TVM_HOME and PYTHONPATH.")
        return False


def create_nvfp4_gemm_tiled(
    M: int,
    N: int,
    K: int,
    block_size: int = 32,
    tile_m: int = 8,
    tile_n: int = 64,
    tile_k: int = 32,
):
    """
    Create tiled nvFP4 GEMM kernel with shared memory.

    Args:
        M, N, K: Matrix dimensions
        block_size: Block size for scaling (32)
        tile_m, tile_n, tile_k: Tile sizes for shared memory

    Returns:
        TVM TensorIR function
    """
    import tvm
    from tvm.script import tir as T

    num_blocks_k = (K + block_size - 1) // block_size

    # Thread block configuration
    # Each block computes a tile_m x tile_n output tile
    THREADS_X = tile_n  # One thread per column in output tile
    THREADS_Y = tile_m  # One thread per row in output tile
    THREADS_PER_BLOCK = THREADS_X * THREADS_Y

    # Grid dimensions
    grid_m = (M + tile_m - 1) // tile_m
    grid_n = (N + tile_n - 1) // tile_n

    @T.prim_func
    def nvfp4_gemm_tiled_func(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({
            "global_symbol": "nvfp4_gemm_tiled",
            "tir.noalias": True,
        })

        # Shared memory for tiles
        A_shared = T.alloc_buffer((tile_m, tile_k), "float32", scope="shared")
        W_shared = T.alloc_buffer((tile_n, tile_k), "float32", scope="shared")
        scale_A_shared = T.alloc_buffer((tile_m,), "float32", scope="shared")
        scale_W_shared = T.alloc_buffer((tile_n,), "float32", scope="shared")

        for bx in T.thread_binding(grid_n, thread="blockIdx.x"):
            for by in T.thread_binding(grid_m, thread="blockIdx.y"):
                # Local accumulator (in registers)
                C_local = T.alloc_buffer((tile_m, tile_n), "float32", scope="local")

                # Initialize accumulator
                for i, j in T.grid(tile_m, tile_n):
                    C_local[i, j] = T.float32(0)

                # Iterate over K dimension in tiles
                for k_tile in T.serial((K + tile_k - 1) // tile_k):
                    k_base = k_tile * tile_k

                    # Cooperative loading of A tile to shared memory
                    for ty in T.thread_binding(tile_m, thread="threadIdx.y"):
                        for tx in T.thread_binding(tile_n, thread="threadIdx.x"):
                            # Each thread loads multiple elements
                            if tx < tile_k:
                                row = by * tile_m + ty
                                col = k_base + tx
                                if row < M and col < K:
                                    A_shared[ty, tx] = A[row, col]
                                else:
                                    A_shared[ty, tx] = T.float32(0)

                    # Cooperative loading of W tile to shared memory
                    for ty in T.thread_binding(tile_m, thread="threadIdx.y"):
                        for tx in T.thread_binding(tile_n, thread="threadIdx.x"):
                            if ty == 0:  # First row of threads loads W
                                row = bx * tile_n + tx
                                for kk in T.serial(tile_k):
                                    col = k_base + kk
                                    if row < N and col < K:
                                        W_shared[tx, kk] = W[row, col]
                                    else:
                                        W_shared[tx, kk] = T.float32(0)

                    # Load scales for this tile
                    for ty in T.thread_binding(tile_m, thread="threadIdx.y"):
                        for tx in T.thread_binding(tile_n, thread="threadIdx.x"):
                            if tx == 0 and ty < tile_m:
                                row = by * tile_m + ty
                                block_idx = k_base // block_size
                                if row < M and block_idx < num_blocks_k:
                                    scale_A_shared[ty] = scale_A[row, block_idx]
                                else:
                                    scale_A_shared[ty] = T.float32(1)

                            if ty == 0 and tx < tile_n:
                                row = bx * tile_n + tx
                                block_idx = k_base // block_size
                                if row < N and block_idx < num_blocks_k:
                                    scale_W_shared[tx] = scale_W[row, block_idx]
                                else:
                                    scale_W_shared[tx] = T.float32(1)

                    # Synchronize after loading
                    T.tvm_storage_sync("shared")

                    # Compute partial products
                    for ty in T.thread_binding(tile_m, thread="threadIdx.y"):
                        for tx in T.thread_binding(tile_n, thread="threadIdx.x"):
                            for k in T.serial(tile_k):
                                if k_base + k < K:
                                    a_val = A_shared[ty, k] * scale_A_shared[ty]
                                    w_val = W_shared[tx, k] * scale_W_shared[tx]
                                    C_local[ty, tx] = C_local[ty, tx] + a_val * w_val

                    # Synchronize before next tile
                    T.tvm_storage_sync("shared")

                # Write back results
                for ty in T.thread_binding(tile_m, thread="threadIdx.y"):
                    for tx in T.thread_binding(tile_n, thread="threadIdx.x"):
                        row = by * tile_m + ty
                        col = bx * tile_n + tx
                        if row < M and col < N:
                            C[row, col] = C_local[ty, tx]

    return nvfp4_gemm_tiled_func


def create_nvfp4_gemm_simple_optimized(
    M: int,
    N: int,
    K: int,
    block_size: int = 32,
):
    """
    Simpler optimized kernel with loop unrolling.
    Uses in-place accumulation (TVM 0.24 compatible).
    """
    import tvm
    from tvm.script import tir as T

    num_blocks_k = (K + block_size - 1) // block_size
    THREADS_PER_BLOCK = 256
    total_elements = M * N
    num_blocks_x = (total_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    # Unroll factor for K loop (must divide block_size for correct scaling)
    UNROLL_K = 8  # Process 8 elements per iteration

    @T.prim_func
    def nvfp4_gemm_optimized_func(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({
            "global_symbol": "nvfp4_gemm_optimized",
            "tir.noalias": True,
        })

        for bx in T.thread_binding(num_blocks_x, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):
                idx = bx * THREADS_PER_BLOCK + tx

                if idx < total_elements:
                    i = idx // N
                    j = idx % N

                    # Initialize output
                    C[i, j] = T.float32(0)

                    # Main loop - process 8 elements at a time
                    # K=3072, UNROLL_K=8 -> 384 iterations
                    for k8 in T.serial(K // UNROLL_K):
                        k_base = k8 * UNROLL_K
                        block_idx = k_base // block_size

                        # Load scales once per 8 elements (may cross block boundary at 32)
                        a_scale = scale_A[i, block_idx]
                        w_scale = scale_W[j, block_idx]

                        # Unrolled accumulation (8 elements)
                        C[i, j] = C[i, j] + A[i, k_base + 0] * a_scale * W[j, k_base + 0] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 1] * a_scale * W[j, k_base + 1] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 2] * a_scale * W[j, k_base + 2] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 3] * a_scale * W[j, k_base + 3] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 4] * a_scale * W[j, k_base + 4] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 5] * a_scale * W[j, k_base + 5] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 6] * a_scale * W[j, k_base + 6] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 7] * a_scale * W[j, k_base + 7] * w_scale

    return nvfp4_gemm_optimized_func


def create_nvfp4_gemm_vectorized(
    M: int,
    N: int,
    K: int,
    block_size: int = 32,
):
    """
    Vectorized kernel - processes 4 elements per loop iteration.
    Requires K to be divisible by 4.
    Uses in-place accumulation (TVM 0.24 compatible).
    """
    import tvm
    from tvm.script import tir as T

    assert K % 4 == 0, "K must be divisible by 4 for vectorized access"

    num_blocks_k = (K + block_size - 1) // block_size
    THREADS_PER_BLOCK = 256
    total_elements = M * N
    num_blocks_x = (total_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    @T.prim_func
    def nvfp4_gemm_vectorized_func(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({
            "global_symbol": "nvfp4_gemm_vectorized",
            "tir.noalias": True,
        })

        for bx in T.thread_binding(num_blocks_x, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):
                idx = bx * THREADS_PER_BLOCK + tx

                if idx < total_elements:
                    i = idx // N
                    j = idx % N

                    # Initialize output
                    C[i, j] = T.float32(0)

                    # Process 4 elements at a time (vectorized-style)
                    for k4 in T.serial(K // 4):
                        k_base = k4 * 4
                        block_idx = k_base // block_size

                        # Get scales
                        a_scale = scale_A[i, block_idx]
                        w_scale = scale_W[j, block_idx]

                        # Accumulate 4 products (will be vectorized by TVM)
                        C[i, j] = C[i, j] + A[i, k_base + 0] * a_scale * W[j, k_base + 0] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 1] * a_scale * W[j, k_base + 1] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 2] * a_scale * W[j, k_base + 2] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 3] * a_scale * W[j, k_base + 3] * w_scale

    return nvfp4_gemm_vectorized_func


def export_optimized_kernel(
    kernel_type: str,
    M: int, N: int, K: int,
    output_dir: str,
    target: str = "cuda -arch=sm_110",
):
    """
    Export optimized kernel to CUDA source.

    Args:
        kernel_type: "tiled", "simple_optimized", or "vectorized"
        M, N, K: Matrix dimensions
        output_dir: Output directory
        target: TVM target
    """
    import tvm
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Create kernel based on type
    if kernel_type == "tiled":
        tir_func = create_nvfp4_gemm_tiled(M, N, K)
        name = "nvfp4_gemm_tiled"
    elif kernel_type == "simple_optimized":
        tir_func = create_nvfp4_gemm_simple_optimized(M, N, K)
        name = "nvfp4_gemm_optimized"
    elif kernel_type == "vectorized":
        tir_func = create_nvfp4_gemm_vectorized(M, N, K)
        name = "nvfp4_gemm_vectorized"
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    print(f"Building {name} kernel for M={M}, N={N}, K={K}...")

    target_obj = tvm.target.Target(target)

    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(tir_func, target=target_obj)

    # Export CUDA source
    if hasattr(mod, 'imports_') and len(mod.imports_) > 0:
        cuda_source = mod.imports_[0].inspect_source()

        cuda_file = os.path.join(output_dir, f"{name}.cu")
        with open(cuda_file, "w") as f:
            f.write(cuda_source)
        print(f"[Saved] {cuda_file} ({len(cuda_source)} bytes)")

        return cuda_source
    else:
        print("ERROR: Could not extract CUDA source")
        return None


def run_auto_scheduler(
    M: int, N: int, K: int,
    num_trials: int = 1000,
    output_file: str = "nvfp4_gemm_tuned.json",
):
    """
    Use TVM auto-scheduler to find optimal schedule.

    Args:
        M, N, K: Matrix dimensions
        num_trials: Number of tuning trials
        output_file: File to save tuning results
    """
    import tvm
    from tvm import auto_scheduler

    print(f"Starting auto-scheduler for M={M}, N={N}, K={K}")
    print(f"Trials: {num_trials}")

    # Create the compute definition for auto-scheduler
    # Note: auto-scheduler works with TE (Tensor Expression), not TensorIR directly
    # We need to create a TE version

    from tvm import te

    block_size = 32
    num_blocks_k = (K + block_size - 1) // block_size

    @auto_scheduler.register_workload
    def nvfp4_gemm_te(M, N, K, block_size):
        A = te.placeholder((M, K), name="A", dtype="float32")
        W = te.placeholder((N, K), name="W", dtype="float32")
        scale_A = te.placeholder((M, num_blocks_k), name="scale_A", dtype="float32")
        scale_W = te.placeholder((N, num_blocks_k), name="scale_W", dtype="float32")

        k = te.reduce_axis((0, K), name="k")

        C = te.compute(
            (M, N),
            lambda i, j: te.sum(
                A[i, k] * scale_A[i, k // block_size] *
                W[j, k] * scale_W[j, k // block_size],
                axis=k
            ),
            name="C"
        )

        return [A, W, scale_A, scale_W, C]

    target = tvm.target.Target("cuda -arch=sm_110")

    # Create search task
    task = auto_scheduler.SearchTask(
        func=nvfp4_gemm_te,
        args=(M, N, K, block_size),
        target=target,
    )

    print(f"Task: {task}")

    # Set up tuning options
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_trials,
        runner=auto_scheduler.LocalRunner(timeout=10),
        measure_callbacks=[auto_scheduler.RecordToFile(output_file)],
        verbose=2,
    )

    # Run auto-scheduler
    print("Running auto-scheduler...")
    task.tune(tune_option)

    # Apply best schedule
    print("Applying best schedule...")
    sch, args = task.apply_best(output_file)

    # Build optimized module
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.build(sch, args, target)

    print(f"Auto-scheduler complete. Results saved to {output_file}")

    return lib, sch


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optimized nvFP4 GEMM Kernel Generator")
    parser.add_argument("--type", type=str, default="simple_optimized",
                        choices=["tiled", "simple_optimized", "vectorized", "auto"],
                        help="Kernel type")
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--N", type=int, default=3072)
    parser.add_argument("--K", type=int, default=3072)
    parser.add_argument("--output", type=str, default="/tmp/tvm_optimized")
    parser.add_argument("--trials", type=int, default=1000,
                        help="Auto-scheduler trials (for --type=auto)")

    args = parser.parse_args()

    if not check_tvm():
        sys.exit(1)

    if args.type == "auto":
        run_auto_scheduler(args.M, args.N, args.K, args.trials,
                          os.path.join(args.output, "tuning_records.json"))
    else:
        export_optimized_kernel(args.type, args.M, args.N, args.K, args.output)


if __name__ == "__main__":
    main()
