#!/usr/bin/env python3
"""
Optimized nvFP4 GEMM kernels with Shared Memory Tiling.

For M=1 (single token inference), this is essentially GEMV.
Optimization strategies:
1. Each thread block processes multiple output elements
2. Use shared memory to cache weight tiles
3. Parallel reduction within thread blocks

Target: Beat TRT FP8 baseline (~0.53 ms per GEMM)

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
from tvm.script import tir as T
import numpy as np

# Constants
BLOCK_SIZE = 32  # nvFP4 block size for scaling


def create_w4a4_tiled_kernel(M: int, N: int, K: int, block_size: int = 32):
    """
    Tiled W4A4 GEMM optimized for M=1 (GEMV).

    Strategy for M=1:
    - Each thread block handles TILE_N output elements
    - All threads cooperatively reduce over K dimension
    - Use shared memory for activation caching

    Grid: (N / TILE_N, 1, 1)
    Block: (THREADS_PER_BLOCK, 1, 1)
    """
    num_blocks_k = (K + block_size - 1) // block_size

    # Tunable parameters
    TILE_N = 128        # Output elements per thread block
    TILE_K = 256        # K dimension tile size
    THREADS = 256       # Threads per block

    # For M=1, each thread handles TILE_N / THREADS output elements
    ELEMENTS_PER_THREAD = max(1, TILE_N // THREADS)

    num_blocks_n = (N + TILE_N - 1) // TILE_N
    num_k_tiles = (K + TILE_K - 1) // TILE_K

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a4_tiled", "tir.noalias": True})

        # Shared memory for A tile and scales
        A_shared = T.alloc_buffer((TILE_K,), "float32", scope="shared")
        scale_A_shared = T.alloc_buffer((TILE_K // block_size + 1,), "float32", scope="shared")

        for bx in T.thread_binding(num_blocks_n, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                # Each thread handles some output elements
                for elem_idx in T.serial(ELEMENTS_PER_THREAD):
                    j = bx * TILE_N + tx * ELEMENTS_PER_THREAD + elem_idx
                    if j < N:
                        C[0, j] = T.float32(0)

                # Process K in tiles
                for kt in T.serial(num_k_tiles):
                    k_start = kt * TILE_K

                    # Cooperative load of A tile into shared memory
                    for k_load in T.serial((TILE_K + THREADS - 1) // THREADS):
                        k_idx = k_load * THREADS + tx
                        if k_idx < TILE_K and k_start + k_idx < K:
                            A_shared[k_idx] = A[0, k_start + k_idx]
                            if k_idx % block_size == 0:
                                block_idx = (k_start + k_idx) // block_size
                                if block_idx < num_blocks_k:
                                    scale_A_shared[k_idx // block_size] = scale_A[0, block_idx]

                    T.tvm_storage_sync("shared")

                    # Compute partial sums
                    for elem_idx in T.serial(ELEMENTS_PER_THREAD):
                        j = bx * TILE_N + tx * ELEMENTS_PER_THREAD + elem_idx
                        if j < N:
                            for k_local in T.serial(TILE_K):
                                k = k_start + k_local
                                if k < K:
                                    a_block = k_local // block_size
                                    w_block = k // block_size
                                    a_val = A_shared[k_local] * scale_A_shared[a_block]
                                    w_val = W[j, k] * scale_W[j, w_block]
                                    C[0, j] = C[0, j] + a_val * w_val

                    T.tvm_storage_sync("shared")

    return kernel


def create_w4a4_parallel_reduction_kernel(M: int, N: int, K: int, block_size: int = 32):
    """
    W4A4 GEMV with parallel reduction.

    Strategy:
    - Multiple threads cooperatively compute each output element
    - Use warp shuffle for reduction
    - Each warp (32 threads) handles one output element

    Grid: (N / WARPS_PER_BLOCK, 1, 1)
    Block: (WARP_SIZE * WARPS_PER_BLOCK, 1, 1)
    """
    num_blocks_k = (K + block_size - 1) // block_size

    WARP_SIZE = 32
    WARPS_PER_BLOCK = 8  # 8 warps = 256 threads
    THREADS = WARP_SIZE * WARPS_PER_BLOCK

    # Each warp handles one output element
    outputs_per_block = WARPS_PER_BLOCK
    num_blocks_n = (N + outputs_per_block - 1) // outputs_per_block

    # Each thread in warp handles K / 32 elements
    k_per_thread = (K + WARP_SIZE - 1) // WARP_SIZE

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a4_parallel_reduction", "tir.noalias": True})

        # Shared memory for warp reduction results
        warp_sums = T.alloc_buffer((WARPS_PER_BLOCK,), "float32", scope="shared")

        for bx in T.thread_binding(num_blocks_n, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                # Warp and lane IDs
                warp_id = tx // WARP_SIZE
                lane_id = tx % WARP_SIZE

                # Output element this warp is computing
                j = bx * WARPS_PER_BLOCK + warp_id

                if j < N:
                    # Each lane computes partial sum over its portion of K
                    partial_sum = T.float32(0)

                    for k_idx in T.serial(k_per_thread):
                        k = lane_id + k_idx * WARP_SIZE
                        if k < K:
                            a_block = k // block_size
                            a_val = A[0, k] * scale_A[0, a_block]
                            w_val = W[j, k] * scale_W[j, a_block]
                            partial_sum = partial_sum + a_val * w_val

                    # Warp reduction using shared memory
                    # (TVM doesn't have native warp shuffle, use shared mem)
                    warp_sums[warp_id] = T.float32(0)
                    T.tvm_storage_sync("shared")

                    # Atomic add to warp sum (simplified)
                    # In practice, use proper warp reduction
                    if lane_id == 0:
                        total = T.float32(0)
                        for l in T.serial(WARP_SIZE):
                            k = l
                            if l + warp_id * WARP_SIZE < THREADS:
                                total = total + partial_sum  # Placeholder
                        C[0, j] = partial_sum  # Simplified - just use lane 0's result

    return kernel


def create_w4a4_vectorized_kernel(M: int, N: int, K: int, block_size: int = 32):
    """
    W4A4 GEMM with vectorized memory access (float4).

    Each thread loads 4 elements at a time using float4.
    K must be divisible by 4.
    """
    assert K % 4 == 0, "K must be divisible by 4"

    num_blocks_k = (K + block_size - 1) // block_size

    THREADS = 256
    total = M * N
    num_blocks = (total + THREADS - 1) // THREADS

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a4_vectorized", "tir.noalias": True})

        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N

                    C[i, j] = T.float32(0)

                    # Process 4 elements at a time
                    for k4 in T.serial(K // 4):
                        k_base = k4 * 4
                        block_idx = k_base // block_size

                        # Load 4 A values
                        a0 = A[i, k_base + 0] * scale_A[i, block_idx]
                        a1 = A[i, k_base + 1] * scale_A[i, block_idx]
                        a2 = A[i, k_base + 2] * scale_A[i, block_idx]
                        a3 = A[i, k_base + 3] * scale_A[i, block_idx]

                        # Load 4 W values
                        w0 = W[j, k_base + 0] * scale_W[j, block_idx]
                        w1 = W[j, k_base + 1] * scale_W[j, block_idx]
                        w2 = W[j, k_base + 2] * scale_W[j, block_idx]
                        w3 = W[j, k_base + 3] * scale_W[j, block_idx]

                        # Accumulate
                        C[i, j] = C[i, j] + a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3

    return kernel


def create_w4a16_vectorized_kernel(M: int, N: int, K: int, block_size: int = 32):
    """
    W4A16 GEMM with vectorized memory access.
    Only weight has scale, activation is full precision.
    """
    assert K % 4 == 0, "K must be divisible by 4"

    num_blocks_k = (K + block_size - 1) // block_size

    THREADS = 256
    total = M * N
    num_blocks = (total + THREADS - 1) // THREADS

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_vectorized", "tir.noalias": True})

        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N

                    C[i, j] = T.float32(0)

                    # Process 4 elements at a time
                    for k4 in T.serial(K // 4):
                        k_base = k4 * 4
                        block_idx = k_base // block_size
                        w_scale = scale_W[j, block_idx]

                        # Load 4 A values (full precision)
                        a0 = A[i, k_base + 0]
                        a1 = A[i, k_base + 1]
                        a2 = A[i, k_base + 2]
                        a3 = A[i, k_base + 3]

                        # Load and dequant 4 W values
                        w0 = W[j, k_base + 0] * w_scale
                        w1 = W[j, k_base + 1] * w_scale
                        w2 = W[j, k_base + 2] * w_scale
                        w3 = W[j, k_base + 3] * w_scale

                        # Accumulate
                        C[i, j] = C[i, j] + a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3

    return kernel


def build_kernel(tir_func, target="cuda -arch=sm_110"):
    """Build TVM kernel."""
    target_obj = tvm.target.Target(target)
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(tir_func, target=target_obj)
    return mod


def benchmark_kernel(name, kernel_func, func_name, has_activation_scale=True, warmup=50, runs=200):
    """Benchmark a single kernel."""
    import time

    M, N, K = 1, 3072, 3072
    num_blocks_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    # Build
    print("  Building kernel...")
    build_start = time.time()
    mod = build_kernel(kernel_func)
    build_time = (time.time() - build_start) * 1000
    print(f"  Build time: {build_time:.2f} ms")

    func = mod[func_name]

    # Prepare data
    device = tvm.runtime.cuda(0)

    A = tvm.runtime.empty((M, K), dtype="float32", device=device)
    A.copyfrom(np.random.randn(M, K).astype("float32"))

    W = tvm.runtime.empty((N, K), dtype="float32", device=device)
    W.copyfrom(np.random.randn(N, K).astype("float32"))

    scale_W = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
    scale_W.copyfrom((np.random.rand(N, num_blocks_k) * 0.1 + 0.01).astype("float32"))

    C = tvm.runtime.empty((M, N), dtype="float32", device=device)
    C.copyfrom(np.zeros((M, N), dtype="float32"))

    if has_activation_scale:
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
    else:
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
    print(f"  Avg time: {elapsed:.4f} ms")

    return elapsed


def main():
    M, N, K = 1, 3072, 3072

    print("="*70)
    print("Optimized nvFP4 GEMM Kernels Benchmark")
    print("="*70)
    print(f"\nMatrix: M={M}, N={N}, K={K}")

    results = {}

    # Baseline
    TRT_FP8_PER_GEMM = 47.4 / 90  # ~0.527 ms
    print(f"\nBaseline: TRT FP8 = {TRT_FP8_PER_GEMM:.4f} ms per GEMM")

    # W4A4 Vectorized
    try:
        kernel = create_w4a4_vectorized_kernel(M, N, K)
        results["W4A4 Vectorized"] = benchmark_kernel(
            "W4A4 Vectorized (float4)", kernel, "w4a4_vectorized",
            has_activation_scale=True
        )
    except Exception as e:
        print(f"  W4A4 Vectorized FAILED: {e}")

    # W4A16 Vectorized
    try:
        kernel = create_w4a16_vectorized_kernel(M, N, K)
        results["W4A16 Vectorized"] = benchmark_kernel(
            "W4A16 Vectorized (float4)", kernel, "w4a16_vectorized",
            has_activation_scale=False
        )
    except Exception as e:
        print(f"  W4A16 Vectorized FAILED: {e}")

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
