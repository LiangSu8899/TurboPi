#!/usr/bin/env python3
"""
Benchmark TVM nvFP4 GEMM kernels (naive vs optimized).

Tests:
1. Correctness - compare with PyTorch reference
2. Performance - measure execution time

Run with:
    python benchmark_optimized.py

Requires TVM and PyTorch.
"""

import os
import sys
import time


def check_environment():
    """Check TVM and PyTorch environment."""
    try:
        import tvm
        print(f"TVM version: {tvm.__version__}")
    except ImportError:
        print("ERROR: TVM not found")
        return False

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("ERROR: PyTorch not found")
        return False

    return True


def create_naive_kernel(M, N, K, block_size=32):
    """Create naive nvFP4 GEMM kernel."""
    import tvm
    from tvm.script import tir as T

    num_blocks_k = (K + block_size - 1) // block_size
    THREADS_PER_BLOCK = 256
    total_elements = M * N
    num_blocks_x = (total_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    @T.prim_func
    def nvfp4_gemm_naive(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "nvfp4_gemm_naive", "tir.noalias": True})

        for bx in T.thread_binding(num_blocks_x, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):
                idx = bx * THREADS_PER_BLOCK + tx
                if idx < total_elements:
                    i = idx // N
                    j = idx % N
                    C[i, j] = T.float32(0)
                    for k in T.serial(K):
                        block_idx = k // block_size
                        a_val = A[i, k] * scale_A[i, block_idx]
                        w_val = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + a_val * w_val

    return nvfp4_gemm_naive


def create_unroll8_kernel(M, N, K, block_size=32):
    """Create 8x unrolled nvFP4 GEMM kernel."""
    import tvm
    from tvm.script import tir as T

    num_blocks_k = (K + block_size - 1) // block_size
    THREADS_PER_BLOCK = 256
    total_elements = M * N
    num_blocks_x = (total_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    @T.prim_func
    def nvfp4_gemm_unroll8(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "nvfp4_gemm_unroll8", "tir.noalias": True})

        for bx in T.thread_binding(num_blocks_x, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):
                idx = bx * THREADS_PER_BLOCK + tx
                if idx < total_elements:
                    i = idx // N
                    j = idx % N
                    C[i, j] = T.float32(0)

                    for k8 in T.serial(K // 8):
                        k_base = k8 * 8
                        block_idx = k_base // block_size
                        a_scale = scale_A[i, block_idx]
                        w_scale = scale_W[j, block_idx]

                        C[i, j] = C[i, j] + A[i, k_base + 0] * a_scale * W[j, k_base + 0] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 1] * a_scale * W[j, k_base + 1] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 2] * a_scale * W[j, k_base + 2] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 3] * a_scale * W[j, k_base + 3] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 4] * a_scale * W[j, k_base + 4] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 5] * a_scale * W[j, k_base + 5] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 6] * a_scale * W[j, k_base + 6] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 7] * a_scale * W[j, k_base + 7] * w_scale

    return nvfp4_gemm_unroll8


def reference_gemm(A, W, scale_A, scale_W, block_size=32):
    """PyTorch reference implementation."""
    import torch

    M, K = A.shape
    N = W.shape[0]
    num_blocks_k = (K + block_size - 1) // block_size

    # Expand scales to match K dimension
    # scale_A: [M, num_blocks_k] -> [M, K]
    scale_A_expanded = scale_A.repeat_interleave(block_size, dim=1)[:, :K]
    # scale_W: [N, num_blocks_k] -> [N, K]
    scale_W_expanded = scale_W.repeat_interleave(block_size, dim=1)[:, :K]

    # Dequantize
    A_dequant = A * scale_A_expanded
    W_dequant = W * scale_W_expanded

    # GEMM: C = A @ W^T
    C = torch.matmul(A_dequant, W_dequant.T)

    return C


def build_and_test_kernel(kernel_func, name, A, W, scale_A, scale_W, C_ref, target="cuda -arch=sm_110"):
    """Build and test a kernel."""
    import tvm
    import torch

    M, K = A.shape
    N = W.shape[0]

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  M={M}, N={N}, K={K}")

    # Build
    print("  Building...")
    build_start = time.time()
    target_obj = tvm.target.Target(target)
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(kernel_func, target=target_obj)
    build_time = time.time() - build_start
    print(f"  Build time: {build_time*1000:.2f} ms")

    # Get function
    func_name = list(mod.get_global_func_table().keys())[0]
    func = mod[func_name]

    # Prepare TVM arrays using DLPack
    C_tvm = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    a_tvm = tvm.runtime.from_dlpack(A.contiguous())
    w_tvm = tvm.runtime.from_dlpack(W.contiguous())
    scale_a_tvm = tvm.runtime.from_dlpack(scale_A.contiguous())
    scale_w_tvm = tvm.runtime.from_dlpack(scale_W.contiguous())
    c_tvm = tvm.runtime.from_dlpack(C_tvm)

    # Warmup
    print("  Warming up...")
    for _ in range(10):
        func(a_tvm, w_tvm, scale_a_tvm, scale_w_tvm, c_tvm)
    torch.cuda.synchronize()

    # Correctness check
    print("  Checking correctness...")
    func(a_tvm, w_tvm, scale_a_tvm, scale_w_tvm, c_tvm)
    torch.cuda.synchronize()

    max_diff = torch.abs(C_tvm - C_ref).max().item()
    rel_error = max_diff / (torch.abs(C_ref).max().item() + 1e-8)
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Rel error: {rel_error:.6f}")

    if max_diff < 1e-3:
        print(f"  Correctness: PASSED")
    else:
        print(f"  Correctness: FAILED")
        # Print first few elements for debugging
        print(f"  First 5 TVM: {C_tvm[0, :5].tolist()}")
        print(f"  First 5 Ref: {C_ref[0, :5].tolist()}")

    # Benchmark
    print("  Benchmarking...")
    runs = 100
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        func(a_tvm, w_tvm, scale_a_tvm, scale_w_tvm, c_tvm)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_ms = elapsed / runs * 1000

    # Calculate TFLOPS
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12

    print(f"  Avg time: {avg_ms:.4f} ms")
    print(f"  Throughput: {tflops:.2f} TFLOPS")

    return {
        "name": name,
        "build_time_ms": build_time * 1000,
        "avg_time_ms": avg_ms,
        "tflops": tflops,
        "max_diff": max_diff,
        "passed": max_diff < 1e-3,
    }


def benchmark_cublas(A, W, scale_A, scale_W, C_ref, block_size=32):
    """Benchmark cuBLAS FP32 GEMM (dequant + matmul)."""
    import torch

    M, K = A.shape
    N = W.shape[0]

    print(f"\n{'='*60}")
    print(f"Testing: cuBLAS FP32 (dequant + matmul)")
    print(f"  M={M}, N={N}, K={K}")

    # Expand scales
    scale_A_expanded = scale_A.repeat_interleave(block_size, dim=1)[:, :K]
    scale_W_expanded = scale_W.repeat_interleave(block_size, dim=1)[:, :K]

    # Warmup
    print("  Warming up...")
    for _ in range(10):
        A_dequant = A * scale_A_expanded
        W_dequant = W * scale_W_expanded
        C = torch.matmul(A_dequant, W_dequant.T)
    torch.cuda.synchronize()

    # Benchmark
    print("  Benchmarking...")
    runs = 100
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        A_dequant = A * scale_A_expanded
        W_dequant = W * scale_W_expanded
        C = torch.matmul(A_dequant, W_dequant.T)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_ms = elapsed / runs * 1000

    # Calculate TFLOPS
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12

    print(f"  Avg time: {avg_ms:.4f} ms")
    print(f"  Throughput: {tflops:.2f} TFLOPS")

    return {
        "name": "cuBLAS FP32 (dequant+matmul)",
        "avg_time_ms": avg_ms,
        "tflops": tflops,
    }


def main():
    import torch
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark TVM nvFP4 GEMM kernels")
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--N", type=int, default=3072)
    parser.add_argument("--K", type=int, default=3072)
    parser.add_argument("--block-size", type=int, default=32)

    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    M, N, K = args.M, args.N, args.K
    block_size = args.block_size
    num_blocks_k = (K + block_size - 1) // block_size

    print(f"\n{'='*60}")
    print(f"Benchmark Configuration")
    print(f"  M={M}, N={N}, K={K}")
    print(f"  Block size: {block_size}")
    print(f"  Num K blocks: {num_blocks_k}")
    print(f"{'='*60}")

    # Create test data
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    W = torch.randn(N, K, dtype=torch.float32, device="cuda")
    scale_A = torch.ones(M, num_blocks_k, dtype=torch.float32, device="cuda")
    scale_W = torch.ones(N, num_blocks_k, dtype=torch.float32, device="cuda")

    # Reference
    C_ref = reference_gemm(A, W, scale_A, scale_W, block_size)
    print(f"\nReference output: {C_ref[0, :5].tolist()}")

    results = []

    # Test cuBLAS baseline
    cublas_result = benchmark_cublas(A, W, scale_A, scale_W, C_ref, block_size)
    results.append(cublas_result)

    # Test naive kernel
    naive_kernel = create_naive_kernel(M, N, K, block_size)
    naive_result = build_and_test_kernel(naive_kernel, "TVM Naive", A, W, scale_A, scale_W, C_ref)
    results.append(naive_result)

    # Test 8x unroll kernel
    unroll8_kernel = create_unroll8_kernel(M, N, K, block_size)
    unroll8_result = build_and_test_kernel(unroll8_kernel, "TVM Unroll 8x", A, W, scale_A, scale_W, C_ref)
    results.append(unroll8_result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Kernel':<30} {'Time (ms)':<12} {'TFLOPS':<12} {'Speedup':<10}")
    print(f"{'-'*60}")

    baseline_time = results[0]["avg_time_ms"]
    for r in results:
        speedup = baseline_time / r["avg_time_ms"] if r["avg_time_ms"] > 0 else 0
        print(f"{r['name']:<30} {r['avg_time_ms']:<12.4f} {r['tflops']:<12.2f} {speedup:<10.2f}x")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
