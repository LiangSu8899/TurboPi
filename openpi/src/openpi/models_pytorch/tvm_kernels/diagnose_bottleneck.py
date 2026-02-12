#!/usr/bin/env python3
"""
TVM nvFP4 Kernel 瓶颈诊断脚本

测试目标：
1. 诊断当前 kernel 的性能瓶颈
2. 对比不同优化级别的效果
3. 为后续优化提供数据支撑

运行方式：
    python diagnose_bottleneck.py

需要 TVM 环境。

Author: Claude Code
Date: 2026-02-10
"""

import sys
import os
import time
import argparse

# TVM path
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)


def check_env():
    """检查环境."""
    try:
        import tvm
        print(f"TVM: {tvm.__version__}")
        print(f"CUDA: {tvm.cuda().exist}")
        return True
    except ImportError:
        print("ERROR: TVM not found")
        return False


# ==============================================================================
# Kernel 实现：从最差到最优
# TVM 0.24 兼容版本 - 所有 buffer 在函数级别分配
# ==============================================================================

def create_kernel_v0_naive(M, N, K, block_size=32):
    """
    V0: 最朴素实现 - 累加到 global memory
    预期：最慢，因为每次循环都读写 global memory
    """
    import tvm
    from tvm.script import tir as T

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
        T.func_attr({"global_symbol": "kernel_v0", "tir.noalias": True})
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N
                    # 问题：累加到 global memory
                    C[i, j] = T.float32(0)
                    for k in T.serial(K):
                        block_idx = k // block_size
                        a_val = A[i, k] * scale_A[i, block_idx]
                        w_val = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + a_val * w_val  # ← 每次都读写 global!

    return kernel, "V0_naive_global_acc"


def create_kernel_v1_unroll8(M, N, K, block_size=32):
    """
    V1: 8x 循环展开

    优化：减少循环开销，可能触发 ILP
    仍然累加到 global memory（TVM 0.24 限制）
    """
    import tvm
    from tvm.script import tir as T

    assert K % 8 == 0, "K must be divisible by 8"

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
        T.func_attr({"global_symbol": "kernel_v1", "tir.noalias": True})

        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N
                    C[i, j] = T.float32(0)

                    # 8x 展开
                    for k8 in T.serial(K // 8):
                        k_base = k8 * 8
                        block_idx = k_base // block_size
                        a_scale = scale_A[i, block_idx]
                        w_scale = scale_W[j, block_idx]

                        # 手动展开
                        C[i, j] = C[i, j] + A[i, k_base + 0] * a_scale * W[j, k_base + 0] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 1] * a_scale * W[j, k_base + 1] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 2] * a_scale * W[j, k_base + 2] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 3] * a_scale * W[j, k_base + 3] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 4] * a_scale * W[j, k_base + 4] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 5] * a_scale * W[j, k_base + 5] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 6] * a_scale * W[j, k_base + 6] * w_scale
                        C[i, j] = C[i, j] + A[i, k_base + 7] * a_scale * W[j, k_base + 7] * w_scale

    return kernel, "V1_unroll8"


def create_kernel_v2_shared_a_tiled(M, N, K, block_size=32, tile_k=256):
    """
    V2: Shared Memory 缓存 A (针对 GEMV M=1)

    对于 M=1，A 只有一行，所有线程都需要读取同一行的数据。
    将 A 缓存到 shared memory 可以避免大量重复 global memory 访问。

    注意：这里累加器使用 shared memory 中的临时空间。
    """
    import tvm
    from tvm.script import tir as T

    num_blocks_k = (K + block_size - 1) // block_size
    THREADS = 256
    total = M * N
    num_n_blocks = (total + THREADS - 1) // THREADS
    num_k_tiles = (K + tile_k - 1) // tile_k

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "kernel_v2", "tir.noalias": True})

        # Shared memory - 在 prim_func 级别分配
        A_shared = T.alloc_buffer((tile_k,), "float32", scope="shared")
        scale_A_shared = T.alloc_buffer((tile_k // block_size + 1,), "float32", scope="shared")
        # 临时累加空间 (每个线程一个位置)
        acc_shared = T.alloc_buffer((THREADS,), "float32", scope="shared")

        for bx in T.thread_binding(num_n_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                # 初始化累加器
                acc_shared[tx] = T.float32(0)

                idx = bx * THREADS + tx

                for k_tile in T.serial(num_k_tiles):
                    k_start = k_tile * tile_k

                    # 协作加载 A 到 shared memory
                    # 每个线程负责加载 tile_k / THREADS 个元素
                    for load_iter in T.serial((tile_k + THREADS - 1) // THREADS):
                        k_local = tx + load_iter * THREADS
                        if k_local < tile_k:
                            k_global = k_start + k_local
                            if k_global < K:
                                A_shared[k_local] = A[0, k_global]
                            else:
                                A_shared[k_local] = T.float32(0)

                    # 加载 scale_A
                    num_scale_per_tile = (tile_k + block_size - 1) // block_size
                    if tx < num_scale_per_tile:
                        scale_idx = k_start // block_size + tx
                        if scale_idx < num_blocks_k:
                            scale_A_shared[tx] = scale_A[0, scale_idx]
                        else:
                            scale_A_shared[tx] = T.float32(1)

                    # 同步
                    T.tvm_storage_sync("shared")

                    # 计算
                    if idx < total:
                        j = idx % N
                        for k_local in T.serial(tile_k):
                            k_global = k_start + k_local
                            if k_global < K:
                                scale_local_idx = k_local // block_size
                                a_val = A_shared[k_local] * scale_A_shared[scale_local_idx]
                                w_block_idx = k_global // block_size
                                w_val = W[j, k_global] * scale_W[j, w_block_idx]
                                acc_shared[tx] = acc_shared[tx] + a_val * w_val

                    T.tvm_storage_sync("shared")

                # 写出
                if idx < total:
                    C[0, idx] = acc_shared[tx]

    return kernel, "V2_shared_A_tiled"


def create_kernel_v3_multi_output(M, N, K, block_size=32, outputs_per_thread=4):
    """
    V3: 每个线程处理多个输出元素

    对于 GEMV，让每个线程计算多个输出可以增加 arithmetic intensity。
    """
    import tvm
    from tvm.script import tir as T

    num_blocks_k = (K + block_size - 1) // block_size
    THREADS = 256
    outputs_per_block = THREADS * outputs_per_thread
    num_n_blocks = (N + outputs_per_block - 1) // outputs_per_block

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "kernel_v3", "tir.noalias": True})

        # 临时累加空间
        acc_shared = T.alloc_buffer((THREADS * outputs_per_thread,), "float32", scope="shared")

        for bx in T.thread_binding(num_n_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                # 初始化
                for e in T.serial(outputs_per_thread):
                    acc_shared[tx * outputs_per_thread + e] = T.float32(0)

                T.tvm_storage_sync("shared")

                # 计算
                for e in T.serial(outputs_per_thread):
                    j = bx * outputs_per_block + tx * outputs_per_thread + e
                    if j < N:
                        for k in T.serial(K):
                            k_block = k // block_size
                            a_val = A[0, k] * scale_A[0, k_block]
                            w_val = W[j, k] * scale_W[j, k_block]
                            acc_shared[tx * outputs_per_thread + e] = acc_shared[tx * outputs_per_thread + e] + a_val * w_val

                T.tvm_storage_sync("shared")

                # 写出
                for e in T.serial(outputs_per_thread):
                    j = bx * outputs_per_block + tx * outputs_per_thread + e
                    if j < N:
                        C[0, j] = acc_shared[tx * outputs_per_thread + e]

    return kernel, "V3_multi_output_4x"


def create_kernel_v4_warp_coalesced(M, N, K, block_size=32):
    """
    V4: Warp 协同访问优化

    让 warp 内的线程访问连续内存位置，提高 memory coalescing。
    每个线程处理 N 维度上连续的元素。
    """
    import tvm
    from tvm.script import tir as T

    num_blocks_k = (K + block_size - 1) // block_size
    THREADS = 256
    WARP_SIZE = 32
    # 每个 warp 处理 32 个连续输出
    outputs_per_warp = WARP_SIZE
    warps_per_block = THREADS // WARP_SIZE
    outputs_per_block = warps_per_block * outputs_per_warp
    num_n_blocks = (N + outputs_per_block - 1) // outputs_per_block

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "kernel_v4", "tir.noalias": True})

        for bx in T.thread_binding(num_n_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                warp_id = tx // WARP_SIZE
                lane_id = tx % WARP_SIZE

                # 每个线程负责一个输出
                j = bx * outputs_per_block + warp_id * WARP_SIZE + lane_id

                if j < N:
                    # 直接累加到输出（利用 TVM 编译器优化）
                    C[0, j] = T.float32(0)
                    for k in T.serial(K):
                        k_block = k // block_size
                        a_val = A[0, k] * scale_A[0, k_block]
                        w_val = W[j, k] * scale_W[j, k_block]
                        C[0, j] = C[0, j] + a_val * w_val

    return kernel, "V4_warp_coalesced"


# ==============================================================================
# 测试框架
# ==============================================================================

def build_and_benchmark(kernel_func, name, M, N, K, warmup=50, runs=200):
    """构建并测试 kernel."""
    import tvm
    import numpy as np

    block_size = 32
    num_blocks_k = (K + block_size - 1) // block_size

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    # Build
    print("  Building...")
    target = tvm.target.Target("cuda -arch=sm_110")
    build_start = time.time()
    try:
        with tvm.transform.PassContext(opt_level=3):
            mod = tvm.build(kernel_func, target=target)
        build_time = (time.time() - build_start) * 1000
        print(f"  Build OK: {build_time:.2f} ms")
    except Exception as e:
        print(f"  Build FAILED: {e}")
        return {"name": name, "status": "build_failed", "error": str(e)}

    # 获取导出的 CUDA 源码长度
    if hasattr(mod, 'imports_') and len(mod.imports_) > 0:
        cuda_src = mod.imports_[0].inspect_source()
        print(f"  CUDA source: {len(cuda_src)} bytes")

    # 获取函数
    func_name = kernel_func.attrs["global_symbol"]
    func = mod[func_name]

    # 准备数据
    device = tvm.runtime.cuda(0)

    A = tvm.runtime.empty((M, K), dtype="float32", device=device)
    A.copyfrom(np.random.randn(M, K).astype("float32"))

    W = tvm.runtime.empty((N, K), dtype="float32", device=device)
    W.copyfrom(np.random.randn(N, K).astype("float32"))

    scale_A = tvm.runtime.empty((M, num_blocks_k), dtype="float32", device=device)
    scale_A.copyfrom((np.random.rand(M, num_blocks_k) * 0.1 + 0.01).astype("float32"))

    scale_W = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
    scale_W.copyfrom((np.random.rand(N, num_blocks_k) * 0.1 + 0.01).astype("float32"))

    C = tvm.runtime.empty((M, N), dtype="float32", device=device)
    C.copyfrom(np.zeros((M, N)).astype("float32"))

    # Warmup
    print(f"  Warming up ({warmup} runs)...")
    try:
        for _ in range(warmup):
            func(A, W, scale_A, scale_W, C)
        tvm.runtime.cuda(0).sync()
    except Exception as e:
        print(f"  Runtime FAILED: {e}")
        return {"name": name, "status": "runtime_failed", "error": str(e)}

    # Benchmark
    print(f"  Benchmarking ({runs} runs)...")
    tvm.runtime.cuda(0).sync()
    start = time.time()
    for _ in range(runs):
        func(A, W, scale_A, scale_W, C)
    tvm.runtime.cuda(0).sync()
    elapsed = time.time() - start
    avg_ms = elapsed / runs * 1000

    # 计算 TFLOPS
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12

    # Memory bandwidth
    # Read: A[M,K] + W[N,K] + scale_A[M,num_blocks] + scale_W[N,num_blocks]
    # Write: C[M,N]
    read_bytes = (M * K + N * K + M * num_blocks_k + N * num_blocks_k) * 4
    write_bytes = M * N * 4
    total_bytes = read_bytes + write_bytes
    bandwidth_gb = total_bytes / (avg_ms / 1000) / 1e9

    print(f"  Avg time:   {avg_ms:.4f} ms")
    print(f"  Throughput: {tflops:.4f} TFLOPS")
    print(f"  Bandwidth:  {bandwidth_gb:.2f} GB/s")

    return {
        "name": name,
        "status": "ok",
        "time_ms": avg_ms,
        "tflops": tflops,
        "bandwidth_gb": bandwidth_gb,
        "build_time_ms": build_time,
    }


def main():
    parser = argparse.ArgumentParser(description="TVM nvFP4 Kernel Bottleneck Diagnosis")
    parser.add_argument("--M", type=int, default=1, help="Batch size (rows)")
    parser.add_argument("--N", type=int, default=3072, help="Output features")
    parser.add_argument("--K", type=int, default=3072, help="Input features")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)

    args = parser.parse_args()

    if not check_env():
        sys.exit(1)

    M, N, K = args.M, args.N, args.K

    print("\n" + "="*70)
    print("TVM nvFP4 Kernel Bottleneck Diagnosis")
    print("="*70)
    print(f"Matrix: M={M}, N={N}, K={K}")
    print(f"FLOPs: {2.0 * M * N * K / 1e9:.2f} GFLOPS")

    # TRT FP8 baseline
    trt_fp8_per_gemm = 0.53  # ms
    print(f"\nBaseline: TRT FP8 = {trt_fp8_per_gemm} ms")

    results = []

    # 测试不同版本的 kernel
    kernels = [
        create_kernel_v0_naive(M, N, K),
        create_kernel_v1_unroll8(M, N, K),
        create_kernel_v2_shared_a_tiled(M, N, K),
        create_kernel_v3_multi_output(M, N, K),
        create_kernel_v4_warp_coalesced(M, N, K),
    ]

    for kernel_func, name in kernels:
        result = build_and_benchmark(
            kernel_func, name, M, N, K,
            warmup=args.warmup, runs=args.runs
        )
        results.append(result)

    # 汇总
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Kernel':<25} {'Time (ms)':<12} {'vs TRT FP8':<12} {'Status'}")
    print("-"*60)

    for r in results:
        if r["status"] == "ok":
            speedup = trt_fp8_per_gemm / r["time_ms"]
            status = "✅ FASTER" if speedup > 1 else "❌ SLOWER"
            print(f"{r['name']:<25} {r['time_ms']:<12.4f} {speedup:<12.2f}x {status}")
        else:
            print(f"{r['name']:<25} {'N/A':<12} {'N/A':<12} ❌ {r['status']}")

    print("-"*60)
    print(f"{'TRT FP8 (baseline)':<25} {trt_fp8_per_gemm:<12} {1.0:<12.2f}x")

    # 诊断结论
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        v0 = next((r for r in ok_results if "v0" in r["name"].lower() or "naive" in r["name"].lower()), None)
        v1 = next((r for r in ok_results if "v1" in r["name"].lower() or "unroll" in r["name"].lower()), None)
        v2 = next((r for r in ok_results if "v2" in r["name"].lower() or "shared" in r["name"].lower()), None)

        if v0 and v1:
            unroll_speedup = v0["time_ms"] / v1["time_ms"]
            print(f"1. Unroll 8x speedup: {unroll_speedup:.2f}x")
            if unroll_speedup < 1.1:
                print("   → Loop unrolling alone doesn't help much")
            else:
                print("   → Loop unrolling provides some benefit")

        if v0 and v2:
            shared_speedup = v0["time_ms"] / v2["time_ms"]
            print(f"2. Shared memory speedup: {shared_speedup:.2f}x")
            if shared_speedup > 1.5:
                print("   → Shared memory is effective for A reuse!")
            else:
                print("   → Shared memory overhead may be too high")

        best = min(ok_results, key=lambda x: x["time_ms"])
        print(f"\n3. Best kernel: {best['name']} ({best['time_ms']:.4f} ms)")

        gap = best["time_ms"] / trt_fp8_per_gemm
        print(f"4. Gap to TRT FP8: {gap:.2f}x slower")

        if gap > 1:
            print("\n5. Remaining optimizations needed:")
            print("   - Use local/register accumulator (TVM 0.24 workaround)")
            print("   - Async memory loads (cp.async)")
            print("   - Software pipelining (double buffering)")
            print("   - Tensor Core if available")
            print("   - Consider Triton as alternative")

    print("="*70)


if __name__ == "__main__":
    main()
