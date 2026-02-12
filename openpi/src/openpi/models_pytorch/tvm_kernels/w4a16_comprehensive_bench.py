#!/usr/bin/env python3
"""
W4A16 Comprehensive Benchmark - Test various approaches and batch sizes.

Author: Claude Code
Date: 2026-02-11
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import te, tir, runtime
from tvm.script import tir as T
import numpy as np
import time


QUANT_BLOCK = 32


# ============= TIR GEMV Kernel =============
def create_tir_gemv_kernel(N, K, THREADS=256):
    """Direct TIR GEMV kernel."""
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid
                if n < N:
                    C[0, n] = T.float32(0)

                    for k in range(K):
                        byte_idx = k // 2
                        is_high = k % 2
                        packed = W_packed[n, byte_idx]

                        int4_val = T.if_then_else(
                            is_high == 0,
                            packed & T.uint8(0xF),
                            (packed >> 4) & T.uint8(0xF)
                        )
                        signed_val = T.Cast("float16", int4_val) - T.float16(8.0)
                        scale_idx = k // QUANT_BLOCK
                        scale = scales[n, scale_idx]
                        w = signed_val * scale

                        a = A[0, k]
                        C[0, n] = C[0, n] + T.Cast("float32", a * w)

    return gemv


# ============= TIR GEMM Kernel =============
def create_tir_gemm_kernel(M, N, K, BLOCK_M=32, BLOCK_N=32, THREADS=256):
    """Direct TIR GEMM kernel for larger M."""
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N

    @T.prim_func
    def gemm(
        A: T.Buffer((M, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemm", "tir.noalias": True})

        for block_m in T.thread_binding(num_blocks_m, thread="blockIdx.y"):
            for block_n in T.thread_binding(num_blocks_n, thread="blockIdx.x"):
                for tid_m in T.thread_binding(BLOCK_M, thread="threadIdx.y"):
                    for tid_n in T.thread_binding(BLOCK_N // 4, thread="threadIdx.x"):
                        m = block_m * BLOCK_M + tid_m
                        n_base = block_n * BLOCK_N + tid_n * 4

                        if m < M:
                            for n_off in range(4):
                                n = n_base + n_off
                                if n < N:
                                    C[m, n] = T.float32(0)

                                    for k in range(K):
                                        byte_idx = k // 2
                                        is_high = k % 2
                                        packed = W_packed[n, byte_idx]

                                        int4_val = T.if_then_else(
                                            is_high == 0,
                                            packed & T.uint8(0xF),
                                            (packed >> 4) & T.uint8(0xF)
                                        )
                                        signed_val = T.Cast("float16", int4_val) - T.float16(8.0)
                                        scale_idx = k // QUANT_BLOCK
                                        scale = scales[n, scale_idx]
                                        w = signed_val * scale

                                        a = A[m, k]
                                        C[m, n] = C[m, n] + T.Cast("float32", a * w)

    return gemm


# ============= TE + dlight =============
def create_te_compute(M, N, K):
    """Create TE compute for dlight scheduling."""
    num_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    A = te.placeholder((M, K), dtype="float16", name="A")
    W_packed = te.placeholder((N, K_packed), dtype="uint8", name="W_packed")
    scales = te.placeholder((N, num_blocks), dtype="float16", name="scales")

    def dequant_func(n, k):
        byte_idx = k // 2
        is_high = k % 2
        packed = W_packed[n, byte_idx]
        int4_val = tir.if_then_else(
            is_high == 0,
            packed & tir.const(0xF, "uint8"),
            (packed >> 4) & tir.const(0xF, "uint8")
        )
        signed_val = int4_val.astype("float16") - tir.const(8.0, "float16")
        block_idx = k // QUANT_BLOCK
        scale = scales[n, block_idx]
        return signed_val * scale

    W_dequant = te.compute((N, K), dequant_func, name="dequantize")

    k_axis = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(
            A[m, k_axis].astype("float32") * W_dequant[n, k_axis].astype("float32"),
            axis=k_axis
        ),
        name="C"
    )

    return A, W_packed, scales, C


def quantize_int4(weight, block_size=QUANT_BLOCK):
    N, K = weight.shape
    num_blocks = (K + block_size - 1) // block_size
    K_packed = K // 2
    W_packed = np.zeros((N, K_packed), dtype=np.uint8)
    scales = np.zeros((N, num_blocks), dtype=np.float16)

    for n in range(N):
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, K)
            block = weight[n, start:end]
            max_abs = np.max(np.abs(block))
            scale = max_abs / 7.0 if max_abs > 0 else 1.0
            scales[n, b] = scale
            for k in range(start, end):
                val = block[k - start] / scale if scale > 0 else 0
                quantized = int(np.clip(np.round(val + 8), 0, 15))
                byte_idx = k // 2
                if k % 2 == 0:
                    W_packed[n, byte_idx] = (W_packed[n, byte_idx] & 0xF0) | quantized
                else:
                    W_packed[n, byte_idx] = (W_packed[n, byte_idx] & 0x0F) | (quantized << 4)

    return W_packed, scales


def dequant_int4(W_packed, scales, K, block_size=QUANT_BLOCK):
    N = W_packed.shape[0]
    W = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            byte_idx = k // 2
            packed = W_packed[n, byte_idx]
            int4_val = (packed & 0xF) if k % 2 == 0 else ((packed >> 4) & 0xF)
            block_idx = k // block_size
            W[n, k] = (int4_val - 8) * scales[n, block_idx]
    return W


def benchmark_kernel(name, lib, func_name, A_tvm, W_packed_tvm, scales_tvm, C_tvm, C_ref, device, warmup=50, runs=200):
    """Benchmark a kernel."""
    try:
        func = lib[func_name]
    except:
        print(f"  {name}: Function not found")
        return None

    func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    C_result = C_tvm.numpy()
    if np.isnan(C_result).any():
        print(f"  {name}: NaN in output")
        return None

    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)

    if cos_sim <= 0.99:
        print(f"  {name}: cos_sim={cos_sim:.4f} FAIL")
        return None

    for _ in range(warmup):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000
    return avg_ms


def test_batch_size(M, N=16384, K=2048):
    """Test all kernels for a given batch size."""
    print(f"\n{'='*60}")
    print(f"M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    # Data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    device = runtime.cuda(0)
    target = tvm.target.Target("cuda -arch=sm_110")

    results = {}

    # 1. TIR GEMV (only for M=1)
    if M == 1:
        try:
            print("Building TIR GEMV...")
            kernel = create_tir_gemv_kernel(N, K)
            mod = tvm.IRModule({"main": kernel})
            with tvm.transform.PassContext(opt_level=3):
                lib = tvm.build(mod, target=target)

            A_tvm = runtime.empty(A_np.shape, "float16", device)
            A_tvm.copyfrom(A_np)
            W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
            W_packed_tvm.copyfrom(W_packed_np)
            scales_tvm = runtime.empty(scales_np.shape, "float16", device)
            scales_tvm.copyfrom(scales_np)
            C_tvm = runtime.empty((M, N), "float32", device)

            ms = benchmark_kernel("TIR GEMV", lib, "gemv", A_tvm, W_packed_tvm, scales_tvm, C_tvm, C_ref, device)
            if ms:
                results["TIR GEMV"] = ms
                print(f"  TIR GEMV: {ms:.4f} ms")
        except Exception as e:
            print(f"  TIR GEMV failed: {e}")

    # 2. dlight Matmul
    try:
        print("Building dlight Matmul...")
        from tvm import dlight as dl

        A_te, W_packed_te, scales_te, C_te = create_te_compute(M, N, K)
        func = te.create_prim_func([A_te, W_packed_te, scales_te, C_te])
        mod = tvm.IRModule({"main": func})

        with tvm.transform.PassContext(opt_level=3):
            with target:
                mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
            lib = tvm.build(mod, target=target)

        A_tvm = runtime.empty(A_np.shape, "float16", device)
        A_tvm.copyfrom(A_np)
        W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
        W_packed_tvm.copyfrom(W_packed_np)
        scales_tvm = runtime.empty(scales_np.shape, "float16", device)
        scales_tvm.copyfrom(scales_np)
        C_tvm = runtime.empty((M, N), "float32", device)

        ms = benchmark_kernel("dlight Matmul", lib, "main", A_tvm, W_packed_tvm, scales_tvm, C_tvm, C_ref, device)
        if ms:
            results["dlight Matmul"] = ms
            print(f"  dlight Matmul: {ms:.4f} ms")
    except Exception as e:
        print(f"  dlight Matmul failed: {e}")

    # 3. dlight Fallback
    try:
        print("Building dlight Fallback...")
        from tvm import dlight as dl

        A_te, W_packed_te, scales_te, C_te = create_te_compute(M, N, K)
        func = te.create_prim_func([A_te, W_packed_te, scales_te, C_te])
        mod = tvm.IRModule({"main": func})

        with tvm.transform.PassContext(opt_level=3):
            with target:
                mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
            lib = tvm.build(mod, target=target)

        A_tvm = runtime.empty(A_np.shape, "float16", device)
        A_tvm.copyfrom(A_np)
        W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
        W_packed_tvm.copyfrom(W_packed_np)
        scales_tvm = runtime.empty(scales_np.shape, "float16", device)
        scales_tvm.copyfrom(scales_np)
        C_tvm = runtime.empty((M, N), "float32", device)

        ms = benchmark_kernel("dlight Fallback", lib, "main", A_tvm, W_packed_tvm, scales_tvm, C_tvm, C_ref, device)
        if ms:
            results["dlight Fallback"] = ms
            print(f"  dlight Fallback: {ms:.4f} ms")
    except Exception as e:
        print(f"  dlight Fallback failed: {e}")

    # Summary
    if results:
        best_name = min(results, key=results.get)
        best_ms = results[best_name]
        print(f"\n  Best: {best_name} at {best_ms:.4f} ms")

        flops = 2.0 * M * N * K
        tflops = flops / (best_ms / 1000) / 1e12
        print(f"  TFLOPS: {tflops:.4f}")

    return results


def main():
    print("\n" + "="*60)
    print("W4A16 Comprehensive Benchmark")
    print("N=16384, K=2048")
    print("="*60)

    all_results = {}

    # Test various batch sizes
    for M in [1, 4, 16, 64, 256, 712]:
        results = test_batch_size(M)
        all_results[M] = results

    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'M':>6} | {'Best Kernel':>20} | {'Time (ms)':>10} | {'TFLOPS':>8}")
    print("-"*60)

    N, K = 16384, 2048
    for M, results in all_results.items():
        if results:
            best_name = min(results, key=results.get)
            best_ms = results[best_name]
            flops = 2.0 * M * N * K
            tflops = flops / (best_ms / 1000) / 1e12
            print(f"{M:>6} | {best_name:>20} | {best_ms:>10.4f} | {tflops:>8.4f}")


if __name__ == "__main__":
    main()
