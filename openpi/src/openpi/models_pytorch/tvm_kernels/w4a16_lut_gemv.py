#!/usr/bin/env python3
"""
W4A16 GEMV with LUT (Look-Up Table) for dequantization.

Key insight: INT4 has only 16 possible values (0-15).
Instead of computing (int4 - 8) * scale at runtime,
precompute all 16 possible outputs for each scale and use LUT.

This trades memory for compute:
- Extra memory: 16 * num_scale_blocks * N * sizeof(float16) = 16 * 64 * 16384 * 2 = 32 MB
- But eliminates: shift, mask, sub, mul per element

Author: Claude Code
Date: 2026-02-11
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import tir, runtime
from tvm.script import tir as T
import numpy as np
import time


QUANT_BLOCK = 32


def create_lut_gemv(N, K, THREADS=256):
    """
    GEMV with precomputed LUT for dequantization.

    LUT shape: (N, num_scale_blocks, 16)
    LUT[n, qb, int4_val] = (int4_val - 8) * scales[n, qb]
    """
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        LUT: T.Buffer((N, num_scale_blocks, 16), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_lut", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid
                if n < N:
                    C[0, n] = T.float32(0)

                    for qb in range(num_scale_blocks):
                        byte_start = qb * (QUANT_BLOCK // 2)
                        k_start = qb * QUANT_BLOCK

                        for byte_offset in range(QUANT_BLOCK // 2):
                            byte_idx = byte_start + byte_offset
                            packed = W_packed[n, byte_idx]

                            k_lo = k_start + byte_offset * 2
                            k_hi = k_lo + 1

                            # LUT lookup instead of compute
                            int4_lo = packed & T.uint8(0xF)
                            int4_hi = (packed >> 4) & T.uint8(0xF)

                            w_lo = LUT[n, qb, T.Cast("int32", int4_lo)]
                            w_hi = LUT[n, qb, T.Cast("int32", int4_hi)]

                            C[0, n] = C[0, n] + T.Cast("float32", A[0, k_lo] * w_lo)
                            C[0, n] = C[0, n] + T.Cast("float32", A[0, k_hi] * w_hi)

    return gemv


def create_lut_shared_a_gemv(N, K, THREADS=256):
    """
    LUT GEMV with A in shared memory.
    """
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        LUT: T.Buffer((N, num_scale_blocks, 16), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_lut_shared", "tir.noalias": True})

        A_shared = T.alloc_buffer((K,), "float16", scope="shared")

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                for i in range((K + THREADS - 1) // THREADS):
                    k = tid + i * THREADS
                    if k < K:
                        A_shared[k] = A[0, k]

            T.tvm_storage_sync("shared")

            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid
                if n < N:
                    C[0, n] = T.float32(0)

                    for qb in range(num_scale_blocks):
                        byte_start = qb * (QUANT_BLOCK // 2)
                        k_start = qb * QUANT_BLOCK

                        for byte_offset in range(QUANT_BLOCK // 2):
                            byte_idx = byte_start + byte_offset
                            packed = W_packed[n, byte_idx]

                            k_lo = k_start + byte_offset * 2
                            k_hi = k_lo + 1

                            int4_lo = packed & T.uint8(0xF)
                            int4_hi = (packed >> 4) & T.uint8(0xF)

                            w_lo = LUT[n, qb, T.Cast("int32", int4_lo)]
                            w_hi = LUT[n, qb, T.Cast("int32", int4_hi)]

                            C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_lo] * w_lo)
                            C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_hi] * w_hi)

    return gemv


def create_transposed_lut_gemv(N, K, THREADS=256):
    """
    LUT GEMV with transposed LUT for better memory access.

    Original LUT: (N, num_scale_blocks, 16) - N is slowest changing
    Transposed LUT: (num_scale_blocks, 16, N) - N is fastest changing (coalesced)
    """
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        LUT_T: T.Buffer((num_scale_blocks, 16, N), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_lut_t", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid
                if n < N:
                    C[0, n] = T.float32(0)

                    for qb in range(num_scale_blocks):
                        byte_start = qb * (QUANT_BLOCK // 2)
                        k_start = qb * QUANT_BLOCK

                        for byte_offset in range(QUANT_BLOCK // 2):
                            byte_idx = byte_start + byte_offset
                            packed = W_packed[n, byte_idx]

                            k_lo = k_start + byte_offset * 2
                            k_hi = k_lo + 1

                            int4_lo = packed & T.uint8(0xF)
                            int4_hi = (packed >> 4) & T.uint8(0xF)

                            # Transposed LUT access
                            w_lo = LUT_T[qb, T.Cast("int32", int4_lo), n]
                            w_hi = LUT_T[qb, T.Cast("int32", int4_hi), n]

                            C[0, n] = C[0, n] + T.Cast("float32", A[0, k_lo] * w_lo)
                            C[0, n] = C[0, n] + T.Cast("float32", A[0, k_hi] * w_hi)

    return gemv


# ============= Helpers =============

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


def create_lut(scales):
    """Create LUT from scales."""
    N, num_scale_blocks = scales.shape
    LUT = np.zeros((N, num_scale_blocks, 16), dtype=np.float16)
    for n in range(N):
        for qb in range(num_scale_blocks):
            scale = scales[n, qb]
            for int4_val in range(16):
                LUT[n, qb, int4_val] = (int4_val - 8) * scale
    return LUT


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


def benchmark_lut_kernel(kernel_fn, name, N=16384, K=2048, warmup=50, runs=200, transposed=False):
    """Benchmark LUT GEMV kernel."""
    print(f"\n--- {name} ---")

    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)
    W_packed_np, scales_np = quantize_int4(W_np)
    LUT_np = create_lut(scales_np)

    if transposed:
        LUT_np = LUT_np.transpose(1, 2, 0).copy()

    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    try:
        kernel = kernel_fn(N, K)
        mod = tvm.IRModule({"main": kernel})
        target = tvm.target.Target("cuda -arch=sm_110")
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(mod, target=target)
    except Exception as e:
        print(f"Build failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    func_names = ["gemv_lut", "gemv_lut_shared", "gemv_lut_t", "main"]
    func_name = None
    for name_try in func_names:
        try:
            func = lib[name_try]
            func_name = name_try
            break
        except:
            continue

    if func_name is None:
        print("No function found!")
        return None

    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    LUT_tvm = runtime.empty(LUT_np.shape, "float16", device)
    LUT_tvm.copyfrom(LUT_np)
    C_tvm = runtime.empty((1, N), "float32", device)

    func = lib[func_name]
    func(A_tvm, W_packed_tvm, LUT_tvm, C_tvm)
    device.sync()

    C_result = C_tvm.numpy()
    if np.isnan(C_result).any():
        print("NaN!")
        return None

    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)
    print(f"Cos sim: {cos_sim:.6f}")

    if cos_sim <= 0.99:
        print("FAIL")
        return None

    for _ in range(warmup):
        func(A_tvm, W_packed_tvm, LUT_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        func(A_tvm, W_packed_tvm, LUT_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000

    # Memory analysis (including LUT)
    weight_bytes = N * K // 2
    lut_bytes = LUT_np.nbytes
    a_bytes = K * 2
    total_bytes = weight_bytes + lut_bytes + a_bytes
    bandwidth = total_bytes / (avg_ms / 1000) / 1e9

    print(f"Time: {avg_ms:.4f} ms")
    print(f"Memory (W+LUT+A): {total_bytes / 1e6:.2f} MB")
    print(f"  W_packed: {weight_bytes / 1e6:.2f} MB")
    print(f"  LUT: {lut_bytes / 1e6:.2f} MB")
    print(f"Bandwidth: {bandwidth:.1f} GB/s")

    return avg_ms


def main():
    N, K = 16384, 2048

    print("="*60)
    print("W4A16 LUT GEMV Benchmark")
    print(f"N={N}, K={K}")
    print("Target: < 0.2ms")
    print("="*60)

    # Memory analysis
    num_scale_blocks = K // QUANT_BLOCK
    weight_bytes = N * K // 2
    lut_bytes = N * num_scale_blocks * 16 * 2  # float16
    total_bytes = weight_bytes + lut_bytes + K * 2

    print(f"\nMemory footprint:")
    print(f"  W_packed: {weight_bytes / 1e6:.2f} MB")
    print(f"  LUT: {lut_bytes / 1e6:.2f} MB")
    print(f"  Total: {total_bytes / 1e6:.2f} MB")
    print(f"\nTheoretical:")
    print(f"  DRAM (55 GB/s): {total_bytes / (55e9) * 1000:.4f} ms")
    print(f"  L2 (230 GB/s): {total_bytes / (230e9) * 1000:.4f} ms")

    results = {}

    # Test LUT kernels
    kernels = [
        (create_lut_gemv, "LUT Basic", False),
        (create_lut_shared_a_gemv, "LUT + Shared A", False),
        (create_transposed_lut_gemv, "LUT Transposed", True),
    ]

    for kernel_fn, name, transposed in kernels:
        ms = benchmark_lut_kernel(kernel_fn, name, N, K, transposed=transposed)
        if ms is not None:
            results[name] = ms

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if results:
        print(f"{'Kernel':<20} | {'Time (ms)':<12} | {'vs 0.2ms':<10}")
        print("-"*50)
        for name, ms in sorted(results.items(), key=lambda x: x[1]):
            ratio = ms / 0.2
            status = "OK" if ratio <= 1.0 else f"{ratio:.1f}x"
            print(f"{name:<20} | {ms:<12.4f} | {status:<10}")

        print(f"\nBest: {min(results.values()):.4f} ms")


if __name__ == "__main__":
    main()
