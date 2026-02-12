#!/usr/bin/env python3
"""
W4A16 GEMV with Transposed Weight Layout for Coalesced Memory Access.

Key insight: Current layout (N, K_packed) causes non-coalesced access
because adjacent threads access different rows.

Transposed layout (K_packed, N):
- Adjacent threads access adjacent N elements in same column
- This enables coalesced 128-bit loads

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


def create_transposed_gemv_v1(N, K, THREADS=256):
    """
    GEMV with transposed W_packed layout.

    W_packed_T: (K_packed, N) - K is slowest, N is fastest
    scales_T: (num_scale_blocks, N) - also transposed

    Memory access pattern:
    - Thread n accesses W_packed_T[k, n] for all k
    - Adjacent threads access W_packed_T[k, n], W_packed_T[k, n+1], ...
    - This is COALESCED!
    """
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS
    BYTES_PER_QB = QUANT_BLOCK // 2

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed_T: T.Buffer((K_packed, N), "uint8"),
        scales_T: T.Buffer((num_scale_blocks, N), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_t_v1", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid
                if n < N:
                    C[0, n] = T.float32(0)

                    for qb in range(num_scale_blocks):
                        scale = scales_T[qb, n]  # Coalesced access!
                        byte_start = qb * BYTES_PER_QB
                        k_start = qb * QUANT_BLOCK

                        for byte_offset in range(BYTES_PER_QB):
                            byte_idx = byte_start + byte_offset
                            packed = W_packed_T[byte_idx, n]  # Coalesced access!

                            k_lo = k_start + byte_offset * 2
                            k_hi = k_lo + 1

                            int4_lo = packed & T.uint8(0xF)
                            int4_hi = (packed >> 4) & T.uint8(0xF)

                            w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                            w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale

                            C[0, n] = C[0, n] + T.Cast("float32", A[0, k_lo] * w_lo)
                            C[0, n] = C[0, n] + T.Cast("float32", A[0, k_hi] * w_hi)

    return gemv


def create_transposed_gemv_v2(N, K, THREADS=256):
    """
    V2: Transposed layout + shared memory for A.
    """
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS
    BYTES_PER_QB = QUANT_BLOCK // 2

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed_T: T.Buffer((K_packed, N), "uint8"),
        scales_T: T.Buffer((num_scale_blocks, N), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_t_v2", "tir.noalias": True})

        A_shared = T.alloc_buffer((K,), "float16", scope="shared")

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            # Load A to shared memory
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
                        scale = scales_T[qb, n]
                        byte_start = qb * BYTES_PER_QB
                        k_start = qb * QUANT_BLOCK

                        for byte_offset in range(BYTES_PER_QB):
                            byte_idx = byte_start + byte_offset
                            packed = W_packed_T[byte_idx, n]

                            k_lo = k_start + byte_offset * 2
                            k_hi = k_lo + 1

                            int4_lo = packed & T.uint8(0xF)
                            int4_hi = (packed >> 4) & T.uint8(0xF)

                            w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                            w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale

                            C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_lo] * w_lo)
                            C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_hi] * w_hi)

    return gemv


def create_transposed_gemv_v3(N, K, THREADS=256, TILE_N=256, TILE_K=64):
    """
    V3: Tiled transposed GEMV with prefetching.

    Process N in tiles of TILE_N, K in tiles of TILE_K.
    This improves cache utilization.
    """
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + TILE_N - 1) // TILE_N
    BYTES_PER_QB = QUANT_BLOCK // 2

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed_T: T.Buffer((K_packed, N), "uint8"),
        scales_T: T.Buffer((num_scale_blocks, N), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_t_v3", "tir.noalias": True})

        # Shared memory for A tile
        A_tile = T.alloc_buffer((TILE_K,), "float16", scope="shared")

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * TILE_N + tid
                if n < N:
                    C[0, n] = T.float32(0)

            # Process K in tiles
            for k_tile in range(K // TILE_K):
                k_tile_start = k_tile * TILE_K

                # Load A tile to shared memory
                for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                    for i in range((TILE_K + THREADS - 1) // THREADS):
                        k_local = tid + i * THREADS
                        if k_local < TILE_K:
                            A_tile[k_local] = A[0, k_tile_start + k_local]

                T.tvm_storage_sync("shared")

                # Process this K tile
                for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                    n = block_idx * TILE_N + tid
                    if n < N:
                        # How many quant blocks in this K tile?
                        qb_start = k_tile_start // QUANT_BLOCK
                        qb_count = TILE_K // QUANT_BLOCK

                        for qb_offset in range(qb_count):
                            qb = qb_start + qb_offset
                            scale = scales_T[qb, n]

                            local_k_start = qb_offset * QUANT_BLOCK
                            byte_start = (k_tile_start + local_k_start) // 2

                            for byte_offset in range(BYTES_PER_QB):
                                byte_idx = byte_start + byte_offset
                                packed = W_packed_T[byte_idx, n]

                                k_lo = local_k_start + byte_offset * 2
                                k_hi = k_lo + 1

                                int4_lo = packed & T.uint8(0xF)
                                int4_hi = (packed >> 4) & T.uint8(0xF)

                                w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                                w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale

                                C[0, n] = C[0, n] + T.Cast("float32", A_tile[k_lo] * w_lo)
                                C[0, n] = C[0, n] + T.Cast("float32", A_tile[k_hi] * w_hi)

                T.tvm_storage_sync("shared")

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


def benchmark_transposed_kernel(kernel_fn, name, N=16384, K=2048, warmup=50, runs=200):
    """Benchmark transposed GEMV kernel."""
    print(f"\n--- {name} ---")

    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)
    W_packed_np, scales_np = quantize_int4(W_np)

    # Transpose for kernel
    W_packed_T = W_packed_np.T.copy()
    scales_T = scales_np.T.copy()

    # Reference
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

    func_names = ["gemv_t_v1", "gemv_t_v2", "gemv_t_v3", "main"]
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
    W_packed_T_tvm = runtime.empty(W_packed_T.shape, "uint8", device)
    W_packed_T_tvm.copyfrom(W_packed_T)
    scales_T_tvm = runtime.empty(scales_T.shape, "float16", device)
    scales_T_tvm.copyfrom(scales_T)
    C_tvm = runtime.empty((1, N), "float32", device)

    func = lib[func_name]
    func(A_tvm, W_packed_T_tvm, scales_T_tvm, C_tvm)
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
        func(A_tvm, W_packed_T_tvm, scales_T_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        func(A_tvm, W_packed_T_tvm, scales_T_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000

    # Memory
    weight_bytes = N * K // 2
    scale_bytes = N * (K // QUANT_BLOCK) * 2
    total_bytes = weight_bytes + scale_bytes + K * 2
    bandwidth = total_bytes / (avg_ms / 1000) / 1e9

    print(f"Time: {avg_ms:.4f} ms")
    print(f"Bandwidth: {bandwidth:.1f} GB/s")
    print(f"DRAM theoretical: {total_bytes / (55e9) * 1000:.4f} ms")
    print(f"Efficiency vs DRAM: {total_bytes / (55e9) * 1000 / avg_ms * 100:.1f}%")

    return avg_ms


def main():
    N, K = 16384, 2048

    print("="*60)
    print("W4A16 Transposed GEMV Benchmark")
    print(f"N={N}, K={K}")
    print("Target: < 0.2ms")
    print("="*60)

    weight_bytes = N * K // 2
    scale_bytes = N * (K // QUANT_BLOCK) * 2
    total_bytes = weight_bytes + scale_bytes + K * 2

    print(f"\nMemory: {total_bytes / 1e6:.2f} MB")
    print(f"DRAM theoretical (55 GB/s): {total_bytes / (55e9) * 1000:.4f} ms")
    print(f"L2 theoretical (230 GB/s): {total_bytes / (230e9) * 1000:.4f} ms")

    results = {}

    kernels = [
        (create_transposed_gemv_v1, "V1: Transposed Basic"),
        (create_transposed_gemv_v2, "V2: Transposed + Shared A"),
        (create_transposed_gemv_v3, "V3: Tiled Transposed"),
    ]

    for kernel_fn, name in kernels:
        ms = benchmark_transposed_kernel(kernel_fn, name, N, K)
        if ms is not None:
            results[name] = ms

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if results:
        print(f"{'Kernel':<25} | {'Time (ms)':<12} | {'BW (GB/s)':<10} | {'vs 0.2ms':<10}")
        print("-"*65)
        for name, ms in sorted(results.items(), key=lambda x: x[1]):
            bw = total_bytes / (ms / 1000) / 1e9
            ratio = ms / 0.2
            status = "OK" if ratio <= 1.0 else f"{ratio:.1f}x"
            print(f"{name:<25} | {ms:<12.4f} | {bw:<10.1f} | {status:<10}")

        print(f"\nBest: {min(results.values()):.4f} ms")

        # Compare with non-transposed
        print("\nCompare with original layout (Shared A): 0.7468 ms")
        best = min(results.values())
        if best < 0.7468:
            print(f"Transposed is {0.7468 / best:.2f}x FASTER")
        else:
            print(f"Transposed is {best / 0.7468:.2f}x SLOWER")


if __name__ == "__main__":
    main()
