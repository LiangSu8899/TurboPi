#!/usr/bin/env python3
"""
W4A16 GEMV - Optimized for Thor (SM110)

Target: < 0.2ms for decode (M=1, N=16384, K=2048)

Key insight: TIR Script accumulation must use buffer writes, not variable reassignment.

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


def create_baseline_gemv(N, K, THREADS=256):
    """Baseline: simple GEMV with direct buffer accumulation."""
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
        T.func_attr({"global_symbol": "gemv_baseline", "tir.noalias": True})

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
                        scale_idx = k // QUANT_BLOCK
                        scale = scales[n, scale_idx]
                        signed_val = T.Cast("float16", int4_val) - T.float16(8.0)
                        w = signed_val * scale
                        a = A[0, k]
                        C[0, n] = C[0, n] + T.Cast("float32", a * w)

    return gemv


def create_qblock_gemv(N, K, THREADS=256):
    """
    Process K by quant blocks for scale hoisting.
    This reduces scale loads from K to K/32.
    """
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
        T.func_attr({"global_symbol": "gemv_qblock", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid
                if n < N:
                    C[0, n] = T.float32(0)

                    # Process K in quant blocks
                    for qb in range(num_scale_blocks):
                        scale = scales[n, qb]
                        k_start = qb * QUANT_BLOCK

                        # Process 32 elements with same scale
                        for local_k in range(QUANT_BLOCK):
                            k = k_start + local_k
                            byte_idx = k // 2
                            is_high = k % 2
                            packed = W_packed[n, byte_idx]
                            int4_val = T.if_then_else(
                                is_high == 0,
                                packed & T.uint8(0xF),
                                (packed >> 4) & T.uint8(0xF)
                            )
                            signed_val = T.Cast("float16", int4_val) - T.float16(8.0)
                            w = signed_val * scale
                            a = A[0, k]
                            C[0, n] = C[0, n] + T.Cast("float32", a * w)

    return gemv


def create_byte_unroll_gemv(N, K, THREADS=256):
    """
    Process 2 INT4 per byte explicitly to avoid T.if_then_else.
    """
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS
    BYTES_PER_QB = QUANT_BLOCK // 2

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_byte_unroll", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid
                if n < N:
                    C[0, n] = T.float32(0)

                    for qb in range(num_scale_blocks):
                        scale = scales[n, qb]
                        byte_start = qb * BYTES_PER_QB
                        k_start = qb * QUANT_BLOCK

                        # Process 16 bytes = 32 INT4
                        for byte_offset in range(BYTES_PER_QB):
                            byte_idx = byte_start + byte_offset
                            packed = W_packed[n, byte_idx]

                            # Low nibble
                            k_lo = k_start + byte_offset * 2
                            int4_lo = packed & T.uint8(0xF)
                            w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                            C[0, n] = C[0, n] + T.Cast("float32", A[0, k_lo] * w_lo)

                            # High nibble
                            k_hi = k_lo + 1
                            int4_hi = (packed >> 4) & T.uint8(0xF)
                            w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale
                            C[0, n] = C[0, n] + T.Cast("float32", A[0, k_hi] * w_hi)

    return gemv


def create_shared_a_gemv(N, K, THREADS=256):
    """
    Load A to shared memory for L1 cache hits.
    """
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS
    BYTES_PER_QB = QUANT_BLOCK // 2

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_shared_a", "tir.noalias": True})

        A_shared = T.alloc_buffer((K,), "float16", scope="shared")

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            # Cooperative load A
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
                        scale = scales[n, qb]
                        byte_start = qb * BYTES_PER_QB
                        k_start = qb * QUANT_BLOCK

                        for byte_offset in range(BYTES_PER_QB):
                            byte_idx = byte_start + byte_offset
                            packed = W_packed[n, byte_idx]

                            k_lo = k_start + byte_offset * 2
                            k_hi = k_lo + 1

                            int4_lo = packed & T.uint8(0xF)
                            int4_hi = (packed >> 4) & T.uint8(0xF)

                            w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                            w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale

                            C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_lo] * w_lo)
                            C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_hi] * w_hi)

    return gemv


def create_warp_reduce_gemv(N, K, WARPS_PER_BLOCK=8):
    """
    Warp-cooperative GEMV: each warp computes one output.
    32 threads split K reduction, then reduce via shared memory.
    """
    WARP_SIZE = 32
    THREADS = WARPS_PER_BLOCK * WARP_SIZE
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    K_PER_THREAD = K // WARP_SIZE  # 64 for K=2048
    BYTES_PER_THREAD = K_packed // WARP_SIZE  # 32 bytes
    QB_PER_THREAD = K_PER_THREAD // QUANT_BLOCK  # 2

    num_blocks = (N + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_warp", "tir.noalias": True})

        warp_sums = T.alloc_buffer((WARPS_PER_BLOCK, WARP_SIZE), "float32", scope="shared")

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                warp_id = tid // WARP_SIZE
                lane_id = tid % WARP_SIZE

                n = block_idx * WARPS_PER_BLOCK + warp_id

                # Initialize partial sum
                warp_sums[warp_id, lane_id] = T.float32(0)

                if n < N:
                    # Thread's portion of K
                    k_thread_start = lane_id * K_PER_THREAD

                    for qb_offset in range(QB_PER_THREAD):
                        qb = (k_thread_start // QUANT_BLOCK) + qb_offset
                        scale = scales[n, qb]

                        qb_k_start = k_thread_start + qb_offset * QUANT_BLOCK
                        qb_byte_start = (qb_k_start // 2)

                        for byte_offset in range(QUANT_BLOCK // 2):
                            byte_idx = qb_byte_start + byte_offset
                            packed = W_packed[n, byte_idx]

                            k_lo = qb_k_start + byte_offset * 2
                            k_hi = k_lo + 1

                            int4_lo = packed & T.uint8(0xF)
                            int4_hi = (packed >> 4) & T.uint8(0xF)

                            w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                            w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale

                            warp_sums[warp_id, lane_id] = warp_sums[warp_id, lane_id] + T.Cast("float32", A[0, k_lo] * w_lo)
                            warp_sums[warp_id, lane_id] = warp_sums[warp_id, lane_id] + T.Cast("float32", A[0, k_hi] * w_hi)

                T.tvm_storage_sync("shared")

                # Reduction: lane 0 sums all
                if n < N and lane_id == 0:
                    total = T.float32(0)
                    for i in range(WARP_SIZE):
                        total = total + warp_sums[warp_id, i]
                    C[0, n] = total

    return gemv


def create_multi_output_gemv(N, K, THREADS=64, OUTPUTS_PER_THREAD=8):
    """
    Each thread handles multiple outputs.
    Reduces thread count, increases work per thread.
    """
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    OUTPUTS_PER_BLOCK = THREADS * OUTPUTS_PER_THREAD
    num_blocks = (N + OUTPUTS_PER_BLOCK - 1) // OUTPUTS_PER_BLOCK
    BYTES_PER_QB = QUANT_BLOCK // 2

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_multi", "tir.noalias": True})

        # Shared buffer for partial results
        results = T.alloc_buffer((OUTPUTS_PER_BLOCK,), "float32", scope="shared")

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            # Init all results to 0
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                for out_idx in range(OUTPUTS_PER_THREAD):
                    local_n = tid * OUTPUTS_PER_THREAD + out_idx
                    results[local_n] = T.float32(0)

            T.tvm_storage_sync("shared")

            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                for out_idx in range(OUTPUTS_PER_THREAD):
                    local_n = tid * OUTPUTS_PER_THREAD + out_idx
                    n = block_idx * OUTPUTS_PER_BLOCK + local_n

                    if n < N:
                        for qb in range(num_scale_blocks):
                            scale = scales[n, qb]
                            byte_start = qb * BYTES_PER_QB
                            k_start = qb * QUANT_BLOCK

                            for byte_offset in range(BYTES_PER_QB):
                                byte_idx = byte_start + byte_offset
                                packed = W_packed[n, byte_idx]

                                k_lo = k_start + byte_offset * 2
                                k_hi = k_lo + 1

                                int4_lo = packed & T.uint8(0xF)
                                int4_hi = (packed >> 4) & T.uint8(0xF)

                                w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                                w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale

                                results[local_n] = results[local_n] + T.Cast("float32", A[0, k_lo] * w_lo)
                                results[local_n] = results[local_n] + T.Cast("float32", A[0, k_hi] * w_hi)

            T.tvm_storage_sync("shared")

            # Write back
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                for out_idx in range(OUTPUTS_PER_THREAD):
                    local_n = tid * OUTPUTS_PER_THREAD + out_idx
                    n = block_idx * OUTPUTS_PER_BLOCK + local_n
                    if n < N:
                        C[0, n] = results[local_n]

    return gemv


# ============= Quantization helpers =============

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


def benchmark_kernel(kernel_fn, kernel_name, N=16384, K=2048, warmup=50, runs=200):
    """Benchmark a GEMV kernel."""
    print(f"\n--- {kernel_name} ---")

    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)
    W_packed_np, scales_np = quantize_int4(W_np)
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
        return None

    func_names = ["gemv_baseline", "gemv_qblock", "gemv_byte_unroll",
                  "gemv_shared_a", "gemv_warp", "gemv_multi", "main"]
    func_name = None
    for name in func_names:
        try:
            func = lib[name]
            func_name = name
            break
        except:
            continue

    if func_name is None:
        print("No kernel function found!")
        return None

    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = runtime.empty((1, N), "float32", device)

    func = lib[func_name]
    func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    C_result = C_tvm.numpy()
    if np.isnan(C_result).any():
        print("NaN in output!")
        return None

    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)
    print(f"Cos sim: {cos_sim:.6f}")

    if cos_sim <= 0.99:
        print("FAIL: accuracy")
        return None

    for _ in range(warmup):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000

    weight_bytes = N * K // 2
    scale_bytes = N * (K // QUANT_BLOCK) * 2
    a_bytes = K * 2
    total_bytes = weight_bytes + scale_bytes + a_bytes
    bandwidth_gbps = total_bytes / (avg_ms / 1000) / 1e9

    print(f"Time: {avg_ms:.4f} ms")
    print(f"Bandwidth: {bandwidth_gbps:.1f} GB/s")

    l2_bw = 230
    dram_bw = 55
    dram_ms = total_bytes / (dram_bw * 1e9) * 1000
    print(f"DRAM theoretical: {dram_ms:.4f} ms")
    print(f"Efficiency vs DRAM: {dram_ms / avg_ms * 100:.1f}%")

    return avg_ms


def main():
    N, K = 16384, 2048

    print("="*60)
    print("W4A16 Optimized GEMV Benchmark")
    print(f"N={N}, K={K}")
    print("Target: < 0.2ms")
    print("="*60)

    weight_bytes = N * K // 2
    scale_bytes = N * (K // QUANT_BLOCK) * 2
    a_bytes = K * 2
    total_bytes = weight_bytes + scale_bytes + a_bytes

    print(f"\nMemory: {total_bytes / 1e6:.2f} MB")
    print(f"L2 theoretical (230 GB/s): {total_bytes / (230e9) * 1000:.4f} ms")
    print(f"DRAM theoretical (55 GB/s): {total_bytes / (55e9) * 1000:.4f} ms")

    results = {}

    kernels = [
        (create_baseline_gemv, "Baseline"),
        (create_qblock_gemv, "QBlock (scale hoist)"),
        (create_byte_unroll_gemv, "Byte Unroll"),
        (create_shared_a_gemv, "Shared A"),
        (create_warp_reduce_gemv, "Warp Reduce"),
        (create_multi_output_gemv, "Multi-Output (8)"),
    ]

    for kernel_fn, name in kernels:
        ms = benchmark_kernel(kernel_fn, name, N, K)
        if ms is not None:
            results[name] = ms

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


if __name__ == "__main__":
    main()
