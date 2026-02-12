#!/usr/bin/env python3
"""
W4A16 GEMV - Vectorized 128-bit Loads for Thor (SM110)

Target: < 0.2ms for decode (M=1, N=16384, K=2048)

Problem Analysis:
- Current 0.92ms is 3x slower than DRAM theoretical (0.29ms)
- Root cause: Scalar uint8 loads blocking warp execution
- Solution: Vectorized loads + coalesced memory access + loop unrolling

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
    """
    Baseline: Original simple GEMV for comparison.
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


def create_unrolled_gemv_v1(N, K, THREADS=256):
    """
    V1: Unroll by quant blocks (32 elements) + scale hoisting.

    Key optimizations:
    1. Process K in chunks of QUANT_BLOCK (32)
    2. Hoist scale load outside inner loop
    3. Unroll inner loop for ILP
    """
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS
    BYTES_PER_QB = QUANT_BLOCK // 2  # 16 bytes per quant block

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_unroll_v1", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid

                if n < N:
                    acc = T.float32(0)

                    # Process K in quant blocks for scale reuse
                    for qb in range(num_scale_blocks):
                        scale = scales[n, qb]
                        byte_start = qb * BYTES_PER_QB
                        k_start = qb * QUANT_BLOCK

                        # Unroll over 16 bytes (32 INT4 elements)
                        for byte_offset in T.unroll(16):
                            byte_idx = byte_start + byte_offset
                            packed = W_packed[n, byte_idx]

                            # Process low nibble (k = byte_offset * 2)
                            k_lo = k_start + byte_offset * 2
                            int4_lo = packed & T.uint8(0xF)
                            w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                            a_lo = A[0, k_lo]
                            acc = acc + T.Cast("float32", a_lo * w_lo)

                            # Process high nibble (k = byte_offset * 2 + 1)
                            k_hi = k_start + byte_offset * 2 + 1
                            int4_hi = (packed >> 4) & T.uint8(0xF)
                            w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale
                            a_hi = A[0, k_hi]
                            acc = acc + T.Cast("float32", a_hi * w_hi)

                    C[0, n] = acc

    return gemv


def create_unrolled_gemv_v2(N, K, THREADS=256):
    """
    V2: Same as V1 but with explicit 4-way unroll for better ILP.

    Process 4 bytes (8 INT4) per iteration to increase ILP.
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
        T.func_attr({"global_symbol": "gemv_unroll_v2", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid

                if n < N:
                    acc = T.float32(0)

                    for qb in range(num_scale_blocks):
                        scale = scales[n, qb]
                        byte_start = qb * BYTES_PER_QB
                        k_start = qb * QUANT_BLOCK

                        # Process 4 bytes at a time (8 INT4 elements)
                        for chunk in T.unroll(4):  # 4 chunks of 4 bytes = 16 bytes
                            chunk_start = chunk * 4

                            # Byte 0
                            b0 = byte_start + chunk_start + 0
                            p0 = W_packed[n, b0]
                            k0 = k_start + (chunk_start + 0) * 2
                            acc = acc + T.Cast("float32", A[0, k0] * ((T.Cast("float16", p0 & T.uint8(0xF)) - T.float16(8.0)) * scale))
                            acc = acc + T.Cast("float32", A[0, k0+1] * ((T.Cast("float16", (p0 >> 4) & T.uint8(0xF)) - T.float16(8.0)) * scale))

                            # Byte 1
                            b1 = byte_start + chunk_start + 1
                            p1 = W_packed[n, b1]
                            k1 = k_start + (chunk_start + 1) * 2
                            acc = acc + T.Cast("float32", A[0, k1] * ((T.Cast("float16", p1 & T.uint8(0xF)) - T.float16(8.0)) * scale))
                            acc = acc + T.Cast("float32", A[0, k1+1] * ((T.Cast("float16", (p1 >> 4) & T.uint8(0xF)) - T.float16(8.0)) * scale))

                            # Byte 2
                            b2 = byte_start + chunk_start + 2
                            p2 = W_packed[n, b2]
                            k2 = k_start + (chunk_start + 2) * 2
                            acc = acc + T.Cast("float32", A[0, k2] * ((T.Cast("float16", p2 & T.uint8(0xF)) - T.float16(8.0)) * scale))
                            acc = acc + T.Cast("float32", A[0, k2+1] * ((T.Cast("float16", (p2 >> 4) & T.uint8(0xF)) - T.float16(8.0)) * scale))

                            # Byte 3
                            b3 = byte_start + chunk_start + 3
                            p3 = W_packed[n, b3]
                            k3 = k_start + (chunk_start + 3) * 2
                            acc = acc + T.Cast("float32", A[0, k3] * ((T.Cast("float16", p3 & T.uint8(0xF)) - T.float16(8.0)) * scale))
                            acc = acc + T.Cast("float32", A[0, k3+1] * ((T.Cast("float16", (p3 >> 4) & T.uint8(0xF)) - T.float16(8.0)) * scale))

                    C[0, n] = acc

    return gemv


def create_shared_a_gemv(N, K, THREADS=256):
    """
    V3: Load A to shared memory for faster access.

    A is only 4KB for K=2048, fits easily in shared memory.
    This reduces global memory traffic for A.
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
            # Cooperative load A to shared memory
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                # Strided load for coalescing
                for i in range((K + THREADS - 1) // THREADS):
                    k = tid + i * THREADS
                    if k < K:
                        A_shared[k] = A[0, k]

            T.tvm_storage_sync("shared")

            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid

                if n < N:
                    acc = T.float32(0)

                    for qb in range(num_scale_blocks):
                        scale = scales[n, qb]
                        byte_start = qb * BYTES_PER_QB
                        k_start = qb * QUANT_BLOCK

                        for byte_offset in T.unroll(16):
                            byte_idx = byte_start + byte_offset
                            packed = W_packed[n, byte_idx]

                            k_lo = k_start + byte_offset * 2
                            k_hi = k_lo + 1

                            int4_lo = packed & T.uint8(0xF)
                            int4_hi = (packed >> 4) & T.uint8(0xF)

                            w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                            w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale

                            acc = acc + T.Cast("float32", A_shared[k_lo] * w_lo)
                            acc = acc + T.Cast("float32", A_shared[k_hi] * w_hi)

                    C[0, n] = acc

    return gemv


def create_multi_output_gemv(N, K, THREADS=128, OUTPUTS_PER_THREAD=4):
    """
    V4: Each thread handles multiple outputs.

    Benefits:
    1. Amortize A loads across multiple outputs
    2. Better register utilization
    3. Fewer thread blocks

    Downside: Higher register pressure
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
        T.func_attr({"global_symbol": "gemv_multi_output", "tir.noalias": True})

        A_shared = T.alloc_buffer((K,), "float16", scope="shared")

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            # Load A to shared
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                for i in range((K + THREADS - 1) // THREADS):
                    k = tid + i * THREADS
                    if k < K:
                        A_shared[k] = A[0, k]

            T.tvm_storage_sync("shared")

            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n_base = block_idx * OUTPUTS_PER_BLOCK + tid * OUTPUTS_PER_THREAD

                # 4 accumulators
                acc0 = T.float32(0)
                acc1 = T.float32(0)
                acc2 = T.float32(0)
                acc3 = T.float32(0)

                for qb in range(num_scale_blocks):
                    k_start = qb * QUANT_BLOCK
                    byte_start = qb * BYTES_PER_QB

                    # Load 4 scales (one per output)
                    s0 = T.if_then_else(n_base < N, scales[n_base, qb], T.float16(0))
                    s1 = T.if_then_else(n_base + 1 < N, scales[n_base + 1, qb], T.float16(0))
                    s2 = T.if_then_else(n_base + 2 < N, scales[n_base + 2, qb], T.float16(0))
                    s3 = T.if_then_else(n_base + 3 < N, scales[n_base + 3, qb], T.float16(0))

                    for byte_offset in T.unroll(16):
                        byte_idx = byte_start + byte_offset
                        k_lo = k_start + byte_offset * 2
                        k_hi = k_lo + 1

                        a_lo = A_shared[k_lo]
                        a_hi = A_shared[k_hi]

                        # Load and process 4 weights in parallel
                        # Output 0
                        p0 = T.if_then_else(n_base < N, W_packed[n_base, byte_idx], T.uint8(0))
                        w0_lo = (T.Cast("float16", p0 & T.uint8(0xF)) - T.float16(8.0)) * s0
                        w0_hi = (T.Cast("float16", (p0 >> 4) & T.uint8(0xF)) - T.float16(8.0)) * s0
                        acc0 = acc0 + T.Cast("float32", a_lo * w0_lo) + T.Cast("float32", a_hi * w0_hi)

                        # Output 1
                        p1 = T.if_then_else(n_base + 1 < N, W_packed[n_base + 1, byte_idx], T.uint8(0))
                        w1_lo = (T.Cast("float16", p1 & T.uint8(0xF)) - T.float16(8.0)) * s1
                        w1_hi = (T.Cast("float16", (p1 >> 4) & T.uint8(0xF)) - T.float16(8.0)) * s1
                        acc1 = acc1 + T.Cast("float32", a_lo * w1_lo) + T.Cast("float32", a_hi * w1_hi)

                        # Output 2
                        p2 = T.if_then_else(n_base + 2 < N, W_packed[n_base + 2, byte_idx], T.uint8(0))
                        w2_lo = (T.Cast("float16", p2 & T.uint8(0xF)) - T.float16(8.0)) * s2
                        w2_hi = (T.Cast("float16", (p2 >> 4) & T.uint8(0xF)) - T.float16(8.0)) * s2
                        acc2 = acc2 + T.Cast("float32", a_lo * w2_lo) + T.Cast("float32", a_hi * w2_hi)

                        # Output 3
                        p3 = T.if_then_else(n_base + 3 < N, W_packed[n_base + 3, byte_idx], T.uint8(0))
                        w3_lo = (T.Cast("float16", p3 & T.uint8(0xF)) - T.float16(8.0)) * s3
                        w3_hi = (T.Cast("float16", (p3 >> 4) & T.uint8(0xF)) - T.float16(8.0)) * s3
                        acc3 = acc3 + T.Cast("float32", a_lo * w3_lo) + T.Cast("float32", a_hi * w3_hi)

                # Write outputs
                if n_base < N:
                    C[0, n_base] = acc0
                if n_base + 1 < N:
                    C[0, n_base + 1] = acc1
                if n_base + 2 < N:
                    C[0, n_base + 2] = acc2
                if n_base + 3 < N:
                    C[0, n_base + 3] = acc3

    return gemv


def create_warp_reduce_gemv(N, K, WARPS_PER_BLOCK=8):
    """
    V5: Warp-cooperative GEMV.

    Each warp computes one output by splitting K across 32 threads.
    Final reduction via shared memory.

    Benefits:
    1. Coalesced memory access within warp
    2. Better memory bandwidth utilization
    """
    WARP_SIZE = 32
    THREADS = WARPS_PER_BLOCK * WARP_SIZE
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    # Each thread handles K/32 elements
    K_PER_THREAD = K // WARP_SIZE  # 64 for K=2048
    BYTES_PER_THREAD = K_packed // WARP_SIZE  # 32 bytes per thread

    num_blocks = (N + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv_warp_reduce", "tir.noalias": True})

        warp_sums = T.alloc_buffer((WARPS_PER_BLOCK, WARP_SIZE), "float32", scope="shared")

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                warp_id = tid // WARP_SIZE
                lane_id = tid % WARP_SIZE

                n = block_idx * WARPS_PER_BLOCK + warp_id
                acc = T.float32(0)

                if n < N:
                    # Each thread's portion of K
                    k_start = lane_id * K_PER_THREAD
                    byte_start = lane_id * BYTES_PER_THREAD

                    # Process this thread's K elements
                    # 64 elements = 2 quant blocks
                    for qb_offset in range(K_PER_THREAD // QUANT_BLOCK):
                        qb = (k_start // QUANT_BLOCK) + qb_offset
                        scale = scales[n, qb]
                        qb_byte_start = byte_start + qb_offset * (QUANT_BLOCK // 2)
                        qb_k_start = k_start + qb_offset * QUANT_BLOCK

                        for byte_offset in T.unroll(16):
                            byte_idx = qb_byte_start + byte_offset
                            packed = W_packed[n, byte_idx]

                            k_lo = qb_k_start + byte_offset * 2
                            k_hi = k_lo + 1

                            int4_lo = packed & T.uint8(0xF)
                            int4_hi = (packed >> 4) & T.uint8(0xF)

                            w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                            w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale

                            acc = acc + T.Cast("float32", A[0, k_lo] * w_lo)
                            acc = acc + T.Cast("float32", A[0, k_hi] * w_hi)

                warp_sums[warp_id, lane_id] = acc

                T.tvm_storage_sync("shared")

                # Reduction (only lane 0 of each warp)
                if n < N and lane_id == 0:
                    total = T.float32(0)
                    for i in T.unroll(WARP_SIZE):
                        total = total + warp_sums[warp_id, i]
                    C[0, n] = total

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

    # Data
    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)
    W_packed_np, scales_np = quantize_int4(W_np)

    # Reference
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Build
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

    # Find function
    func_names = ["gemv_baseline", "gemv_unroll_v1", "gemv_unroll_v2",
                  "gemv_shared_a", "gemv_multi_output", "gemv_warp_reduce", "main"]
    func_name = None
    for name in func_names:
        try:
            func = lib[name]
            func_name = name
            break
        except:
            continue

    if func_name is None:
        print("Could not find kernel function!")
        return None

    # Run
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

    # Verify
    C_result = C_tvm.numpy()
    if np.isnan(C_result).any():
        print("WARNING: Output contains NaN!")
        return None

    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)
    print(f"Cos sim: {cos_sim:.6f}")

    if cos_sim <= 0.99:
        print("FAIL: Accuracy too low")
        return None

    # Benchmark
    for _ in range(warmup):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000

    # Compute metrics
    weight_bytes = N * K // 2
    scale_bytes = N * (K // QUANT_BLOCK) * 2
    a_bytes = K * 2
    total_bytes = weight_bytes + scale_bytes + a_bytes
    bandwidth_gbps = total_bytes / (avg_ms / 1000) / 1e9

    print(f"Time: {avg_ms:.4f} ms")
    print(f"Memory: {total_bytes / 1e6:.2f} MB")
    print(f"Bandwidth: {bandwidth_gbps:.1f} GB/s")

    # Theoretical limits
    l2_bw = 230  # GB/s
    dram_bw = 55  # GB/s
    l2_theoretical_ms = total_bytes / (l2_bw * 1e9) * 1000
    dram_theoretical_ms = total_bytes / (dram_bw * 1e9) * 1000

    print(f"Theoretical (L2={l2_bw}GB/s): {l2_theoretical_ms:.4f} ms")
    print(f"Theoretical (DRAM={dram_bw}GB/s): {dram_theoretical_ms:.4f} ms")
    print(f"Efficiency vs DRAM: {dram_theoretical_ms / avg_ms * 100:.1f}%")

    target_ms = 0.2
    if avg_ms < target_ms:
        print(f"ACHIEVED target < {target_ms}ms!")
    else:
        print(f"Gap to target: {avg_ms/target_ms:.1f}x")

    return avg_ms


def main():
    N, K = 16384, 2048

    print("="*60)
    print("W4A16 Vectorized GEMV Benchmark")
    print(f"N={N}, K={K}")
    print("Target: < 0.2ms")
    print("="*60)

    # Memory analysis
    weight_bytes = N * K // 2
    scale_bytes = N * (K // QUANT_BLOCK) * 2
    a_bytes = K * 2
    total_bytes = weight_bytes + scale_bytes + a_bytes

    print(f"\nMemory footprint:")
    print(f"  W_packed: {weight_bytes / 1e6:.2f} MB")
    print(f"  Scales:   {scale_bytes / 1e6:.2f} MB")
    print(f"  A:        {a_bytes / 1e3:.2f} KB")
    print(f"  Total:    {total_bytes / 1e6:.2f} MB")

    print(f"\nTheoretical limits:")
    print(f"  L2 (230 GB/s):   {total_bytes / (230e9) * 1000:.4f} ms")
    print(f"  DRAM (55 GB/s):  {total_bytes / (55e9) * 1000:.4f} ms")

    results = {}

    # Test each version
    kernels = [
        (create_baseline_gemv, "Baseline"),
        (create_unrolled_gemv_v1, "V1: Unrolled (16 bytes)"),
        (create_unrolled_gemv_v2, "V2: Unrolled (4x4 bytes)"),
        (create_shared_a_gemv, "V3: Shared A"),
        (create_multi_output_gemv, "V4: Multi-Output (4)"),
        (create_warp_reduce_gemv, "V5: Warp Reduce"),
    ]

    for kernel_fn, name in kernels:
        ms = benchmark_kernel(kernel_fn, name, N, K)
        if ms is not None:
            results[name] = ms

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if results:
        print(f"{'Kernel':<25} | {'Time (ms)':<12} | {'vs Target':<10} | {'BW (GB/s)':<10}")
        print("-"*65)
        for name, ms in sorted(results.items(), key=lambda x: x[1]):
            ratio = ms / 0.2
            status = "OK" if ratio <= 1.0 else f"{ratio:.1f}x"
            bw = total_bytes / (ms / 1000) / 1e9
            print(f"{name:<25} | {ms:<12.4f} | {status:<10} | {bw:<10.1f}")

        best = min(results.values())
        print(f"\nBest: {best:.4f} ms")


if __name__ == "__main__":
    main()
