#!/usr/bin/env python3
"""
W4A16 GEMV - Direct TIR Script with Register-Level Dequant

For decode (M=1), this is a GEMV: C[1, N] = A[1, K] @ W[N, K]^T

Strategy:
- Each thread block handles a chunk of N outputs
- K reduction is done across threads with shared memory reduction
- Dequant happens in registers (local scope)

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


def create_w4a16_gemv_kernel(N, K, BLOCK_N=128, BLOCK_K=128, THREADS=256):
    """Create W4A16 GEMV kernel (M=1)."""
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    num_k_iters = (K + BLOCK_K - 1) // BLOCK_K

    # Each thread handles BLOCK_N / THREADS outputs per block
    OUTPUTS_PER_THREAD = max(1, BLOCK_N // THREADS)

    @T.prim_func
    def w4a16_gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_gemv", "tir.noalias": True})

        # Shared memory for reduction
        A_shared = T.alloc_buffer((BLOCK_K,), "float16", scope="shared")
        partial_sum = T.alloc_buffer((THREADS, OUTPUTS_PER_THREAD), "float32", scope="shared")

        for block_n in T.thread_binding(num_blocks_n, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                # Initialize partial sums
                for out_idx in range(OUTPUTS_PER_THREAD):
                    partial_sum[tid, out_idx] = T.float32(0)

                # Loop over K chunks
                for k_iter in range(num_k_iters):
                    k_start = k_iter * BLOCK_K

                    # Cooperative load of A to shared memory
                    for load_k in range(BLOCK_K // THREADS):
                        k = k_start + tid * (BLOCK_K // THREADS) + load_k
                        if k < K:
                            A_shared[tid * (BLOCK_K // THREADS) + load_k] = A[0, k]
                        else:
                            A_shared[tid * (BLOCK_K // THREADS) + load_k] = T.float16(0)

                    T.tvm_storage_sync("shared")

                    # Each thread computes its outputs
                    for out_idx in range(OUTPUTS_PER_THREAD):
                        n = block_n * BLOCK_N + tid * OUTPUTS_PER_THREAD + out_idx

                        if n < N:
                            # Accumulate over this K chunk
                            for local_k in range(BLOCK_K):
                                k = k_start + local_k
                                if k < K:
                                    # Dequant in register
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

                                    a = A_shared[local_k]
                                    partial_sum[tid, out_idx] = partial_sum[tid, out_idx] + T.Cast("float32", a * w)

                    T.tvm_storage_sync("shared")

                # Write results
                for out_idx in range(OUTPUTS_PER_THREAD):
                    n = block_n * BLOCK_N + tid * OUTPUTS_PER_THREAD + out_idx
                    if n < N:
                        C[0, n] = partial_sum[tid, out_idx]

    return w4a16_gemv


def create_w4a16_gemv_simple(N, K, THREADS=256):
    """Simpler W4A16 GEMV - one thread per output."""
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS

    @T.prim_func
    def w4a16_gemv_simple(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_gemv_simple", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid
                if n < N:
                    acc = T.float32(0)

                    # Accumulate over K
                    for k in range(K):
                        # Dequant
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
                        acc = acc + T.Cast("float32", a * w)

                    C[0, n] = acc

    return w4a16_gemv_simple


def create_w4a16_gemv_vectorized(N, K, THREADS=256, VEC_K=8):
    """Vectorized W4A16 GEMV - vector loads for K."""
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS

    @T.prim_func
    def w4a16_gemv_vec(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_gemv_vec", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid
                if n < N:
                    acc = T.float32(0)

                    # Loop over K in chunks of 32 (quantization block size)
                    for k_block in range(K // QUANT_BLOCK):
                        k_start = k_block * QUANT_BLOCK
                        scale = scales[n, k_block]

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
                            acc = acc + T.Cast("float32", a * w)

                    C[0, n] = acc

    return w4a16_gemv_vec


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


def test_gemv_kernel(kernel_fn, kernel_name, N=16384, K=2048):
    """Test a GEMV kernel."""
    print(f"\n--- {kernel_name} ---")

    # Data
    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    W_packed_np, scales_np = quantize_int4(W_np)
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

    # Run
    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = runtime.empty((1, N), "float32", device)

    # Get function name
    func_name = None
    for name in ["w4a16_gemv", "w4a16_gemv_simple", "w4a16_gemv_vec", "main"]:
        try:
            func = lib[name]
            func_name = name
            break
        except:
            continue

    if func_name is None:
        print("Could not find kernel function!")
        return None

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
        print("FAIL")
        return None

    # Benchmark
    warmup = 50
    runs = 200

    for _ in range(warmup):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000
    flops = 2.0 * N * K
    gflops = flops / (avg_ms / 1000) / 1e9

    print(f"Time: {avg_ms:.4f} ms")
    print(f"GFLOPS: {gflops:.2f}")

    return avg_ms


def main():
    print(f"\n{'='*60}")
    print(f"W4A16 TIR GEMV Kernels (M=1)")
    print(f"N=16384, K=2048")
    print(f"{'='*60}")

    results = {}

    # Test simple kernel
    ms = test_gemv_kernel(create_w4a16_gemv_simple, "Simple GEMV (1 thread/output)")
    if ms is not None:
        results["Simple"] = ms

    # Test vectorized kernel
    ms = test_gemv_kernel(create_w4a16_gemv_vectorized, "Vectorized GEMV (scale-aligned)")
    if ms is not None:
        results["Vectorized"] = ms

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")

    if results:
        best_name = min(results, key=results.get)
        best_ms = results[best_name]

        for name, ms in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name:20s}: {ms:.4f} ms")

        target_ms = 1.0
        if best_ms < target_ms:
            print(f"\n  ACHIEVED target of < {target_ms}ms!")
        else:
            print(f"\n  Need {best_ms/target_ms:.1f}x speedup to reach {target_ms}ms target")


if __name__ == "__main__":
    main()
