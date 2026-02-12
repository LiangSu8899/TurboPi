#!/usr/bin/env python3
"""
Standalone W4A16 TVM Kernel Test (no PyTorch custom op framework).

This script tests the core TVM kernel functionality without requiring
torch.library registration, which may not work in all environments.
"""

import sys
import os
import time
import numpy as np

# Add TVM to path
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import runtime
from tvm.script import tir as T

# Constants
N, K = 16384, 2048
QUANT_BLOCK = 32
num_scale_blocks = K // QUANT_BLOCK
THREADS = 256
num_blocks = (N + THREADS - 1) // THREADS


@T.prim_func
def gemv_vec128(
    A: T.Buffer((1, K), "float16"),
    W_packed: T.Buffer((num_scale_blocks, N, 4), "uint32"),
    scales_T: T.Buffer((num_scale_blocks, N), "float16"),
    C: T.Buffer((1, N), "float32"),
):
    T.func_attr({"global_symbol": "w4a16_gemv", "tir.noalias": True})

    A_shared = T.alloc_buffer((K,), "float16", scope="shared")
    W_local = T.alloc_buffer((4,), "uint32", scope="local")

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
                    scale = scales_T[qb, n]
                    k_base = qb * QUANT_BLOCK

                    for v in T.vectorized(4):
                        W_local[v] = W_packed[qb, n, v]

                    for u_idx in range(4):
                        u = W_local[u_idx]
                        k_offset = u_idx * 8

                        for i in range(8):
                            int4_val = (u >> T.uint32(i * 4)) & T.uint32(0xF)
                            k_idx = k_base + k_offset + i
                            w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                            C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_idx] * w)


def quantize_to_block_interleaved(W, block_size=QUANT_BLOCK):
    """Quantize weights to block-interleaved layout."""
    N_dim, K_dim = W.shape
    num_blocks_k = K_dim // block_size

    W_int4 = np.zeros((N_dim, K_dim), dtype=np.int8)
    scales = np.zeros((num_blocks_k, N_dim), dtype=np.float16)

    for n in range(N_dim):
        for b in range(num_blocks_k):
            start = b * block_size
            end = start + block_size
            block = W[n, start:end]

            max_abs = np.max(np.abs(block))
            scale = max_abs / 7.0 if max_abs > 0 else 1.0
            scales[b, n] = scale

            for k in range(block_size):
                val = block[k] / scale if scale > 0 else 0
                quantized = int(np.clip(np.round(val + 8), 0, 15))
                W_int4[n, start + k] = quantized

    W_packed = np.zeros((num_blocks_k, N_dim, 4), dtype=np.uint32)

    for n in range(N_dim):
        for qb in range(num_blocks_k):
            k_base = qb * block_size

            for u_idx in range(4):
                val = np.uint32(0)
                for i in range(8):
                    k = k_base + u_idx * 8 + i
                    int4_val = W_int4[n, k]
                    val |= np.uint32(int4_val) << np.uint32(i * 4)
                W_packed[qb, n, u_idx] = val

    return W_packed, scales


def dequantize(W_packed, scales, K_dim):
    """Dequantize for reference."""
    num_blocks_k, N_dim, _ = W_packed.shape
    W = np.zeros((N_dim, K_dim), dtype=np.float32)

    for n in range(N_dim):
        for qb in range(num_blocks_k):
            scale = float(scales[qb, n])
            k_base = qb * QUANT_BLOCK

            for u_idx in range(4):
                u = int(W_packed[qb, n, u_idx])
                for i in range(8):
                    int4_val = (u >> (i * 4)) & 0xF
                    k = k_base + u_idx * 8 + i
                    # Cast to signed int before subtracting
                    W[n, k] = (int(int4_val) - 8) * scale

    return W


def main():
    print("=" * 60)
    print("W4A16 Standalone TVM Kernel Test")
    print(f"N={N}, K={K}")
    print("=" * 60)

    # Build kernel
    print("\nBuilding TVM kernel...")
    mod = tvm.IRModule({"main": gemv_vec128})
    target = tvm.target.Target("cuda -arch=sm_110")

    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.build(mod, target=target)

    func = lib["w4a16_gemv"]
    print("Kernel built successfully!")

    # Prepare data
    print("\nPreparing test data...")
    np.random.seed(42)

    A_np = np.random.randn(1, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    W_packed, scales = quantize_to_block_interleaved(W_np)
    print(f"W_packed shape: {W_packed.shape}")
    print(f"scales shape: {scales.shape}")

    # Reference computation
    W_dequant = dequantize(W_packed, scales, K)
    C_ref = A_np.astype(np.float32) @ W_dequant.T

    # Create TVM tensors
    device = runtime.cuda(0)

    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)

    W_tvm = runtime.empty(W_packed.shape, "uint32", device)
    W_tvm.copyfrom(W_packed)

    scales_tvm = runtime.empty(scales.shape, "float16", device)
    scales_tvm.copyfrom(scales)

    C_tvm = runtime.empty((1, N), "float32", device)

    # Run kernel
    print("\nRunning kernel...")
    func(A_tvm, W_tvm, scales_tvm, C_tvm)
    device.sync()

    C_result = C_tvm.numpy()

    # Verify
    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Correctness: {'PASS' if cos_sim > 0.99 else 'FAIL'}")

    # Benchmark
    print("\n--- Benchmark ---")
    warmup, runs = 100, 500

    for _ in range(warmup):
        func(A_tvm, W_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        func(A_tvm, W_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000

    # Metrics
    weight_bytes = num_scale_blocks * N * 4 * 4  # uint32
    scale_bytes = num_scale_blocks * N * 2  # float16
    total_bytes = weight_bytes + scale_bytes + K * 2
    bandwidth = total_bytes / (avg_ms / 1000) / 1e9

    print(f"Average latency: {avg_ms:.4f} ms")
    print(f"Bandwidth: {bandwidth:.1f} GB/s")
    print(f"Target (< 0.2ms): {'ACHIEVED!' if avg_ms < 0.2 else f'{avg_ms/0.2:.1f}x slower'}")

    # Multi-run stability
    print("\n--- Stability Check (5 runs) ---")
    results = []
    for run in range(5):
        start = time.time()
        for _ in range(300):
            func(A_tvm, W_tvm, scales_tvm, C_tvm)
        device.sync()
        ms = (time.time() - start) / 300 * 1000
        results.append(ms)
        print(f"Run {run+1}: {ms:.4f} ms")

    print(f"\nMean: {np.mean(results):.4f} ms")
    print(f"Std:  {np.std(results):.4f} ms")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
