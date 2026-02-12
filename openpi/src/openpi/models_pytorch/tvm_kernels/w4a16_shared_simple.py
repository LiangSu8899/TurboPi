#!/usr/bin/env python3
"""
W4A16 GEMM - Simple shared memory version

Simpler implementation for debugging:
1. One thread per output element
2. K-loop with tiling
3. Shared memory for A and W_packed tiles

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
BLOCK_K = 32


def create_w4a16_shared_kernel(M, N, K):
    """Create W4A16 kernel with shared memory K-tiling."""
    num_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_tiles_k = (K + BLOCK_K - 1) // BLOCK_K

    # Simple: one thread per output element
    # Use small tile for M/N to fit in thread block limits
    TILE_M = 16
    TILE_N = 16

    num_tiles_m = (M + TILE_M - 1) // TILE_M
    num_tiles_n = (N + TILE_N - 1) // TILE_N

    @T.prim_func
    def w4a16_shared(
        A: T.Buffer((M, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_blocks), "float16"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_shared", "tir.noalias": True})

        # Shared memory for A tile and W_packed tile
        A_shared = T.alloc_buffer((TILE_M, BLOCK_K), "float16", scope="shared")
        W_packed_shared = T.alloc_buffer((TILE_N, BLOCK_K // 2), "uint8", scope="shared")

        for tile_m in T.thread_binding(num_tiles_m, thread="blockIdx.y"):
            for tile_n in T.thread_binding(num_tiles_n, thread="blockIdx.x"):
                for tm in T.thread_binding(TILE_M, thread="threadIdx.y"):
                    for tn in T.thread_binding(TILE_N, thread="threadIdx.x"):
                        # Global indices
                        m = tile_m * TILE_M + tm
                        n = tile_n * TILE_N + tn

                        # Initialize output
                        if m < M and n < N:
                            C[m, n] = T.float32(0)

                        # Loop over K tiles
                        for tile_k in range(num_tiles_k):
                            k_start = tile_k * BLOCK_K

                            # Load A tile to shared memory
                            # Thread (tm, tn) loads A[m, k_start + tn] for multiple k values
                            for k_local in range(BLOCK_K // TILE_N):
                                k = k_start + tn + k_local * TILE_N
                                if m < M and k < K:
                                    A_shared[tm, tn + k_local * TILE_N] = A[m, k]
                                else:
                                    A_shared[tm, tn + k_local * TILE_N] = T.float16(0)

                            # Load W_packed tile to shared memory
                            # Thread (tm, tn) loads W_packed[n, k_packed_start + idx]
                            k_packed_start = k_start // 2
                            for k_local in range((BLOCK_K // 2) // TILE_N):
                                k_packed = k_packed_start + tn + k_local * TILE_N
                                if n < N and k_packed < K_packed:
                                    W_packed_shared[tm, tn + k_local * TILE_N] = W_packed[n, k_packed]
                                else:
                                    W_packed_shared[tm, tn + k_local * TILE_N] = T.uint8(0x88)

                            T.tvm_storage_sync("shared")

                            # Compute
                            if m < M and n < N:
                                block_idx = k_start // QUANT_BLOCK
                                scale = scales[n, block_idx] if block_idx < num_blocks else T.float16(1)

                                for k_local in range(BLOCK_K):
                                    k = k_start + k_local
                                    if k < K:
                                        # Load A from shared
                                        a_val = A_shared[tm, k_local]

                                        # Dequant W from shared
                                        byte_idx = k_local // 2
                                        is_high = k_local % 2
                                        packed = W_packed_shared[tn, byte_idx]

                                        int4_val = T.if_then_else(
                                            is_high == 0,
                                            packed & T.uint8(0xF),
                                            (packed >> 4) & T.uint8(0xF)
                                        )
                                        signed_val = T.Cast("float16", int4_val) - T.float16(8.0)
                                        w = signed_val * scale

                                        # Accumulate
                                        C[m, n] = C[m, n] + T.Cast("float32", a_val * w)

                            T.tvm_storage_sync("shared")

    return w4a16_shared


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


def test_shared_kernel(M=64, N=64, K=64):
    """Test simple shared memory kernel."""
    print(f"\n{'='*60}")
    print(f"W4A16 Simple Shared Memory Test")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    # Data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Build
    print("Building kernel...")
    kernel = create_w4a16_shared_kernel(M, N, K)
    mod = tvm.IRModule({"main": kernel})

    print("\n--- IR ---")
    # print(mod.script())

    target = tvm.target.Target("cuda -arch=sm_110")

    try:
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(mod, target=target)
        print("Build successful!")
    except Exception as e:
        print(f"Build failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run
    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = runtime.empty((M, N), "float32", device)

    func = lib["w4a16_shared"]
    func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    # Verify
    C_result = C_tvm.numpy()
    print(f"\nC_result range: [{C_result.min():.4f}, {C_result.max():.4f}]")
    print(f"C_ref range: [{C_ref.min():.4f}, {C_ref.max():.4f}]")

    max_diff = np.abs(C_result - C_ref).max()
    if np.isnan(C_result).any():
        print("WARNING: Output contains NaN!")
        cos_sim = 0
    else:
        cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
            np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)

    print(f"\nResults:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Cos sim:  {cos_sim:.6f}")
    print(f"  Status: {'PASS' if cos_sim > 0.99 else 'FAIL'}")

    if cos_sim <= 0.99:
        print(f"\n  Ref sample:\n{C_ref[:2, :4]}")
        print(f"  TVM sample:\n{C_result[:2, :4]}")


if __name__ == "__main__":
    test_shared_kernel(64, 64, 64)
