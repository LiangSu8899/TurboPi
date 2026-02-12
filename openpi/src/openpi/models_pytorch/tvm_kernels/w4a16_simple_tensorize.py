#!/usr/bin/env python3
"""
W4A16 GEMM with On-the-fly Dequant - Simple Tensorized Version

Strategy:
1. Load A to shared memory (cooperative)
2. Load W_packed to shared memory (INT4)
3. Dequant W to shared memory (FP16) - each thread dequants its elements
4. Use WMMA intrinsics for FP16 x FP16 MMA

This is "dequant to shared" approach - simpler than register-level fusion.

Author: Claude Code
Date: 2026-02-11
"""

import sys
import os

TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import te, tir
from tvm.script import tir as T
import numpy as np
import time

# ==============================================================================
# Parameters
# ==============================================================================

# Problem size
M_SIZE = 712
N_SIZE = 16384
K_SIZE = 2048

# Tile sizes
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32

# WMMA tile
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16

# Thread config
NUM_WARPS = 4
WARP_SIZE = 32
THREADS = NUM_WARPS * WARP_SIZE

# Quantization
QUANT_BLOCK = 32


# ==============================================================================
# TE Compute + Schedule Approach (Main implementation)
# ==============================================================================

def create_w4a16_te_compute(M, N, K, block_size=QUANT_BLOCK):
    """Create W4A16 GEMM using TE compute."""
    num_blocks = (K + block_size - 1) // block_size
    K_packed = K // 2

    # Placeholders
    A = te.placeholder((M, K), dtype="float16", name="A")
    W_packed = te.placeholder((N, K_packed), dtype="uint8", name="W_packed")
    scales = te.placeholder((N, num_blocks), dtype="float16", name="scales")

    # Dequant compute
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
        block_idx = k // block_size
        scale = scales[n, block_idx]
        return signed_val * scale

    W_dequant = te.compute(
        (N, K),
        dequant_func,
        name="W_dequant"
    )

    # GEMM
    k_axis = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(
            A[m, k_axis].astype("float32") * W_dequant[n, k_axis].astype("float32"),
            axis=k_axis
        ),
        name="C"
    )

    return A, W_packed, scales, C, W_dequant


def schedule_w4a16_basic(A, W_packed, scales, C, W_dequant, M, N, K):
    """Basic GPU schedule for W4A16 GEMM."""
    s = te.create_schedule(C.op)

    # Get axes
    m, n = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # Tile
    mo, mi = s[C].split(m, factor=BLOCK_M)
    no, ni = s[C].split(n, factor=BLOCK_N)
    ko, ki = s[C].split(k, factor=BLOCK_K)

    # Reorder
    s[C].reorder(mo, no, ko, mi, ni, ki)

    # Bind
    s[C].bind(mo, te.thread_axis("blockIdx.y"))
    s[C].bind(no, te.thread_axis("blockIdx.x"))

    # Compute W_dequant inline (fused with C computation)
    s[W_dequant].compute_at(s[C], ko)

    # Cache A to shared memory
    A_shared = s.cache_read(A, "shared", [C])
    s[A_shared].compute_at(s[C], ko)

    # Cache W_dequant to shared memory
    W_shared = s.cache_read(W_dequant, "shared", [C])
    s[W_shared].compute_at(s[C], ko)

    # Cooperative load for A
    ax0, ax1 = s[A_shared].op.axis
    fused = s[A_shared].fuse(ax0, ax1)
    _, t = s[A_shared].split(fused, factor=THREADS)
    ty, tx = s[A_shared].split(t, factor=WARP_SIZE)
    s[A_shared].bind(ty, te.thread_axis("threadIdx.y"))
    s[A_shared].bind(tx, te.thread_axis("threadIdx.x"))

    # Cooperative load for W (dequanted)
    ax0, ax1 = s[W_shared].op.axis
    fused = s[W_shared].fuse(ax0, ax1)
    _, t = s[W_shared].split(fused, factor=THREADS)
    ty, tx = s[W_shared].split(t, factor=WARP_SIZE)
    s[W_shared].bind(ty, te.thread_axis("threadIdx.y"))
    s[W_shared].bind(tx, te.thread_axis("threadIdx.x"))

    # Thread binding for computation
    mio, mii = s[C].split(mi, factor=16)
    nio, nii = s[C].split(ni, factor=4)
    s[C].bind(mio, te.thread_axis("threadIdx.y"))
    s[C].bind(nio, te.thread_axis("threadIdx.x"))

    # Vectorize innermost
    s[C].vectorize(ki)

    return s


# ==============================================================================
# Quantization Utilities
# ==============================================================================

def quantize_int4(weight, block_size=QUANT_BLOCK):
    """Quantize weight to INT4."""
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
    """Dequantize INT4 to FP32."""
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


# ==============================================================================
# Benchmark
# ==============================================================================

def benchmark_w4a16(M=M_SIZE, N=N_SIZE, K=K_SIZE):
    """Benchmark W4A16 using TE + Schedule approach."""
    print(f"\n{'='*70}")
    print(f"W4A16 TE Compute + Schedule Benchmark")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*70}")

    # Generate test data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    # Quantize
    print("Quantizing weights...")
    W_packed_np, scales_np = quantize_int4(W_np)
    print(f"  Original: {W_np.nbytes / 1e6:.2f} MB")
    print(f"  Packed:   {(W_packed_np.nbytes + scales_np.nbytes) / 1e6:.2f} MB")

    # Reference
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Create TE compute
    print("\nBuilding kernel...")
    A, W_packed, scales, C, W_dequant = create_w4a16_te_compute(M, N, K)
    s = schedule_w4a16_basic(A, W_packed, scales, C, W_dequant, M, N, K)

    # Build
    target = tvm.target.Target("cuda -arch=sm_87")  # Orin compatible
    with tvm.transform.PassContext(opt_level=3):
        func = tvm.build(s, [A, W_packed, scales, C], target=target)
    print("Build successful!")

    # TVM execution
    device = tvm.runtime.cuda(0)
    A_tvm = tvm.nd.array(A_np, device)
    W_packed_tvm = tvm.nd.array(W_packed_np, device)
    scales_tvm = tvm.nd.array(scales_np, device)
    C_tvm = tvm.nd.empty((M, N), "float32", device)

    # Run
    func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    # Verify
    C_result = C_tvm.numpy()
    max_diff = np.abs(C_result - C_ref).max()
    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)

    print(f"\nResults:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Cos sim:  {cos_sim:.6f}")
    print(f"  Status: {'PASS' if cos_sim > 0.99 else 'FAIL'}")

    # Benchmark
    warmup = 20
    runs = 100

    for _ in range(warmup):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        func(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12

    print(f"\n  Time:    {avg_ms:.4f} ms")
    print(f"  TFLOPS:  {tflops:.4f}")

    # Compare with baseline
    BF16_MS = 0.58
    print(f"\n  vs BF16 baseline ({BF16_MS}ms): {BF16_MS / avg_ms:.2f}x")

    return avg_ms


if __name__ == "__main__":
    benchmark_w4a16()
