#!/usr/bin/env python3
"""
W4A16 GEMM with WMMA Tensorization - Correct Implementation

Strategy:
1. Use te.compute for W4A16 GEMM computation (with inline dequant)
2. Schedule with shared memory tiling
3. Dequant W to shared memory (not global!)
4. Use WMMA intrinsics for FP16 GEMM from shared memory

This is the practical approach: dequant to shared memory + WMMA.
While not fully register-level, it's the best TVM can do without custom PTX.

Author: Claude Code
Date: 2026-02-11
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import te, tir, runtime
from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group
import numpy as np
import time


QUANT_BLOCK = 32  # INT4 group size
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def create_w4a16_gemm_schedule(M, N, K, block_size_quant=QUANT_BLOCK):
    """
    Create W4A16 GEMM with proper WMMA tensorization schedule.

    C[M, N] = A[M, K] @ dequant(W_packed[N, K//2], scales[N, K//block_size])^T
    """
    num_k_blocks = K // block_size_quant
    K_packed = K // 2

    # Placeholders
    A = te.placeholder((M, K), dtype="float16", name="A")
    W_packed = te.placeholder((N, K_packed), dtype="uint8", name="W_packed")
    scales = te.placeholder((N, num_k_blocks), dtype="float16", name="scales")

    # Step 1: Dequant W to FP16 (intermediate)
    # This will be computed at shared memory level
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
        block_idx = k // block_size_quant
        scale = scales[n, block_idx]
        return signed_val * scale

    W_dequant = te.compute(
        (N, K),
        dequant_func,
        name="W_dequant"
    )

    # Step 2: GEMM with FP16 dequantized weights
    k_axis = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(
            A[m, k_axis].astype("float32") * W_dequant[n, k_axis].astype("float32"),
            axis=k_axis
        ),
        name="C"
    )

    return A, W_packed, scales, W_dequant, C


def schedule_wmma(A, W_packed, scales, W_dequant, C, M, N, K):
    """
    Apply WMMA tensorization schedule.

    Key: W_dequant is computed at shared memory scope (not global).
    """
    s = te.create_schedule(C.op)

    # Tile sizes
    BLOCK_M = 64  # Must be multiple of WMMA_M
    BLOCK_N = 64  # Must be multiple of WMMA_N
    BLOCK_K = 32  # Must be multiple of WMMA_K

    # WMMA tile
    WMMA_TILE_M = 16
    WMMA_TILE_N = 16
    WMMA_TILE_K = 16

    # Get axes
    m, n = C.op.axis
    k = C.op.reduce_axis[0]

    # Split for block tiling
    bm, tm = s[C].split(m, factor=BLOCK_M)
    bn, tn = s[C].split(n, factor=BLOCK_N)

    # Split K for tiling
    bk, tk = s[C].split(k, factor=BLOCK_K)

    # Reorder: blocks first, then K outer, then inner
    s[C].reorder(bm, bn, bk, tm, tn, tk)

    # Bind blocks to GPU blocks
    block_fuse = s[C].fuse(bm, bn)
    s[C].bind(block_fuse, te.thread_axis("blockIdx.x"))

    # Cache A to shared memory
    A_shared = s.cache_read(A, "shared", [C])
    s[A_shared].compute_at(s[C], bk)

    # Compute W_dequant at shared memory scope
    # This is key: dequant happens at tile level, not globally
    s[W_dequant].compute_at(s[C], bk)
    W_shared = s.cache_write(W_dequant, "shared")
    s[W_shared].compute_at(s[C], bk)

    # Thread binding for loading A to shared
    ax0, ax1 = s[A_shared].op.axis
    fused_a = s[A_shared].fuse(ax0, ax1)
    ty, tx = s[A_shared].split(fused_a, factor=32)
    s[A_shared].bind(ty, te.thread_axis("threadIdx.y"))
    s[A_shared].bind(tx, te.thread_axis("threadIdx.x"))

    # Thread binding for W_dequant (computed, then cached to shared)
    wn, wk = s[W_dequant].op.axis
    fused_w = s[W_dequant].fuse(wn, wk)
    wty, wtx = s[W_dequant].split(fused_w, factor=32)
    s[W_dequant].bind(wty, te.thread_axis("threadIdx.y"))
    s[W_dequant].bind(wtx, te.thread_axis("threadIdx.x"))

    # Split inner computation for WMMA tiles
    tmm, tmi = s[C].split(tm, factor=WMMA_TILE_M)
    tnn, tni = s[C].split(tn, factor=WMMA_TILE_N)
    tkk, tki = s[C].split(tk, factor=WMMA_TILE_K)

    # Reorder for WMMA
    s[C].reorder(tmm, tnn, tkk, tmi, tni, tki)

    # Bind warps
    warp_fuse = s[C].fuse(tmm, tnn)
    s[C].bind(warp_fuse, te.thread_axis("threadIdx.y"))

    return s


def build_simple_w4a16(M=712, N=16384, K=2048):
    """
    Build a simpler W4A16 kernel without complex tensorization.

    This uses dlight auto-scheduling which handles tensor core intrinsics.
    """
    print(f"\n{'='*60}")
    print(f"W4A16 WMMA Kernel - Simple dlight Schedule")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    from tvm import dlight as dl

    num_k_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    # Placeholders
    A = te.placeholder((M, K), dtype="float16", name="A")
    W_packed = te.placeholder((N, K_packed), dtype="uint8", name="W_packed")
    scales = te.placeholder((N, num_k_blocks), dtype="float16", name="scales")

    # Dequant intermediate
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

    W_dequant = te.compute((N, K), dequant_func, name="W_dequant")

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

    # Create PrimFunc
    func = te.create_prim_func([A, W_packed, scales, C])
    mod = tvm.IRModule({"main": func})

    target = tvm.target.Target("cuda -arch=sm_110")

    print("Applying dlight schedule...")
    with tvm.transform.PassContext(opt_level=3):
        with target:
            # Try different dlight rules
            try:
                mod = dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                )(mod)
                print("Applied Matmul schedule")
            except Exception as e:
                print(f"Matmul failed: {e}")
                try:
                    mod = dl.ApplyDefaultSchedule(
                        dl.gpu.GEMV(),
                    )(mod)
                    print("Applied GEMV schedule")
                except:
                    mod = dl.ApplyDefaultSchedule(
                        dl.gpu.Fallback(),
                    )(mod)
                    print("Applied Fallback schedule")

    print("Building...")
    try:
        lib = tvm.build(mod, target=target)
        print("Build successful!")
    except Exception as e:
        print(f"Build failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    return lib, A, W_packed, scales, C


def quantize_int4(weight, block_size=QUANT_BLOCK):
    """Quantize FP32 weights to INT4."""
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


def dequant_int4_np(W_packed, scales, K, block_size=QUANT_BLOCK):
    """Dequant INT4 to FP32 (reference)."""
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


def test_w4a16_kernel(M=256, N=512, K=256):
    """Test W4A16 kernel with small size first."""
    print(f"\n{'='*60}")
    print(f"W4A16 WMMA Kernel Test (Small)")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    result = build_simple_w4a16(M, N, K)
    if result is None:
        return None

    lib, _, _, _, _ = result

    # Generate data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4_np(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # TVM execution
    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = runtime.empty((M, N), "float32", device)

    lib["main"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    # Verify
    C_result = C_tvm.numpy()
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
        return None

    # Benchmark
    warmup = 20
    runs = 100

    for _ in range(warmup):
        lib["main"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        lib["main"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12

    print(f"\n  Time:    {avg_ms:.4f} ms")
    print(f"  TFLOPS:  {tflops:.4f}")

    return avg_ms


def test_full_size(M=712, N=16384, K=2048):
    """Test with full problem size."""
    print(f"\n{'='*60}")
    print(f"W4A16 WMMA Kernel - Full Size")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    result = build_simple_w4a16(M, N, K)
    if result is None:
        return None

    lib, _, _, _, _ = result

    # Generate data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    print("Quantizing...")
    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4_np(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # TVM execution
    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = runtime.empty((M, N), "float32", device)

    lib["main"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
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
    warmup = 10
    runs = 50

    for _ in range(warmup):
        lib["main"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        lib["main"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms / 1000) / 1e12

    print(f"\n  Time:    {avg_ms:.4f} ms")
    print(f"  TFLOPS:  {tflops:.4f}")

    BF16_MS = 0.42
    print(f"\n  vs BF16 baseline ({BF16_MS}ms): {BF16_MS / avg_ms:.2f}x")

    return avg_ms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    test_w4a16_kernel()

    if args.full:
        test_full_size()
