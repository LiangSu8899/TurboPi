#!/usr/bin/env python3
"""
W4A16 GEMV using Tensor Expression + Schedule for proper vectorization.

Target: < 0.2ms for decode (M=1, N=16384, K=2048)

Key insight: TIR Script range loops don't get unrolled by TVM.
Using TE + schedule gives more control over loop transformations.

Author: Claude Code
Date: 2026-02-11
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import te, tir, runtime
import numpy as np
import time


QUANT_BLOCK = 32


def create_te_gemv_basic(N, K):
    """Basic TE compute definition."""
    K_packed = K // 2
    num_scale_blocks = K // QUANT_BLOCK

    A = te.placeholder((1, K), dtype="float16", name="A")
    W_packed = te.placeholder((N, K_packed), dtype="uint8", name="W_packed")
    scales = te.placeholder((N, num_scale_blocks), dtype="float16", name="scales")

    # Dequantize compute
    def dequant_fn(n, k):
        byte_idx = k // 2
        is_high = k % 2
        packed = W_packed[n, byte_idx]
        int4_val = tir.if_then_else(
            is_high == 0,
            packed & tir.const(0xF, "uint8"),
            (packed >> 4) & tir.const(0xF, "uint8")
        )
        scale_idx = k // QUANT_BLOCK
        scale = scales[n, scale_idx]
        return (int4_val.astype("float16") - tir.const(8.0, "float16")) * scale

    W_dequant = te.compute((N, K), dequant_fn, name="dequant")

    # GEMV compute
    k_axis = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (1, N),
        lambda m, n: te.sum(
            A[m, k_axis].astype("float32") * W_dequant[n, k_axis].astype("float32"),
            axis=k_axis
        ),
        name="C"
    )

    return A, W_packed, scales, C, W_dequant


def schedule_gemv_v1(A, W_packed, scales, C, W_dequant, N, K, THREADS=256):
    """
    V1: Basic schedule with dequant in local scope.
    """
    s = te.create_schedule(C.op)

    # Put dequant in local memory (registers)
    s[W_dequant].compute_inline()

    # Bind output
    _, n = C.op.axis
    k = C.op.reduce_axis[0]

    # Split n for blocks and threads
    n_outer, n_inner = s[C].split(n, factor=THREADS)
    s[C].bind(n_outer, te.thread_axis("blockIdx.x"))
    s[C].bind(n_inner, te.thread_axis("threadIdx.x"))

    # Split k for tiling
    k_outer, k_inner = s[C].split(k, factor=QUANT_BLOCK)
    s[C].reorder(n_outer, n_inner, k_outer, k_inner)

    return s


def schedule_gemv_v2(A, W_packed, scales, C, W_dequant, N, K, THREADS=256):
    """
    V2: Unroll inner k loop.
    """
    s = te.create_schedule(C.op)
    s[W_dequant].compute_inline()

    _, n = C.op.axis
    k = C.op.reduce_axis[0]

    n_outer, n_inner = s[C].split(n, factor=THREADS)
    s[C].bind(n_outer, te.thread_axis("blockIdx.x"))
    s[C].bind(n_inner, te.thread_axis("threadIdx.x"))

    k_outer, k_inner = s[C].split(k, factor=QUANT_BLOCK)
    s[C].reorder(n_outer, n_inner, k_outer, k_inner)

    # Unroll inner k
    s[C].unroll(k_inner)

    return s


def schedule_gemv_v3(A, W_packed, scales, C, W_dequant, N, K, THREADS=256, TILE_K=64):
    """
    V3: Larger k tile with vectorized load.
    """
    s = te.create_schedule(C.op)
    s[W_dequant].compute_inline()

    _, n = C.op.axis
    k = C.op.reduce_axis[0]

    n_outer, n_inner = s[C].split(n, factor=THREADS)
    s[C].bind(n_outer, te.thread_axis("blockIdx.x"))
    s[C].bind(n_inner, te.thread_axis("threadIdx.x"))

    # Split k for larger tile
    k_outer, k_inner = s[C].split(k, factor=TILE_K)
    k_inner_outer, k_inner_inner = s[C].split(k_inner, factor=8)

    s[C].reorder(n_outer, n_inner, k_outer, k_inner_outer, k_inner_inner)
    s[C].unroll(k_inner_outer)
    s[C].vectorize(k_inner_inner)

    return s


def build_and_benchmark(schedule_fn, name, N=16384, K=2048, warmup=50, runs=200):
    """Build and benchmark a scheduled kernel."""
    print(f"\n--- {name} ---")

    A, W_packed, scales, C, W_dequant = create_te_gemv_basic(N, K)

    try:
        s = schedule_fn(A, W_packed, scales, C, W_dequant, N, K)
    except Exception as e:
        print(f"Schedule failed: {e}")
        return None

    target = tvm.target.Target("cuda -arch=sm_110")

    try:
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(s, [A, W_packed, scales, C], target=target)
    except Exception as e:
        print(f"Build failed: {e}")
        return None

    # Test data
    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)
    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = runtime.empty((1, N), "float32", device)

    lib(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
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
        lib(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        lib(A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000

    weight_bytes = N * K // 2
    scale_bytes = N * (K // QUANT_BLOCK) * 2
    total_bytes = weight_bytes + scale_bytes + K * 2
    bandwidth = total_bytes / (avg_ms / 1000) / 1e9

    print(f"Time: {avg_ms:.4f} ms")
    print(f"Bandwidth: {bandwidth:.1f} GB/s")

    return avg_ms


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


def use_dlight_schedule(N=16384, K=2048):
    """Use dlight auto-scheduler."""
    print("\n--- dlight Auto-Schedule ---")

    from tvm import dlight as dl

    K_packed = K // 2
    num_scale_blocks = K // QUANT_BLOCK

    A = te.placeholder((1, K), dtype="float16", name="A")
    W_packed = te.placeholder((N, K_packed), dtype="uint8", name="W_packed")
    scales = te.placeholder((N, num_scale_blocks), dtype="float16", name="scales")

    def dequant_fn(n, k):
        byte_idx = k // 2
        is_high = k % 2
        packed = W_packed[n, byte_idx]
        int4_val = tir.if_then_else(
            is_high == 0,
            packed & tir.const(0xF, "uint8"),
            (packed >> 4) & tir.const(0xF, "uint8")
        )
        scale_idx = k // QUANT_BLOCK
        scale = scales[n, scale_idx]
        return (int4_val.astype("float16") - tir.const(8.0, "float16")) * scale

    W_dequant = te.compute((N, K), dequant_fn, name="dequantize")

    k_axis = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (1, N),
        lambda m, n: te.sum(
            A[m, k_axis].astype("float32") * W_dequant[n, k_axis].astype("float32"),
            axis=k_axis
        ),
        name="C"
    )

    func = te.create_prim_func([A, W_packed, scales, C])
    mod = tvm.IRModule({"main": func})
    target = tvm.target.Target("cuda -arch=sm_110")

    try:
        with tvm.transform.PassContext(opt_level=3):
            with target:
                # Try different schedules
                for scheduler in [dl.gpu.GEMV(), dl.gpu.LowBatchGEMV(), dl.gpu.Matmul(), dl.gpu.Fallback()]:
                    try:
                        mod_scheduled = dl.ApplyDefaultSchedule(scheduler)(mod)
                        lib = tvm.build(mod_scheduled, target=target)
                        print(f"  Using {scheduler.__class__.__name__}")
                        break
                    except:
                        continue
                else:
                    print("  No scheduler worked")
                    return None
    except Exception as e:
        print(f"Build failed: {e}")
        return None

    # Benchmark
    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)
    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = runtime.empty((1, N), "float32", device)

    lib["main"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    C_result = C_tvm.numpy()
    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)
    print(f"Cos sim: {cos_sim:.6f}")

    if cos_sim <= 0.99:
        print("FAIL")
        return None

    warmup, runs = 50, 200
    for _ in range(warmup):
        lib["main"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        lib["main"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000

    weight_bytes = N * K // 2
    scale_bytes = N * (K // QUANT_BLOCK) * 2
    total_bytes = weight_bytes + scale_bytes + K * 2
    bandwidth = total_bytes / (avg_ms / 1000) / 1e9

    print(f"Time: {avg_ms:.4f} ms")
    print(f"Bandwidth: {bandwidth:.1f} GB/s")

    return avg_ms


def main():
    N, K = 16384, 2048

    print("="*60)
    print("W4A16 TE GEMV Benchmark")
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

    # Test TE schedules
    schedules = [
        (schedule_gemv_v1, "V1: Basic"),
        (schedule_gemv_v2, "V2: Unroll K"),
        (schedule_gemv_v3, "V3: Tile + Vector"),
    ]

    for sched_fn, name in schedules:
        ms = build_and_benchmark(sched_fn, name, N, K)
        if ms is not None:
            results[name] = ms

    # Test dlight
    ms = use_dlight_schedule(N, K)
    if ms is not None:
        results["dlight Auto"] = ms

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if results:
        print(f"{'Schedule':<20} | {'Time (ms)':<12} | {'BW (GB/s)':<10} | {'vs 0.2ms':<10}")
        print("-"*60)
        for name, ms in sorted(results.items(), key=lambda x: x[1]):
            bw = total_bytes / (ms / 1000) / 1e9
            ratio = ms / 0.2
            status = "OK" if ratio <= 1.0 else f"{ratio:.1f}x"
            print(f"{name:<20} | {ms:<12.4f} | {bw:<10.1f} | {status:<10}")

        print(f"\nBest: {min(results.values()):.4f} ms")


if __name__ == "__main__":
    main()
