#!/usr/bin/env python3
"""
W4A16 GEMM - LowBatchGEMV with proper dequantize block naming.

Key insight from dlight's low_batch_gemv.py:
- Block named "dequantize" gets detected and scheduled with `set_scope(..., "local")`
- This puts dequantized weights in registers (local scope) instead of global memory

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


def create_w4a16_compute(M, N, K):
    """Create W4A16 GEMM with properly named dequantize block."""
    num_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    # Placeholders
    A = te.placeholder((M, K), dtype="float16", name="A")
    W_packed = te.placeholder((N, K_packed), dtype="uint8", name="W_packed")
    scales = te.placeholder((N, num_blocks), dtype="float16", name="scales")

    # Dequantize compute - name must contain "dequantize"!
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

    # Named "dequantize" so LowBatchGEMV can detect it
    W_dequantize = te.compute(
        (N, K),
        dequant_func,
        name="dequantize"  # Critical: must contain "dequantize"
    )

    # GEMM: C = A @ W^T
    k_axis = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(
            A[m, k_axis].astype("float32") * W_dequantize[n, k_axis].astype("float32"),
            axis=k_axis
        ),
        name="C"
    )

    return A, W_packed, scales, C


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


def test_lowbatch_v2(M=712, N=16384, K=2048):
    """Test LowBatchGEMV with properly named dequantize block."""
    print(f"\n{'='*60}")
    print(f"W4A16 LowBatchGEMV v2 (with dequantize naming)")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    # Data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    print("Quantizing...")
    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Create compute
    print("Creating compute...")
    A, W_packed, scales, C = create_w4a16_compute(M, N, K)

    # Build with dlight schedules
    print("Building with dlight...")
    from tvm import dlight as dl

    func = te.create_prim_func([A, W_packed, scales, C])
    mod = tvm.IRModule({"main": func})
    target = tvm.target.Target("cuda -arch=sm_110")

    # Print IR before scheduling to check block names
    print("\nIR blocks before scheduling:")
    for gv, f in mod.functions.items():
        if isinstance(f, tir.PrimFunc):
            script = f.script()
            # Check if dequantize is in the script
            if "dequantize" in script:
                print("  Found 'dequantize' in IR - good!")
            else:
                print("  WARNING: 'dequantize' not found in IR")

    with tvm.transform.PassContext(opt_level=3):
        with target:
            # Try LowBatchGEMV first
            try:
                scheduled = dl.ApplyDefaultSchedule(
                    dl.gpu.LowBatchGEMV(bucket=4),
                )(mod)
                print("Applied LowBatchGEMV schedule!")

                # Check scheduled IR for local scope
                for gv, f in scheduled.functions.items():
                    if isinstance(f, tir.PrimFunc):
                        script = f.script()
                        if "local" in script and "dequantize" in script.lower():
                            print("  Dequantize in local scope - register-level!")
                        if 'scope="global"' in script and "dequantize" in script.lower():
                            print("  WARNING: Dequantize in global scope!")

                mod = scheduled
            except Exception as e:
                print(f"LowBatchGEMV failed: {e}")
                # Try GEMV
                try:
                    mod = dl.ApplyDefaultSchedule(
                        dl.gpu.GEMV(),
                    )(mod)
                    print("Applied GEMV schedule!")
                except Exception as e2:
                    print(f"GEMV failed: {e2}")
                    # Fallback
                    mod = dl.ApplyDefaultSchedule(
                        dl.gpu.Fallback(),
                    )(mod)
                    print("Applied Fallback schedule")

    try:
        lib = tvm.build(mod, target=target)
        print("Build successful!")
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

    BF16_MS = 0.42  # Baseline
    print(f"\n  vs BF16 baseline ({BF16_MS}ms): {BF16_MS / avg_ms:.2f}x")

    return avg_ms


def test_small(M=4, N=512, K=256):
    """Test with small size for debugging."""
    print(f"\n{'='*60}")
    print(f"W4A16 LowBatchGEMV v2 - Small Test")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    # Data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Create compute
    A, W_packed, scales, C = create_w4a16_compute(M, N, K)

    # Build
    from tvm import dlight as dl

    func = te.create_prim_func([A, W_packed, scales, C])
    mod = tvm.IRModule({"main": func})
    target = tvm.target.Target("cuda -arch=sm_110")

    with tvm.transform.PassContext(opt_level=3):
        with target:
            try:
                mod = dl.ApplyDefaultSchedule(
                    dl.gpu.LowBatchGEMV(bucket=4),
                )(mod)
                print("Applied LowBatchGEMV schedule!")
            except Exception as e:
                print(f"LowBatchGEMV failed: {e}")
                try:
                    mod = dl.ApplyDefaultSchedule(
                        dl.gpu.GEMV(),
                    )(mod)
                    print("Applied GEMV schedule!")
                except:
                    mod = dl.ApplyDefaultSchedule(
                        dl.gpu.Fallback(),
                    )(mod)
                    print("Applied Fallback schedule")

    lib = tvm.build(mod, target=target)

    # Run
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
    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)

    print(f"\n  Cos sim: {cos_sim:.6f}")
    print(f"  Status: {'PASS' if cos_sim > 0.99 else 'FAIL'}")

    return cos_sim > 0.99


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Run small test first")
    parser.add_argument("--full", action="store_true", help="Run full size test")
    args = parser.parse_args()

    if args.small:
        test_small()

    if args.full or not args.small:
        test_lowbatch_v2()
