#!/usr/bin/env python3
"""
W4A16 GEMV for Decode Phase - Using dlight LowBatchGEMV

For decode (seq_len=1), M is small (batch size) and we want GEMV-style execution.
dlight's LowBatchGEMV rule specifically handles dequantize + GEMV fusion.

Key: Name the dequant compute "dequantize" so LowBatchGEMV recognizes it.

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


def create_w4a16_gemv(M, N, K, block_size_quant=QUANT_BLOCK):
    """
    Create W4A16 GEMV compute definition.

    Important: Name the dequant stage "dequantize" for LowBatchGEMV recognition.
    """
    num_k_blocks = K // block_size_quant
    K_packed = K // 2

    # Placeholders
    A = te.placeholder((M, K), dtype="float16", name="A")
    W_packed = te.placeholder((N, K_packed), dtype="uint8", name="W_packed")
    scales = te.placeholder((N, num_k_blocks), dtype="float16", name="scales")

    # Dequant stage - MUST be named "dequantize" for LowBatchGEMV
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

    W_dequantize = te.compute(
        (N, K),
        dequant_func,
        name="dequantize"  # Critical: LowBatchGEMV looks for this name
    )

    # GEMV: C[m, n] = sum_k A[m, k] * W[n, k]
    k_axis = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(
            A[m, k_axis].astype("float32") * W_dequantize[n, k_axis].astype("float32"),
            axis=k_axis
        ),
        name="C"
    )

    return A, W_packed, scales, W_dequantize, C


def build_w4a16_gemv_dlight(M, N, K):
    """Build W4A16 GEMV using dlight auto-scheduling."""
    print(f"\n{'='*60}")
    print(f"W4A16 GEMV (dlight) - Decode Phase")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    from tvm import dlight as dl

    A, W_packed, scales, W_dequantize, C = create_w4a16_gemv(M, N, K)

    func = te.create_prim_func([A, W_packed, scales, C])
    mod = tvm.IRModule({"main": func})

    target = tvm.target.Target("cuda -arch=sm_110")

    print("Applying dlight schedule...")
    with tvm.transform.PassContext(opt_level=3):
        with target:
            # Try different schedules in order of preference
            schedules = [
                ("LowBatchGEMV", dl.gpu.LowBatchGEMV()),
                ("GEMV", dl.gpu.GEMV()),
                ("Matmul", dl.gpu.Matmul()),
                ("Fallback", dl.gpu.Fallback()),
            ]

            for name, rule in schedules:
                try:
                    mod_scheduled = dl.ApplyDefaultSchedule(rule)(mod)
                    print(f"Applied {name} schedule successfully!")
                    mod = mod_scheduled
                    break
                except Exception as e:
                    print(f"{name} failed: {e}")
                    continue

    print("Building...")
    try:
        lib = tvm.build(mod, target=target)
        print("Build successful!")
    except Exception as e:
        print(f"Build failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    return lib


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


def test_decode_gemv(M=1, N=16384, K=2048):
    """Test W4A16 GEMV for decode (seq_len=1)."""
    print(f"\n{'='*60}")
    print(f"W4A16 GEMV Decode Test")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    lib = build_w4a16_gemv_dlight(M, N, K)
    if lib is None:
        return None

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
        return None

    # Benchmark
    warmup = 50
    runs = 200

    for _ in range(warmup):
        lib["main"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    start = time.time()
    for _ in range(runs):
        lib["main"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    avg_ms = (time.time() - start) / runs * 1000
    flops = 2.0 * M * N * K
    gflops = flops / (avg_ms / 1000) / 1e9

    print(f"\n  Time:    {avg_ms:.4f} ms")
    print(f"  GFLOPS:  {gflops:.4f}")

    # Compare to cuBLAS baseline for GEMV
    print(f"\n  Target: < 0.1 ms for decode")

    return avg_ms


def test_batch_decode(M=712, N=16384, K=2048):
    """Test W4A16 for batched decode (typical batch size)."""
    print(f"\n{'='*60}")
    print(f"W4A16 Batched Decode Test")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    lib = build_w4a16_gemv_dlight(M, N, K)
    if lib is None:
        return None

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

    BF16_MS = 0.42
    print(f"\n  vs BF16 baseline ({BF16_MS}ms): {BF16_MS / avg_ms:.2f}x")

    return avg_ms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq1", action="store_true", help="Test seq_len=1 decode")
    parser.add_argument("--batch", action="store_true", help="Test batched decode (M=712)")
    args = parser.parse_args()

    if args.seq1:
        test_decode_gemv(M=1)
    elif args.batch:
        test_batch_decode(M=712)
    else:
        # Default: test both
        test_decode_gemv(M=1)
        test_batch_decode(M=712)
