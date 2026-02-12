#!/usr/bin/env python3
"""
W4A16 Decode GEMV - True seq_len=1 decode scenario.

For decode, M is very small (1-8 tokens per forward).
This is a GEMV (matrix-vector) operation, not GEMM.

Target: < 1ms for M=1, N=16384, K=2048

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
    """Create W4A16 GEMM compute."""
    num_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    A = te.placeholder((M, K), dtype="float16", name="A")
    W_packed = te.placeholder((N, K_packed), dtype="uint8", name="W_packed")
    scales = te.placeholder((N, num_blocks), dtype="float16", name="scales")

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

    W_dequant = te.compute((N, K), dequant_func, name="dequantize")

    k_axis = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(
            A[m, k_axis].astype("float32") * W_dequant[n, k_axis].astype("float32"),
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


def test_decode_gemv(M=1, N=16384, K=2048):
    """Test true decode GEMV scenario."""
    print(f"\n{'='*60}")
    print(f"W4A16 Decode GEMV (seq_len={M})")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    from tvm import dlight as dl

    # Data
    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    print("Quantizing...")
    W_packed_np, scales_np = quantize_int4(W_np)
    W_dequant_np = dequant_int4(W_packed_np, scales_np, K)
    C_ref = A_np.astype(np.float32) @ W_dequant_np.T

    # Create compute
    A, W_packed, scales, C = create_w4a16_compute(M, N, K)
    func = te.create_prim_func([A, W_packed, scales, C])
    mod = tvm.IRModule({"main": func})
    target = tvm.target.Target("cuda -arch=sm_110")

    # Try different schedules
    schedules_to_try = [
        ("GEMV", lambda: dl.gpu.GEMV()),
        ("LowBatchGEMV(4)", lambda: dl.gpu.LowBatchGEMV(bucket=4)),
        ("LowBatchGEMV(1)", lambda: dl.gpu.LowBatchGEMV(bucket=1)),
        ("Matmul", lambda: dl.gpu.Matmul()),
        ("Fallback", lambda: dl.gpu.Fallback()),
    ]

    results = {}

    for name, sched_fn in schedules_to_try:
        print(f"\nTrying {name} schedule...")

        # Fresh module
        func = te.create_prim_func([A, W_packed, scales, C])
        mod = tvm.IRModule({"main": func})

        try:
            with tvm.transform.PassContext(opt_level=3):
                with target:
                    mod = dl.ApplyDefaultSchedule(sched_fn())(mod)
            print(f"  Schedule applied!")

            lib = tvm.build(mod, target=target)
            print(f"  Build successful!")

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
            if np.isnan(C_result).any():
                print(f"  WARNING: Output contains NaN!")
                continue

            cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
                np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)
            print(f"  Cos sim: {cos_sim:.6f}")

            if cos_sim <= 0.99:
                print(f"  FAIL - skipping benchmark")
                continue

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

            print(f"  Time: {avg_ms:.4f} ms")
            print(f"  GFLOPS: {gflops:.2f}")

            results[name] = avg_ms

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary for M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    if results:
        best_name = min(results, key=results.get)
        best_ms = results[best_name]

        for name, ms in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name:20s}: {ms:.4f} ms")

        print(f"\n  Best: {best_name} at {best_ms:.4f} ms")

        # Compare to target
        target_ms = 1.0
        if best_ms < target_ms:
            print(f"  ACHIEVED target of < {target_ms}ms!")
        else:
            print(f"  Need {best_ms/target_ms:.1f}x speedup to reach {target_ms}ms target")

    return results


def test_various_batch_sizes():
    """Test with various batch sizes for decode."""
    print("\n" + "="*60)
    print("Testing Various Batch Sizes")
    print("="*60)

    N, K = 16384, 2048

    for M in [1, 2, 4, 8, 16, 32]:
        print(f"\n--- M={M} ---")
        results = test_decode_gemv(M, N, K)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="Batch size (M)")
    parser.add_argument("--all", action="store_true", help="Test all batch sizes")
    args = parser.parse_args()

    if args.all:
        test_various_batch_sizes()
    else:
        test_decode_gemv(M=args.batch)
