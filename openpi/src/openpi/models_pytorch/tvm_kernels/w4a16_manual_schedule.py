#!/usr/bin/env python3
"""
W4A16 GEMM - Manual Schedule with Register-Level Dequant

Strategy:
1. Create compute graph with dequantize as intermediate
2. Apply Matmul schedule for Tensor Core usage
3. Manually set dequantize scope to "local" (registers)

Author: Claude Code
Date: 2026-02-11
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import te, tir, runtime
from tvm.script import tir as T
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


def try_manual_schedule(M=712, N=16384, K=2048):
    """Try manually scheduled kernel with local dequant."""
    print(f"\n{'='*60}")
    print(f"W4A16 Manual Schedule with Local Dequant")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    from tvm import dlight as dl

    # Create compute
    A, W_packed, scales, C = create_w4a16_compute(M, N, K)
    func = te.create_prim_func([A, W_packed, scales, C])
    mod = tvm.IRModule({"main": func})
    target = tvm.target.Target("cuda -arch=sm_110")

    print("Creating schedule...")
    sch = tir.Schedule(mod)

    # Find blocks
    blocks = []
    def collect_blocks(stmt):
        if isinstance(stmt, tir.Block):
            blocks.append(stmt.name_hint)
    tir.stmt_functor.post_order_visit(sch.mod["main"].body, collect_blocks)
    print(f"Blocks found: {blocks}")

    # Get block references
    try:
        dequant_block = sch.get_block("dequantize")
        c_block = sch.get_block("C")
        print("Found dequantize and C blocks")

        # Try to set dequant to local scope first
        try:
            sch.set_scope(dequant_block, 0, "local")
            print("Set dequantize to local scope")
        except Exception as e:
            print(f"Could not set scope: {e}")

        # Try compute_inline
        try:
            sch.compute_inline(dequant_block)
            print("Inlined dequantize block")
        except Exception as e:
            print(f"Could not inline dequantize: {e}")

    except Exception as e:
        print(f"Block lookup failed: {e}")

    # Apply dlight schedule
    print("\nApplying dlight Matmul schedule...")
    with tvm.transform.PassContext(opt_level=3):
        with target:
            mod = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(),
            )(mod)

    print("Building...")
    lib = tvm.build(mod, target=target)
    print("Build successful!")

    return lib


def test_current_best(M=712, N=16384, K=2048):
    """Run the current best implementation (dlight Matmul)."""
    print(f"\n{'='*60}")
    print(f"W4A16 Current Best (dlight Matmul)")
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

    # Apply Matmul schedule
    print("Applying Matmul schedule...")
    with tvm.transform.PassContext(opt_level=3):
        with target:
            mod = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(),
            )(mod)

    lib = tvm.build(mod, target=target)
    print("Build successful!")

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

    print(f"\n  Cos sim:  {cos_sim:.6f}")
    print(f"  Status: {'PASS' if cos_sim > 0.99 else 'FAIL'}")

    if cos_sim <= 0.99:
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

    BF16_MS = 0.42
    print(f"\n  vs BF16 baseline ({BF16_MS}ms): {BF16_MS / avg_ms:.2f}x")

    return avg_ms


def analyze_scheduled_ir(M=712, N=16384, K=2048):
    """Analyze the scheduled IR to understand memory hierarchy."""
    print(f"\n{'='*60}")
    print(f"Analyzing Scheduled IR")
    print(f"{'='*60}")

    from tvm import dlight as dl

    # Create compute
    A, W_packed, scales, C = create_w4a16_compute(M, N, K)
    func = te.create_prim_func([A, W_packed, scales, C])
    mod = tvm.IRModule({"main": func})
    target = tvm.target.Target("cuda -arch=sm_110")

    # Apply Matmul schedule
    with tvm.transform.PassContext(opt_level=3):
        with target:
            mod = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(),
            )(mod)

    # Print scheduled IR
    print("\nScheduled IR (first 5000 chars):")
    ir_script = mod.script()
    print(ir_script[:5000])

    # Check for WMMA usage
    has_wmma = "wmma" in ir_script.lower()
    has_shared = "shared" in ir_script
    has_local = '"local"' in ir_script or "'local'" in ir_script

    print(f"\n\nAnalysis:")
    print(f"  Uses WMMA intrinsics: {'YES' if has_wmma else 'NO'}")
    print(f"  Uses shared memory: {'YES' if has_shared else 'NO'}")
    print(f"  Uses local memory (registers): {'YES' if has_local else 'NO'}")

    # Count memory allocations
    lines = ir_script.split('\n')
    for line in lines:
        if 'alloc_buffer' in line.lower() or 'allocate' in line.lower():
            if 'dequant' in line.lower() or 'W_' in line:
                print(f"  {line.strip()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="Analyze scheduled IR")
    parser.add_argument("--run", action="store_true", help="Run benchmark")
    args = parser.parse_args()

    if args.analyze:
        analyze_scheduled_ir()

    if args.run or not args.analyze:
        test_current_best()
