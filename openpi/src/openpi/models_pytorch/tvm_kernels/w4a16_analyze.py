#!/usr/bin/env python3
"""
W4A16 Analysis - Check generated CUDA code and Tensor Core usage.

This script analyzes the TVM-generated kernel to understand
whether Tensor Cores are being used.

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


QUANT_BLOCK = 32


def analyze_w4a16(M=712, N=16384, K=2048):
    """Analyze W4A16 kernel - check for Tensor Core usage."""
    print(f"\n{'='*60}")
    print(f"W4A16 Analysis")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}")

    from tvm import dlight as dl

    num_k_blocks = K // QUANT_BLOCK
    K_packed = K // 2

    # Placeholders
    A = te.placeholder((M, K), dtype="float16", name="A")
    W_packed = te.placeholder((N, K_packed), dtype="uint8", name="W_packed")
    scales = te.placeholder((N, num_k_blocks), dtype="float16", name="scales")

    # Dequant
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

    func = te.create_prim_func([A, W_packed, scales, C])
    mod = tvm.IRModule({"main": func})

    target = tvm.target.Target("cuda -arch=sm_110")

    print("\n1. Applying dlight Matmul schedule...")
    with tvm.transform.PassContext(opt_level=3):
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)

    print("\n2. Scheduled IR:")
    print(mod.script()[:3000])  # Print first 3000 chars

    print("\n3. Building...")
    lib = tvm.build(mod, target=target)

    print("\n4. Generated CUDA Code (checking for mma/wmma):")
    cuda_source = lib.imported_modules[0].get_source()

    # Check for Tensor Core indicators
    has_mma = "mma" in cuda_source.lower()
    has_wmma = "wmma" in cuda_source.lower()
    has_ldmatrix = "ldmatrix" in cuda_source.lower()

    print(f"\n   MMA instructions: {'YES' if has_mma else 'NO'}")
    print(f"   WMMA intrinsics: {'YES' if has_wmma else 'NO'}")
    print(f"   ldmatrix: {'YES' if has_ldmatrix else 'NO'}")

    if has_mma or has_wmma:
        print("\n   => Tensor Cores ARE being used!")
    else:
        print("\n   => Tensor Cores NOT being used (scalar FMA only)")

    # Print relevant snippets
    lines = cuda_source.split('\n')
    print("\n5. Key code snippets:")

    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ['mma', 'wmma', 'ldmatrix', '__shfl']):
            print(f"   L{i}: {line[:100]}")

    return lib, cuda_source


if __name__ == "__main__":
    analyze_w4a16()
