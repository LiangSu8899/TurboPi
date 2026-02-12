#!/usr/bin/env python3
"""
Verify accuracy of nvFP4 TVM kernels (W4A4, W4A8, W4A16).

This script compares TVM kernel outputs against PyTorch reference implementation
to ensure numerical correctness before performance optimization.

Verification metrics:
- Max absolute error
- Mean absolute error
- Relative error
- Correlation coefficient

Run with TVM environment:
    source /home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/venv/bin/activate
    python verify_kernel_accuracy.py

Author: Claude Code
Date: 2026-02-10
"""

import sys
import os

# Add TVM to path
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm.script import tir as T
import numpy as np
import torch

print(f"TVM Version: {tvm.__version__}")
print(f"CUDA Available: {tvm.cuda().exist}")
print(f"PyTorch CUDA: {torch.cuda.is_available()}")

# Constants
BLOCK_SIZE = 32
M = 1  # Batch size (inference)
K = 3072  # Hidden dim (Pi0)
N = 3072  # Output features

# nvFP4 E2M1 valid values
NVFP4_VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)

# FP8 E4M3 range
FP8_E4M3_MAX = 448.0


def create_nvfp4_gemm_kernel(M, N, K, block_size=32):
    """W4A4: Both activation and weight are nvFP4 with block scales."""
    num_blocks_k = (K + block_size - 1) // block_size
    THREADS = 256
    total = M * N
    num_blocks = (total + THREADS - 1) // THREADS

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),
        W: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "nvfp4_gemm", "tir.noalias": True})
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N
                    C[i, j] = T.float32(0)
                    for k in T.serial(K):
                        block_idx = k // block_size
                        a_val = A[i, k] * scale_A[i, block_idx]
                        w_val = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + a_val * w_val
    return kernel


def create_w4a8_gemm_kernel(M, N, K, block_size=32):
    """W4A8: Activation is FP8, Weight is nvFP4, both with block scales."""
    num_blocks_k = (K + block_size - 1) // block_size
    THREADS = 256
    total = M * N
    num_blocks = (total + THREADS - 1) // THREADS

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),  # FP8 values
        W: T.Buffer((N, K), "float32"),  # nvFP4 values
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a8_gemm", "tir.noalias": True})
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N
                    C[i, j] = T.float32(0)
                    for k in T.serial(K):
                        block_idx = k // block_size
                        a_val = A[i, k] * scale_A[i, block_idx]
                        w_val = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + a_val * w_val
    return kernel


def create_w4a16_gemm_kernel(M, N, K, block_size=32):
    """W4A16: Activation is full precision, Weight is nvFP4 with scale."""
    num_blocks_k = (K + block_size - 1) // block_size
    THREADS = 256
    total = M * N
    num_blocks = (total + THREADS - 1) // THREADS

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), "float32"),  # Full precision
        W: T.Buffer((N, K), "float32"),  # nvFP4 values
        scale_W: T.Buffer((N, num_blocks_k), "float32"),  # Only weight scale
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_gemm", "tir.noalias": True})
        for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS, thread="threadIdx.x"):
                idx = bx * THREADS + tx
                if idx < total:
                    i = idx // N
                    j = idx % N
                    C[i, j] = T.float32(0)
                    for k in T.serial(K):
                        block_idx = k // block_size
                        w_dequant = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + A[i, k] * w_dequant
    return kernel


def build_kernel(tir_func, target="cuda -arch=sm_110"):
    """Build TVM kernel."""
    target_obj = tvm.target.Target(target)
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(tir_func, target=target_obj)
    return mod


def quantize_to_nvfp4(x: np.ndarray) -> np.ndarray:
    """Quantize float array to nearest nvFP4 value."""
    x_flat = x.flatten()
    # Find nearest nvFP4 value for each element
    distances = np.abs(x_flat[:, np.newaxis] - NVFP4_VALUES)
    nearest_idx = np.argmin(distances, axis=1)
    result = NVFP4_VALUES[nearest_idx].reshape(x.shape)
    return result


def quantize_to_fp8_e4m3(x: np.ndarray) -> np.ndarray:
    """Quantize to FP8 E4M3 range."""
    return np.clip(x, -FP8_E4M3_MAX, FP8_E4M3_MAX)


def reference_gemm_w4a4(A, W, scale_A, scale_W, block_size=BLOCK_SIZE):
    """PyTorch reference for W4A4 GEMM."""
    M, K = A.shape
    N = W.shape[0]
    num_blocks_k = (K + block_size - 1) // block_size

    # Expand scales to full K dimension
    scale_A_exp = np.repeat(scale_A, block_size, axis=1)[:, :K]
    scale_W_exp = np.repeat(scale_W, block_size, axis=1)[:, :K]

    # Dequantize
    A_dq = A * scale_A_exp
    W_dq = W * scale_W_exp

    # GEMM
    C = np.matmul(A_dq, W_dq.T)
    return C


def reference_gemm_w4a8(A, W, scale_A, scale_W, block_size=BLOCK_SIZE):
    """PyTorch reference for W4A8 GEMM (same as W4A4)."""
    return reference_gemm_w4a4(A, W, scale_A, scale_W, block_size)


def reference_gemm_w4a16(A, W, scale_W, block_size=BLOCK_SIZE):
    """PyTorch reference for W4A16 GEMM (only weight has scale)."""
    M, K = A.shape
    N = W.shape[0]

    # Expand weight scale
    scale_W_exp = np.repeat(scale_W, block_size, axis=1)[:, :K]

    # Dequantize weight only
    W_dq = W * scale_W_exp

    # GEMM
    C = np.matmul(A, W_dq.T)
    return C


def verify_kernel(name, kernel_func, func_name, has_activation_scale=True):
    """Verify a single kernel against PyTorch reference."""
    print(f"\n{'='*60}")
    print(f"Verifying: {name}")
    print(f"{'='*60}")

    num_blocks_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Generate test data
    np.random.seed(42)

    if name == "W4A4":
        # Both A and W are nvFP4 quantized
        A_raw = np.random.randn(M, K).astype(np.float32)
        A = quantize_to_nvfp4(A_raw)
        scale_A = (np.random.rand(M, num_blocks_k) * 0.1 + 0.01).astype(np.float32)
    elif name == "W4A8":
        # A is FP8, W is nvFP4
        A_raw = np.random.randn(M, K).astype(np.float32)
        A = quantize_to_fp8_e4m3(A_raw)
        scale_A = (np.random.rand(M, num_blocks_k) * 0.1 + 0.01).astype(np.float32)
    else:  # W4A16
        # A is full precision
        A = np.random.randn(M, K).astype(np.float32)
        scale_A = None

    # W is always nvFP4
    W_raw = np.random.randn(N, K).astype(np.float32)
    W = quantize_to_nvfp4(W_raw)
    scale_W = (np.random.rand(N, num_blocks_k) * 0.1 + 0.01).astype(np.float32)

    # Build TVM kernel
    print("  Building TVM kernel...")
    mod = build_kernel(kernel_func)
    func = mod[func_name]

    # Prepare TVM arrays
    device = tvm.runtime.cuda(0)

    A_tvm = tvm.runtime.empty((M, K), dtype="float32", device=device)
    A_tvm.copyfrom(A)

    W_tvm = tvm.runtime.empty((N, K), dtype="float32", device=device)
    W_tvm.copyfrom(W)

    scale_W_tvm = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
    scale_W_tvm.copyfrom(scale_W)

    C_tvm = tvm.runtime.empty((M, N), dtype="float32", device=device)
    C_tvm.copyfrom(np.zeros((M, N), dtype=np.float32))

    if has_activation_scale:
        scale_A_tvm = tvm.runtime.empty((M, num_blocks_k), dtype="float32", device=device)
        scale_A_tvm.copyfrom(scale_A)

        # Run TVM kernel
        func(A_tvm, W_tvm, scale_A_tvm, scale_W_tvm, C_tvm)

        # Reference
        C_ref = reference_gemm_w4a4(A, W, scale_A, scale_W)
    else:
        # W4A16: no activation scale
        func(A_tvm, W_tvm, scale_W_tvm, C_tvm)

        # Reference
        C_ref = reference_gemm_w4a16(A, W, scale_W)

    tvm.runtime.cuda(0).sync()

    # Get TVM result
    C_result = C_tvm.numpy()

    # Calculate metrics
    abs_diff = np.abs(C_result - C_ref)
    max_abs_error = np.max(abs_diff)
    mean_abs_error = np.mean(abs_diff)

    # Relative error (avoid div by zero)
    rel_diff = abs_diff / (np.abs(C_ref) + 1e-10)
    max_rel_error = np.max(rel_diff)
    mean_rel_error = np.mean(rel_diff)

    # Correlation
    correlation = np.corrcoef(C_result.flatten(), C_ref.flatten())[0, 1]

    # Print results
    print(f"\n  {'Metric':<25} {'Value':<20}")
    print(f"  {'-'*45}")
    print(f"  {'Max Absolute Error':<25} {max_abs_error:.6e}")
    print(f"  {'Mean Absolute Error':<25} {mean_abs_error:.6e}")
    print(f"  {'Max Relative Error':<25} {max_rel_error:.6e}")
    print(f"  {'Mean Relative Error':<25} {mean_rel_error:.6e}")
    print(f"  {'Correlation':<25} {correlation:.10f}")

    # Sample values
    print(f"\n  Sample values (first 5 elements):")
    print(f"  TVM:  {C_result.flatten()[:5]}")
    print(f"  Ref:  {C_ref.flatten()[:5]}")
    print(f"  Diff: {abs_diff.flatten()[:5]}")

    # Pass/Fail threshold
    # For FP32 GEMM with K=3072, accumulated floating point errors are expected
    # Reasonable thresholds:
    # - Max relative error < 1% (0.01) - accounts for accumulation over 3072 elements
    # - Correlation > 0.9999 - ensures outputs are strongly correlated
    TOLERANCE = 0.01  # 1% relative tolerance
    CORR_THRESHOLD = 0.9999

    if max_rel_error < TOLERANCE and correlation > CORR_THRESHOLD:
        print(f"\n  ✅ PASS: {name} kernel is accurate!")
        print(f"     (Max rel error {max_rel_error:.2e} < {TOLERANCE}, correlation {correlation:.8f})")
        status = "PASS"
    else:
        print(f"\n  ❌ FAIL: {name} kernel has accuracy issues!")
        print(f"     (Max rel error {max_rel_error:.2e} >= {TOLERANCE} or correlation {correlation:.8f} < {CORR_THRESHOLD})")
        status = "FAIL"

    return {
        "name": name,
        "status": status,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
        "mean_rel_error": mean_rel_error,
        "correlation": correlation,
    }


def main():
    print("="*70)
    print("nvFP4 TVM Kernel Accuracy Verification")
    print("="*70)
    print(f"\nMatrix: M={M}, N={N}, K={K}")
    print(f"Block size: {BLOCK_SIZE}")

    results = []

    # W4A4
    w4a4_kernel = create_nvfp4_gemm_kernel(M, N, K)
    r = verify_kernel("W4A4", w4a4_kernel, "nvfp4_gemm", has_activation_scale=True)
    results.append(r)

    # W4A8
    w4a8_kernel = create_w4a8_gemm_kernel(M, N, K)
    r = verify_kernel("W4A8", w4a8_kernel, "w4a8_gemm", has_activation_scale=True)
    results.append(r)

    # W4A16
    w4a16_kernel = create_w4a16_gemm_kernel(M, N, K)
    r = verify_kernel("W4A16", w4a16_kernel, "w4a16_gemm", has_activation_scale=False)
    results.append(r)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Kernel':<10} {'Status':<10} {'Max Rel Err':<15} {'Correlation':<15}")
    print("-"*50)
    for r in results:
        print(f"{r['name']:<10} {r['status']:<10} {r['max_rel_error']:<15.6e} {r['correlation']:<15.10f}")

    all_pass = all(r['status'] == 'PASS' for r in results)
    print("\n" + "="*70)
    if all_pass:
        print("✅ ALL KERNELS PASSED ACCURACY VERIFICATION")
        print("   Ready to proceed with performance optimization!")
    else:
        print("❌ SOME KERNELS FAILED - Need to fix accuracy issues first")
    print("="*70)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
