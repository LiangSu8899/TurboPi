"""
TVM TensorIR: nvFP4 to BF16 Dequantization Kernel (W4A16)

Solves: Need efficient nvFP4->BF16 dequant kernel for W4A16 mixed precision

Flow: nvFP4 weight -> dequant to BF16 -> standard cuBLAS GEMM with BF16 activation

This allows using cuBLAS for compute while still saving weight memory bandwidth.

Author: Claude Code
Date: 2026-02-10
"""

import tvm
from tvm import te, tir
from tvm.script import tir as T
import numpy as np
from typing import Tuple, Optional

# Constants
BLOCK_SIZE = 32
NVFP4_MAX = 6.0


def create_dequant_nvfp4_te(
    N: int,  # out_features (num rows)
    K: int,  # in_features (num cols)
    block_size: int = BLOCK_SIZE,
):
    """
    Create TVM Tensor Expression for nvFP4 -> BF16 dequantization.

    out[N, K] = weight_fp4[N, K] * scale[N, K // block_size]
    """
    num_blocks = (K + block_size - 1) // block_size

    # Inputs
    W = te.placeholder((N, K), name="W", dtype="float32")  # nvFP4 values as float32
    scales = te.placeholder((N, num_blocks), name="scales", dtype="float32")

    # Dequantize
    W_dequant = te.compute(
        (N, K),
        lambda i, j: W[i, j] * scales[i, j // block_size],
        name="W_dequant"
    )

    return W, scales, W_dequant


def create_dequant_nvfp4_tir(
    N: int,
    K: int,
    block_size: int = BLOCK_SIZE,
):
    """
    TensorIR version with CUDA thread bindings.
    """
    num_blocks = (K + block_size - 1) // block_size

    # Thread configuration
    THREADS_PER_BLOCK = 256
    total_elements = N * K

    @T.prim_func
    def dequant_nvfp4_func(
        W: T.Buffer((N, K), "float32"),
        scales: T.Buffer((N, num_blocks), "float32"),
        W_out: T.Buffer((N, K), "float32"),  # Use float32 for output compatibility
    ):
        T.func_attr({"global_symbol": "dequant_nvfp4", "tir.noalias": True})

        num_blocks_x = (total_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

        for bx in T.thread_binding(num_blocks_x, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):
                idx = bx * THREADS_PER_BLOCK + tx
                if idx < total_elements:
                    i = idx // K
                    j = idx % K
                    block_idx = j // block_size
                    scale = scales[i, block_idx]
                    W_out[i, j] = W[i, j] * scale

    return dequant_nvfp4_func


def build_w4a16_dequant_kernel(
    N: int,
    K: int,
    target: str = "cuda -arch=sm_110",
    opt_level: int = 3,
) -> tvm.runtime.Module:
    """
    Build optimized nvFP4 dequantization kernel.
    """
    # Create TensorIR function directly
    func = create_dequant_nvfp4_tir(N, K)

    # Get target
    target_obj = tvm.target.Target(target)

    # Build with TensorIR
    with tvm.transform.PassContext(opt_level=opt_level):
        mod = tvm.build(func, target=target_obj)

    return mod


def dequant_nvfp4_to_bf16_tvm(
    W: "torch.Tensor",      # [N, K] nvFP4 values
    scales: "torch.Tensor",  # [N, num_blocks]
    module: Optional[tvm.runtime.Module] = None,
) -> "torch.Tensor":
    """
    Dequantize nvFP4 weight to BF16 using TVM kernel.
    """
    import torch

    N, K = W.shape

    if module is None:
        module = build_w4a16_dequant_kernel(N, K)

    device = tvm.cuda(0)
    w_tvm = tvm.nd.array(W.cpu().numpy().astype("float32"), device)
    scales_tvm = tvm.nd.array(scales.cpu().numpy().astype("float32"), device)
    w_out_tvm = tvm.nd.empty((N, K), dtype="float32", device=device)

    func = module["dequant_nvfp4"]
    func(w_tvm, scales_tvm, w_out_tvm)

    return torch.from_numpy(w_out_tvm.numpy()).to(W.device).to(torch.bfloat16)


# ============================================================================
# W4A16 Full GEMM (dequant + cuBLAS)
# ============================================================================

def w4a16_gemm_hybrid(
    A: "torch.Tensor",      # [M, K] BF16 activation
    W: "torch.Tensor",      # [N, K] nvFP4 weight
    scales: "torch.Tensor",  # [N, num_blocks]
    dequant_module: Optional[tvm.runtime.Module] = None,
) -> "torch.Tensor":
    """
    W4A16 GEMM: TVM dequant + cuBLAS GEMM.

    This combines:
    1. TVM kernel for fast nvFP4 -> BF16 dequantization
    2. cuBLAS for optimized BF16 GEMM

    Expected to be faster than full nvFP4 compute due to cuBLAS optimization.
    """
    import torch

    # Step 1: Dequantize weight to BF16
    W_bf16 = dequant_nvfp4_to_bf16_tvm(W, scales, dequant_module)

    # Step 2: cuBLAS GEMM
    # A: [M, K], W_bf16: [N, K] -> A @ W^T = [M, N]
    C = torch.matmul(A, W_bf16.T)

    return C


# ============================================================================
# Fused Dequant + GEMM (Alternative)
# ============================================================================

def create_w4a16_fused_gemm_te(
    M: int,
    N: int,
    K: int,
    block_size: int = BLOCK_SIZE,
):
    """
    Fused dequant + GEMM for W4A16.

    C[M, N] = A_bf16[M, K] @ (W_fp4[N, K] * scale[N, K//block_size])^T
    """
    num_blocks = (K + block_size - 1) // block_size

    A = te.placeholder((M, K), name="A", dtype="bfloat16")
    W = te.placeholder((N, K), name="W", dtype="float32")  # nvFP4 values
    scales = te.placeholder((N, num_blocks), name="scales", dtype="float32")

    k = te.reduce_axis((0, K), name="k")

    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k].astype("float32") *
            W[j, k] * scales[j, k // block_size],
            axis=k
        ),
        name="C"
    )

    return A, W, scales, C


def create_w4a16_fused_gemm_tir(
    M: int,
    N: int,
    K: int,
    block_size: int = BLOCK_SIZE,
):
    """
    TensorIR version of fused W4A16 GEMM with CUDA thread bindings.
    NOTE: Accumulates directly to output buffer (TVM 0.24 local var bug workaround).
    """
    num_blocks = (K + block_size - 1) // block_size

    # Thread configuration
    THREADS_PER_BLOCK = 256
    total_elements = M * N

    @T.prim_func
    def w4a16_fused_gemm_func(
        A: T.Buffer((M, K), "float32"),  # Use float32 for compatibility
        W: T.Buffer((N, K), "float32"),
        scales: T.Buffer((N, num_blocks), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a16_fused_gemm", "tir.noalias": True})

        num_blocks_x = (total_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

        for bx in T.thread_binding(num_blocks_x, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):
                idx = bx * THREADS_PER_BLOCK + tx
                if idx < total_elements:
                    i = idx // N
                    j = idx % N

                    # Initialize output (accumulate in place)
                    C[i, j] = T.float32(0)

                    # Fused dequant + GEMM
                    for k in T.serial(K):
                        block_idx = k // block_size
                        w_dequant = W[j, k] * scales[j, block_idx]
                        C[i, j] = C[i, j] + A[i, k] * w_dequant

    return w4a16_fused_gemm_func


def build_w4a16_fused_kernel(
    M: int,
    N: int,
    K: int,
    target: str = "cuda -arch=sm_110",
    opt_level: int = 3,
) -> tvm.runtime.Module:
    """
    Build fused W4A16 kernel (dequant + GEMM in one pass).
    """
    # Create TensorIR function directly
    func = create_w4a16_fused_gemm_tir(M, N, K)

    # Get target
    target_obj = tvm.target.Target(target)

    # Build with TensorIR
    with tvm.transform.PassContext(opt_level=opt_level):
        mod = tvm.build(func, target=target_obj)

    return mod


def w4a16_fused_gemm_tvm(
    A: "torch.Tensor",
    W: "torch.Tensor",
    scales: "torch.Tensor",
    module: Optional[tvm.runtime.Module] = None,
) -> "torch.Tensor":
    """
    Execute fused W4A16 GEMM.
    """
    import torch

    M, K = A.shape
    N = W.shape[0]

    if module is None:
        module = build_w4a16_fused_kernel(M, N, K)

    device = tvm.cuda(0)
    a_tvm = tvm.nd.array(A.cpu().numpy(), device)
    w_tvm = tvm.nd.array(W.cpu().numpy().astype("float32"), device)
    scales_tvm = tvm.nd.array(scales.cpu().numpy().astype("float32"), device)
    c_tvm = tvm.nd.empty((M, N), dtype="float32", device=device)

    func = module["w4a16_fused_gemm"]
    func(a_tvm, w_tvm, scales_tvm, c_tvm)

    return torch.from_numpy(c_tvm.numpy()).to(A.device)


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_w4a16(M: int = 1, N: int = 3072, K: int = 3072, warmup: int = 10, runs: int = 100):
    """Benchmark W4A16 approaches."""
    import torch
    import time

    print(f"\n{'='*60}")
    print(f"TVM W4A16 Benchmark")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}\n")

    num_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create inputs
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    # nvFP4 weight
    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6])
    W = fp4_values[torch.randint(0, len(fp4_values), (N, K))].float().cuda()
    scales = torch.rand(N, num_blocks, device="cuda") * 0.1 + 0.01

    # Build kernels
    print("Building kernels...")
    dequant_mod = build_w4a16_dequant_kernel(N, K)
    fused_mod = build_w4a16_fused_kernel(M, N, K)
    print("Done\n")

    # Warmup
    for _ in range(warmup):
        _ = w4a16_gemm_hybrid(A, W, scales, dequant_mod)
        _ = w4a16_fused_gemm_tvm(A, W, scales, fused_mod)
    torch.cuda.synchronize()

    # Benchmark hybrid (dequant + cuBLAS)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        C_hybrid = w4a16_gemm_hybrid(A, W, scales, dequant_mod)
    torch.cuda.synchronize()
    hybrid_time = (time.time() - start) / runs * 1000

    # Benchmark fused
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        C_fused = w4a16_fused_gemm_tvm(A, W, scales, fused_mod)
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / runs * 1000

    # Reference: full BF16
    W_bf16 = W * scales.repeat_interleave(BLOCK_SIZE, dim=1)[:, :K]
    W_bf16 = W_bf16.to(torch.bfloat16)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        C_ref = torch.matmul(A, W_bf16.T)
    torch.cuda.synchronize()
    bf16_time = (time.time() - start) / runs * 1000

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Hybrid (TVM dequant + cuBLAS): {hybrid_time:.4f} ms")
    print(f"  Fused (TVM dequant+GEMM):      {fused_time:.4f} ms")
    print(f"  Reference (BF16 cuBLAS):       {bf16_time:.4f} ms")
    print(f"")
    print(f"  Hybrid speedup vs BF16:  {bf16_time/hybrid_time:.2f}x")
    print(f"  Fused speedup vs BF16:   {bf16_time/fused_time:.2f}x")
    print(f"{'='*60}\n")

    # Verify correctness
    diff_hybrid = torch.abs(C_hybrid.float() - C_ref.float()).max().item()
    diff_fused = torch.abs(C_fused.float() - C_ref.float()).max().item()
    print(f"Hybrid max diff: {diff_hybrid:.6f}")
    print(f"Fused max diff:  {diff_fused:.6f}")

    return hybrid_time, fused_time


if __name__ == "__main__":
    benchmark_w4a16()
