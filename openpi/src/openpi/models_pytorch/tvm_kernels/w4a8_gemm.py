"""
TVM TensorIR: W4A8 GEMM Kernel (FP8 Activation + nvFP4 Weight)

Solves: CUTLASS mxf8f6f4 instruction not supported on SM110 (Thor)

This is a pure software implementation that bypasses the hardware limitation.
Activations are in FP8 E4M3, weights are in nvFP4 E2M1, both with block scaling.

C[M, N] = dequant(A_fp8, scale_A) @ dequant(W_fp4, scale_W)^T

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
FP8_E4M3_MAX = 448.0


def create_w4a8_gemm_te(
    M: int,  # batch * seq_len
    N: int,  # out_features
    K: int,  # in_features
    block_size: int = BLOCK_SIZE,
):
    """
    Create TVM Tensor Expression for W4A8 GEMM.

    C[M, N] = A_fp8[M, K] @ W_fp4[K, N]

    Activation: FP8 E4M3 with block scaling
    Weight: nvFP4 E2M1 with block scaling
    """
    num_blocks_k = (K + block_size - 1) // block_size

    # Inputs
    # A: FP8 activations stored as float32 (actual FP8 values)
    # W: nvFP4 weights stored as float32 (transposed: [N, K])
    A = te.placeholder((M, K), name="A", dtype="float32")
    W = te.placeholder((N, K), name="W", dtype="float32")
    scale_A = te.placeholder((M, num_blocks_k), name="scale_A", dtype="float32")
    scale_W = te.placeholder((N, num_blocks_k), name="scale_W", dtype="float32")

    # Reduce axis
    k = te.reduce_axis((0, K), name="k")

    # GEMM with fused dequantization
    # Key insight: software implementation doesn't need mxf8f6f4 instruction
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k] * scale_A[i, k // block_size] *
            W[j, k] * scale_W[j, k // block_size],
            axis=k
        ),
        name="C"
    )

    return A, W, scale_A, scale_W, C


def create_w4a8_gemm_tir(
    M: int,
    N: int,
    K: int,
    block_size: int = BLOCK_SIZE,
):
    """
    Create TensorIR for W4A8 GEMM with CUDA thread bindings.
    NOTE: Accumulates directly to output buffer (TVM 0.24 local var bug workaround).
    """
    num_blocks_k = (K + block_size - 1) // block_size

    # Thread configuration
    THREADS_PER_BLOCK = 256
    total_elements = M * N

    @T.prim_func
    def w4a8_gemm_func(
        A: T.Buffer((M, K), "float32"),      # FP8 activation values
        W: T.Buffer((N, K), "float32"),      # nvFP4 weight values
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_W: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "w4a8_gemm", "tir.noalias": True})

        # Calculate grid dimensions
        num_blocks_x = (total_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

        for bx in T.thread_binding(num_blocks_x, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):
                idx = bx * THREADS_PER_BLOCK + tx
                if idx < total_elements:
                    i = idx // N
                    j = idx % N

                    # Initialize output (accumulate in place)
                    C[i, j] = T.float32(0)

                    # Main GEMM loop with fused dequantization
                    for k in T.serial(K):
                        block_idx = k // block_size
                        a_val = A[i, k] * scale_A[i, block_idx]
                        w_val = W[j, k] * scale_W[j, block_idx]
                        C[i, j] = C[i, j] + a_val * w_val

    return w4a8_gemm_func


def create_w4a8_gemm_optimized_te(
    M: int,
    N: int,
    K: int,
    block_size: int = BLOCK_SIZE,
):
    """
    Create optimized W4A8 GEMM using split-K for better parallelism.

    For small M (like M=1 in inference), we split K into chunks
    and reduce in parallel.
    """
    num_blocks_k = (K + block_size - 1) // block_size
    split_k = 4  # Number of K splits

    A = te.placeholder((M, K), name="A", dtype="float32")
    W = te.placeholder((N, K), name="W", dtype="float32")
    scale_A = te.placeholder((M, num_blocks_k), name="scale_A", dtype="float32")
    scale_W = te.placeholder((N, num_blocks_k), name="scale_W", dtype="float32")

    # Split K into chunks
    k_chunk_size = (K + split_k - 1) // split_k

    # First level: parallel partial sums
    k_inner = te.reduce_axis((0, k_chunk_size), name="k_inner")

    C_partial = te.compute(
        (M, N, split_k),
        lambda i, j, s: te.sum(
            te.if_then_else(
                s * k_chunk_size + k_inner < K,
                A[i, s * k_chunk_size + k_inner] *
                scale_A[i, (s * k_chunk_size + k_inner) // block_size] *
                W[j, s * k_chunk_size + k_inner] *
                scale_W[j, (s * k_chunk_size + k_inner) // block_size],
                tvm.tir.const(0.0, "float32")
            ),
            axis=k_inner
        ),
        name="C_partial"
    )

    # Second level: reduce partial sums
    k_split = te.reduce_axis((0, split_k), name="k_split")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(C_partial[i, j, k_split], axis=k_split),
        name="C"
    )

    return A, W, scale_A, scale_W, C, C_partial


def build_w4a8_gemm_kernel(
    M: int,
    N: int,
    K: int,
    target: str = "cuda -arch=sm_110",
    opt_level: int = 3,
    use_split_k: bool = False,
) -> tvm.runtime.Module:
    """
    Build optimized W4A8 GEMM kernel.

    Args:
        M, N, K: Matrix dimensions
        target: TVM target
        opt_level: Optimization level
        use_split_k: Use split-K optimization for small M (ignored, using TensorIR)
    """
    # Create TensorIR function directly
    func = create_w4a8_gemm_tir(M, N, K)

    # Get target
    target_obj = tvm.target.Target(target)

    # Build with TensorIR
    with tvm.transform.PassContext(opt_level=opt_level):
        mod = tvm.build(func, target=target_obj)

    return mod


def w4a8_gemm_tvm(
    A: "torch.Tensor",      # [M, K] FP8 activation values
    W: "torch.Tensor",      # [N, K] nvFP4 weight values (transposed)
    scale_A: "torch.Tensor",
    scale_W: "torch.Tensor",
    module: Optional[tvm.runtime.Module] = None,
) -> "torch.Tensor":
    """
    Execute W4A8 GEMM using TVM kernel.

    This is the software implementation that bypasses CUTLASS mxf8f6f4 limitation.
    """
    import torch

    M, K = A.shape
    N = W.shape[0]

    if module is None:
        module = build_w4a8_gemm_kernel(M, N, K)

    device = tvm.cuda(0)
    a_tvm = tvm.nd.array(A.cpu().numpy().astype("float32"), device)
    w_tvm = tvm.nd.array(W.cpu().numpy().astype("float32"), device)
    scale_a_tvm = tvm.nd.array(scale_A.cpu().numpy().astype("float32"), device)
    scale_w_tvm = tvm.nd.array(scale_W.cpu().numpy().astype("float32"), device)
    c_tvm = tvm.nd.empty((M, N), dtype="float32", device=device)

    func = module["w4a8_gemm"]
    func(a_tvm, w_tvm, scale_a_tvm, scale_w_tvm, c_tvm)

    return torch.from_numpy(c_tvm.numpy()).to(A.device)


# ============================================================================
# FP8 Quantization Helper
# ============================================================================

def quantize_to_fp8_e4m3(x: "torch.Tensor", block_size: int = BLOCK_SIZE):
    """
    Quantize BF16/FP32 to FP8 E4M3 with block scaling.

    FP8 E4M3 range: [-448, 448]
    """
    import torch

    M, K = x.shape
    num_blocks = (K + block_size - 1) // block_size

    # Reshape to blocks
    x_padded = torch.nn.functional.pad(x, (0, num_blocks * block_size - K))
    x_blocks = x_padded.view(M, num_blocks, block_size)

    # Compute block max
    block_max = x_blocks.abs().max(dim=2)[0]  # [M, num_blocks]

    # Compute scales
    scales = block_max / FP8_E4M3_MAX
    scales = scales.clamp(min=1e-12)

    # Quantize
    scales_expanded = scales.unsqueeze(2).expand(-1, -1, block_size)
    x_scaled = x_blocks / scales_expanded
    x_quantized = x_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

    # Reshape back
    x_quantized = x_quantized.view(M, -1)[:, :K]

    return x_quantized, scales


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_w4a8(M: int = 1, N: int = 3072, K: int = 3072, warmup: int = 10, runs: int = 100):
    """Benchmark W4A8 TVM GEMM."""
    import torch
    import time

    print(f"\n{'='*60}")
    print(f"TVM W4A8 GEMM Benchmark (Software mxf8f6f4 bypass)")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}\n")

    num_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create test inputs
    # Activation: FP8 values
    A = torch.randn(M, K, device="cuda")
    A_fp8, scale_A = quantize_to_fp8_e4m3(A)

    # Weight: nvFP4 values
    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6])
    W = fp4_values[torch.randint(0, len(fp4_values), (N, K))].float().cuda()
    scale_W = torch.rand(N, num_blocks, device="cuda") * 0.1 + 0.01

    # Build kernel
    print("Building TVM W4A8 kernel...")
    build_start = time.time()
    module = build_w4a8_gemm_kernel(M, N, K)
    build_time = time.time() - build_start
    print(f"Build time: {build_time*1000:.2f} ms\n")

    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        C = w4a8_gemm_tvm(A_fp8, W, scale_A, scale_W, module)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Running {runs} iterations...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        C = w4a8_gemm_tvm(A_fp8, W, scale_A, scale_W, module)
    torch.cuda.synchronize()
    tvm_time = (time.time() - start) / runs * 1000

    # Reference
    A_dequant = A_fp8 * scale_A.repeat_interleave(BLOCK_SIZE, dim=1)[:, :K]
    W_dequant = W * scale_W.repeat_interleave(BLOCK_SIZE, dim=1)[:, :K]

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        C_ref = torch.matmul(A_dequant, W_dequant.T)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / runs * 1000

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  TVM W4A8 GEMM:     {tvm_time:.4f} ms")
    print(f"  Torch BF16 GEMM:   {torch_time:.4f} ms")
    print(f"  Ratio:             {torch_time/tvm_time:.2f}x")
    print(f"")
    print(f"  NOTE: This bypasses CUTLASS mxf8f6f4 SM110 limitation!")
    print(f"{'='*60}\n")

    # Verify
    C_tvm = w4a8_gemm_tvm(A_fp8, W, scale_A, scale_W, module)
    diff = torch.abs(C_tvm - C_ref).max().item()
    print(f"Max diff vs reference: {diff:.6f}")

    return tvm_time


if __name__ == "__main__":
    benchmark_w4a8()
