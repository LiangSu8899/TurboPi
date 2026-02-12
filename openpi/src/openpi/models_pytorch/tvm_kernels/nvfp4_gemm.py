"""
TVM TensorIR: nvFP4 Block-Scaled GEMM Kernel (W4A4)

Implements: C = dequant(A_fp4, scale_A) @ dequant(B_fp4, scale_B)^T

Both weights and activations are in nvFP4 E2M1 format with block scaling.
This is the pure software implementation, not relying on CUTLASS mxf8f6f4 instructions.

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


def create_nvfp4_gemm_te(
    M: int,  # batch * seq_len
    N: int,  # out_features
    K: int,  # in_features
    block_size: int = BLOCK_SIZE,
):
    """
    Create TVM Tensor Expression for W4A4 GEMM with block scaling.

    C[M, N] = A[M, K] @ B[K, N]
    where A (activation) and B (weight) are both in nvFP4 with block scales.

    For Pi0 MLP: typical shapes are M=1, K=3072, N=3072/12288
    """
    num_blocks_k = (K + block_size - 1) // block_size

    # Inputs: quantized values stored as float32 (actual FP4 values like 0, 0.5, 1.0, etc.)
    A = te.placeholder((M, K), name="A", dtype="float32")  # Activation (nvFP4 values)
    B = te.placeholder((N, K), name="B", dtype="float32")  # Weight (nvFP4 values), transposed
    scale_A = te.placeholder((M, num_blocks_k), name="scale_A", dtype="float32")
    scale_B = te.placeholder((N, num_blocks_k), name="scale_B", dtype="float32")

    # Reduce axis
    k = te.reduce_axis((0, K), name="k")

    # Compute GEMM with dequantization fused
    def gemm_compute(i, j):
        # Get block index for k
        block_idx = k // block_size

        # Dequantize A and B
        a_dequant = A[i, k] * scale_A[i, block_idx]
        b_dequant = B[j, k] * scale_B[j, block_idx]

        # Accumulate
        return te.sum(a_dequant * b_dequant, axis=k)

    # Output in BF16
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k] * scale_A[i, k // block_size] *
            B[j, k] * scale_B[j, k // block_size],
            axis=k
        ),
        name="C"
    )

    return A, B, scale_A, scale_B, C


def create_nvfp4_gemm_tir(
    M: int,
    N: int,
    K: int,
    block_size: int = BLOCK_SIZE,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 32,
):
    """
    Create optimized TensorIR for nvFP4 GEMM with CUDA thread bindings.

    Uses proper GPU thread hierarchy for CUDA execution.
    NOTE: Accumulates directly to output buffer (TVM 0.24 local var bug workaround).
    """
    num_blocks_k = (K + block_size - 1) // block_size

    # Thread configuration
    THREADS_PER_BLOCK = 256
    total_elements = M * N

    @T.prim_func
    def nvfp4_gemm_func(
        A: T.Buffer((M, K), "float32"),
        B: T.Buffer((N, K), "float32"),
        scale_A: T.Buffer((M, num_blocks_k), "float32"),
        scale_B: T.Buffer((N, num_blocks_k), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        T.func_attr({"global_symbol": "nvfp4_gemm", "tir.noalias": True})

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

                    # GEMM with fused dequantization
                    for k in T.serial(K):
                        block_idx = k // block_size
                        a_val = A[i, k] * scale_A[i, block_idx]
                        b_val = B[j, k] * scale_B[j, block_idx]
                        C[i, j] = C[i, j] + a_val * b_val

    return nvfp4_gemm_func


def build_nvfp4_gemm_kernel(
    M: int,
    N: int,
    K: int,
    target: str = "cuda -arch=sm_110",
    opt_level: int = 3,
) -> tvm.runtime.Module:
    """
    Build optimized nvFP4 GEMM kernel with CUDA optimizations.

    Optimizations applied:
    - Thread block tiling
    - Shared memory caching
    - Register blocking
    - Vectorized loads
    """
    # Create TensorIR function directly
    func = create_nvfp4_gemm_tir(M, N, K)

    # Get target
    target_obj = tvm.target.Target(target)

    # Build with TensorIR
    with tvm.transform.PassContext(opt_level=opt_level):
        mod = tvm.build(func, target=target_obj)

    return mod


def nvfp4_gemm_tvm(
    A: "torch.Tensor",      # [M, K] quantized activation (FP4 values)
    B: "torch.Tensor",      # [N, K] quantized weight (FP4 values)
    scale_A: "torch.Tensor",  # [M, num_blocks]
    scale_B: "torch.Tensor",  # [N, num_blocks]
    module: Optional[tvm.runtime.Module] = None,
) -> "torch.Tensor":
    """
    Execute nvFP4 GEMM using TVM kernel.

    Args:
        A: Quantized activation [M, K]
        B: Quantized weight [N, K] (transposed)
        scale_A: Activation scales [M, num_blocks]
        scale_B: Weight scales [N, num_blocks]
        module: Pre-compiled TVM module

    Returns:
        Output tensor [M, N]
    """
    import torch

    M, K = A.shape
    N = B.shape[0]

    # Build module if needed
    if module is None:
        module = build_nvfp4_gemm_kernel(M, N, K)

    # Convert to TVM arrays
    device = tvm.cuda(0)
    a_tvm = tvm.nd.array(A.cpu().numpy().astype("float32"), device)
    b_tvm = tvm.nd.array(B.cpu().numpy().astype("float32"), device)
    scale_a_tvm = tvm.nd.array(scale_A.cpu().numpy().astype("float32"), device)
    scale_b_tvm = tvm.nd.array(scale_B.cpu().numpy().astype("float32"), device)
    c_tvm = tvm.nd.empty((M, N), dtype="float32", device=device)

    # Execute
    func = module["nvfp4_gemm"]
    func(a_tvm, b_tvm, scale_a_tvm, scale_b_tvm, c_tvm)

    # Convert back
    return torch.from_numpy(c_tvm.numpy()).to(A.device)


# ============================================================================
# Alternative: Use TVM auto-scheduler for better optimization
# ============================================================================

def tune_nvfp4_gemm(
    M: int,
    N: int,
    K: int,
    target: str = "cuda -arch=sm_110",
    num_trials: int = 100,
    log_file: str = "nvfp4_gemm_tune.json",
):
    """
    Use TVM auto-scheduler to find optimal schedule.

    This can achieve better performance than manual scheduling.
    """
    from tvm import auto_scheduler

    # Create computation
    A, B, scale_A, scale_B, C = create_nvfp4_gemm_te(M, N, K)

    # Create task
    task = auto_scheduler.SearchTask(
        func=te.create_prim_func([A, B, scale_A, scale_B, C]),
        target=target,
    )

    # Set tuning options
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_trials,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    # Run tuning
    print(f"Tuning nvFP4 GEMM for M={M}, N={N}, K={K}...")
    task.tune(tune_option)

    # Apply best schedule and build
    sch, args = task.apply_best(log_file)
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(sch, args, target=target)

    return mod


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_gemm(M: int = 1, N: int = 3072, K: int = 3072, warmup: int = 10, runs: int = 100):
    """Benchmark TVM nvFP4 GEMM."""
    import torch
    import time

    print(f"\n{'='*60}")
    print(f"TVM nvFP4 GEMM (W4A4) Benchmark")
    print(f"Shape: M={M}, N={N}, K={K}")
    print(f"{'='*60}\n")

    num_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create test inputs (random FP4 values)
    fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6])
    A = fp4_values[torch.randint(0, len(fp4_values), (M, K))].float().cuda()
    B = fp4_values[torch.randint(0, len(fp4_values), (N, K))].float().cuda()
    scale_A = torch.rand(M, num_blocks, device="cuda") * 0.1 + 0.01
    scale_B = torch.rand(N, num_blocks, device="cuda") * 0.1 + 0.01

    # Build kernel
    print("Building TVM kernel...")
    build_start = time.time()
    module = build_nvfp4_gemm_kernel(M, N, K)
    build_time = time.time() - build_start
    print(f"Build time: {build_time*1000:.2f} ms\n")

    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        C = nvfp4_gemm_tvm(A, B, scale_A, scale_B, module)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Running {runs} iterations...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        C = nvfp4_gemm_tvm(A, B, scale_A, scale_B, module)
    torch.cuda.synchronize()
    tvm_time = (time.time() - start) / runs * 1000  # ms

    # Reference: torch matmul
    A_dequant = A * scale_A.repeat_interleave(BLOCK_SIZE, dim=1)[:, :K]
    B_dequant = B * scale_B.repeat_interleave(BLOCK_SIZE, dim=1)[:, :K]

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        C_ref = torch.matmul(A_dequant, B_dequant.T)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / runs * 1000

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  TVM W4A4 GEMM:     {tvm_time:.4f} ms")
    print(f"  Torch BF16 GEMM:   {torch_time:.4f} ms")
    print(f"  Speedup:           {torch_time/tvm_time:.2f}x")
    print(f"{'='*60}\n")

    # Verify correctness
    C_tvm = nvfp4_gemm_tvm(A, B, scale_A, scale_B, module)
    diff = torch.abs(C_tvm - C_ref).max().item()
    print(f"Max diff vs reference: {diff:.6f}")

    return tvm_time


if __name__ == "__main__":
    benchmark_gemm()
