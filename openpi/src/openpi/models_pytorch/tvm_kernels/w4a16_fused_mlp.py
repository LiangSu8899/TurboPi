"""
TVM TensorIR: W4A16 Fused MLP Kernel

Fuses gate_proj + up_proj + SiLU*mul into a single kernel.

Benefits:
1. A is loaded once instead of twice (gate and up share input)
2. No intermediate storage for gate/up outputs
3. Single kernel launch instead of 3

MLP computation:
  gate = A @ W_gate.T     # [1, 16384]
  up = A @ W_up.T         # [1, 16384]
  intermediate = SiLU(gate) * up
  output = intermediate @ W_down.T  # [1, 2048]

This kernel implements: intermediate = SiLU(A @ W_gate.T) * (A @ W_up.T)

Author: Claude Code
Date: 2026-02-10
"""

import tvm
from tvm import te, tir
from tvm.script import tir as T
import numpy as np
from typing import Tuple, Optional
import argparse
import time

# Constants
BLOCK_SIZE = 32  # nvFP4 scaling block size

# nvFP4 E2M1 lookup table
NVFP4_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # positive
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0  # negative
], dtype=np.float32)


def quantize_to_nvfp4_packed(weight: np.ndarray, block_size: int = BLOCK_SIZE):
    """Quantize weight to packed nvFP4 format."""
    N, K = weight.shape
    num_blocks = (K + block_size - 1) // block_size

    # Compute scales per block
    scales = np.zeros((N, num_blocks), dtype=np.float32)
    for n in range(N):
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, K)
            block_max = np.abs(weight[n, start:end]).max()
            scales[n, b] = block_max / 6.0 if block_max > 0 else 1.0

    # Quantize to nvFP4 indices
    W_quant = np.zeros((N, K), dtype=np.int32)
    for n in range(N):
        for k in range(K):
            block_idx = k // block_size
            scaled_val = weight[n, k] / scales[n, block_idx]

            # Find closest nvFP4 value
            best_idx = 0
            best_diff = abs(scaled_val - NVFP4_LUT[0])
            for i in range(1, 16):
                diff = abs(scaled_val - NVFP4_LUT[i])
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            W_quant[n, k] = best_idx

    # Pack to uint8 (2 FP4 per byte)
    K_packed = K // 2
    W_packed = np.zeros((N, K_packed), dtype=np.uint8)
    for n in range(N):
        for k in range(0, K, 2):
            low = W_quant[n, k] & 0xF
            high = W_quant[n, k + 1] & 0xF
            W_packed[n, k // 2] = low | (high << 4)

    return W_packed, scales


# ==============================================================================
# Fused Gate+Up+SiLU*Mul Kernel
# ==============================================================================

def create_w4a16_fused_gate_up_silu(N: int, K: int, block_size: int = BLOCK_SIZE):
    """
    Fused gate_proj + up_proj + SiLU * mul kernel.

    Computes: out[1, N] = SiLU(A @ W_gate.T) * (A @ W_up.T)

    Both gate and up projections share the same input A, so we load A once
    and compute both projections in parallel.
    """
    num_blocks_k = (K + block_size - 1) // block_size
    K_packed = K // 2

    # Configuration: each thread computes one output element
    # For each output, we compute both gate and up, then fuse SiLU*mul
    THREADS_PER_BLOCK = 256
    num_thread_blocks = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    @T.prim_func
    def kernel(
        A: T.Buffer((1, K), "float32"),                        # Input activation
        W_gate_packed: T.Buffer((N, K_packed), "uint8"),       # Gate proj weights
        W_up_packed: T.Buffer((N, K_packed), "uint8"),         # Up proj weights
        scales_gate: T.Buffer((N, num_blocks_k), "float32"),   # Gate scales
        scales_up: T.Buffer((N, num_blocks_k), "float32"),     # Up scales
        out: T.Buffer((1, N), "float32"),                      # Output
    ):
        T.func_attr({
            "global_symbol": "w4a16_fused_gate_up_silu",
            "tir.noalias": True,
        })

        # nvFP4 lookup table in shared memory
        lut = T.alloc_buffer((16,), "float32", scope="shared")

        for bx in T.thread_binding(num_thread_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):

                # Initialize LUT
                if tx < 16:
                    if tx == 0:
                        lut[0] = T.float32(0.0)
                    elif tx == 1:
                        lut[1] = T.float32(0.5)
                    elif tx == 2:
                        lut[2] = T.float32(1.0)
                    elif tx == 3:
                        lut[3] = T.float32(1.5)
                    elif tx == 4:
                        lut[4] = T.float32(2.0)
                    elif tx == 5:
                        lut[5] = T.float32(3.0)
                    elif tx == 6:
                        lut[6] = T.float32(4.0)
                    elif tx == 7:
                        lut[7] = T.float32(6.0)
                    elif tx == 8:
                        lut[8] = T.float32(0.0)
                    elif tx == 9:
                        lut[9] = T.float32(-0.5)
                    elif tx == 10:
                        lut[10] = T.float32(-1.0)
                    elif tx == 11:
                        lut[11] = T.float32(-1.5)
                    elif tx == 12:
                        lut[12] = T.float32(-2.0)
                    elif tx == 13:
                        lut[13] = T.float32(-3.0)
                    elif tx == 14:
                        lut[14] = T.float32(-4.0)
                    elif tx == 15:
                        lut[15] = T.float32(-6.0)

                T.tvm_storage_sync("shared")

                j = bx * THREADS_PER_BLOCK + tx  # Output column

                if j < N:
                    # Compute both gate and up projections
                    gate_acc = T.float32(0)
                    up_acc = T.float32(0)

                    for k in T.serial(K):
                        # Get packed byte index
                        byte_idx = k // 2
                        is_high = k % 2

                        # Load both gate and up weights from same position
                        gate_packed = W_gate_packed[j, byte_idx]
                        up_packed = W_up_packed[j, byte_idx]

                        # Extract FP4 indices
                        gate_fp4_idx = T.if_then_else(
                            is_high == 0,
                            gate_packed & T.uint8(0xF),
                            (gate_packed >> 4) & T.uint8(0xF)
                        )
                        up_fp4_idx = T.if_then_else(
                            is_high == 0,
                            up_packed & T.uint8(0xF),
                            (up_packed >> 4) & T.uint8(0xF)
                        )

                        # Lookup and dequant
                        gate_w = lut[T.Cast("int32", gate_fp4_idx)]
                        up_w = lut[T.Cast("int32", up_fp4_idx)]

                        block_idx = k // block_size
                        gate_scale = scales_gate[j, block_idx]
                        up_scale = scales_up[j, block_idx]

                        gate_dequant = gate_w * gate_scale
                        up_dequant = up_w * up_scale

                        # Load A once, use for both
                        a_val = A[0, k]

                        # Accumulate
                        gate_acc = gate_acc + a_val * gate_dequant
                        up_acc = up_acc + a_val * up_dequant

                    # Apply SiLU(gate) * up
                    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                    sigmoid_gate = T.float32(1.0) / (T.float32(1.0) + T.exp(-gate_acc))
                    silu_gate = gate_acc * sigmoid_gate
                    out[0, j] = silu_gate * up_acc

    return kernel


def create_w4a16_fused_gate_up_silu_fast(N: int, K: int, block_size: int = BLOCK_SIZE):
    """
    Fused gate_proj + up_proj + SiLU * mul kernel - Optimized with K tiling.

    Same as above but with parallel reduction for better performance.
    """
    num_blocks_k = (K + block_size - 1) // block_size
    K_packed = K // 2

    # Configuration
    REDUCE_THREADS = 64
    OUTPUTS_PER_BLOCK = 4
    THREADS_PER_BLOCK = REDUCE_THREADS * OUTPUTS_PER_BLOCK  # 256

    TILE_K = min(K, 2048)
    num_thread_blocks = (N + OUTPUTS_PER_BLOCK - 1) // OUTPUTS_PER_BLOCK
    num_k_tiles = (K + TILE_K - 1) // TILE_K
    K_PER_THREAD_PER_TILE = (TILE_K + REDUCE_THREADS - 1) // REDUCE_THREADS

    @T.prim_func
    def kernel(
        A: T.Buffer((1, K), "float32"),
        W_gate_packed: T.Buffer((N, K_packed), "uint8"),
        W_up_packed: T.Buffer((N, K_packed), "uint8"),
        scales_gate: T.Buffer((N, num_blocks_k), "float32"),
        scales_up: T.Buffer((N, num_blocks_k), "float32"),
        out: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({
            "global_symbol": "w4a16_fused_gate_up_silu_fast",
            "tir.noalias": True,
        })

        # Shared memory
        lut = T.alloc_buffer((16,), "float32", scope="shared")
        A_shared = T.alloc_buffer((TILE_K,), "float32", scope="shared")
        gate_partial = T.alloc_buffer((OUTPUTS_PER_BLOCK, REDUCE_THREADS), "float32", scope="shared")
        up_partial = T.alloc_buffer((OUTPUTS_PER_BLOCK, REDUCE_THREADS), "float32", scope="shared")

        for bx in T.thread_binding(num_thread_blocks, thread="blockIdx.x"):
            for tx in T.thread_binding(THREADS_PER_BLOCK, thread="threadIdx.x"):

                # Initialize LUT
                if tx < 16:
                    if tx == 0:
                        lut[0] = T.float32(0.0)
                    elif tx == 1:
                        lut[1] = T.float32(0.5)
                    elif tx == 2:
                        lut[2] = T.float32(1.0)
                    elif tx == 3:
                        lut[3] = T.float32(1.5)
                    elif tx == 4:
                        lut[4] = T.float32(2.0)
                    elif tx == 5:
                        lut[5] = T.float32(3.0)
                    elif tx == 6:
                        lut[6] = T.float32(4.0)
                    elif tx == 7:
                        lut[7] = T.float32(6.0)
                    elif tx == 8:
                        lut[8] = T.float32(0.0)
                    elif tx == 9:
                        lut[9] = T.float32(-0.5)
                    elif tx == 10:
                        lut[10] = T.float32(-1.0)
                    elif tx == 11:
                        lut[11] = T.float32(-1.5)
                    elif tx == 12:
                        lut[12] = T.float32(-2.0)
                    elif tx == 13:
                        lut[13] = T.float32(-3.0)
                    elif tx == 14:
                        lut[14] = T.float32(-4.0)
                    elif tx == 15:
                        lut[15] = T.float32(-6.0)

                T.tvm_storage_sync("shared")

                # Thread assignment
                output_idx = tx // REDUCE_THREADS
                reduce_idx = tx % REDUCE_THREADS
                j = bx * OUTPUTS_PER_BLOCK + output_idx

                # Initialize partial sums
                gate_partial[output_idx, reduce_idx] = T.float32(0)
                up_partial[output_idx, reduce_idx] = T.float32(0)

                # Process K in tiles
                for kt in T.serial(num_k_tiles):
                    k_tile_start = kt * TILE_K

                    # Cooperative load of A tile
                    for load_iter in T.serial((TILE_K + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK):
                        k_local = tx + load_iter * THREADS_PER_BLOCK
                        if k_local < TILE_K:
                            k_global = k_tile_start + k_local
                            if k_global < K:
                                A_shared[k_local] = A[0, k_global]
                            else:
                                A_shared[k_local] = T.float32(0)

                    T.tvm_storage_sync("shared")

                    # Each thread processes K_PER_THREAD_PER_TILE elements
                    if j < N:
                        for k_iter in T.serial(K_PER_THREAD_PER_TILE):
                            k_local = reduce_idx + k_iter * REDUCE_THREADS
                            k_global = k_tile_start + k_local

                            if k_local < TILE_K:
                                if k_global < K:
                                    byte_idx = k_global // 2
                                    is_high = k_global % 2

                                    gate_packed = W_gate_packed[j, byte_idx]
                                    up_packed = W_up_packed[j, byte_idx]

                                    gate_fp4_idx = T.if_then_else(
                                        is_high == 0,
                                        gate_packed & T.uint8(0xF),
                                        (gate_packed >> 4) & T.uint8(0xF)
                                    )
                                    up_fp4_idx = T.if_then_else(
                                        is_high == 0,
                                        up_packed & T.uint8(0xF),
                                        (up_packed >> 4) & T.uint8(0xF)
                                    )

                                    gate_w = lut[T.Cast("int32", gate_fp4_idx)]
                                    up_w = lut[T.Cast("int32", up_fp4_idx)]

                                    block_idx_k = k_global // block_size
                                    gate_scale = scales_gate[j, block_idx_k]
                                    up_scale = scales_up[j, block_idx_k]

                                    gate_dequant = gate_w * gate_scale
                                    up_dequant = up_w * up_scale

                                    a_val = A_shared[k_local]

                                    gate_partial[output_idx, reduce_idx] = (
                                        gate_partial[output_idx, reduce_idx] + a_val * gate_dequant
                                    )
                                    up_partial[output_idx, reduce_idx] = (
                                        up_partial[output_idx, reduce_idx] + a_val * up_dequant
                                    )

                    T.tvm_storage_sync("shared")

                # Parallel reduction
                if reduce_idx < 32:
                    gate_partial[output_idx, reduce_idx] = (
                        gate_partial[output_idx, reduce_idx] +
                        gate_partial[output_idx, reduce_idx + 32]
                    )
                    up_partial[output_idx, reduce_idx] = (
                        up_partial[output_idx, reduce_idx] +
                        up_partial[output_idx, reduce_idx + 32]
                    )
                T.tvm_storage_sync("shared")

                if reduce_idx < 16:
                    gate_partial[output_idx, reduce_idx] = (
                        gate_partial[output_idx, reduce_idx] +
                        gate_partial[output_idx, reduce_idx + 16]
                    )
                    up_partial[output_idx, reduce_idx] = (
                        up_partial[output_idx, reduce_idx] +
                        up_partial[output_idx, reduce_idx + 16]
                    )
                T.tvm_storage_sync("shared")

                if reduce_idx < 8:
                    gate_partial[output_idx, reduce_idx] = (
                        gate_partial[output_idx, reduce_idx] +
                        gate_partial[output_idx, reduce_idx + 8]
                    )
                    up_partial[output_idx, reduce_idx] = (
                        up_partial[output_idx, reduce_idx] +
                        up_partial[output_idx, reduce_idx + 8]
                    )
                T.tvm_storage_sync("shared")

                if reduce_idx < 4:
                    gate_partial[output_idx, reduce_idx] = (
                        gate_partial[output_idx, reduce_idx] +
                        gate_partial[output_idx, reduce_idx + 4]
                    )
                    up_partial[output_idx, reduce_idx] = (
                        up_partial[output_idx, reduce_idx] +
                        up_partial[output_idx, reduce_idx + 4]
                    )
                T.tvm_storage_sync("shared")

                if reduce_idx < 2:
                    gate_partial[output_idx, reduce_idx] = (
                        gate_partial[output_idx, reduce_idx] +
                        gate_partial[output_idx, reduce_idx + 2]
                    )
                    up_partial[output_idx, reduce_idx] = (
                        up_partial[output_idx, reduce_idx] +
                        up_partial[output_idx, reduce_idx + 2]
                    )
                T.tvm_storage_sync("shared")

                # Final reduction and SiLU * mul
                if reduce_idx == 0:
                    if j < N:
                        gate_sum = gate_partial[output_idx, 0] + gate_partial[output_idx, 1]
                        up_sum = up_partial[output_idx, 0] + up_partial[output_idx, 1]

                        # SiLU(gate) * up
                        sigmoid_gate = T.float32(1.0) / (T.float32(1.0) + T.exp(-gate_sum))
                        silu_gate = gate_sum * sigmoid_gate
                        out[0, j] = silu_gate * up_sum

    return kernel


# ==============================================================================
# Build and Benchmark
# ==============================================================================

def build_kernel(kernel_func, target="cuda -arch=sm_110"):
    """Build TVM kernel."""
    target_obj = tvm.target.Target(target)
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.build(kernel_func, target=target_obj)
    return mod


def benchmark_fused_mlp(N=16384, K=2048, warmup=50, runs=200):
    """Benchmark fused gate+up+SiLU kernel vs separate kernels."""
    print(f"\n{'='*70}")
    print(f"W4A16 Fused MLP Benchmark (gate + up + SiLU*mul)")
    print(f"Shape: M=1, N={N}, K={K}")
    print(f"{'='*70}")

    # Generate test data
    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float32)
    W_gate_np = np.random.randn(N, K).astype(np.float32) * 0.1
    W_up_np = np.random.randn(N, K).astype(np.float32) * 0.1

    # Quantize
    W_gate_packed, scales_gate = quantize_to_nvfp4_packed(W_gate_np)
    W_up_packed, scales_up = quantize_to_nvfp4_packed(W_up_np)

    num_blocks_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    print(f"\nWeight memory:")
    print(f"  Original (2x BF16): {2 * N * K * 2 / 1e6:.2f} MB")
    print(f"  Packed (2x FP4):    {2 * W_gate_packed.nbytes / 1e6:.2f} MB ({2 * N * K * 2 / (2 * W_gate_packed.nbytes):.1f}x compression)")

    # Prepare TVM arrays
    device = tvm.runtime.cuda(0)

    A_tvm = tvm.runtime.empty((1, K), dtype="float32", device=device)
    A_tvm.copyfrom(A_np)

    W_gate_packed_tvm = tvm.runtime.empty((N, K // 2), dtype="uint8", device=device)
    W_gate_packed_tvm.copyfrom(W_gate_packed)

    W_up_packed_tvm = tvm.runtime.empty((N, K // 2), dtype="uint8", device=device)
    W_up_packed_tvm.copyfrom(W_up_packed)

    scales_gate_tvm = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
    scales_gate_tvm.copyfrom(scales_gate)

    scales_up_tvm = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
    scales_up_tvm.copyfrom(scales_up)

    out_tvm = tvm.runtime.empty((1, N), dtype="float32", device=device)

    # Reference: separate computation
    def dequant_weight(W_packed, scales, K):
        N = W_packed.shape[0]
        W_dequant = np.zeros((N, K), dtype=np.float32)
        for n in range(N):
            for k in range(K):
                byte_idx = k // 2
                is_high = k % 2
                packed = W_packed[n, byte_idx]
                fp4_idx = ((packed >> 4) & 0xF) if is_high else (packed & 0xF)
                w_val = NVFP4_LUT[fp4_idx]
                block_idx = k // BLOCK_SIZE
                W_dequant[n, k] = w_val * scales[n, block_idx]
        return W_dequant

    W_gate_dequant = dequant_weight(W_gate_packed, scales_gate, K)
    W_up_dequant = dequant_weight(W_up_packed, scales_up, K)

    gate_ref = A_np @ W_gate_dequant.T
    up_ref = A_np @ W_up_dequant.T
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_ref))
    out_ref = (gate_ref * sigmoid_gate) * up_ref

    results = []

    # Test simple fused kernel
    print(f"\n--- Simple Fused (1 thread/output) ---")
    try:
        kernel_func = create_w4a16_fused_gate_up_silu(N, K)
        mod = build_kernel(kernel_func)
        func = mod["w4a16_fused_gate_up_silu"]
        print("  Build successful!")

        # Warmup
        for _ in range(warmup):
            func(A_tvm, W_gate_packed_tvm, W_up_packed_tvm, scales_gate_tvm, scales_up_tvm, out_tvm)
        tvm.runtime.cuda(0).sync()

        # Benchmark
        tvm.runtime.cuda(0).sync()
        start = time.time()
        for _ in range(runs):
            func(A_tvm, W_gate_packed_tvm, W_up_packed_tvm, scales_gate_tvm, scales_up_tvm, out_tvm)
        tvm.runtime.cuda(0).sync()
        simple_ms = (time.time() - start) / runs * 1000

        # Verify
        out_np = out_tvm.numpy()
        cos_sim = np.dot(out_np.flatten(), out_ref.flatten()) / (
            np.linalg.norm(out_np) * np.linalg.norm(out_ref) + 1e-8)

        print(f"  Time:    {simple_ms:.4f} ms")
        print(f"  Cos sim: {cos_sim:.6f}")

        results.append(("Simple Fused", simple_ms, cos_sim > 0.99))
    except Exception as e:
        print(f"  Failed: {e}")

    # Test fast fused kernel
    print(f"\n--- Fast Fused (parallel reduction) ---")
    try:
        kernel_func = create_w4a16_fused_gate_up_silu_fast(N, K)
        mod = build_kernel(kernel_func)
        func = mod["w4a16_fused_gate_up_silu_fast"]
        print("  Build successful!")

        # Warmup
        for _ in range(warmup):
            func(A_tvm, W_gate_packed_tvm, W_up_packed_tvm, scales_gate_tvm, scales_up_tvm, out_tvm)
        tvm.runtime.cuda(0).sync()

        # Benchmark
        tvm.runtime.cuda(0).sync()
        start = time.time()
        for _ in range(runs):
            func(A_tvm, W_gate_packed_tvm, W_up_packed_tvm, scales_gate_tvm, scales_up_tvm, out_tvm)
        tvm.runtime.cuda(0).sync()
        fast_ms = (time.time() - start) / runs * 1000

        # Verify
        out_np = out_tvm.numpy()
        cos_sim = np.dot(out_np.flatten(), out_ref.flatten()) / (
            np.linalg.norm(out_np) * np.linalg.norm(out_ref) + 1e-8)

        print(f"  Time:    {fast_ms:.4f} ms")
        print(f"  Cos sim: {cos_sim:.6f}")

        results.append(("Fast Fused", fast_ms, cos_sim > 0.99))
    except Exception as e:
        print(f"  Failed: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")

    # Compare with separate kernels baseline
    SEPARATE_GATE_UP_MS = 0.224 * 2  # Two separate kernels
    SILU_MUL_OVERHEAD = 0.02  # Estimated SiLU*mul kernel

    for name, time_ms, correct in results:
        status = "✅" if correct else "❌"
        speedup = (SEPARATE_GATE_UP_MS + SILU_MUL_OVERHEAD) / time_ms
        print(f"  {name:<25} {time_ms:.4f}ms  vs Separate: {speedup:.2f}x  {status}")

    print(f"\nBaseline (separate kernels):")
    print(f"  gate + up:   2 x 0.224ms = 0.448ms")
    print(f"  SiLU * mul:  ~0.02ms")
    print(f"  Total:       ~0.47ms")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=16384)
    parser.add_argument("--K", type=int, default=2048)
    args = parser.parse_args()

    benchmark_fused_mlp(N=args.N, K=args.K)
