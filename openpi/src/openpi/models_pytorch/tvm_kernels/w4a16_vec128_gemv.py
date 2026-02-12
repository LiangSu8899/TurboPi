#!/usr/bin/env python3
"""
W4A16 GEMV with 128-bit Vectorized Loads.

Key insight: Previous kernel used scalar 8-bit loads, causing instruction bottleneck.
This kernel uses 128-bit coalesced loads via block-interleaved layout.

Layout change:
- Old: (K_packed, N) uint8 - one byte at a time, 16 loads per quant block
- New: (num_scale_blocks, N, 4) uint32 - one 128-bit load per quant block

Memory: 4 contiguous uint32 = 128 bits = 16 bytes = 32 int4 = 1 quant block

Author: Claude Code
Date: 2026-02-11
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import runtime
from tvm.script import tir as T
import numpy as np
import time


# ============= Kernel Parameters =============
N, K = 16384, 2048
QUANT_BLOCK = 32
num_scale_blocks = K // QUANT_BLOCK  # 64
THREADS = 256
num_blocks = (N + THREADS - 1) // THREADS


# ============= Version 1: Block-Interleaved Layout =============
@T.prim_func
def gemv_vec128_v1(
    A: T.Buffer((1, K), "float16"),
    # Block-interleaved: (num_scale_blocks, N, 4) uint32
    # Adjacent threads access adjacent N, enabling coalescing
    # 4 uint32 per thread per qb = 128-bit per access
    W_packed: T.Buffer((num_scale_blocks, N, 4), "uint32"),
    scales_T: T.Buffer((num_scale_blocks, N), "float16"),
    C: T.Buffer((1, N), "float32"),
):
    T.func_attr({"global_symbol": "gemv_vec128_v1", "tir.noalias": True})

    A_shared = T.alloc_buffer((K,), "float16", scope="shared")

    for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
        # Load A to shared memory
        for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
            for i in range((K + THREADS - 1) // THREADS):
                k = tid + i * THREADS
                if k < K:
                    A_shared[k] = A[0, k]

        T.tvm_storage_sync("shared")

        for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
            n = block_idx * THREADS + tid
            if n < N:
                C[0, n] = T.float32(0)

                for qb in range(num_scale_blocks):
                    scale = scales_T[qb, n]
                    k_base = qb * QUANT_BLOCK

                    # Load 4 x uint32 = 128 bits
                    # These are contiguous in memory: W_packed[qb, n, 0:4]
                    u0 = W_packed[qb, n, 0]
                    u1 = W_packed[qb, n, 1]
                    u2 = W_packed[qb, n, 2]
                    u3 = W_packed[qb, n, 3]

                    # Decode u0: int4[0:7]
                    for i in range(8):
                        int4_val = (u0 >> T.uint32(i * 4)) & T.uint32(0xF)
                        k_idx = k_base + i
                        w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                        C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_idx] * w)

                    # Decode u1: int4[8:15]
                    for i in range(8):
                        int4_val = (u1 >> T.uint32(i * 4)) & T.uint32(0xF)
                        k_idx = k_base + 8 + i
                        w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                        C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_idx] * w)

                    # Decode u2: int4[16:23]
                    for i in range(8):
                        int4_val = (u2 >> T.uint32(i * 4)) & T.uint32(0xF)
                        k_idx = k_base + 16 + i
                        w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                        C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_idx] * w)

                    # Decode u3: int4[24:31]
                    for i in range(8):
                        int4_val = (u3 >> T.uint32(i * 4)) & T.uint32(0xF)
                        k_idx = k_base + 24 + i
                        w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                        C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_idx] * w)


# ============= Version 2: Explicit uint32x4 Vector Type =============
@T.prim_func
def gemv_vec128_v2(
    A: T.Buffer((1, K), "float16"),
    # Explicit vector type: one 128-bit load per (qb, n)
    W_packed_vec: T.Buffer((num_scale_blocks, N), "uint32x4"),
    scales_T: T.Buffer((num_scale_blocks, N), "float16"),
    C: T.Buffer((1, N), "float32"),
):
    T.func_attr({"global_symbol": "gemv_vec128_v2", "tir.noalias": True})

    A_shared = T.alloc_buffer((K,), "float16", scope="shared")

    for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
        for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
            for i in range((K + THREADS - 1) // THREADS):
                k = tid + i * THREADS
                if k < K:
                    A_shared[k] = A[0, k]

        T.tvm_storage_sync("shared")

        for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
            n = block_idx * THREADS + tid
            if n < N:
                C[0, n] = T.float32(0)

                for qb in range(num_scale_blocks):
                    scale = scales_T[qb, n]
                    k_base = qb * QUANT_BLOCK

                    # Single 128-bit vectorized load!
                    vec = W_packed_vec[qb, n]

                    # Extract 4 x uint32 using Shuffle
                    u0 = T.Shuffle([vec], [0])
                    u1 = T.Shuffle([vec], [1])
                    u2 = T.Shuffle([vec], [2])
                    u3 = T.Shuffle([vec], [3])

                    # Decode and accumulate (same as v1)
                    for i in range(8):
                        int4_val = (u0 >> T.uint32(i * 4)) & T.uint32(0xF)
                        w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                        C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_base + i] * w)

                    for i in range(8):
                        int4_val = (u1 >> T.uint32(i * 4)) & T.uint32(0xF)
                        w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                        C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_base + 8 + i] * w)

                    for i in range(8):
                        int4_val = (u2 >> T.uint32(i * 4)) & T.uint32(0xF)
                        w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                        C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_base + 16 + i] * w)

                    for i in range(8):
                        int4_val = (u3 >> T.uint32(i * 4)) & T.uint32(0xF)
                        w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                        C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_base + 24 + i] * w)


# ============= Version 3: Vectorized Loop Annotation =============
@T.prim_func
def gemv_vec128_v3(
    A: T.Buffer((1, K), "float16"),
    W_packed: T.Buffer((num_scale_blocks, N, 4), "uint32"),
    scales_T: T.Buffer((num_scale_blocks, N), "float16"),
    C: T.Buffer((1, N), "float32"),
):
    T.func_attr({"global_symbol": "gemv_vec128_v3", "tir.noalias": True})

    A_shared = T.alloc_buffer((K,), "float16", scope="shared")
    W_local = T.alloc_buffer((4,), "uint32", scope="local")

    for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
        for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
            for i in range((K + THREADS - 1) // THREADS):
                k = tid + i * THREADS
                if k < K:
                    A_shared[k] = A[0, k]

        T.tvm_storage_sync("shared")

        for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
            n = block_idx * THREADS + tid
            if n < N:
                C[0, n] = T.float32(0)

                for qb in range(num_scale_blocks):
                    scale = scales_T[qb, n]
                    k_base = qb * QUANT_BLOCK

                    # Vectorized load annotation - hint to compiler
                    for v in T.vectorized(4):
                        W_local[v] = W_packed[qb, n, v]

                    # Process 4 uint32 from local buffer
                    for u_idx in range(4):
                        u = W_local[u_idx]
                        k_offset = u_idx * 8

                        for i in range(8):
                            int4_val = (u >> T.uint32(i * 4)) & T.uint32(0xF)
                            k_idx = k_base + k_offset + i
                            w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                            C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_idx] * w)


# ============= Quantization Helper =============
def quantize_to_block_interleaved(W, block_size=QUANT_BLOCK):
    """
    Quantize weights to block-interleaved layout for vectorized loads.

    Input: W (N, K) float32
    Output:
        W_packed: (num_scale_blocks, N, 4) uint32 - block interleaved
        scales: (num_scale_blocks, N) float16
    """
    N_dim, K_dim = W.shape
    num_blocks_k = K_dim // block_size

    # First quantize to standard format
    W_int4 = np.zeros((N_dim, K_dim), dtype=np.int8)
    scales = np.zeros((num_blocks_k, N_dim), dtype=np.float16)

    for n in range(N_dim):
        for b in range(num_blocks_k):
            start = b * block_size
            end = start + block_size
            block = W[n, start:end]

            max_abs = np.max(np.abs(block))
            scale = max_abs / 7.0 if max_abs > 0 else 1.0
            scales[b, n] = scale

            for k in range(block_size):
                val = block[k] / scale if scale > 0 else 0
                quantized = int(np.clip(np.round(val + 8), 0, 15))
                W_int4[n, start + k] = quantized

    # Pack to block-interleaved uint32 layout
    # Each quant block = 32 int4 = 16 bytes = 4 uint32
    W_packed = np.zeros((num_blocks_k, N_dim, 4), dtype=np.uint32)

    for n in range(N_dim):
        for qb in range(num_blocks_k):
            k_base = qb * block_size

            for u_idx in range(4):  # 4 uint32 per quant block
                val = np.uint32(0)
                for i in range(8):  # 8 int4 per uint32
                    k = k_base + u_idx * 8 + i
                    int4_val = W_int4[n, k]
                    val |= np.uint32(int4_val) << np.uint32(i * 4)
                W_packed[qb, n, u_idx] = val

    return W_packed, scales


def dequantize_block_interleaved(W_packed, scales, K_dim):
    """Dequantize for verification."""
    num_blocks_k, N_dim, _ = W_packed.shape
    W = np.zeros((N_dim, K_dim), dtype=np.float32)

    for n in range(N_dim):
        for qb in range(num_blocks_k):
            scale = scales[qb, n]
            k_base = qb * QUANT_BLOCK

            for u_idx in range(4):
                u = W_packed[qb, n, u_idx]
                for i in range(8):
                    int4_val = (u >> (i * 4)) & 0xF
                    k = k_base + u_idx * 8 + i
                    W[n, k] = (int4_val - 8) * scale

    return W


# ============= Benchmark =============
def benchmark_kernel(kernel_func, kernel_name, W_packed, scales, use_vec_type=False):
    """Benchmark a kernel and check correctness."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {kernel_name}")
    print(f"{'='*60}")

    # Prepare data
    np.random.seed(42)
    A_np = np.random.randn(1, K).astype(np.float16)

    # Reference
    W_dequant = dequantize_block_interleaved(W_packed, scales, K)
    C_ref = A_np.astype(np.float32) @ W_dequant.T

    # Build kernel
    try:
        mod = tvm.IRModule({"main": kernel_func})
        target = tvm.target.Target("cuda -arch=sm_110")
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.build(mod, target=target)
    except Exception as e:
        print(f"Build FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Get function
    func_names = ["gemv_vec128_v1", "gemv_vec128_v2", "gemv_vec128_v3", "main"]
    func = None
    for name in func_names:
        try:
            func = lib[name]
            break
        except:
            continue

    if func is None:
        print("Function not found!")
        return None

    # Prepare TVM tensors
    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)

    if use_vec_type:
        # For uint32x4, flatten to (num_scale_blocks * N * 4,) and reshape
        # Actually TVM expects the same underlying memory
        W_flat = W_packed.reshape(-1).view(np.uint32)
        # Reinterpret as vector: (num_scale_blocks, N) with 4 uint32 each
        W_tvm = runtime.empty((num_scale_blocks, N), "uint32x4", device)
        # Copy raw bytes
        W_tvm_array = W_tvm.numpy()  # This might not work for vector types
        # Alternative: copy via flat buffer
        W_tvm_flat = runtime.empty((num_scale_blocks * N * 4,), "uint32", device)
        W_tvm_flat.copyfrom(W_packed.reshape(-1))
        # Try to use the flat buffer directly? This is tricky...
        # For now, skip vec type version if it fails
        print("Note: Vector type buffer allocation is complex, using standard layout")
        W_tvm = runtime.empty(W_packed.shape, "uint32", device)
        W_tvm.copyfrom(W_packed)
    else:
        W_tvm = runtime.empty(W_packed.shape, "uint32", device)
        W_tvm.copyfrom(W_packed)

    scales_tvm = runtime.empty(scales.shape, "float16", device)
    scales_tvm.copyfrom(scales)
    C_tvm = runtime.empty((1, N), "float32", device)

    # Test run
    try:
        func(A_tvm, W_tvm, scales_tvm, C_tvm)
        device.sync()
    except Exception as e:
        print(f"Execution FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Check correctness
    C_result = C_tvm.numpy()
    if np.isnan(C_result).any():
        print("Result contains NaN!")
        return None

    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)
    print(f"Cosine similarity: {cos_sim:.6f}")

    if cos_sim < 0.99:
        print("FAILED: Low accuracy!")
        return None

    # Warmup
    warmup, runs = 50, 200
    for _ in range(warmup):
        func(A_tvm, W_tvm, scales_tvm, C_tvm)
    device.sync()

    # Benchmark
    start = time.time()
    for _ in range(runs):
        func(A_tvm, W_tvm, scales_tvm, C_tvm)
    device.sync()
    avg_ms = (time.time() - start) / runs * 1000

    # Calculate bandwidth
    weight_bytes = num_scale_blocks * N * 4 * 4  # uint32
    scale_bytes = num_scale_blocks * N * 2  # float16
    a_bytes = K * 2  # float16
    total_bytes = weight_bytes + scale_bytes + a_bytes
    bandwidth = total_bytes / (avg_ms / 1000) / 1e9

    print(f"\nPerformance:")
    print(f"  Latency: {avg_ms:.4f} ms")
    print(f"  Bandwidth: {bandwidth:.1f} GB/s")
    print(f"  Data moved: {total_bytes / 1e6:.2f} MB")
    print(f"  Target (< 0.2ms): {'ACHIEVED!' if avg_ms < 0.2 else f'{avg_ms/0.2:.1f}x slower'}")

    return avg_ms


def analyze_cuda_code(kernel_func, kernel_name):
    """Analyze generated CUDA code for vectorized loads."""
    print(f"\n{'='*60}")
    print(f"CUDA Analysis: {kernel_name}")
    print(f"{'='*60}")

    mod = tvm.IRModule({"main": kernel_func})
    target = tvm.target.Target("cuda -arch=sm_110")

    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.build(mod, target=target)

    # Try different ways to get CUDA source
    cuda_src = None
    try:
        # New TVM API
        for m in lib.get_lib().get_modules():
            if hasattr(m, 'get_source'):
                src = m.get_source()
                if src and len(src) > 100:
                    cuda_src = src
                    break
    except:
        pass

    if cuda_src is None:
        try:
            # Try direct method
            cuda_src = lib.get_source("cuda")
        except:
            pass

    if cuda_src is None:
        print("Could not extract CUDA source code")
        return None

    # Count load instructions
    lines = cuda_src.split('\n')

    ld_global_count = sum(1 for line in lines if 'ld.global' in line)
    ld_v4_count = sum(1 for line in lines if 'ld.global.v4' in line or 'ld.global.nc.v4' in line)
    ld_v2_count = sum(1 for line in lines if 'ld.global.v2' in line or 'ld.global.nc.v2' in line)
    ld_u32_count = sum(1 for line in lines if '.u32' in line and 'ld.global' in line)

    print(f"Load instruction analysis:")
    print(f"  Total ld.global: {ld_global_count}")
    print(f"  ld.global.v4 (128-bit): {ld_v4_count}")
    print(f"  ld.global.v2 (64-bit): {ld_v2_count}")
    print(f"  ld.global.*.u32: {ld_u32_count}")

    # Show relevant PTX lines
    print(f"\nLoad instruction samples:")
    count = 0
    for line in lines:
        if 'ld.global' in line and len(line.strip()) > 0:
            print(f"  {line.strip()[:100]}")
            count += 1
            if count >= 10:
                break

    return cuda_src


def main():
    print("="*70)
    print("W4A16 GEMV with 128-bit Vectorized Loads")
    print(f"N={N}, K={K}, QUANT_BLOCK={QUANT_BLOCK}")
    print("="*70)

    # Memory analysis
    weight_bytes_old = N * K // 2  # Original uint8 layout
    weight_bytes_new = num_scale_blocks * N * 4 * 4  # New uint32 layout (same size)
    scale_bytes = num_scale_blocks * N * 2
    total_bytes = weight_bytes_new + scale_bytes + K * 2

    print(f"\nMemory footprint:")
    print(f"  Weights (uint32): {weight_bytes_new / 1e6:.2f} MB")
    print(f"  Scales: {scale_bytes / 1e6:.2f} MB")
    print(f"  Total: {total_bytes / 1e6:.2f} MB")
    print(f"\nTheoretical limits:")
    print(f"  DRAM (55 GB/s): {total_bytes / (55e9) * 1000:.4f} ms")
    print(f"  L2 (230 GB/s): {total_bytes / (230e9) * 1000:.4f} ms")

    # Create test data
    print("\nQuantizing weights...")
    np.random.seed(42)
    W_np = np.random.randn(N, K).astype(np.float32)
    W_packed, scales = quantize_to_block_interleaved(W_np)
    print(f"W_packed shape: {W_packed.shape} (block-interleaved)")
    print(f"scales shape: {scales.shape}")

    # Analyze CUDA code first (skip if fails)
    try:
        analyze_cuda_code(gemv_vec128_v1, "V1: Block-Interleaved")
    except Exception as e:
        print(f"CUDA analysis skipped: {e}")

    # Benchmark kernels
    results = {}

    # V1: Block-interleaved layout
    ms = benchmark_kernel(gemv_vec128_v1, "V1: Block-Interleaved uint32", W_packed, scales)
    if ms:
        results["V1-Interleaved"] = ms

    # V3: Vectorized loop annotation
    ms = benchmark_kernel(gemv_vec128_v3, "V3: Vectorized Loop", W_packed, scales)
    if ms:
        results["V3-VecLoop"] = ms

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if results:
        print(f"\n{'Kernel':<25} | {'Latency (ms)':<12} | {'vs 0.2ms':<10} | {'Bandwidth':<12}")
        print("-"*70)
        for name, ms in sorted(results.items(), key=lambda x: x[1]):
            status = "ACHIEVED" if ms <= 0.2 else f"{ms/0.2:.1f}x slower"
            # Estimate bandwidth
            bw = total_bytes / (ms / 1000) / 1e9
            print(f"{name:<25} | {ms:<12.4f} | {status:<10} | {bw:.1f} GB/s")

        best = min(results.values())
        print(f"\nBest: {best:.4f} ms")
        if best <= 0.2:
            print("TARGET ACHIEVED!")
        else:
            print(f"Gap to target: {best / 0.2:.1f}x")


if __name__ == "__main__":
    main()
