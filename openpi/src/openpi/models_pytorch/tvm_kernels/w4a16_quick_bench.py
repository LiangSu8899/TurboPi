#!/usr/bin/env python3
"""Quick benchmark to verify transposed layout performance."""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import runtime
from tvm.script import tir as T
import numpy as np
import time

N, K = 16384, 2048
QUANT_BLOCK = 32
num_scale_blocks = K // QUANT_BLOCK
K_packed = K // 2
THREADS = 256
num_blocks = (N + THREADS - 1) // THREADS

@T.prim_func
def gemv_transposed(
    A: T.Buffer((1, K), "float16"),
    W_packed_T: T.Buffer((K_packed, N), "uint8"),
    scales_T: T.Buffer((num_scale_blocks, N), "float16"),
    C: T.Buffer((1, N), "float32"),
):
    T.func_attr({"global_symbol": "gemv_t", "tir.noalias": True})

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
                    byte_start = qb * 16
                    k_start = qb * QUANT_BLOCK
                    for byte_offset in range(16):
                        byte_idx = byte_start + byte_offset
                        packed = W_packed_T[byte_idx, n]
                        k_lo = k_start + byte_offset * 2
                        k_hi = k_lo + 1
                        int4_lo = packed & T.uint8(0xF)
                        int4_hi = (packed >> 4) & T.uint8(0xF)
                        w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                        w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale
                        C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_lo] * w_lo)
                        C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_hi] * w_hi)

print("Building...")
mod = tvm.IRModule({"main": gemv_transposed})
target = tvm.target.Target("cuda -arch=sm_110")
with tvm.transform.PassContext(opt_level=3):
    lib = tvm.build(mod, target=target)

print("Preparing data...")
np.random.seed(42)
A_np = np.random.randn(1, K).astype(np.float16)
W_packed_T = np.random.randint(0, 255, (K_packed, N), dtype=np.uint8)
scales_T = np.random.randn(num_scale_blocks, N).astype(np.float16)

device = runtime.cuda(0)
A_tvm = runtime.empty(A_np.shape, "float16", device)
A_tvm.copyfrom(A_np)
W_tvm = runtime.empty(W_packed_T.shape, "uint8", device)
W_tvm.copyfrom(W_packed_T)
scales_tvm = runtime.empty(scales_T.shape, "float16", device)
scales_tvm.copyfrom(scales_T)
C_tvm = runtime.empty((1, N), "float32", device)

print("Warming up...")
for _ in range(20):
    lib["gemv_t"](A_tvm, W_tvm, scales_tvm, C_tvm)
device.sync()

print("Benchmarking...")
start = time.time()
for _ in range(100):
    lib["gemv_t"](A_tvm, W_tvm, scales_tvm, C_tvm)
device.sync()
avg_ms = (time.time() - start) / 100 * 1000

total_bytes = N * K // 2 + N * (K // QUANT_BLOCK) * 2 + K * 2
bw = total_bytes / (avg_ms / 1000) / 1e9

print(f"\nResult: {avg_ms:.4f} ms, {bw:.1f} GB/s")
print(f"Target: < 0.2 ms, {'ACHIEVED' if avg_ms < 0.2 else f'Need {avg_ms/0.2:.1f}x speedup'}")
