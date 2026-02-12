#!/usr/bin/env python3
"""Dump PTX code for vectorized kernel analysis."""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm.script import tir as T
import subprocess
import tempfile
import os

N, K = 16384, 2048
QUANT_BLOCK = 32
num_scale_blocks = K // QUANT_BLOCK
THREADS = 256
num_blocks = (N + THREADS - 1) // THREADS


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

                    for v in T.vectorized(4):
                        W_local[v] = W_packed[qb, n, v]

                    for u_idx in range(4):
                        u = W_local[u_idx]
                        k_offset = u_idx * 8

                        for i in range(8):
                            int4_val = (u >> T.uint32(i * 4)) & T.uint32(0xF)
                            k_idx = k_base + k_offset + i
                            w = (T.Cast("float16", int4_val) - T.float16(8.0)) * scale
                            C[0, n] = C[0, n] + T.Cast("float32", A_shared[k_idx] * w)


print("Building with CUDA source export...")
mod = tvm.IRModule({"main": gemv_vec128_v3})
target = tvm.target.Target("cuda -arch=sm_110")

with tvm.transform.PassContext(opt_level=3):
    lib = tvm.build(mod, target=target)

    # Export to file
    temp_dir = tempfile.mkdtemp()
    lib_path = os.path.join(temp_dir, "kernel.so")
    cuda_path = os.path.join(temp_dir, "kernel.cu")

    lib.export_library(lib_path)

    # Get the CUDA code via cubin disassembly
    print("\nExported library to:", lib_path)

# Try to get CUDA source via nvcc -cubin -dryrun
print("\nPrinting TIR Module...")
print(str(mod)[:4000])

print("\n" + "="*60)
print("KEY OBSERVATIONS:")
print("="*60)
print("1. T.vectorized(4) annotates 4-wide vector load")
print("2. W_local buffer in 'local' scope should be in registers")
print("3. Memory access pattern: W_packed[qb, n, 0:4] is contiguous")
print("4. Adjacent threads access adjacent n (coalesced)")
print("5. 4 x uint32 (128-bit) per thread per quant block")

# Try to disassemble
import subprocess
print("\n" + "="*60)
print("Disassembling kernel...")
print("="*60)

try:
    result = subprocess.run(
        ["cuobjdump", "-ptx", lib_path],
        capture_output=True, text=True, timeout=30
    )
    ptx = result.stdout
    if ptx:
        # Count load types
        lines = ptx.split('\n')
        ld_v4 = sum(1 for l in lines if 'ld.global.v4' in l or 'ld.global.nc.v4' in l)
        ld_v2 = sum(1 for l in lines if 'ld.global.v2' in l or 'ld.global.nc.v2' in l)
        ld_32 = sum(1 for l in lines if 'ld.global' in l and '.u32' in l and 'v4' not in l and 'v2' not in l)
        ld_total = sum(1 for l in lines if 'ld.global' in l)

        print(f"PTX Load Instructions:")
        print(f"  ld.global.v4 (128-bit): {ld_v4}")
        print(f"  ld.global.v2 (64-bit):  {ld_v2}")
        print(f"  ld.global.u32 (32-bit): {ld_32}")
        print(f"  Total: {ld_total}")

        # Show sample loads
        print("\nSample ld.global instructions:")
        for line in lines:
            if 'ld.global' in line:
                print(f"  {line.strip()}")
                break
    else:
        print("No PTX output")
except Exception as e:
    print(f"Disassembly failed: {e}")
