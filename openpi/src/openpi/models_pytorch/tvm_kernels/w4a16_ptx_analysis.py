#!/usr/bin/env python3
"""
Analyze generated PTX for W4A16 GEMV kernel.
"""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import tir
from tvm.script import tir as T


QUANT_BLOCK = 32


def create_baseline_gemv(N, K, THREADS=256):
    num_scale_blocks = K // QUANT_BLOCK
    K_packed = K // 2
    num_blocks = (N + THREADS - 1) // THREADS
    BYTES_PER_QB = QUANT_BLOCK // 2

    @T.prim_func
    def gemv(
        A: T.Buffer((1, K), "float16"),
        W_packed: T.Buffer((N, K_packed), "uint8"),
        scales: T.Buffer((N, num_scale_blocks), "float16"),
        C: T.Buffer((1, N), "float32"),
    ):
        T.func_attr({"global_symbol": "gemv", "tir.noalias": True})

        for block_idx in T.thread_binding(num_blocks, thread="blockIdx.x"):
            for tid in T.thread_binding(THREADS, thread="threadIdx.x"):
                n = block_idx * THREADS + tid
                if n < N:
                    C[0, n] = T.float32(0)

                    for qb in range(num_scale_blocks):
                        scale = scales[n, qb]
                        byte_start = qb * BYTES_PER_QB
                        k_start = qb * QUANT_BLOCK

                        for byte_offset in range(BYTES_PER_QB):
                            byte_idx = byte_start + byte_offset
                            packed = W_packed[n, byte_idx]

                            k_lo = k_start + byte_offset * 2
                            k_hi = k_lo + 1

                            int4_lo = packed & T.uint8(0xF)
                            int4_hi = (packed >> 4) & T.uint8(0xF)

                            w_lo = (T.Cast("float16", int4_lo) - T.float16(8.0)) * scale
                            w_hi = (T.Cast("float16", int4_hi) - T.float16(8.0)) * scale

                            C[0, n] = C[0, n] + T.Cast("float32", A[0, k_lo] * w_lo)
                            C[0, n] = C[0, n] + T.Cast("float32", A[0, k_hi] * w_hi)

    return gemv


def analyze_ir(N=16384, K=2048):
    kernel = create_baseline_gemv(N, K)
    mod = tvm.IRModule({"main": kernel})
    target = tvm.target.Target("cuda -arch=sm_110")

    print("="*60)
    print("Original TIR")
    print("="*60)
    print(mod.script()[:3000])

    print("\n" + "="*60)
    print("After lowering")
    print("="*60)

    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.build(mod, target=target)

    # Get CUDA source
    try:
        cuda_src = lib.imported_modules[0].get_source()
    except:
        cuda_src = lib.get_source()

    print("\n" + "="*60)
    print("Generated CUDA (first 5000 chars)")
    print("="*60)
    print(cuda_src[:5000])

    # Count key instructions
    print("\n" + "="*60)
    print("Instruction Analysis")
    print("="*60)

    counts = {
        "ld.global": cuda_src.count("ld.global"),
        "ld.global.v2": cuda_src.count("ld.global.v2"),
        "ld.global.v4": cuda_src.count("ld.global.v4"),
        "st.global": cuda_src.count("st.global"),
        "fma.rn.f32": cuda_src.count("fma.rn.f32"),
        "fma.rn.f16": cuda_src.count("fma.rn.f16"),
        "cvt.": cuda_src.count("cvt."),
        "and.b32": cuda_src.count("and.b32"),
        "shr.": cuda_src.count("shr."),
        "bra": cuda_src.count("bra"),  # branches
    }

    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    # Check for loop unrolling
    if ".pragma" in cuda_src or "#pragma" in cuda_src:
        print("\n  Pragmas found (potential unrolling)")
    else:
        print("\n  WARNING: No pragmas found - loops may not be unrolled")


if __name__ == "__main__":
    analyze_ir()
