#!/usr/bin/env python3
"""Debug W4A16 kernel - print IR."""

import sys
TVM_PATH = "/home/heima-thor/suliang/Robot-llm/mlc-llm_tvm/tvm/python"
if TVM_PATH not in sys.path:
    sys.path.insert(0, TVM_PATH)

import tvm
from tvm import tir, runtime
from tvm.script import tir as T
import numpy as np

# Very small size for debugging
M, N, K = 8, 16, 16
QUANT_BLOCK = 32
num_blocks = (K + QUANT_BLOCK - 1) // QUANT_BLOCK
K_packed = K // 2

print(f"Shape: M={M}, N={N}, K={K}")
print(f"num_blocks={num_blocks}, K_packed={K_packed}")


@T.prim_func
def w4a16_debug(
    A: T.Buffer((8, 16), "float16"),
    W_packed: T.Buffer((16, 8), "uint8"),
    scales: T.Buffer((16, 1), "float16"),
    C: T.Buffer((8, 16), "float32"),
):
    T.func_attr({"global_symbol": "w4a16", "tir.noalias": True})

    for m in T.thread_binding(8, thread="blockIdx.x"):
        for n in T.thread_binding(16, thread="threadIdx.x"):
            # Use local storage for accumulator
            with T.block("compute"):
                T.reads(A[m, 0:16], W_packed[n, 0:8], scales[n, 0])
                T.writes(C[m, n])
                acc: T.float32 = T.float32(0)
                for k in range(16):
                    byte_idx = k // 2
                    is_high = k % 2
                    packed = W_packed[n, byte_idx]

                    int4_val = T.if_then_else(
                        is_high == 0,
                        packed & T.uint8(0xF),
                        (packed >> 4) & T.uint8(0xF)
                    )
                    signed_val = T.Cast("float16", int4_val) - T.float16(8.0)
                    scale = scales[n, 0]
                    w = signed_val * scale

                    acc = acc + T.Cast("float32", A[m, k] * w)

                C[m, n] = acc


def main():
    print("Building...")
    mod = tvm.IRModule({"main": w4a16_debug})

    print("\n--- Original IR ---")
    print(mod.script())

    target = tvm.target.Target("cuda -arch=sm_110")

    # Try without dlight first
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.build(mod, target=target)

    print("\nBuild OK")

    # Print generated CUDA code
    print("\n--- Generated CUDA ---")
    print(lib.imported_modules[0].get_source())

    # Test data
    np.random.seed(42)
    A_np = np.ones((M, K), dtype=np.float16)
    W_packed_np = np.zeros((N, K_packed), dtype=np.uint8)  # All zeros
    scales_np = np.ones((N, num_blocks), dtype=np.float16)

    # W_packed = 0 means both nibbles are 0
    # signed_val = 0 - 8 = -8
    # Expected: sum(1.0 * (-8.0)) over K = -8 * K = -128
    print(f"\nExpected output: {-8 * K}")

    device = runtime.cuda(0)
    A_tvm = runtime.empty(A_np.shape, "float16", device)
    A_tvm.copyfrom(A_np)
    W_packed_tvm = runtime.empty(W_packed_np.shape, "uint8", device)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm = runtime.empty(scales_np.shape, "float16", device)
    scales_tvm.copyfrom(scales_np)
    C_tvm = runtime.empty((M, N), "float32", device)

    lib["w4a16"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    C_result = C_tvm.numpy()
    print(f"Actual output[0,0]: {C_result[0, 0]}")
    print(f"Output range: [{C_result.min()}, {C_result.max()}]")
    print(f"Output sample:\n{C_result[:4, :4]}")


if __name__ == "__main__":
    main()
