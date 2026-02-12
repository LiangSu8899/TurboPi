#!/usr/bin/env python3
"""Debug W4A16 kernel - using proper TIR reduction."""

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
            # Initialize output to zero
            C[m, n] = T.float32(0)

            # Reduction loop
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

                # Accumulate directly to output buffer
                C[m, n] = C[m, n] + T.Cast("float32", A[m, k] * w)


def main():
    print("Building...")
    mod = tvm.IRModule({"main": w4a16_debug})

    print("\n--- Original IR ---")
    print(mod.script())

    target = tvm.target.Target("cuda -arch=sm_110")

    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.build(mod, target=target)

    print("\nBuild OK")

    # Print generated CUDA code
    print("\n--- Generated CUDA ---")
    try:
        # Try different TVM APIs
        if hasattr(lib, 'get_source'):
            print(lib.get_source())
        elif hasattr(lib, 'get_function'):
            # Try to get the source from the main function
            fn = lib.get_function("w4a16")
            if hasattr(fn, 'get_source'):
                print(fn.get_source())
    except Exception as e:
        print(f"Could not get source: {e}")

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

    # Test with real data
    print("\n--- Test with real data ---")
    A_np = np.random.randn(M, K).astype(np.float16)
    W_np = np.random.randn(N, K).astype(np.float32)

    # Quantize
    scales_np = np.zeros((N, 1), dtype=np.float16)
    W_packed_np = np.zeros((N, K_packed), dtype=np.uint8)
    for n in range(N):
        max_abs = np.max(np.abs(W_np[n]))
        scale = max_abs / 7.0 if max_abs > 0 else 1.0
        scales_np[n, 0] = scale
        for k in range(K):
            val = W_np[n, k] / scale if scale > 0 else 0
            quantized = int(np.clip(np.round(val + 8), 0, 15))
            byte_idx = k // 2
            if k % 2 == 0:
                W_packed_np[n, byte_idx] = (W_packed_np[n, byte_idx] & 0xF0) | quantized
            else:
                W_packed_np[n, byte_idx] = (W_packed_np[n, byte_idx] & 0x0F) | (quantized << 4)

    # Reference
    W_dequant = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            byte_idx = k // 2
            packed = W_packed_np[n, byte_idx]
            int4_val = (packed & 0xF) if k % 2 == 0 else ((packed >> 4) & 0xF)
            W_dequant[n, k] = (int4_val - 8) * scales_np[n, 0]

    C_ref = A_np.astype(np.float32) @ W_dequant.T

    # TVM
    A_tvm.copyfrom(A_np)
    W_packed_tvm.copyfrom(W_packed_np)
    scales_tvm.copyfrom(scales_np)

    lib["w4a16"](A_tvm, W_packed_tvm, scales_tvm, C_tvm)
    device.sync()

    C_result = C_tvm.numpy()
    max_diff = np.abs(C_result - C_ref).max()
    cos_sim = np.dot(C_result.flatten(), C_ref.flatten()) / (
        np.linalg.norm(C_result) * np.linalg.norm(C_ref) + 1e-8)

    print(f"Max diff: {max_diff:.6f}")
    print(f"Cos sim: {cos_sim:.6f}")
    print(f"Ref sample:\n{C_ref[:2, :4]}")
    print(f"TVM sample:\n{C_result[:2, :4]}")


if __name__ == "__main__":
    main()
