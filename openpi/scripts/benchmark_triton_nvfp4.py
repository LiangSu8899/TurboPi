#!/usr/bin/env python3
"""
Benchmark Triton NVFP4 Kernels - Simplified for Triton 3.5.

Must be run from a file (not stdin) for Triton JIT to work.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time

print("=" * 70)
print("Triton NVFP4 Benchmark")
print("=" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print(f"Device: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda')

NVFP4_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def quantize_weight_nvfp4(weight, block_size=32):
    """Quantize weight to NVFP4 format."""
    N, K = weight.shape
    num_blocks = K // block_size
    weight_blocked = weight.view(N, num_blocks, block_size)

    scales = weight_blocked.abs().amax(dim=-1) / 6.0
    scales = scales.clamp(min=1e-8)

    weight_norm = weight_blocked / scales.unsqueeze(-1)

    nvfp4_positive = torch.tensor(NVFP4_MAGNITUDES, device=device, dtype=weight.dtype)
    signs = (weight_norm < 0).to(torch.uint8) * 8
    abs_vals = weight_norm.abs()

    diffs = (abs_vals.unsqueeze(-1) - nvfp4_positive).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)
    fp4_vals = (signs + indices).view(N, K)

    low = fp4_vals[:, 0::2]
    high = fp4_vals[:, 1::2]
    packed = (high << 4) | low

    return packed.to(torch.uint8), scales.to(weight.dtype)


@triton.jit
def decode_fp4_mag(idx):
    """Decode FP4 magnitude from index 0-7."""
    return tl.where(idx == 0, 0.0,
           tl.where(idx == 1, 0.5,
           tl.where(idx == 2, 1.0,
           tl.where(idx == 3, 1.5,
           tl.where(idx == 4, 2.0,
           tl.where(idx == 5, 3.0,
           tl.where(idx == 6, 4.0, 6.0)))))))


@triton.jit
def _nvfp4_gemv_simple_kernel(
    x_ptr, w_packed_ptr, w_scale_ptr, out_ptr,
    N, K, num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple: One program per output element."""
    n_idx = tl.program_id(0)

    if n_idx >= N:
        return

    acc = 0.0

    for block_idx in range(num_blocks):
        k_start = block_idx * BLOCK_SIZE
        scale = tl.load(w_scale_ptr + n_idx * num_blocks + block_idx)

        block_sum = 0.0
        for k_pair in range(BLOCK_SIZE // 2):
            k = k_start + k_pair * 2
            byte_idx = k // 2

            packed = tl.load(w_packed_ptr + n_idx * (K // 2) + byte_idx)
            x_low = tl.load(x_ptr + k)
            x_high = tl.load(x_ptr + k + 1)

            fp4_low = packed & 0xF
            fp4_high = (packed >> 4) & 0xF

            sign_low = tl.where(fp4_low >= 8, -1.0, 1.0)
            mag_low = decode_fp4_mag(fp4_low & 0x7)

            sign_high = tl.where(fp4_high >= 8, -1.0, 1.0)
            mag_high = decode_fp4_mag(fp4_high & 0x7)

            block_sum += sign_low * mag_low * x_low + sign_high * mag_high * x_high

        acc += block_sum * scale

    tl.store(out_ptr + n_idx, acc)


def benchmark_kernel(name, run_fn, warmup=100, runs=500):
    """Benchmark a kernel."""
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        run_fn()
    torch.cuda.synchronize()

    return (time.time() - start) / runs * 1000


def main():
    configs = [
        (2048, 16384, "gate/up: 2048 -> 16384"),
        (16384, 2048, "down: 16384 -> 2048"),
    ]

    TRT_FP8_GEMM = 0.53
    warmup = 100
    runs = 500

    all_results = {}

    for K, N, desc in configs:
        print(f"\n{'=' * 60}")
        print(f"{desc}")
        print(f"{'=' * 60}")

        weight = torch.randn(N, K, device=device, dtype=torch.float32)
        w_packed, w_scale = quantize_weight_nvfp4(weight)
        x = torch.randn(K, device=device, dtype=torch.float32)
        num_blocks = K // 32

        results = {}

        # Triton NVFP4
        out_triton = torch.empty(N, device=device, dtype=torch.float32)

        def run_triton():
            grid = (N,)
            _nvfp4_gemv_simple_kernel[grid](
                x, w_packed, w_scale, out_triton,
                N, K, num_blocks,
                BLOCK_SIZE=32,
            )

        try:
            results["Triton_NVFP4"] = benchmark_kernel("Triton", run_triton, warmup, runs)
            print(f"  Triton NVFP4: {results['Triton_NVFP4']:.4f} ms")
        except Exception as e:
            print(f"  Triton NVFP4 failed: {e}")
            results["Triton_NVFP4"] = float('inf')

        # cuBLAS BF16 baseline
        w_bf16 = weight.to(torch.bfloat16)
        x_bf16 = x.to(torch.bfloat16).unsqueeze(0)

        def run_cublas():
            return F.linear(x_bf16, w_bf16)

        results["cuBLAS_BF16"] = benchmark_kernel("cuBLAS", run_cublas, warmup, runs)
        print(f"  cuBLAS BF16: {results['cuBLAS_BF16']:.4f} ms")

        # cuBLAS FP32 (for comparison)
        w_fp32 = weight
        x_fp32 = x.unsqueeze(0)

        def run_cublas_fp32():
            return F.linear(x_fp32, w_fp32)

        results["cuBLAS_FP32"] = benchmark_kernel("cuBLAS FP32", run_cublas_fp32, warmup, runs)
        print(f"  cuBLAS FP32: {results['cuBLAS_FP32']:.4f} ms")

        all_results[desc] = results

        # Summary
        print(f"\n  vs TRT FP8 ({TRT_FP8_GEMM:.2f} ms):")
        for name, t in results.items():
            if t < float('inf'):
                speedup = TRT_FP8_GEMM / t
                status = "FASTER" if t < TRT_FP8_GEMM else "slower"
                print(f"    {name}: {speedup:.2f}x ({status})")

    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    gate_times = all_results["gate/up: 2048 -> 16384"]
    down_times = all_results["down: 16384 -> 2048"]

    print(f"\n{'Approach':<25} {'Full MLP (ms)':<15} {'vs TRT FP8':<12} {'Status'}")
    print("-" * 65)

    trt_mlp = TRT_FP8_GEMM * 3
    print(f"{'TRT FP8 (estimated)':<25} {trt_mlp:<15.4f} {'1.00x':<12} baseline")

    for kernel in ["Triton_NVFP4", "cuBLAS_BF16", "cuBLAS_FP32"]:
        if gate_times.get(kernel, float('inf')) < float('inf'):
            full_mlp = gate_times[kernel] * 2 + down_times[kernel]
            speedup = trt_mlp / full_mlp
            status = "FASTER" if full_mlp < trt_mlp else "slower"
            print(f"{kernel:<25} {full_mlp:<15.4f} {speedup:.2f}x{' ':<8} {status}")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    cublas_mlp = gate_times["cuBLAS_BF16"] * 2 + down_times["cuBLAS_BF16"]

    if cublas_mlp < trt_mlp:
        print(f"\ncuBLAS BF16 is {trt_mlp/cublas_mlp:.2f}x FASTER than TRT FP8!")
        print("Recommendation: Use BF16 for MLP - it's faster AND simpler.")

    triton_mlp = gate_times.get("Triton_NVFP4", float('inf'))
    if triton_mlp < float('inf'):
        triton_full = triton_mlp * 2 + down_times.get("Triton_NVFP4", float('inf'))
        if triton_full < cublas_mlp:
            print(f"\nTriton NVFP4 is {cublas_mlp/triton_full:.2f}x faster than cuBLAS BF16")
        else:
            print(f"\nTriton NVFP4 is {triton_full/cublas_mlp:.2f}x slower than cuBLAS BF16")
            print("Further optimization needed for NVFP4 to be competitive.")


if __name__ == "__main__":
    main()
