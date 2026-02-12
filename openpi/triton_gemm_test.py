#!/usr/bin/env python3
"""
Triton GEMM Benchmark on Jetson Thor (SM110)
"""

import torch
import triton
import triton.language as tl
import numpy as np

print(f'Triton version: {triton.__version__}')
print(f'GPU: {torch.cuda.get_device_name()}')
print(f'Compute Capability: {torch.cuda.get_device_capability()}')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def benchmark(fn, warmup=20, iters=100):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        _ = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.mean(times), np.std(times)


def main():
    print("\n" + "="*60)
    print("Triton GEMM Benchmark on Thor")
    print("="*60)

    # Test sizes (Pi0.5 model dimensions)
    test_sizes = [
        (1, 1536, 1536),    # Attention
        (8, 1536, 1536),
        (1, 1536, 6144),    # MLP up
        (8, 1536, 6144),
        (1, 6144, 1536),    # MLP down
        (8, 6144, 1536),
    ]

    results = []

    for M, K, N in test_sizes:
        print(f"\n--- M={M}, K={K}, N={N} ---")

        a = torch.randn(M, K, dtype=torch.float16, device='cuda')
        b = torch.randn(K, N, dtype=torch.float16, device='cuda')

        # Reference
        ref = torch.matmul(a, b)

        # Triton (first call triggers autotune)
        try:
            triton_out = triton_matmul(a, b)
            diff = (triton_out - ref).abs().max().item()

            # Benchmark
            pytorch_ms, pt_std = benchmark(lambda: torch.matmul(a, b))
            triton_ms, tr_std = benchmark(lambda: triton_matmul(a, b))

            flops = 2 * M * K * N
            pytorch_tflops = flops / (pytorch_ms * 1e-3) / 1e12
            triton_tflops = flops / (triton_ms * 1e-3) / 1e12
            speedup = pytorch_ms / triton_ms

            print(f"  PyTorch: {pytorch_ms:.3f}ms ({pytorch_tflops:.2f} TFLOPS)")
            print(f"  Triton:  {triton_ms:.3f}ms ({triton_tflops:.2f} TFLOPS)")
            print(f"  Speedup: {speedup:.2f}x, MaxDiff: {diff:.6f}")

            results.append({
                'M': M, 'K': K, 'N': N,
                'pytorch_ms': pytorch_ms,
                'triton_ms': triton_ms,
                'pytorch_tflops': pytorch_tflops,
                'triton_tflops': triton_tflops,
                'speedup': speedup,
                'max_diff': diff,
                'success': True
            })
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({'M': M, 'K': K, 'N': N, 'success': False, 'error': str(e)})

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful = [r for r in results if r['success']]
    if successful:
        print(f"\n{'Size':<20} {'PyTorch':>12} {'Triton':>12} {'Speedup':>10}")
        print("-"*60)
        for r in successful:
            size = f"{r['M']}x{r['K']}x{r['N']}"
            print(f"{size:<20} {r['pytorch_tflops']:>10.2f}T {r['triton_tflops']:>10.2f}T {r['speedup']:>9.2f}x")

        best = max(successful, key=lambda x: x['triton_tflops'])
        print(f"\n🏆 Best Triton: {best['triton_tflops']:.2f} TFLOPS @ {best['M']}x{best['K']}x{best['N']}")


if __name__ == "__main__":
    main()
