#!/usr/bin/env python3
"""
Triton 性能分析脚本

目的: 确认 Triton 在 Thor 上的基础性能
如果 Triton FP16 MatMul 很慢，说明 Triton 不适合 Thor
如果 Triton FP16 MatMul 正常，说明 INT4 实现有问题
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def simple_fp16_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """简单的 FP16 MatMul kernel"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def benchmark_triton_fp16():
    """测试 Triton FP16 MatMul 性能"""
    print("=" * 60)
    print("Triton FP16 MatMul 性能分析")
    print("=" * 60)

    print(f"Triton version: {triton.__version__}")

    M, N, K = 712, 16384, 2048
    print(f"\nShape: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")

    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    # torch.matmul baseline
    for _ in range(20):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / 100 * 1000

    print(f"\ntorch.matmul: {torch_time:.3f} ms")

    # 测试不同的 block size
    print("\n" + "-" * 40)
    print("测试不同 Block Size 的 Triton FP16 MatMul")
    print("-" * 40)

    block_configs = [
        (32, 64, 32),
        (64, 64, 64),
        (64, 128, 64),
        (128, 128, 64),
        (128, 256, 64),
    ]

    best_time = float('inf')
    best_config = None

    for BLOCK_M, BLOCK_N, BLOCK_K in block_configs:
        c = torch.empty(M, N, device='cuda', dtype=torch.float16)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        try:
            # Warmup
            for _ in range(10):
                simple_fp16_matmul_kernel[grid](
                    a, b, c,
                    M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                )
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                simple_fp16_matmul_kernel[grid](
                    a, b, c,
                    M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                )
            torch.cuda.synchronize()

            triton_time = (time.perf_counter() - start) / 100 * 1000

            if triton_time < best_time:
                best_time = triton_time
                best_config = (BLOCK_M, BLOCK_N, BLOCK_K)

            speedup = torch_time / triton_time
            print(f"Block ({BLOCK_M:3d}, {BLOCK_N:3d}, {BLOCK_K:3d}): {triton_time:.3f} ms, vs torch: {speedup:.2f}x")

        except Exception as e:
            print(f"Block ({BLOCK_M:3d}, {BLOCK_N:3d}, {BLOCK_K:3d}): FAILED - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"torch.matmul:     {torch_time:.3f} ms")
    print(f"Best Triton:      {best_time:.3f} ms (Block {best_config})")
    print(f"Triton vs torch:  {torch_time/best_time:.2f}x")

    if best_time > torch_time * 2:
        print("\n⚠️ Triton 在 Thor 上性能显著差于 cuBLAS")
        print("   这意味着 Triton 可能不适合 Thor 平台")
        print("   建议: 考虑使用 CUTLASS 或其他方案")
    elif best_time > torch_time:
        print("\n⚠️ Triton 略慢于 cuBLAS，但可以接受")
        print("   INT4 kernel 需要更多优化")
    else:
        print("\n✅ Triton 性能正常")
        print("   INT4 kernel 实现有问题，需要重写")


def test_triton_autotune():
    """测试 Triton autotune"""
    print("\n" + "=" * 60)
    print("Triton Autotune 测试")
    print("=" * 60)

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def autotuned_matmul_kernel(
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

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
            b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
            b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)

            acc += tl.dot(a, b)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    M, N, K = 712, 16384, 2048
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    print("Running autotune...")
    # Warmup (triggers autotune)
    for _ in range(5):
        autotuned_matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        autotuned_matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
    torch.cuda.synchronize()
    autotuned_time = (time.perf_counter() - start) / 100 * 1000

    print(f"Autotuned Triton: {autotuned_time:.3f} ms")


if __name__ == "__main__":
    benchmark_triton_fp16()
    test_triton_autotune()
