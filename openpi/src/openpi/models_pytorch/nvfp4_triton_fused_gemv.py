#!/usr/bin/env python3
"""
Triton Fused NVFP4 GEMV Kernel

真正的 Fused Dequant + GEMV:
- 直接从 packed FP4 权重读取
- 在寄存器中解码 FP4 (使用 LUT)
- 累加到寄存器
- 避免任何中间存储

目标: 超越 TRT FP8 的 0.53ms per-GEMM

Author: Claude Code
Date: 2026-02-10
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional
import time

# NVFP4 配置
BLOCK_SIZE = 32  # Block scaling 的块大小
NVFP4_MAX = 6.0

# NVFP4 E2M1 LUT: index -> value
# 0-7: positive values (0, 0.5, 1, 1.5, 2, 3, 4, 6)
# 8-15: negative values (0, -0.5, -1, -1.5, -2, -3, -4, -6)


@triton.jit
def _nvfp4_gemv_fused_kernel(
    # Input activation [M, K] - assumed M=1 for GEMV
    x_ptr,
    # Packed FP4 weights [N, K//2]
    w_packed_ptr,
    # Weight scales [N, num_blocks]
    w_scale_ptr,
    # Output [M, N]
    out_ptr,
    # Bias (optional) [N]
    bias_ptr,
    # Dimensions
    M: tl.constexpr,  # Always 1 for GEMV
    N,
    K,
    # Strides
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    stride_om,
    stride_on,
    # Config
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    # Tile sizes
    BLOCK_N: tl.constexpr,  # 每个 block 处理的 N 维度大小
    BLOCK_K: tl.constexpr,  # K 维度 tile 大小
):
    """
    Fused NVFP4 GEMV: y = x @ W^T (+ bias)

    每个 thread block 处理 BLOCK_N 个输出元素。
    K 维度分 tile 处理，使用寄存器累加。
    """
    # Block ID
    pid_n = tl.program_id(0)

    # 计算该 block 处理的 N 范围
    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # NVFP4 LUT (在常量内存中)
    # Index 0-7: positive, 8-15: negative
    LUT = tl.full((16,), 0.0, dtype=tl.float32)

    # 初始化累加器 [BLOCK_N] - 在寄存器中
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # 遍历 K 维度 tiles
    num_k_tiles = (K + BLOCK_K - 1) // BLOCK_K

    for k_tile in range(num_k_tiles):
        k_start = k_tile * BLOCK_K
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # 加载输入 x[0, k_start:k_start+BLOCK_K]
        x_ptrs = x_ptr + k_offsets * stride_xk  # M=1, so skip stride_xm
        x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0)

        # 处理每个输出位置
        for local_n in range(BLOCK_N):
            global_n = n_start + local_n

            if global_n < N:
                # 加载该行的 scale blocks
                # 每个 scale 对应 BLOCK_SIZE 个元素
                # 需要计算 k_start 对应哪些 scale blocks

                local_acc = tl.zeros((1,), dtype=tl.float32)

                # 逐元素处理 K 维度
                for local_k in range(BLOCK_K):
                    global_k = k_start + local_k

                    if global_k < K:
                        # 读取 x 值
                        x_val = tl.load(x_ptr + global_k * stride_xk)

                        # 计算 packed 位置
                        packed_k = global_k // 2
                        is_high = (global_k % 2) == 1

                        # 读取 packed byte
                        packed_byte = tl.load(w_packed_ptr + global_n * stride_wn + packed_k * stride_wk)

                        # 解包 FP4
                        if is_high:
                            fp4_idx = (packed_byte >> 4) & 0xF
                        else:
                            fp4_idx = packed_byte & 0xF

                        # 解码 FP4 值 (inline LUT)
                        # sign = fp4_idx >= 8
                        # abs_idx = fp4_idx % 8
                        # abs_vals = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
                        sign_bit = (fp4_idx >> 3) & 1
                        abs_idx = fp4_idx & 0x7

                        # 手动 LUT 展开
                        w_abs = tl.where(abs_idx == 0, 0.0,
                                tl.where(abs_idx == 1, 0.5,
                                tl.where(abs_idx == 2, 1.0,
                                tl.where(abs_idx == 3, 1.5,
                                tl.where(abs_idx == 4, 2.0,
                                tl.where(abs_idx == 5, 3.0,
                                tl.where(abs_idx == 6, 4.0, 6.0)))))))

                        w_val = tl.where(sign_bit == 1, -w_abs, w_abs)

                        # 读取 scale
                        scale_idx = global_k // BLOCK_SIZE
                        w_scale = tl.load(w_scale_ptr + global_n * stride_sn + scale_idx * stride_sk)

                        # 累加
                        local_acc += x_val * w_val * w_scale

                # 更新全局累加器
                acc = tl.where(tl.arange(0, BLOCK_N) == local_n,
                              acc + tl.load(tl.zeros((1,), dtype=tl.float32)),  # placeholder
                              acc)

    # 写出结果
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
        acc = acc + bias_vals

    out_ptrs = out_ptr + n_offsets * stride_on
    tl.store(out_ptrs, acc, mask=n_mask)


# ============================================================================
# Vectorized Fused GEMV - 更高效的实现
# ============================================================================

@triton.jit
def _nvfp4_gemv_vectorized_kernel(
    # Input [1, K]
    x_ptr,
    # Packed weights [N, K//2]
    w_packed_ptr,
    # Scales [N, K//BLOCK_SIZE]
    w_scale_ptr,
    # Output [1, N]
    out_ptr,
    # Bias [N] or None
    bias_ptr,
    # Dims
    N,
    K,
    num_scale_blocks,
    # Config
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
):
    """
    Vectorized Fused NVFP4 GEMV

    优化:
    1. 每个 thread 处理一个输出元素
    2. K 维度向量化加载 (4 bytes = 8 FP4 values)
    3. 寄存器累加
    """
    # 每个 program 处理 BLOCK_N 个输出
    pid = tl.program_id(0)

    # 输出索引
    n_idx = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_idx < N

    # 初始化累加器
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # K 维度每次处理 8 个元素 (对应 4 packed bytes)
    num_k_iters = K // 8

    for k_iter in range(num_k_iters):
        k_base = k_iter * 8

        # 加载 8 个 x 值
        x_vals = tl.load(x_ptr + k_base + tl.arange(0, 8))

        # 加载 4 packed bytes (包含 8 个 FP4 值)
        # 对于每个 n_idx
        for local_n in range(BLOCK_N):
            global_n = pid * BLOCK_N + local_n

            if global_n < N:
                # 加载 4 bytes = 8 FP4 values
                packed_base = global_n * (K // 2) + k_base // 2

                # 加载 4 bytes
                byte0 = tl.load(w_packed_ptr + packed_base + 0)
                byte1 = tl.load(w_packed_ptr + packed_base + 1)
                byte2 = tl.load(w_packed_ptr + packed_base + 2)
                byte3 = tl.load(w_packed_ptr + packed_base + 3)

                # 解包每个 byte 的低高 nibble
                # byte0: idx 0, 1
                # byte1: idx 2, 3
                # byte2: idx 4, 5
                # byte3: idx 6, 7

                # 获取 scale (假设 k_base 在同一个 scale block 内)
                scale_idx = k_base // SCALE_BLOCK_SIZE
                w_scale = tl.load(w_scale_ptr + global_n * num_scale_blocks + scale_idx)

                # 解码并累加
                local_sum = 0.0

                # 使用内联 LUT
                def decode_fp4(idx):
                    sign = (idx >> 3) & 1
                    abs_idx = idx & 0x7
                    abs_val = tl.where(abs_idx == 0, 0.0,
                              tl.where(abs_idx == 1, 0.5,
                              tl.where(abs_idx == 2, 1.0,
                              tl.where(abs_idx == 3, 1.5,
                              tl.where(abs_idx == 4, 2.0,
                              tl.where(abs_idx == 5, 3.0,
                              tl.where(abs_idx == 6, 4.0, 6.0)))))))
                    return tl.where(sign == 1, -abs_val, abs_val)

                # idx 0-7 from bytes
                idx0 = byte0 & 0xF
                idx1 = (byte0 >> 4) & 0xF
                idx2 = byte1 & 0xF
                idx3 = (byte1 >> 4) & 0xF
                idx4 = byte2 & 0xF
                idx5 = (byte2 >> 4) & 0xF
                idx6 = byte3 & 0xF
                idx7 = (byte3 >> 4) & 0xF

                # 加载 x 值
                x0 = tl.load(x_ptr + k_base + 0)
                x1 = tl.load(x_ptr + k_base + 1)
                x2 = tl.load(x_ptr + k_base + 2)
                x3 = tl.load(x_ptr + k_base + 3)
                x4 = tl.load(x_ptr + k_base + 4)
                x5 = tl.load(x_ptr + k_base + 5)
                x6 = tl.load(x_ptr + k_base + 6)
                x7 = tl.load(x_ptr + k_base + 7)

                # 累加
                local_sum += x0 * decode_fp4(idx0) * w_scale
                local_sum += x1 * decode_fp4(idx1) * w_scale
                local_sum += x2 * decode_fp4(idx2) * w_scale
                local_sum += x3 * decode_fp4(idx3) * w_scale
                local_sum += x4 * decode_fp4(idx4) * w_scale
                local_sum += x5 * decode_fp4(idx5) * w_scale
                local_sum += x6 * decode_fp4(idx6) * w_scale
                local_sum += x7 * decode_fp4(idx7) * w_scale

                # 更新累加器
                acc = tl.where(tl.arange(0, BLOCK_N) == local_n,
                              acc + local_sum, acc)

    # 添加 bias
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + n_idx, mask=n_mask, other=0.0)
        acc = acc + bias_vals

    # 写出
    tl.store(out_ptr + n_idx, acc, mask=n_mask)


# ============================================================================
# 简化版: 每个线程处理一个输出元素
# ============================================================================

@triton.jit
def _nvfp4_gemv_simple_kernel(
    x_ptr,           # [K]
    w_packed_ptr,    # [N, K//2]
    w_scale_ptr,     # [N, num_blocks]
    out_ptr,         # [N]
    bias_ptr,        # [N] or None
    N,
    K,
    num_blocks,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    最简单的实现: 每个线程处理一个输出元素。
    用于验证正确性和作为 baseline。
    """
    n_idx = tl.program_id(0)

    if n_idx >= N:
        return

    # 累加器
    acc = 0.0

    # 遍历 K
    for k in range(K):
        # 读取 x
        x_val = tl.load(x_ptr + k)

        # 读取 packed FP4
        packed_k = k // 2
        packed_byte = tl.load(w_packed_ptr + n_idx * (K // 2) + packed_k)

        # 解包
        if k % 2 == 0:
            fp4_idx = packed_byte & 0xF
        else:
            fp4_idx = (packed_byte >> 4) & 0xF

        # 解码
        sign_bit = (fp4_idx >> 3) & 1
        abs_idx = fp4_idx & 0x7

        abs_val = tl.where(abs_idx == 0, 0.0,
                  tl.where(abs_idx == 1, 0.5,
                  tl.where(abs_idx == 2, 1.0,
                  tl.where(abs_idx == 3, 1.5,
                  tl.where(abs_idx == 4, 2.0,
                  tl.where(abs_idx == 5, 3.0,
                  tl.where(abs_idx == 6, 4.0, 6.0)))))))

        w_val = tl.where(sign_bit == 1, -abs_val, abs_val)

        # Scale
        scale_idx = k // BLOCK_SIZE
        w_scale = tl.load(w_scale_ptr + n_idx * num_blocks + scale_idx)

        # 累加
        acc += x_val * w_val * w_scale

    # Bias
    if HAS_BIAS:
        acc += tl.load(bias_ptr + n_idx)

    # 输出
    tl.store(out_ptr + n_idx, acc)


# ============================================================================
# Python Wrapper
# ============================================================================

def nvfp4_gemv_fused(
    x: torch.Tensor,            # [M, K] or [K] - input (M=1 for GEMV)
    w_packed: torch.Tensor,     # [N, K//2] - packed FP4 weights
    w_scale: torch.Tensor,      # [N, num_blocks] - scales
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused NVFP4 GEMV.

    Args:
        x: Input tensor [M, K] or [K]
        w_packed: Packed FP4 weights [N, K//2]
        w_scale: Scale factors [N, num_blocks]
        bias: Optional bias [N]

    Returns:
        Output tensor [M, N] or [N]
    """
    # Handle input shape
    squeeze_output = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_output = True

    M, K = x.shape
    assert M == 1, f"GEMV requires M=1, got M={M}"

    N = w_packed.shape[0]
    num_blocks = w_scale.shape[1]

    # Ensure contiguous
    x = x.contiguous().view(-1)  # [K]
    w_packed = w_packed.contiguous()
    w_scale = w_scale.contiguous()

    # Output
    out = torch.empty(N, device=x.device, dtype=torch.float32)

    # Bias handling
    has_bias = bias is not None
    if has_bias:
        bias = bias.contiguous().float()
    else:
        bias = torch.empty(1, device=x.device, dtype=torch.float32)  # dummy

    # Launch kernel
    grid = (N,)  # 每个线程处理一个输出
    _nvfp4_gemv_simple_kernel[grid](
        x.float(), w_packed, w_scale.float(), out, bias,
        N, K, num_blocks,
        HAS_BIAS=has_bias,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if squeeze_output:
        return out
    return out.unsqueeze(0)


# ============================================================================
# Weight Quantization Helper
# ============================================================================

def quantize_weight_nvfp4(
    weight: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight to NVFP4 format.

    Args:
        weight: [N, K] weight tensor

    Returns:
        (packed, scales): packed uint8 [N, K//2] and scales [N, num_blocks]
    """
    N, K = weight.shape
    assert K % block_size == 0
    assert K % 2 == 0

    num_blocks = K // block_size
    device = weight.device

    # Reshape to blocks
    w_blocked = weight.view(N, num_blocks, block_size)

    # Compute per-block scales
    abs_max = w_blocked.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    scales = abs_max / NVFP4_MAX
    w_normalized = w_blocked / scales

    # Quantize to NVFP4
    w_flat = w_normalized.view(N, K)
    signs = (w_flat < 0).int()
    abs_vals = w_flat.abs()

    # NVFP4 LUT for quantization
    nvfp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)

    # Find nearest
    distances = (abs_vals.unsqueeze(-1) - nvfp4_values.unsqueeze(0).unsqueeze(0)).abs()
    indices = distances.argmin(dim=-1)

    # Combine sign and index
    quantized = indices + signs * 8  # [N, K]

    # Pack two 4-bit values per byte
    K_half = K // 2
    packed = torch.zeros(N, K_half, dtype=torch.uint8, device=device)
    packed = (quantized[:, 0::2] & 0xF) | ((quantized[:, 1::2] & 0xF) << 4)

    scales = scales.squeeze(-1).float()  # [N, num_blocks]

    return packed, scales


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_fused_gemv():
    """Benchmark fused NVFP4 GEMV."""
    print("=" * 70)
    print("Fused NVFP4 GEMV Benchmark")
    print("=" * 70)

    device = torch.device("cuda")

    # Test dimensions (matching MLP layers)
    test_cases = [
        # (K, N) - input_dim, output_dim
        (2048, 16384, "gate_proj (Gemma 2B)"),
        (2048, 16384, "up_proj (Gemma 2B)"),
        (16384, 2048, "down_proj (Gemma 2B)"),
        (1024, 4096, "gate_proj (Gemma 300M)"),
        (4096, 1024, "down_proj (Gemma 300M)"),
    ]

    warmup = 50
    runs = 200

    print(f"\n{'Layer':<25} {'FP4 Fused (ms)':<15} {'BF16 cuBLAS (ms)':<18} {'Speedup':<10}")
    print("-" * 70)

    for K, N, name in test_cases:
        # Create random weight and quantize
        weight = torch.randn(N, K, device=device, dtype=torch.float32)
        w_packed, w_scale = quantize_weight_nvfp4(weight)

        # Input
        x = torch.randn(1, K, device=device, dtype=torch.float32)
        x_bf16 = x.bfloat16()
        w_bf16 = weight.bfloat16()

        # ================================================================
        # Benchmark FP4 Fused
        # ================================================================
        for _ in range(warmup):
            _ = nvfp4_gemv_fused(x, w_packed, w_scale)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            out = nvfp4_gemv_fused(x, w_packed, w_scale)
        torch.cuda.synchronize()
        fp4_time = (time.perf_counter() - start) / runs * 1000

        # ================================================================
        # Benchmark BF16 cuBLAS
        # ================================================================
        for _ in range(warmup):
            _ = torch.mm(x_bf16, w_bf16.t())
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            out_bf16 = torch.mm(x_bf16, w_bf16.t())
        torch.cuda.synchronize()
        bf16_time = (time.perf_counter() - start) / runs * 1000

        speedup = bf16_time / fp4_time
        status = "✅" if speedup > 1.0 else "❌"

        print(f"{name:<25} {fp4_time:<15.4f} {bf16_time:<18.4f} {speedup:<8.2f}x {status}")

    print("=" * 70)

    # TRT FP8 comparison
    trt_fp8_per_gemm = 0.53  # ms
    print(f"\nTRT FP8 baseline: {trt_fp8_per_gemm:.2f} ms per GEMM")


if __name__ == "__main__":
    benchmark_fused_gemv()
