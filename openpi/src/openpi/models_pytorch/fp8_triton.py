#!/usr/bin/env python3
"""
Triton-based FP8 Dynamic Quantization Kernels

极速将 BF16 转为 FP8，用于 W4A8 Pipeline。
核心优化：消灭 Python 开销，全部在 GPU 上完成量化。

FP8 E4M3 格式:
- 范围: [-448, 448]
- 精度: ~2^-10 相对精度
- Tensor Core 友好

性能目标:
- Python cast: ~0.5ms
- Triton kernel: <0.05ms (10x 加速)

Author: Claude Code
Date: 2026-02-09
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional

# FP8 E4M3 配置
FP8_E4M3_MAX = 448.0  # FP8 E4M3 最大值
BLOCK_SIZE_DEFAULT = 1024  # 默认 block 大小


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def _quantize_fp8_per_tensor_kernel(
    x_ptr,              # Input: BF16/FP32 [N]
    y_ptr,              # Output: FP8 [N]
    scale_ptr,          # Output: Scale (FP32) [1]
    n_elements,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Per-Tensor FP8 量化 kernel。

    第一步：所有 block 计算局部 max
    第二步：原子更新全局 max
    第三步：使用全局 scale 量化

    注意：这个 kernel 需要两次 launch，简化起见先用 Per-Block。
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load BF16/FP32
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute block abs max
    abs_x = tl.abs(x)
    block_max = tl.max(abs_x)

    # Compute scale (per-block for now)
    scale = block_max / FP8_MAX + 1e-12

    # Quantize to FP8
    y_scaled = x / scale
    # Clamp to FP8 range
    y_clamped = tl.minimum(tl.maximum(y_scaled, -FP8_MAX), FP8_MAX)
    # Cast to FP8 (Triton handles saturation/rounding)
    y_fp8 = y_clamped.to(tl.float8e4nv)

    # Store
    tl.store(y_ptr + offsets, y_fp8, mask=mask)

    # Store scale (per-block)
    if scale_ptr is not None:
        tl.store(scale_ptr + pid, scale)


@triton.jit
def _quantize_fp8_per_row_kernel(
    x_ptr,              # Input: BF16/FP32 [M, K]
    y_ptr,              # Output: FP8 [M, K]
    scale_ptr,          # Output: Scale (FP32) [M]
    M,                  # 行数
    K,                  # 列数
    stride_xm,          # x 的 row stride
    stride_xk,          # x 的 col stride
    stride_ym,          # y 的 row stride
    stride_yk,          # y 的 col stride
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Per-Row (Per-Token) FP8 动态量化 kernel。

    每个 program 处理一行，计算该行的 abs max，然后量化整行。
    这是 W4A8 最常用的量化模式：每个 token 独立量化。
    """
    pid_m = tl.program_id(0)  # 行索引

    # 初始化行最大值 (scalar)
    row_max = 0.0

    # 第一遍：计算行的 abs max
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask = k_offsets < K
        x_ptrs = x_ptr + pid_m * stride_xm + k_offsets * stride_xk
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        block_max = tl.max(abs_x)
        row_max = tl.maximum(row_max, block_max)

    # 计算 scale (row_max 现在是 scalar)
    scale = row_max / FP8_MAX + 1e-12

    # 存储 scale
    tl.store(scale_ptr + pid_m, scale.to(tl.float32))

    # 第二遍：量化并存储
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask = k_offsets < K
        x_ptrs = x_ptr + pid_m * stride_xm + k_offsets * stride_xk
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # 量化
        y_scaled = x / scale
        y_clamped = tl.minimum(tl.maximum(y_scaled, -FP8_MAX), FP8_MAX)
        y_fp8 = y_clamped.to(tl.float8e4nv)

        # 存储
        y_ptrs = y_ptr + pid_m * stride_ym + k_offsets * stride_yk
        tl.store(y_ptrs, y_fp8, mask=mask)


@triton.jit
def _quantize_fp8_fused_kernel(
    x_ptr,              # Input: BF16/FP32 [M, K]
    y_ptr,              # Output: FP8 [M, K]
    scale_ptr,          # Output: Scale (FP32) [M]
    M,                  # 行数
    K,                  # 列数
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused Per-Row FP8 量化 kernel (单遍优化版本)。

    假设 K <= BLOCK_SIZE_K，可以一遍完成。
    适用于 hidden_size <= 4096 的常见场景。
    """
    pid_m = tl.program_id(0)

    # 加载整行
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    mask = k_offsets < K
    x_ptrs = x_ptr + pid_m * stride_xm + k_offsets * stride_xk
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # 计算 abs max
    abs_x = tl.abs(x)
    row_max = tl.max(abs_x)

    # 计算 scale
    scale = row_max / FP8_MAX + 1e-12

    # 存储 scale
    tl.store(scale_ptr + pid_m, scale)

    # 量化
    y_scaled = x / scale
    y_clamped = tl.minimum(tl.maximum(y_scaled, -FP8_MAX), FP8_MAX)
    y_fp8 = y_clamped.to(tl.float8e4nv)

    # 存储
    y_ptrs = y_ptr + pid_m * stride_ym + k_offsets * stride_yk
    tl.store(y_ptrs, y_fp8, mask=mask)


@triton.jit
def _dequantize_fp8_kernel(
    y_ptr,              # Input: FP8 [M, K]
    scale_ptr,          # Input: Scale (FP32) [M]
    out_ptr,            # Output: BF16/FP32 [M, K]
    M,
    K,
    stride_ym,
    stride_yk,
    stride_om,
    stride_ok,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    FP8 反量化 kernel。
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_start = pid_k * BLOCK_SIZE_K
    k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
    mask = k_offsets < K

    # 加载 FP8 数据
    y_ptrs = y_ptr + pid_m * stride_ym + k_offsets * stride_yk
    y = tl.load(y_ptrs, mask=mask, other=0.0)

    # 加载 scale
    scale = tl.load(scale_ptr + pid_m)

    # 反量化
    out = y.to(tl.float32) * scale

    # 存储
    out_ptrs = out_ptr + pid_m * stride_om + k_offsets * stride_ok
    tl.store(out_ptrs, out, mask=mask)


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def quantize_to_fp8_triton(
    x: torch.Tensor,
    per_row: bool = True,
    return_scale: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    使用 Triton 进行 FP8 动态量化。

    Args:
        x: [M, K] 输入张量 (BF16/FP32)
        per_row: 是否 Per-Row 量化 (推荐 True)
        return_scale: 是否返回 scale

    Returns:
        (y_fp8, scales): FP8 张量和 scale factors
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, K = x.shape
    x_f32 = x.float().contiguous()

    # 输出 tensors
    y_fp8 = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(M, device=x.device, dtype=torch.float32) if per_row else None

    if per_row:
        # 选择合适的 block size
        if K <= 1024:
            block_size_k = 1024
            kernel = _quantize_fp8_fused_kernel
        elif K <= 2048:
            block_size_k = 2048
            kernel = _quantize_fp8_fused_kernel
        elif K <= 4096:
            block_size_k = 4096
            kernel = _quantize_fp8_fused_kernel
        else:
            block_size_k = 1024
            kernel = _quantize_fp8_per_row_kernel

        grid = (M,)
        kernel[grid](
            x_f32, y_fp8, scales,
            M, K,
            x_f32.stride(0), x_f32.stride(1),
            y_fp8.stride(0), y_fp8.stride(1),
            FP8_MAX=FP8_E4M3_MAX,
            BLOCK_SIZE_K=block_size_k,
        )
    else:
        # Per-tensor 量化
        n_elements = x.numel()
        num_blocks = triton.cdiv(n_elements, BLOCK_SIZE_DEFAULT)
        scales = torch.empty(num_blocks, device=x.device, dtype=torch.float32)

        grid = (num_blocks,)
        _quantize_fp8_per_tensor_kernel[grid](
            x_f32.flatten(), y_fp8.flatten(), scales,
            n_elements,
            FP8_MAX=FP8_E4M3_MAX,
            BLOCK_SIZE=BLOCK_SIZE_DEFAULT,
        )

    if return_scale:
        return y_fp8, scales
    return y_fp8, None


def dequantize_fp8_triton(
    y_fp8: torch.Tensor,
    scales: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    使用 Triton 进行 FP8 反量化。

    Args:
        y_fp8: [M, K] FP8 张量
        scales: [M] scale factors
        target_dtype: 输出 dtype

    Returns:
        反量化后的张量
    """
    M, K = y_fp8.shape
    out = torch.empty(M, K, device=y_fp8.device, dtype=torch.float32)

    block_size_k = min(1024, K)
    num_k_blocks = triton.cdiv(K, block_size_k)

    grid = (M, num_k_blocks)
    _dequantize_fp8_kernel[grid](
        y_fp8, scales, out,
        M, K,
        y_fp8.stride(0), y_fp8.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_K=block_size_k,
    )

    return out.to(target_dtype)


def quantize_to_fp8_fast(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    快速 FP8 量化接口 (智元风格 API)。

    Args:
        x: [M, K] 或 [B, S, H] 输入张量

    Returns:
        (y_fp8, scales): FP8 张量和 per-row scales
    """
    original_shape = x.shape
    if x.dim() == 3:
        x = x.view(-1, x.shape[-1])
    elif x.dim() != 2:
        x = x.view(-1, x.shape[-1])

    y_fp8, scales = quantize_to_fp8_triton(x, per_row=True, return_scale=True)

    # 保持原始 batch 形状
    if len(original_shape) > 2:
        y_fp8 = y_fp8.view(*original_shape[:-1], -1)

    return y_fp8, scales


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_fp8_quantization():
    """Benchmark Triton FP8 vs PyTorch cast."""
    import time

    print("=" * 70)
    print("FP8 Quantization Benchmark: Triton vs PyTorch")
    print("=" * 70)

    device = torch.device("cuda")

    # 典型的 Pi0 activation 尺寸
    test_cases = [
        (1, 2048),      # Single token
        (256, 2048),    # Typical batch
        (256, 16384),   # Intermediate size
        (1024, 2048),   # Large batch
        (1, 4096),      # Gemma hidden
    ]

    for M, K in test_cases:
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(5):
            _ = quantize_to_fp8_triton(x)
            _ = x.to(torch.float8_e4m3fn)
        torch.cuda.synchronize()

        iterations = 100

        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            y_triton, s_triton = quantize_to_fp8_triton(x)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - start) / iterations * 1000

        # Benchmark PyTorch cast (no dynamic scaling)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            y_pytorch = x.float().clamp(-448, 448).to(torch.float8_e4m3fn)
        torch.cuda.synchronize()
        pytorch_ms = (time.perf_counter() - start) / iterations * 1000

        speedup = pytorch_ms / triton_ms if triton_ms > 0 else float('inf')

        print(f"[{M:4d}, {K:5d}] Triton: {triton_ms:.4f} ms | PyTorch: {pytorch_ms:.4f} ms | Speedup: {speedup:.2f}x")

    print("\n" + "=" * 70)
    print("Accuracy Validation (Dynamic Scaling)")
    print("=" * 70)

    # 验证精度
    M, K = 256, 2048
    x = torch.randn(M, K, device=device, dtype=torch.float32) * 100  # Large values to test scaling

    y_fp8, scales = quantize_to_fp8_triton(x.bfloat16())

    # 反量化
    x_reconstructed = dequantize_fp8_triton(y_fp8, scales, torch.float32)

    # 计算误差
    rel_error = ((x - x_reconstructed).abs() / (x.abs() + 1e-8)).mean().item() * 100
    max_error = (x - x_reconstructed).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        x.flatten().unsqueeze(0),
        x_reconstructed.flatten().unsqueeze(0)
    ).item()

    print(f"Shape: [{M}, {K}]")
    print(f"Relative error: {rel_error:.4f}%")
    print(f"Max absolute error: {max_error:.4f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Precision: {'EXCELLENT' if cos_sim > 0.999 else 'GOOD' if cos_sim > 0.99 else 'ACCEPTABLE'}")


if __name__ == "__main__":
    benchmark_fp8_quantization()
