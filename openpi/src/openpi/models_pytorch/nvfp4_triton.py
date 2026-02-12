#!/usr/bin/env python3
"""
Triton-based NVFP4 Quantization Kernels

Optimized kernels for NVFP4 (E2M1) block-scaled quantization on Thor SM110.
Replaces Python-based quantization (~7000ms) with fused Triton kernels (<20ms).

NVFP4 E2M1 可表示的值: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

Performance Target:
- Python版本: ~7000ms per inference (0.14 Hz)
- Triton目标: <20ms per inference (50+ Hz)

Author: Claude Code
Date: 2026-02-09
"""

import torch
import triton
import triton.language as tl
from typing import Tuple

# NVFP4 配置
BLOCK_SIZE = 32  # Block scaling 的块大小
NVFP4_MAX = 6.0  # NVFP4 E2M1 的最大可表示值


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def _quantize_nvfp4_kernel(
    x_ptr,              # 输入: BF16/FP32 [M, K]
    q_ptr,              # 输出: FP32 量化值 [M, K] (用于验证/dequant)
    scale_ptr,          # 输出: FP32 scales [M, K // BLOCK_SIZE]
    M,                  # 行数
    K,                  # 列数
    stride_xm,          # x 的 row stride
    stride_xk,          # x 的 col stride
    stride_qm,          # q 的 row stride
    stride_qk,          # q 的 col stride
    stride_sm,          # scale 的 row stride
    BLOCK_SIZE: tl.constexpr,
    NVFP4_MAX_VAL: tl.constexpr,  # NVFP4 最大值 (6.0)
):
    """
    Fused NVFP4 quantization kernel.

    每个 program 处理一个 block (32 个元素)。
    计算 block 内的 abs max，然后量化到 NVFP4。
    """
    # Program ID
    pid_m = tl.program_id(0)  # 行索引
    pid_k = tl.program_id(1)  # block 索引 (在 K 维度)

    # 计算该 block 的起始偏移
    block_start_k = pid_k * BLOCK_SIZE

    # 加载 block 数据
    offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = block_start_k + offsets
    mask = k_offsets < K

    x_ptrs = x_ptr + pid_m * stride_xm + k_offsets * stride_xk
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # 计算 block abs max (Scale)
    abs_x = tl.abs(x)
    block_max = tl.max(abs_x)

    # 防止除零，并计算 scale
    # scale = block_max / NVFP4_MAX
    scale = tl.maximum(block_max, 1e-12) / NVFP4_MAX_VAL

    # 存储 scale
    scale_ptr_offset = scale_ptr + pid_m * stride_sm + pid_k
    tl.store(scale_ptr_offset, scale)

    # 量化到 NVFP4 范围 [-6, 6]
    x_scaled = x / tl.maximum(scale, 1e-12)
    x_clamped = tl.minimum(tl.maximum(x_scaled, -NVFP4_MAX_VAL), NVFP4_MAX_VAL)

    # NVFP4 E2M1 可表示的正值: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    # 使用分段线性映射找到最近值
    sign = tl.where(x_clamped >= 0, 1.0, -1.0)
    abs_val = tl.abs(x_clamped)

    # 找到最近的 NVFP4 值 (分段逻辑)
    # 0: [0, 0.25)
    # 0.5: [0.25, 0.75)
    # 1: [0.75, 1.25)
    # 1.5: [1.25, 1.75)
    # 2: [1.75, 2.5)
    # 3: [2.5, 3.5)
    # 4: [3.5, 5.0)
    # 6: [5.0, inf)

    q_abs = tl.where(abs_val < 0.25, 0.0,
            tl.where(abs_val < 0.75, 0.5,
            tl.where(abs_val < 1.25, 1.0,
            tl.where(abs_val < 1.75, 1.5,
            tl.where(abs_val < 2.5, 2.0,
            tl.where(abs_val < 3.5, 3.0,
            tl.where(abs_val < 5.0, 4.0, 6.0)))))))

    q_val = sign * q_abs

    # 存储量化后的值 (用于 simulation 验证，实际 CUTLASS 用 packed format)
    q_ptrs = q_ptr + pid_m * stride_qm + k_offsets * stride_qk
    tl.store(q_ptrs, q_val, mask=mask)


@triton.jit
def _quantize_and_pack_nvfp4_kernel(
    x_ptr,              # 输入: BF16/FP32 [M, K]
    packed_ptr,         # 输出: uint8 packed [M, K // 2]
    scale_ptr,          # 输出: FP32 scales [M, K // BLOCK_SIZE]
    M,                  # 行数
    K,                  # 列数
    stride_xm,          # x 的 row stride
    stride_xk,          # x 的 col stride
    stride_pm,          # packed 的 row stride
    stride_pk,          # packed 的 col stride
    stride_sm,          # scale 的 row stride
    BLOCK_SIZE: tl.constexpr,
    NVFP4_MAX_VAL: tl.constexpr,
):
    """
    Fused NVFP4 quantize + pack kernel.

    每个 program 处理一个 block (32 元素)，输出 16 个 packed bytes。

    Packing format:
    - 每个 byte 包含两个 4-bit FP4 值
    - Low nibble: even index element
    - High nibble: odd index element
    - FP4 encoding: sign(1) | magnitude_index(3)
    """
    # Program ID
    pid_m = tl.program_id(0)  # 行索引
    pid_k = tl.program_id(1)  # block 索引

    block_start_k = pid_k * BLOCK_SIZE

    # 加载 block 数据
    offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = block_start_k + offsets
    mask = k_offsets < K

    x_ptrs = x_ptr + pid_m * stride_xm + k_offsets * stride_xk
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # 计算 block abs max
    abs_x = tl.abs(x)
    block_max = tl.max(abs_x)
    scale = tl.maximum(block_max, 1e-12) / NVFP4_MAX_VAL

    # 存储 scale
    tl.store(scale_ptr + pid_m * stride_sm + pid_k, scale)

    # 简化实现：分别提取偶数和奇数位置的值
    half_size: tl.constexpr = BLOCK_SIZE // 2
    even_offsets = tl.arange(0, half_size) * 2
    odd_offsets = even_offsets + 1

    low_vals = tl.load(x_ptr + pid_m * stride_xm + (block_start_k + even_offsets) * stride_xk,
                       mask=(block_start_k + even_offsets) < K, other=0.0)
    high_vals = tl.load(x_ptr + pid_m * stride_xm + (block_start_k + odd_offsets) * stride_xk,
                        mask=(block_start_k + odd_offsets) < K, other=0.0)

    # 量化 low 值
    low_scaled = low_vals / tl.maximum(scale, 1e-12)
    low_clamped = tl.minimum(tl.maximum(low_scaled, -NVFP4_MAX_VAL), NVFP4_MAX_VAL)
    low_sign = tl.where(low_clamped < 0, 1, 0).to(tl.uint8)
    low_abs = tl.abs(low_clamped)
    low_idx = tl.where(low_abs < 0.25, 0,
              tl.where(low_abs < 0.75, 1,
              tl.where(low_abs < 1.25, 2,
              tl.where(low_abs < 1.75, 3,
              tl.where(low_abs < 2.5, 4,
              tl.where(low_abs < 3.5, 5,
              tl.where(low_abs < 5.0, 6, 7))))))).to(tl.uint8)
    low_enc = (low_sign << 3) | low_idx

    # 量化 high 值
    high_scaled = high_vals / tl.maximum(scale, 1e-12)
    high_clamped = tl.minimum(tl.maximum(high_scaled, -NVFP4_MAX_VAL), NVFP4_MAX_VAL)
    high_sign = tl.where(high_clamped < 0, 1, 0).to(tl.uint8)
    high_abs = tl.abs(high_clamped)
    high_idx = tl.where(high_abs < 0.25, 0,
               tl.where(high_abs < 0.75, 1,
               tl.where(high_abs < 1.25, 2,
               tl.where(high_abs < 1.75, 3,
               tl.where(high_abs < 2.5, 4,
               tl.where(high_abs < 3.5, 5,
               tl.where(high_abs < 5.0, 6, 7))))))).to(tl.uint8)
    high_enc = (high_sign << 3) | high_idx

    # 打包: low | (high << 4)
    packed = low_enc | (high_enc << 4)

    # 存储 packed 数据
    pack_offsets = tl.arange(0, half_size)
    packed_k_start = pid_k * half_size
    packed_ptrs = packed_ptr + pid_m * stride_pm + (packed_k_start + pack_offsets) * stride_pk
    pack_mask = (packed_k_start + pack_offsets) < (K // 2)
    tl.store(packed_ptrs, packed, mask=pack_mask)


@triton.jit
def _dequantize_nvfp4_kernel(
    q_ptr,              # 输入: FP32 量化值 [M, K]
    scale_ptr,          # 输入: FP32 scales [M, num_blocks]
    out_ptr,            # 输出: BF16/FP32 [M, K]
    M,
    K,
    stride_qm,
    stride_qk,
    stride_sm,
    stride_om,
    stride_ok,
    BLOCK_SIZE: tl.constexpr,
):
    """
    NVFP4 dequantization kernel.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    block_start_k = pid_k * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = block_start_k + offsets
    mask = k_offsets < K

    # 加载量化值
    q_ptrs = q_ptr + pid_m * stride_qm + k_offsets * stride_qk
    q = tl.load(q_ptrs, mask=mask, other=0.0)

    # 加载 scale
    scale = tl.load(scale_ptr + pid_m * stride_sm + pid_k)

    # 反量化
    out = q * scale

    # 存储
    out_ptrs = out_ptr + pid_m * stride_om + k_offsets * stride_ok
    tl.store(out_ptrs, out, mask=mask)


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def quantize_nvfp4_triton(
    x: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 Triton 进行 NVFP4 量化。

    Args:
        x: [M, K] 输入张量 (BF16/FP32)
        block_size: block scaling 的块大小

    Returns:
        (quantized, scales): 量化后的值和 scale factors
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, K = x.shape
    assert K % block_size == 0, f"K ({K}) must be divisible by block_size ({block_size})"

    num_blocks = K // block_size

    # 输出 tensors
    quantized = torch.empty_like(x, dtype=torch.float32)
    scales = torch.empty(M, num_blocks, device=x.device, dtype=torch.float32)

    # Launch kernel
    grid = (M, num_blocks)
    _quantize_nvfp4_kernel[grid](
        x, quantized, scales,
        M, K,
        x.stride(0), x.stride(1),
        quantized.stride(0), quantized.stride(1),
        scales.stride(0),
        BLOCK_SIZE=block_size,
        NVFP4_MAX_VAL=NVFP4_MAX,
    )

    return quantized.to(x.dtype), scales


def quantize_and_pack_nvfp4_triton(
    x: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 Triton 进行 NVFP4 量化并打包。

    Args:
        x: [M, K] 输入张量
        block_size: block scaling 的块大小

    Returns:
        (packed, scales): packed uint8 tensor 和 scale factors
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, K = x.shape
    assert K % block_size == 0, f"K ({K}) must be divisible by block_size ({block_size})"
    assert K % 2 == 0, f"K ({K}) must be even for packing"

    num_blocks = K // block_size

    # 输出 tensors
    packed = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8)
    scales = torch.empty(M, num_blocks, device=x.device, dtype=torch.float32)

    # Launch kernel
    grid = (M, num_blocks)
    _quantize_and_pack_nvfp4_kernel[grid](
        x, packed, scales,
        M, K,
        x.stride(0), x.stride(1),
        packed.stride(0), packed.stride(1),
        scales.stride(0),
        BLOCK_SIZE=block_size,
        NVFP4_MAX_VAL=NVFP4_MAX,
    )

    return packed, scales


def dequantize_nvfp4_triton(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    使用 Triton 进行 NVFP4 反量化。

    Args:
        quantized: [M, K] 量化后的值
        scales: [M, num_blocks] scale factors
        block_size: block scaling 的块大小
        target_dtype: 输出 dtype

    Returns:
        反量化后的张量
    """
    M, K = quantized.shape
    num_blocks = K // block_size

    out = torch.empty(M, K, device=quantized.device, dtype=torch.float32)

    grid = (M, num_blocks)
    _dequantize_nvfp4_kernel[grid](
        quantized.float(), scales, out,
        M, K,
        quantized.stride(0), quantized.stride(1),
        scales.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=block_size,
    )

    return out.to(target_dtype)


# ============================================================================
# NVFP4 Linear with Triton Quantization
# ============================================================================

class NVFP4LinearTriton(torch.nn.Module):
    """
    使用 Triton 加速的 NVFP4 Linear 层。

    权重预量化为 NVFP4，输入使用 Triton kernel 在线量化。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: int = BLOCK_SIZE,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # 原始权重
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # 量化权重缓存
        self.register_buffer('weight_q', None)
        self.register_buffer('weight_scales', None)

        self._quantized = False
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, block_size: int = BLOCK_SIZE) -> 'NVFP4LinearTriton':
        """从 nn.Linear 创建。"""
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        layer.quantize_weights()
        return layer

    def quantize_weights(self):
        """预量化权重。"""
        with torch.no_grad():
            self.weight_q, self.weight_scales = quantize_nvfp4_triton(
                self.weight.data.float(), self.block_size
            )
            self.weight_q = self.weight_q.to(self.weight.dtype)
            self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        if not self._quantized:
            self.quantize_weights()

        original_dtype = x.dtype
        batch_shape = x.shape[:-1]
        x_2d = x.view(-1, self.in_features).float()

        # 使用 Triton 量化输入
        x_q, x_scales = quantize_nvfp4_triton(x_2d, self.block_size)

        # 反量化并计算 (Simulation mode)
        x_dequant = dequantize_nvfp4_triton(x_q, x_scales, self.block_size, torch.float32)
        w_dequant = dequantize_nvfp4_triton(self.weight_q.float(), self.weight_scales, self.block_size, torch.float32)

        out = torch.nn.functional.linear(x_dequant, w_dequant, self.bias.float() if self.bias is not None else None)

        return out.view(*batch_shape, self.out_features).to(original_dtype)


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_triton_vs_python():
    """Benchmark Triton vs Python NVFP4 quantization."""
    import time

    print("=" * 70)
    print("NVFP4 Quantization Benchmark: Triton vs Python")
    print("=" * 70)

    device = torch.device("cuda")

    # 典型的 Pi0 activation 尺寸
    test_cases = [
        (256, 2048),    # Typical batch
        (256, 16384),   # Intermediate size
        (1, 2048),      # Single sample
        (1024, 2048),   # Large batch
    ]

    for M, K in test_cases:
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(3):
            _ = quantize_nvfp4_triton(x)
        torch.cuda.synchronize()

        # Benchmark Triton
        iterations = 100
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            q, s = quantize_nvfp4_triton(x)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - start) / iterations * 1000

        # Benchmark Python (from nvfp4_mlp.py)
        try:
            from openpi.models_pytorch.nvfp4_mlp import quantize_to_nvfp4_sim

            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                q_py, s_py = quantize_to_nvfp4_sim(x.float(), use_mse_search=False)
            torch.cuda.synchronize()
            python_ms = (time.perf_counter() - start) / iterations * 1000

            speedup = python_ms / triton_ms
        except ImportError:
            python_ms = float('nan')
            speedup = float('nan')

        print(f"[{M:4d}, {K:5d}] Triton: {triton_ms:.3f} ms | Python: {python_ms:.3f} ms | Speedup: {speedup:.1f}x")

    print("\n" + "=" * 70)
    print("Accuracy Validation")
    print("=" * 70)

    # 验证精度
    M, K = 256, 2048
    x = torch.randn(M, K, device=device, dtype=torch.float32)

    q_triton, s_triton = quantize_nvfp4_triton(x)

    try:
        from openpi.models_pytorch.nvfp4_mlp import quantize_to_nvfp4_sim, dequantize_nvfp4_sim
        q_python, s_python = quantize_to_nvfp4_sim(x, use_mse_search=False)

        # 反量化并比较
        x_triton = dequantize_nvfp4_triton(q_triton.float(), s_triton)
        x_python = dequantize_nvfp4_sim(q_python, s_python)

        # 计算误差
        diff = (x_triton - x_python).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # 与原始输入的相对误差
        rel_error_triton = ((x - x_triton.to(x.dtype)).abs() / (x.abs() + 1e-8)).mean().item() * 100
        rel_error_python = ((x - x_python.to(x.dtype)).abs() / (x.abs() + 1e-8)).mean().item() * 100

        print(f"Max diff (Triton vs Python): {max_diff:.6f}")
        print(f"Mean diff (Triton vs Python): {mean_diff:.6f}")
        print(f"Relative error (Triton): {rel_error_triton:.2f}%")
        print(f"Relative error (Python): {rel_error_python:.2f}%")

    except ImportError:
        print("Could not import Python reference for validation")


if __name__ == "__main__":
    benchmark_triton_vs_python()
