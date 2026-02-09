#!/usr/bin/env python3
"""
Triton INT4 Linear Kernel for Thor

W4A16: INT4 权重, BF16/FP16 激活
目标: 将 MLP 带宽从 46.5ms 降到 ~12ms

实现策略:
1. Per-group quantization (每 128 个元素一个 scale)
2. Asymmetric quantization (min-max)
3. 权重打包: 2 个 INT4 → 1 个 UINT8
"""

import torch
import triton
import triton.language as tl
import time
from typing import Optional


# ============================================================
# INT4 Quantization Utils
# ============================================================

def quantize_int4_per_group(weight: torch.Tensor, group_size: int = 128) -> tuple:
    """
    Per-group INT4 quantization

    Args:
        weight: [out_features, in_features] FP16/BF16
        group_size: 量化组大小

    Returns:
        w_packed: [in_features, out_features // 2] UINT8 (打包后)
        w_scale: [in_features // group_size, out_features] FP16
        w_zero: [in_features // group_size, out_features] FP16
    """
    out_features, in_features = weight.shape
    device = weight.device

    # Transpose to [in_features, out_features] for matmul
    w = weight.t().float()  # [in, out]

    # Reshape for per-group quantization
    # [in_features, out_features] -> [num_groups, group_size, out_features]
    num_groups = in_features // group_size
    w_grouped = w.reshape(num_groups, group_size, out_features)

    # Compute min/max per group
    w_min = w_grouped.min(dim=1).values  # [num_groups, out_features]
    w_max = w_grouped.max(dim=1).values

    # Compute scale and zero point (asymmetric)
    scale = (w_max - w_min) / 15.0  # INT4: 0-15
    scale = scale.clamp(min=1e-8)  # Avoid division by zero
    zero = -w_min / scale  # Zero point
    zero = zero.round().clamp(0, 15)

    # Quantize
    w_int4 = ((w_grouped - w_min.unsqueeze(1)) / scale.unsqueeze(1))
    w_int4 = w_int4.round().clamp(0, 15).to(torch.uint8)
    w_int4 = w_int4.reshape(in_features, out_features)

    # Pack: 2 INT4 -> 1 UINT8
    # Even columns in low 4 bits, odd columns in high 4 bits
    w_packed = w_int4[:, 0::2] | (w_int4[:, 1::2] << 4)

    return (
        w_packed.contiguous(),
        scale.half().contiguous(),
        zero.half().contiguous(),
    )


def dequantize_int4_per_group(
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_zero: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Dequantize INT4 weights back to FP16

    Returns:
        weight: [in_features, out_features] FP16
    """
    in_features = w_packed.shape[0]
    out_features = w_packed.shape[1] * 2
    num_groups = in_features // group_size

    # Unpack
    w_low = (w_packed & 0x0F).to(torch.float16)
    w_high = ((w_packed >> 4) & 0x0F).to(torch.float16)

    # Interleave to [in_features, out_features]
    w_int4 = torch.zeros(in_features, out_features, dtype=torch.float16, device=w_packed.device)
    w_int4[:, 0::2] = w_low
    w_int4[:, 1::2] = w_high

    # Dequantize
    w_grouped = w_int4.reshape(num_groups, group_size, out_features)

    # w_fp16 = (w_int4 - zero) * scale
    w_dequant = (w_grouped - w_zero.unsqueeze(1)) * w_scale.unsqueeze(1)

    return w_dequant.reshape(in_features, out_features)


# ============================================================
# Triton INT4 MatMul Kernel (Simple Version)
# ============================================================

@triton.jit
def int4_matmul_kernel(
    # Pointers
    A_ptr,        # Input activation: [M, K] FP16
    W_packed_ptr, # Packed INT4 weights: [K, N//2] UINT8
    W_scale_ptr,  # Scale: [K//group_size, N] FP16
    W_zero_ptr,   # Zero: [K//group_size, N] FP16
    C_ptr,        # Output: [M, N] FP16
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_sk, stride_sn,
    stride_cm, stride_cn,
    # Constants
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    INT4 MatMul: C = A @ dequant(W_packed)

    Key optimizations:
    1. On-the-fly dequantization (no intermediate buffer)
    2. FP32 accumulator for precision
    3. Per-group scale/zero
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator (FP32 for precision)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load activation [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load packed INT4 weights [BLOCK_K, BLOCK_N//2]
        # Note: we need to handle the N dimension carefully
        w_n_offs = offs_n // 2  # Packed column index
        w_ptrs = W_packed_ptr + k_offs[:, None] * stride_wk + w_n_offs[None, :]
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_packed = tl.load(w_ptrs, mask=(k_offs[:, None] < K) & (w_n_offs[None, :] < N // 2), other=0)

        # Unpack INT4
        # Even columns: low 4 bits, Odd columns: high 4 bits
        is_odd = (offs_n[None, :] % 2) == 1
        w_int4 = tl.where(is_odd, (w_packed >> 4) & 0x0F, w_packed & 0x0F)
        w_int4 = w_int4.to(tl.float16)

        # Load scale and zero for this K block
        group_idx = k_offs // GROUP_SIZE
        s_ptrs = W_scale_ptr + group_idx[:, None] * stride_sk + offs_n[None, :] * stride_sn
        z_ptrs = W_zero_ptr + group_idx[:, None] * stride_sk + offs_n[None, :] * stride_sn

        s_mask = (group_idx[:, None] < K // GROUP_SIZE) & (offs_n[None, :] < N)
        scale = tl.load(s_ptrs, mask=s_mask, other=1.0)
        zero = tl.load(z_ptrs, mask=s_mask, other=0.0)

        # Dequantize: w_fp16 = (w_int4 - zero) * scale
        w = (w_int4 - zero) * scale

        # Matrix multiply accumulate
        acc += tl.dot(a.to(tl.float16), w.to(tl.float16)).to(tl.float32)

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# ============================================================
# PyTorch Wrapper
# ============================================================

class TritonINT4Linear(torch.nn.Module):
    """
    Triton INT4 Linear layer

    W4A16: INT4 weights, FP16 activations
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Packed INT4 weights
        self.register_buffer(
            "w_packed",
            torch.zeros(in_features, out_features // 2, dtype=torch.uint8)
        )

        # Per-group scale and zero
        num_groups = in_features // group_size
        self.register_buffer(
            "w_scale",
            torch.ones(num_groups, out_features, dtype=torch.float16)
        )
        self.register_buffer(
            "w_zero",
            torch.zeros(num_groups, out_features, dtype=torch.float16)
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float16)
            )
        else:
            self.bias = None

    @classmethod
    def from_float(
        cls,
        linear: torch.nn.Linear,
        group_size: int = 128,
    ) -> "TritonINT4Linear":
        """Create INT4 Linear from FP16/BF16 Linear"""
        instance = cls(
            linear.in_features,
            linear.out_features,
            group_size,
            bias=linear.bias is not None,
        )

        # Quantize weights
        w_packed, w_scale, w_zero = quantize_int4_per_group(
            linear.weight.data,
            group_size,
        )

        instance.w_packed = w_packed
        instance.w_scale = w_scale
        instance.w_zero = w_zero

        if linear.bias is not None:
            instance.bias = linear.bias.data.half()

        return instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Triton INT4 kernel"""
        # Reshape input to 2D
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        M = x_2d.shape[0]
        N = self.out_features
        K = self.in_features

        # Allocate output
        out = torch.empty(M, N, device=x.device, dtype=torch.float16)

        # Grid configuration
        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = min(128, self.group_size)  # Align with group size

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch kernel
        int4_matmul_kernel[grid](
            x_2d.half(), self.w_packed, self.w_scale, self.w_zero, out,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            self.w_packed.stride(0), self.w_packed.stride(1),
            self.w_scale.stride(0), self.w_scale.stride(1),
            out.stride(0), out.stride(1),
            GROUP_SIZE=self.group_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        # Reshape output
        return out.reshape(*orig_shape[:-1], N)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"group_size={self.group_size}, "
            f"bias={self.bias is not None}"
        )


# ============================================================
# Fallback: PyTorch INT4 Linear (for comparison)
# ============================================================

class PyTorchINT4Linear(torch.nn.Module):
    """
    PyTorch INT4 Linear (fallback implementation)

    Uses pre-dequantized weights for correctness testing
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Store quantization parameters
        self.register_buffer(
            "w_packed",
            torch.zeros(in_features, out_features // 2, dtype=torch.uint8)
        )

        num_groups = in_features // group_size
        self.register_buffer(
            "w_scale",
            torch.ones(num_groups, out_features, dtype=torch.float16)
        )
        self.register_buffer(
            "w_zero",
            torch.zeros(num_groups, out_features, dtype=torch.float16)
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: torch.nn.Linear, group_size: int = 128):
        instance = cls(
            linear.in_features,
            linear.out_features,
            group_size,
            bias=linear.bias is not None,
        )

        w_packed, w_scale, w_zero = quantize_int4_per_group(
            linear.weight.data,
            group_size,
        )

        instance.w_packed = w_packed
        instance.w_scale = w_scale
        instance.w_zero = w_zero

        if linear.bias is not None:
            instance.bias = linear.bias.data.half()

        return instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights on-the-fly
        w = dequantize_int4_per_group(
            self.w_packed,
            self.w_scale,
            self.w_zero,
            self.group_size,
        )

        # MatMul
        out = torch.matmul(x.half(), w)

        if self.bias is not None:
            out = out + self.bias

        return out


# ============================================================
# Benchmark and Validation
# ============================================================

def benchmark_int4_linear(
    in_features: int,
    out_features: int,
    seq_len: int = 712,
    group_size: int = 128,
    num_warmup: int = 20,
    num_iters: int = 100,
):
    """Benchmark INT4 vs FP16 Linear"""
    device = "cuda"

    print(f"\n{'='*60}")
    print(f"Benchmark: ({seq_len}, {in_features}) @ ({in_features}, {out_features})")
    print(f"Group size: {group_size}")
    print(f"{'='*60}")

    # Create FP16 baseline
    fp16_linear = torch.nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16)

    # Create INT4 versions
    pytorch_int4 = PyTorchINT4Linear.from_float(fp16_linear, group_size).to(device)

    # Try Triton INT4
    try:
        triton_int4 = TritonINT4Linear.from_float(fp16_linear, group_size).to(device)
        triton_available = True
    except Exception as e:
        print(f"⚠️ Triton INT4 failed: {e}")
        triton_available = False

    # Test input
    x = torch.randn(1, seq_len, in_features, device=device, dtype=torch.bfloat16)

    # Validate correctness
    print("\nCorrectness validation:")
    print("-" * 40)

    with torch.no_grad():
        fp16_out = fp16_linear(x)
        pytorch_int4_out = pytorch_int4(x)

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            fp16_out.flatten().float(),
            pytorch_int4_out.flatten().float(),
            dim=0,
        ).item()

        # Max error
        max_err = (fp16_out - pytorch_int4_out).abs().max().item()

        print(f"PyTorch INT4 vs FP16:")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max absolute error: {max_err:.6f}")

        if triton_available:
            triton_int4_out = triton_int4(x.half())

            cos_sim_triton = torch.nn.functional.cosine_similarity(
                fp16_out.flatten().float(),
                triton_int4_out.flatten().float(),
                dim=0,
            ).item()

            max_err_triton = (fp16_out.half() - triton_int4_out).abs().max().item()

            print(f"Triton INT4 vs FP16:")
            print(f"  Cosine similarity: {cos_sim_triton:.6f}")
            print(f"  Max absolute error: {max_err_triton:.6f}")

    # Benchmark
    print("\nPerformance benchmark:")
    print("-" * 40)

    # Weight size
    fp16_size = in_features * out_features * 2  # BF16 = 2 bytes
    int4_size = in_features * out_features // 2  # INT4 = 0.5 bytes
    print(f"FP16 weight size: {fp16_size / 1e6:.2f} MB")
    print(f"INT4 weight size: {int4_size / 1e6:.2f} MB (+ scale/zero)")

    # Warmup FP16
    for _ in range(num_warmup):
        _ = fp16_linear(x)
    torch.cuda.synchronize()

    # Benchmark FP16
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = fp16_linear(x)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / num_iters * 1000
    fp16_bw = fp16_size / (fp16_time / 1000) / 1e9

    print(f"\nFP16 Linear:")
    print(f"  Time: {fp16_time:.3f} ms")
    print(f"  Effective BW: {fp16_bw:.1f} GB/s")

    # Warmup and benchmark PyTorch INT4
    for _ in range(num_warmup):
        _ = pytorch_int4(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = pytorch_int4(x)
    torch.cuda.synchronize()
    pytorch_int4_time = (time.perf_counter() - start) / num_iters * 1000

    print(f"\nPyTorch INT4 (dequant + matmul):")
    print(f"  Time: {pytorch_int4_time:.3f} ms")
    print(f"  Speedup vs FP16: {fp16_time/pytorch_int4_time:.2f}x")

    if triton_available:
        # Warmup and benchmark Triton INT4
        x_fp16 = x.half()
        for _ in range(num_warmup):
            _ = triton_int4(x_fp16)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            _ = triton_int4(x_fp16)
        torch.cuda.synchronize()
        triton_int4_time = (time.perf_counter() - start) / num_iters * 1000
        triton_bw = int4_size / (triton_int4_time / 1000) / 1e9

        print(f"\nTriton INT4 (fused dequant+matmul):")
        print(f"  Time: {triton_int4_time:.3f} ms")
        print(f"  Effective BW: {triton_bw:.1f} GB/s")
        print(f"  Speedup vs FP16: {fp16_time/triton_int4_time:.2f}x")

    return {
        "fp16_time_ms": fp16_time,
        "pytorch_int4_time_ms": pytorch_int4_time,
        "triton_int4_time_ms": triton_int4_time if triton_available else None,
        "cosine_similarity": cos_sim,
    }


def main():
    """Run benchmarks for MLP layers"""
    print("=" * 60)
    print("Triton INT4 Linear Kernel Benchmark")
    print("Target: KV Cache MLP acceleration")
    print("=" * 60)

    # MLP configurations (Pi0.5)
    configs = [
        # (in, out, name)
        (2048, 16384, "MLP gate_proj"),
        (2048, 16384, "MLP up_proj"),
        (16384, 2048, "MLP down_proj"),
    ]

    results = {}
    for in_dim, out_dim, name in configs:
        print(f"\n\n{'#'*60}")
        print(f"# {name}")
        print(f"{'#'*60}")

        result = benchmark_int4_linear(in_dim, out_dim, seq_len=712, group_size=128)
        results[name] = result

    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY: 18-layer MLP time estimates")
    print("=" * 60)

    fp16_total = 0
    int4_total = 0

    for name, result in results.items():
        fp16_layer = result["fp16_time_ms"]
        int4_layer = result.get("triton_int4_time_ms") or result["pytorch_int4_time_ms"]

        fp16_total += fp16_layer
        int4_total += int4_layer

        print(f"{name}: {fp16_layer:.2f} ms → {int4_layer:.2f} ms")

    # Scale to 18 layers
    fp16_18 = fp16_total * 18 / 3  # 3 MLP layers per transformer
    int4_18 = int4_total * 18 / 3

    print(f"\nFP16 MLP (18 layers): {fp16_18:.1f} ms")
    print(f"INT4 MLP (18 layers): {int4_18:.1f} ms")
    print(f"Speedup: {fp16_18/int4_18:.2f}x")
    print(f"Time saved: {fp16_18 - int4_18:.1f} ms")


if __name__ == "__main__":
    main()
