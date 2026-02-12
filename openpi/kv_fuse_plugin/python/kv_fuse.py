#!/usr/bin/env python3
"""
KV Fuse: Fused QKV Projection with FP4 Weights

核心优化:
1. QKV projection融合为单个kernel
2. FP4权重在shared memory中decode,减少register pressure
3. 直接写入KV cache,避免transpose
4. GQA优化: K/V weight被所有Q heads共享

Usage:
    from kv_fuse import FusedQKVProjection, quantize_qkv_to_fp4

    # 量化权重
    Wq_packed, scale_Wq = quantize_to_fp4(Wq)
    Wk_packed, scale_Wk = quantize_to_fp4(Wk)
    Wv_packed, scale_Wv = quantize_to_fp4(Wv)

    # 创建fused layer
    fused_qkv = FusedQKVProjection(...)

    # 推理
    Q, K_cache, V_cache = fused_qkv(x, K_cache, V_cache, cache_pos)

Author: Claude Code
Date: 2026-02-10
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from pathlib import Path

# NVFP4 配置
NVFP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
NVFP4_MAX = 6.0
BLOCK_SIZE = 32

# CUDA extension
_cuda_ext = None
_cuda_ext_loaded = False


def _load_cuda_extension():
    """延迟加载CUDA extension."""
    global _cuda_ext, _cuda_ext_loaded

    if _cuda_ext_loaded:
        return _cuda_ext is not None

    _cuda_ext_loaded = True

    try:
        from torch.utils.cpp_extension import load

        plugin_dir = Path(__file__).parent.parent / 'src'

        _cuda_ext = load(
            name='fused_qkv_fp4_ext',
            sources=[str(plugin_dir / 'fused_qkv_fp4.cu')],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--use_fast_math'],  # Let nvcc auto-detect arch
            verbose=False
        )
        print("[KV_FUSE] Fused QKV FP4 kernel loaded successfully")
        return True
    except Exception as e:
        print(f"[KV_FUSE] Warning: CUDA kernel not available: {e}")
        return False


def quantize_to_fp4(
    tensor: torch.Tensor,
    block_size: int = BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    量化张量为NVFP4格式.

    Args:
        tensor: [N, K] 权重张量
        block_size: block scaling块大小

    Returns:
        (packed, scales): packed FP4权重和scale factors
    """
    N, K = tensor.shape
    device = tensor.device
    nvfp4_values = NVFP4_VALUES.to(device)

    # Reshape to blocks
    num_blocks = K // block_size
    tensor_blocked = tensor.view(N, num_blocks, block_size)

    # 计算per-block scale
    abs_max = tensor_blocked.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    scales = abs_max / NVFP4_MAX
    tensor_normalized = tensor_blocked / scales

    # 量化到NVFP4
    tensor_flat = tensor_normalized.view(-1)
    signs = (tensor_flat < 0).int()
    abs_vals = tensor_flat.abs()

    # 找最近的FP4值
    distances = (abs_vals.unsqueeze(-1) - nvfp4_values.unsqueeze(0)).abs()
    indices = distances.argmin(dim=-1)

    # 添加符号位
    quantized = indices + signs * 8
    quantized = quantized.view(N, K).to(torch.uint8)

    # Pack两个4-bit值为一个uint8
    K_half = K // 2
    packed = torch.zeros(N, K_half, dtype=torch.uint8, device=device)
    packed = (quantized[:, 0::2] & 0xF) | ((quantized[:, 1::2] & 0xF) << 4)

    scales = scales.squeeze(-1).float()  # [N, num_blocks]

    return packed, scales


def pack_fp4(quantized: torch.Tensor) -> torch.Tensor:
    """Pack FP4 values: 两个4-bit值 -> 一个uint8."""
    N, K = quantized.shape
    K_half = K // 2
    packed = (quantized[:, 0::2] & 0xF) | ((quantized[:, 1::2] & 0xF) << 4)
    return packed.to(torch.uint8)


def unpack_fp4(packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Unpack FP4 values: 一个uint8 -> 两个4-bit值."""
    device = packed.device
    unpacked = torch.zeros(N, K, dtype=torch.uint8, device=device)
    unpacked[:, 0::2] = packed & 0xF
    unpacked[:, 1::2] = (packed >> 4) & 0xF
    return unpacked


class FusedQKVProjection(nn.Module):
    """
    Fused QKV Projection with FP4 Weights.

    针对GQA优化: num_kv_heads << num_heads
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        block_size: int = BLOCK_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size

        self.q_dim = num_heads * head_dim
        self.kv_dim = num_kv_heads * head_dim

        # FP4 packed weights (registered as buffers)
        self.register_buffer('Wq_packed', None)
        self.register_buffer('scale_Wq', None)
        self.register_buffer('Wk_packed', None)
        self.register_buffer('scale_Wk', None)
        self.register_buffer('Wv_packed', None)
        self.register_buffer('scale_Wv', None)

        self._use_cuda = _load_cuda_extension()

    @classmethod
    def from_qkv_weights(
        cls,
        Wq: torch.Tensor,           # [q_dim, hidden_size]
        Wk: torch.Tensor,           # [kv_dim, hidden_size]
        Wv: torch.Tensor,           # [kv_dim, hidden_size]
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        block_size: int = BLOCK_SIZE,
    ) -> 'FusedQKVProjection':
        """从Q/K/V权重创建FusedQKVProjection."""
        hidden_size = Wq.size(1)

        layer = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            block_size=block_size,
        )

        # 量化权重
        with torch.no_grad():
            Wq_packed, scale_Wq = quantize_to_fp4(Wq.float(), block_size)
            Wk_packed, scale_Wk = quantize_to_fp4(Wk.float(), block_size)
            Wv_packed, scale_Wv = quantize_to_fp4(Wv.float(), block_size)

            layer.Wq_packed = Wq_packed
            layer.scale_Wq = scale_Wq
            layer.Wk_packed = Wk_packed
            layer.scale_Wk = scale_Wk
            layer.Wv_packed = Wv_packed
            layer.scale_Wv = scale_Wv

        return layer

    def forward(
        self,
        x: torch.Tensor,                    # [B, hidden_size]
        K_cache: torch.Tensor,              # [B, max_seq, num_kv_heads, head_dim]
        V_cache: torch.Tensor,              # [B, max_seq, num_kv_heads, head_dim]
        cache_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播.

        Args:
            x: 输入 [B, hidden_size]
            K_cache: K cache [B, max_seq, num_kv_heads, head_dim]
            V_cache: V cache [B, max_seq, num_kv_heads, head_dim]
            cache_pos: 当前token在cache中的位置

        Returns:
            Q: [B, num_heads * head_dim]
            K_cache: 更新后的K cache
            V_cache: 更新后的V cache
        """
        if self._use_cuda and _cuda_ext is not None:
            Q = _cuda_ext.fused_qkv_fp4(
                x.contiguous(),
                self.Wq_packed.contiguous(),
                self.scale_Wq.contiguous(),
                self.Wk_packed.contiguous(),
                self.scale_Wk.contiguous(),
                self.Wv_packed.contiguous(),
                self.scale_Wv.contiguous(),
                K_cache.contiguous(),
                V_cache.contiguous(),
                self.hidden_size,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                cache_pos
            )
            return Q, K_cache, V_cache
        else:
            return self._forward_fallback(x, K_cache, V_cache, cache_pos)

    def _forward_fallback(
        self,
        x: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        cache_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fallback: 反量化 + cuBLAS GEMM."""
        # 反量化权重
        Wq = self._dequantize(self.Wq_packed, self.scale_Wq, self.q_dim, self.hidden_size)
        Wk = self._dequantize(self.Wk_packed, self.scale_Wk, self.kv_dim, self.hidden_size)
        Wv = self._dequantize(self.Wv_packed, self.scale_Wv, self.kv_dim, self.hidden_size)

        # GEMM
        Q = torch.mm(x, Wq.t())
        K = torch.mm(x, Wk.t())
        V = torch.mm(x, Wv.t())

        # 更新KV cache
        B = x.size(0)
        K_cache[:, cache_pos, :, :] = K.view(B, self.num_kv_heads, self.head_dim)
        V_cache[:, cache_pos, :, :] = V.view(B, self.num_kv_heads, self.head_dim)

        return Q, K_cache, V_cache

    def _dequantize(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        N: int,
        K: int
    ) -> torch.Tensor:
        """反量化FP4权重."""
        device = packed.device
        nvfp4_values = NVFP4_VALUES.to(device)

        # Unpack
        unpacked = unpack_fp4(packed, N, K)

        # Decode
        signs = (unpacked >= 8).float()
        indices = (unpacked % 8).long()
        # Use flatten/unflatten to handle 2D tensor indexing
        flat_indices = indices.flatten()
        flat_values = nvfp4_values[flat_indices]
        values = flat_values.view(N, K)
        values = values * (1 - 2 * signs)

        # Apply scales
        num_blocks = K // self.block_size
        values_blocked = values.view(N, num_blocks, self.block_size)
        scales_expanded = scales.unsqueeze(-1)
        values_blocked = values_blocked * scales_expanded
        values = values_blocked.view(N, K)

        return values


def benchmark_qkv_fuse(
    hidden_size: int = 2048,
    num_heads: int = 8,
    num_kv_heads: int = 1,
    head_dim: int = 256,
    batch_size: int = 1,
    max_seq_len: int = 512,
    warmup: int = 50,
    runs: int = 200,
):
    """Benchmark fused vs separate QKV projection."""
    import time

    device = torch.device('cuda')
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(f"QKV Fuse Benchmark")
    print(f"{'='*60}")
    print(f"hidden_size={hidden_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}")
    print(f"head_dim={head_dim}, batch_size={batch_size}")

    # 创建随机权重
    Wq = torch.randn(q_dim, hidden_size, device=device)
    Wk = torch.randn(kv_dim, hidden_size, device=device)
    Wv = torch.randn(kv_dim, hidden_size, device=device)

    # 创建fused layer
    fused_layer = FusedQKVProjection.from_qkv_weights(
        Wq, Wk, Wv,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
    ).cuda()

    # 创建输入和cache
    x = torch.randn(batch_size, hidden_size, device=device)
    K_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim, device=device)
    V_cache = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim, device=device)

    # Benchmark: Fused FP4
    print("\n--- Fused FP4 Kernel ---")
    for _ in range(warmup):
        Q, K_cache, V_cache = fused_layer(x, K_cache, V_cache, 0)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for i in range(runs):
        Q, K_cache, V_cache = fused_layer(x, K_cache, V_cache, i % max_seq_len)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / runs * 1000
    print(f"Fused FP4: {fused_time:.3f} ms")

    # Benchmark: Separate cuBLAS BF16
    print("\n--- Separate cuBLAS BF16 ---")
    Wq_bf16 = Wq.to(torch.bfloat16)
    Wk_bf16 = Wk.to(torch.bfloat16)
    Wv_bf16 = Wv.to(torch.bfloat16)
    x_bf16 = x.to(torch.bfloat16)

    for _ in range(warmup):
        Q_bf16 = torch.mm(x_bf16, Wq_bf16.t())
        K_bf16 = torch.mm(x_bf16, Wk_bf16.t())
        V_bf16 = torch.mm(x_bf16, Wv_bf16.t())
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        Q_bf16 = torch.mm(x_bf16, Wq_bf16.t())
        K_bf16 = torch.mm(x_bf16, Wk_bf16.t())
        V_bf16 = torch.mm(x_bf16, Wv_bf16.t())
    torch.cuda.synchronize()
    separate_time = (time.perf_counter() - start) / runs * 1000
    print(f"Separate BF16: {separate_time:.3f} ms")

    # 结果
    speedup = separate_time / fused_time
    print(f"\n=> Speedup: {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}")

    return fused_time, separate_time, speedup


if __name__ == "__main__":
    # PaLiGemma配置
    print("\n=== PaLiGemma (Gemma 2B) ===")
    benchmark_qkv_fuse(
        hidden_size=2048,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    )

    # Action Expert配置
    print("\n=== Action Expert (Gemma 300M) ===")
    benchmark_qkv_fuse(
        hidden_size=1024,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    )
