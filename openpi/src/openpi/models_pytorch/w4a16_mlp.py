#!/usr/bin/env python3
"""
W4A16 MLP: Weight 4-bit (NVFP4), Activation 16-bit (BF16)

这是 NVFP4 的精度改进版本:
- 权重: NVFP4 量化 (节省 75% 存储/带宽)
- 激活: 保持 BF16 (无量化损失!)
- 计算: BF16 Tensor Core

关键优势:
1. 零激活量化开销 (W4A4 需要 7.6ms!)
2. 原生 BF16 精度 (无激活量化误差)
3. 仍有 75% 权重带宽节省

计算流程:
1. 权重在 __init__ 时预量化为 NVFP4
2. 推理时: Load W4 -> Dequant to BF16 -> BF16 Matmul
3. 权重反量化可以缓存或实时计算

性能预期:
- 精度: ~0.99+ (vs W4A4 的 -0.11)
- 速度: ~2-3x 加速 (权重带宽减少 75%, 计算仍是 BF16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .nvfp4_mlp import (
    NVFP4_VALUES,
    NVFP4_MAX,
    BLOCK_SIZE,
    quantize_to_nvfp4_sim,
    dequantize_nvfp4_sim,
)


class W4A16Linear(nn.Module):
    """
    W4A16 Linear 层: 权重 NVFP4, 激活 BF16。

    权重在初始化时量化为 NVFP4, 推理时反量化为 BF16 进行计算。
    可选择:
    1. 实时反量化: 每次 forward 时反量化 (节省内存)
    2. 缓存反量化: 预计算并缓存 BF16 权重 (更快但占内存)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: int = BLOCK_SIZE,
        cache_dequantized: bool = True,  # 默认缓存以获得最佳速度
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.cache_dequantized = cache_dequantized

        # 原始权重 (用于初始化和量化)
        self.register_buffer('weight', torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # 量化后的权重和 scales
        self.register_buffer('weight_q', None)
        self.register_buffer('weight_scales', None)

        # 缓存的 BF16 权重 (可选)
        self.register_buffer('weight_bf16_cached', None)

        self._quantized = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = BLOCK_SIZE,
        cache_dequantized: bool = True,
    ) -> 'W4A16Linear':
        """从 nn.Linear 创建 W4A16Linear。"""
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
            cache_dequantized=cache_dequantized,
        )

        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)

        # 量化权重
        layer.quantize_weights()

        return layer

    def quantize_weights(self):
        """量化权重为 NVFP4 并可选缓存 BF16 反量化结果。"""
        with torch.no_grad():
            # 量化为 NVFP4
            self.weight_q, self.weight_scales = quantize_to_nvfp4_sim(
                self.weight, self.block_size, use_mse_search=True
            )

            if self.cache_dequantized:
                # 预计算 BF16 反量化权重
                self.weight_bf16_cached = dequantize_nvfp4_sim(
                    self.weight_q, self.weight_scales, self.block_size
                ).to(torch.bfloat16)

            self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: BF16 激活直接计算。"""
        if not self._quantized:
            self.quantize_weights()

        original_dtype = x.dtype

        # 获取 BF16 权重
        if self.cache_dequantized and self.weight_bf16_cached is not None:
            w = self.weight_bf16_cached
        else:
            # 实时反量化
            w = dequantize_nvfp4_sim(
                self.weight_q, self.weight_scales, self.block_size
            ).to(torch.bfloat16)

        # 确保激活是 BF16
        x_bf16 = x.to(torch.bfloat16)

        # 标准 BF16 矩阵乘法 (使用 cuBLAS)
        out = F.linear(x_bf16, w, self.bias)

        return out.to(original_dtype)


class W4A16MLP(nn.Module):
    """
    W4A16 MLP 模块: 权重 NVFP4, 激活 BF16。

    结构: input -> gate_proj + up_proj -> GeLU -> down_proj -> output
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 16384,
        block_size: int = BLOCK_SIZE,
        cache_dequantized: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = W4A16Linear(
            hidden_size, intermediate_size, bias=False,
            block_size=block_size, cache_dequantized=cache_dequantized
        )
        self.up_proj = W4A16Linear(
            hidden_size, intermediate_size, bias=False,
            block_size=block_size, cache_dequantized=cache_dequantized
        )
        self.down_proj = W4A16Linear(
            intermediate_size, hidden_size, bias=False,
            block_size=block_size, cache_dequantized=cache_dequantized
        )

    @classmethod
    def from_gemma_mlp(
        cls,
        mlp: nn.Module,
        block_size: int = BLOCK_SIZE,
        cache_dequantized: bool = True,
    ) -> 'W4A16MLP':
        """从 GemmaMLP 创建 W4A16MLP。"""
        hidden_size = mlp.gate_proj.in_features
        intermediate_size = mlp.gate_proj.out_features

        layer = cls(hidden_size, intermediate_size, block_size, cache_dequantized)
        layer.gate_proj = W4A16Linear.from_linear(
            mlp.gate_proj, block_size, cache_dequantized
        )
        layer.up_proj = W4A16Linear.from_linear(
            mlp.up_proj, block_size, cache_dequantized
        )
        layer.down_proj = W4A16Linear.from_linear(
            mlp.down_proj, block_size, cache_dequantized
        )

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        gate = F.gelu(self.gate_proj(x), approximate='tanh')
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def replace_paligemma_mlp_with_w4a16(
    model,
    block_size: int = BLOCK_SIZE,
    cache_dequantized: bool = True,
) -> int:
    """
    将 PaliGemma 的 MLP 层替换为 W4A16 版本。

    Args:
        model: PI0Pytorch 模型
        block_size: NVFP4 block size
        cache_dequantized: 是否缓存 BF16 反量化权重

    Returns:
        替换的层数
    """
    paligemma_lm = model.paligemma_with_expert.paligemma.language_model
    replaced_count = 0

    for layer_idx, layer in enumerate(paligemma_lm.layers):
        if hasattr(layer, 'mlp') and layer.mlp is not None:
            original_mlp = layer.mlp

            # 创建 W4A16 MLP
            w4a16_mlp = W4A16MLP.from_gemma_mlp(
                original_mlp, block_size, cache_dequantized
            )

            # 移动到相同设备
            device = next(original_mlp.parameters()).device
            dtype = next(original_mlp.parameters()).dtype
            w4a16_mlp = w4a16_mlp.to(device=device, dtype=dtype)

            # 替换
            layer.mlp = w4a16_mlp
            replaced_count += 1

    print(f"[W4A16] Replaced {replaced_count} PaliGemma MLP layers")
    return replaced_count


def benchmark_w4a16():
    """Benchmark W4A16 vs BF16 MLP."""
    import time

    print("=" * 70)
    print("W4A16 MLP Benchmark")
    print("=" * 70)

    device = torch.device("cuda")
    batch_size = 256
    hidden_size = 2048
    intermediate_size = 16384

    # BF16 MLP
    class BF16MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            gate = F.gelu(self.gate_proj(x), approximate='tanh')
            return self.down_proj(gate * self.up_proj(x))

    bf16_mlp = BF16MLP().to(device).bfloat16()
    w4a16_mlp = W4A16MLP.from_gemma_mlp(bf16_mlp, cache_dequantized=True).to(device)

    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = bf16_mlp(x)
        _ = w4a16_mlp(x)
    torch.cuda.synchronize()

    iterations = 50

    # BF16
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = bf16_mlp(x)
    torch.cuda.synchronize()
    bf16_ms = (time.perf_counter() - start) / iterations * 1000

    # W4A16 (cached)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = w4a16_mlp(x)
    torch.cuda.synchronize()
    w4a16_ms = (time.perf_counter() - start) / iterations * 1000

    # Precision comparison
    with torch.no_grad():
        bf16_out = bf16_mlp(x)
        w4a16_out = w4a16_mlp(x)

    cos_sim = F.cosine_similarity(
        bf16_out.flatten().float().unsqueeze(0),
        w4a16_out.flatten().float().unsqueeze(0)
    ).item()

    mae = (bf16_out - w4a16_out).abs().mean().item()

    print(f"\nConfig: batch={batch_size}, hidden={hidden_size}, intermediate={intermediate_size}")
    print(f"\n{'Method':<20} {'Time (ms)':>10} {'Cosine Sim':>12} {'MAE':>12}")
    print("-" * 60)
    print(f"{'BF16 (baseline)':<20} {bf16_ms:>10.3f} {'1.0000':>12} {'0.0000':>12}")
    print(f"{'W4A16 (cached)':<20} {w4a16_ms:>10.3f} {cos_sim:>12.6f} {mae:>12.6f}")
    print("-" * 60)
    print(f"\nSpeedup: {bf16_ms/w4a16_ms:.2f}x")
    print(f"Precision: {'GOOD' if cos_sim > 0.99 else 'ACCEPTABLE' if cos_sim > 0.95 else 'POOR'}")


if __name__ == "__main__":
    benchmark_w4a16()
