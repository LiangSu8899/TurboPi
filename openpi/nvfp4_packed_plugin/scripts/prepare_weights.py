#!/usr/bin/env python3
"""
NVFP4 Packed Weight Preparation Script

离线将模型权重转换为 packed uint8 格式，用于 TRT Plugin 推理。

功能:
1. 量化权重为 NVFP4 (E2M1 format)
2. Pack 为 uint8 (2 个 FP4 值 per byte)
3. 计算 block-scaled factors
4. 支持多种输出格式 (npz, safetensors, bin)

使用方式:
    python prepare_weights.py --input model.safetensors --output weights_packed.npz

Author: Claude Code
Date: 2026-02-10
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

# NVFP4 配置
NVFP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
NVFP4_MAX = 6.0
BLOCK_SIZE = 32


def quantize_to_nvfp4(
    tensor: torch.Tensor,
    block_size: int = BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    量化张量为 NVFP4 格式。

    Args:
        tensor: [N, K] 权重张量
        block_size: block scaling 块大小

    Returns:
        (quantized, scales): 量化后的张量 (仍为 float) 和 scale factors
    """
    N, K = tensor.shape
    device = tensor.device
    nvfp4_values = NVFP4_VALUES.to(device)

    # Reshape to blocks: [N, num_blocks, block_size]
    num_blocks = K // block_size
    tensor_blocked = tensor.view(N, num_blocks, block_size).float()

    # 计算 scale factors (per block max)
    block_max = tensor_blocked.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale_factors = block_max / NVFP4_MAX

    # 量化
    scaled = tensor_blocked / block_max * NVFP4_MAX
    scaled = scaled.clamp(-NVFP4_MAX, NVFP4_MAX)

    signs = scaled.sign()
    abs_scaled = scaled.abs()

    # 找最近的 NVFP4 值
    distances = (abs_scaled.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1)
    quantized_abs = nvfp4_values[indices]
    quantized = (signs * quantized_abs).view(N, K)

    return quantized, scale_factors.squeeze(-1)


def pack_nvfp4(quantized: torch.Tensor) -> torch.Tensor:
    """
    将量化后的 NVFP4 值打包为 uint8。

    Args:
        quantized: [N, K] 量化后的张量 (值在 -6 到 6 范围)

    Returns:
        [N, K//2] packed uint8 tensor
    """
    N, K = quantized.shape
    device = quantized.device
    nvfp4_values = NVFP4_VALUES.to(device)

    # NVFP4 编码: sign(1) + magnitude_index(3)
    signs = (quantized < 0).to(torch.uint8) << 3
    abs_vals = quantized.abs()

    # 找最近值的索引
    distances = (abs_vals.unsqueeze(-1) - nvfp4_values).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    # 编码
    encoded = signs | indices  # [N, K], 值 0-15

    # 打包: 两个 4-bit 值放入一个 byte
    low = encoded[:, 0::2]
    high = encoded[:, 1::2]
    packed = low | (high << 4)

    return packed.to(torch.uint8)


def unpack_nvfp4(packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """
    解包 uint8 为 NVFP4 值 (用于验证)。

    Args:
        packed: [N, K//2] packed uint8 tensor
        N, K: 原始维度

    Returns:
        [N, K] 解包后的张量
    """
    decode_table = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], device=packed.device)

    low = packed & 0xF
    high = (packed >> 4) & 0xF

    # Interleave
    unpacked = torch.zeros(N, K, device=packed.device, dtype=torch.float32)
    unpacked[:, 0::2] = decode_table[low.long()]
    unpacked[:, 1::2] = decode_table[high.long()]

    return unpacked


def prepare_linear_weights(
    weight: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    verify: bool = True
) -> Dict[str, torch.Tensor]:
    """
    准备 Linear 层的权重。

    Args:
        weight: [out_features, in_features] 权重张量
        block_size: block scaling 块大小
        verify: 是否验证 pack/unpack 一致性

    Returns:
        包含以下键的字典:
        - weight_packed: [N, K//2] packed uint8
        - weight_scales: [N, K//block_size] float32 scales
        - out_features: int
        - in_features: int
    """
    N, K = weight.shape

    # 确保 K 是 block_size 的倍数
    if K % block_size != 0:
        pad_size = block_size - (K % block_size)
        weight = torch.nn.functional.pad(weight, (0, pad_size))
        K = weight.shape[1]
        print(f"  Padded K from {K - pad_size} to {K}")

    # 量化
    quantized, scales = quantize_to_nvfp4(weight, block_size)

    # 打包
    packed = pack_nvfp4(quantized)

    # 验证
    if verify:
        unpacked = unpack_nvfp4(packed, N, K)
        max_diff = (quantized - unpacked).abs().max().item()
        assert max_diff < 1e-6, f"Pack/unpack mismatch: {max_diff}"

    return {
        'weight_packed': packed.cpu(),
        'weight_scales': scales.cpu().float(),
        'out_features': N,
        'in_features': K,
    }


def process_model_weights(
    state_dict: Dict[str, torch.Tensor],
    layer_patterns: Optional[list] = None,
    block_size: int = BLOCK_SIZE,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    处理模型权重字典。

    Args:
        state_dict: PyTorch state dict
        layer_patterns: 要处理的层名称模式列表 (None = 所有 Linear 权重)
        block_size: block scaling 块大小
        device: 计算设备

    Returns:
        处理后的权重字典
    """
    processed = {}

    for name, param in state_dict.items():
        # 检查是否是权重矩阵
        if 'weight' not in name:
            continue

        if param.dim() != 2:
            continue

        # 检查是否匹配模式
        if layer_patterns:
            if not any(p in name for p in layer_patterns):
                continue

        print(f"Processing: {name} [{param.shape}]")

        # 移动到设备
        param = param.to(device)

        # 准备权重
        try:
            result = prepare_linear_weights(param, block_size)
            processed[name] = result
            print(f"  -> packed shape: {result['weight_packed'].shape}")
            print(f"  -> scales shape: {result['weight_scales'].shape}")
        except Exception as e:
            print(f"  -> ERROR: {e}")

    return processed


def save_packed_weights(
    processed: Dict[str, Dict],
    output_path: str,
    format: str = 'npz'
):
    """
    保存打包后的权重。

    Args:
        processed: 处理后的权重字典
        output_path: 输出路径
        format: 输出格式 ('npz', 'safetensors', 'bin')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'npz':
        # 展平为 numpy 数组
        arrays = {}
        for name, data in processed.items():
            safe_name = name.replace('.', '_')
            arrays[f'{safe_name}_packed'] = data['weight_packed'].numpy()
            arrays[f'{safe_name}_scales'] = data['weight_scales'].numpy()
            arrays[f'{safe_name}_shape'] = np.array([data['out_features'], data['in_features']])

        np.savez_compressed(output_path, **arrays)
        print(f"\nSaved to {output_path}")

    elif format == 'safetensors':
        try:
            from safetensors.torch import save_file
            tensors = {}
            for name, data in processed.items():
                safe_name = name.replace('.', '_')
                tensors[f'{safe_name}_packed'] = data['weight_packed']
                tensors[f'{safe_name}_scales'] = data['weight_scales']
            save_file(tensors, output_path)
            print(f"\nSaved to {output_path}")
        except ImportError:
            print("safetensors not installed, falling back to npz")
            save_packed_weights(processed, str(output_path).replace('.safetensors', '.npz'), 'npz')

    elif format == 'bin':
        # 简单的二进制格式
        with open(output_path, 'wb') as f:
            import struct
            # 写入层数
            f.write(struct.pack('I', len(processed)))
            for name, data in processed.items():
                # 写入层名称
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                # 写入形状
                f.write(struct.pack('II', data['out_features'], data['in_features']))
                # 写入 packed 权重
                packed_bytes = data['weight_packed'].numpy().tobytes()
                f.write(struct.pack('I', len(packed_bytes)))
                f.write(packed_bytes)
                # 写入 scales
                scales_bytes = data['weight_scales'].numpy().tobytes()
                f.write(struct.pack('I', len(scales_bytes)))
                f.write(scales_bytes)
        print(f"\nSaved to {output_path}")


def compute_quantization_stats(
    original: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = BLOCK_SIZE
) -> Dict[str, float]:
    """
    计算量化统计信息。
    """
    # 反量化
    N, K = original.shape
    num_blocks = K // block_size
    quantized_blocked = quantized.view(N, num_blocks, block_size)
    dequantized = (quantized_blocked * scales.unsqueeze(-1)).view(N, K)

    # 计算误差
    abs_error = (original - dequantized).abs()
    rel_error = abs_error / (original.abs() + 1e-8)

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        original.flatten().unsqueeze(0),
        dequantized.flatten().unsqueeze(0)
    ).item()

    return {
        'mean_abs_error': abs_error.mean().item(),
        'max_abs_error': abs_error.max().item(),
        'mean_rel_error': rel_error.mean().item() * 100,
        'cosine_similarity': cos_sim,
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare NVFP4 packed weights')
    parser.add_argument('--input', type=str, required=True, help='Input model path')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--format', type=str, default='npz', choices=['npz', 'safetensors', 'bin'])
    parser.add_argument('--block-size', type=int, default=32, help='Block size for scaling')
    parser.add_argument('--layers', type=str, nargs='+', help='Layer name patterns to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device for computation')

    args = parser.parse_args()

    print("=" * 60)
    print("NVFP4 Weight Packing")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Block size: {args.block_size}")
    print(f"Format: {args.format}")
    print()

    # 加载权重
    print("Loading weights...")
    input_path = Path(args.input)

    if input_path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state_dict = load_file(args.input)
    else:
        state_dict = torch.load(args.input, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

    print(f"Loaded {len(state_dict)} tensors")
    print()

    # 处理权重
    processed = process_model_weights(
        state_dict,
        layer_patterns=args.layers,
        block_size=args.block_size,
        device=args.device
    )

    # 保存
    save_packed_weights(processed, args.output, args.format)

    # 统计
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_original = 0
    total_packed = 0
    for name, data in processed.items():
        N, K = data['out_features'], data['in_features']
        original_bytes = N * K * 4  # float32
        packed_bytes = N * (K // 2) + N * (K // args.block_size) * 4  # uint8 + float32 scales
        total_original += original_bytes
        total_packed += packed_bytes

    print(f"Processed {len(processed)} layers")
    print(f"Original size: {total_original / 1e6:.2f} MB")
    print(f"Packed size: {total_packed / 1e6:.2f} MB")
    print(f"Compression ratio: {total_original / total_packed:.2f}x")


if __name__ == '__main__':
    main()
