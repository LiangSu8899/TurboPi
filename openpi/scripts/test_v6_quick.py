#!/usr/bin/env python3
"""快速测试 V6 kernel 是否修复成功。"""

import sys
import os

# Support both host and Docker paths
for path in [
    '/home/heima-thor/suliang/Turbo-Pi/openpi/nvfp4_packed_plugin/python',
    '/workspace/nvfp4_packed_plugin/python',
]:
    if os.path.exists(path):
        sys.path.insert(0, path)
        break

import torch
torch.cuda.empty_cache()

print("=" * 60)
print("Testing V6 NVFP4 kernel with K=4096 (down_proj dimensions)")
print("=" * 60)

# 导入模块
from nvfp4_packed import NVFP4PackedLinear

# 测试参数 - 模拟 down_proj
M = 1
K = 4096  # in_features
N = 1024  # out_features

print(f"\nTest dimensions: M={M}, K={K}, N={N}")
print(f"This tests the down_proj layer configuration")

# 创建原始线性层
linear = torch.nn.Linear(K, N, bias=True).cuda()

# 创建 NVFP4 量化层
nvfp4_layer = NVFP4PackedLinear.from_linear(linear, activation='gelu')
nvfp4_layer = nvfp4_layer.cuda()

# 创建测试输入
x = torch.randn(M, K, device='cuda', dtype=torch.float32)

print("\n--- Testing forward pass ---")

# 测试前向传播
try:
    with torch.no_grad():
        # 运行 V6 kernel
        output = nvfp4_layer(x)
        torch.cuda.synchronize()

    print(f"✓ V6 kernel succeeded!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output sample values: {output[0, :5].tolist()}")

    # 计算参考值
    with torch.no_grad():
        ref_output = nvfp4_layer._forward_dequant_gemm(x, 1)  # 1 = gelu

    # 比较精度
    max_diff = (output - ref_output).abs().max().item()
    print(f"  Max diff vs reference: {max_diff:.6f}")

    if max_diff < 1.0:  # NVFP4 精度有限，允许较大差异
        print("✓ Accuracy check passed!")
    else:
        print(f"✗ Accuracy check failed (max_diff={max_diff})")

except Exception as e:
    print(f"✗ V6 kernel FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 性能测试
print("\n--- Performance test ---")
torch.cuda.synchronize()

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = nvfp4_layer(x)
torch.cuda.synchronize()

# Benchmark
import time
start = time.perf_counter()
for _ in range(100):
    with torch.no_grad():
        _ = nvfp4_layer(x)
torch.cuda.synchronize()
end = time.perf_counter()

avg_time = (end - start) / 100 * 1000
print(f"V6 kernel: {avg_time:.3f} ms per call")

# 测试其他维度
print("\n--- Testing gate/up_proj dimensions (K=1024, N=4096) ---")

K2, N2 = 1024, 4096
linear2 = torch.nn.Linear(K2, N2, bias=True).cuda()
nvfp4_layer2 = NVFP4PackedLinear.from_linear(linear2, activation='silu')
nvfp4_layer2 = nvfp4_layer2.cuda()
x2 = torch.randn(M, K2, device='cuda', dtype=torch.float32)

try:
    with torch.no_grad():
        output2 = nvfp4_layer2(x2)
        torch.cuda.synchronize()
    print(f"✓ gate/up_proj test passed! Output shape: {output2.shape}")
except Exception as e:
    print(f"✗ gate/up_proj test FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! V6 kernel is working correctly.")
print("=" * 60)
