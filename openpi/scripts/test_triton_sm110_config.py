#!/usr/bin/env python3
"""
测试正确配置的 Triton 在 Thor SM110 上的性能

关键发现:
1. Thor 是 SM 11.0，需要特殊的架构标志 11.0a
2. 需要设置 TORCH_CUDA_ARCH_LIST=11.0a
3. 需要设置 TRITON_PTXAS_PATH

运行方式 (需要先设置环境变量):
    export TORCH_CUDA_ARCH_LIST=11.0a
    export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    python test_triton_sm110_config.py
"""

import os
import sys
import torch
import time


def check_environment():
    """检查环境变量配置"""
    print("=" * 60)
    print("环境变量检查")
    print("=" * 60)

    required_vars = {
        "TORCH_CUDA_ARCH_LIST": "11.0a",
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    }

    all_set = True
    for var, recommended in required_vars.items():
        value = os.environ.get(var, "NOT SET")
        status = "✅" if value != "NOT SET" else "❌"
        print(f"{status} {var}: {value}")
        if value == "NOT SET":
            print(f"   建议设置: export {var}={recommended}")
            all_set = False

    # 检查 CUDA 路径
    cuda_path = os.environ.get("PATH", "")
    if "/usr/local/cuda/bin" not in cuda_path:
        print("⚠️ PATH 中没有 /usr/local/cuda/bin")

    # 检查 ptxas
    ptxas_path = os.environ.get("TRITON_PTXAS_PATH", "")
    if ptxas_path and os.path.exists(ptxas_path):
        print(f"✅ ptxas 存在: {ptxas_path}")
    elif ptxas_path:
        print(f"❌ ptxas 不存在: {ptxas_path}")

    # 检查 CUDA 版本
    print(f"\nPyTorch CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    return all_set


def check_gpu_arch():
    """检查 GPU 架构"""
    print("\n" + "=" * 60)
    print("GPU 架构检查")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Compute Capability: SM {props.major}.{props.minor}")

    if props.major == 11 and props.minor == 0:
        print("✅ 检测到 Thor (SM 11.0)")
        print("   需要使用架构标志: 11.0a 或 110a")
    else:
        print(f"⚠️ 非 Thor GPU: SM {props.major}.{props.minor}")


def test_triton_import():
    """测试 Triton 导入"""
    print("\n" + "=" * 60)
    print("Triton 导入测试")
    print("=" * 60)

    try:
        import triton
        import triton.language as tl
        print(f"✅ Triton version: {triton.__version__}")

        # 检查 Triton 配置
        if hasattr(triton, 'runtime'):
            print(f"   Triton runtime available")

        return True
    except ImportError as e:
        print(f"❌ Triton 导入失败: {e}")
        return False


def test_simple_triton_kernel():
    """测试简单的 Triton kernel"""
    print("\n" + "=" * 60)
    print("Triton Kernel 编译测试")
    print("=" * 60)

    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    try:
        print("编译 Triton kernel...")
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
        torch.cuda.synchronize()

        # 验证
        expected = x + y
        if torch.allclose(out, expected, atol=1e-5):
            print("✅ Triton kernel 编译和执行成功")
            return True
        else:
            print("❌ Triton kernel 结果不正确")
            return False

    except Exception as e:
        print(f"❌ Triton kernel 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_triton_matmul_performance():
    """测试 Triton MatMul 性能"""
    print("\n" + "=" * 60)
    print("Triton MatMul 性能测试")
    print("=" * 60)

    import triton
    import triton.language as tl

    # 使用 warp specialization 配置
    @triton.autotune(
        configs=[
            # 标准配置
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
            # 尝试不同的 num_stages
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
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

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
            b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    M, N, K = 712, 16384, 2048

    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)

    # torch.matmul baseline
    print("\n1. torch.matmul (cuBLAS) baseline")
    for _ in range(20):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / 100 * 1000

    print(f"   torch.matmul: {torch_time:.3f} ms")

    # Triton MatMul (with autotune)
    print("\n2. Triton MatMul (autotune)")
    print("   正在运行 autotune...")

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    try:
        # Warmup (triggers autotune)
        for _ in range(5):
            matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
            )
        torch.cuda.synchronize()

        # More warmup
        for _ in range(20):
            matmul_kernel[grid](
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
            matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
            )
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / 100 * 1000

        speedup = torch_time / triton_time
        print(f"   Triton MatMul: {triton_time:.3f} ms")
        print(f"   vs cuBLAS: {speedup:.2f}x")

        if speedup > 0.8:
            print("   ✅ Triton 性能可接受")
        elif speedup > 0.5:
            print("   ⚠️ Triton 性能有提升空间")
        else:
            print("   ❌ Triton 性能仍然很差，可能需要:")
            print("      - 确认 TORCH_CUDA_ARCH_LIST=11.0a")
            print("      - 重新安装 Triton")
            print("      - 使用 SGLang/vLLM 的优化版本")

    except Exception as e:
        print(f"   ❌ Triton MatMul 失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 60)
    print("Thor SM110 Triton 配置测试")
    print("=" * 60)

    # 1. 检查环境变量
    env_ok = check_environment()

    # 2. 检查 GPU 架构
    check_gpu_arch()

    # 3. 测试 Triton 导入
    if not test_triton_import():
        return

    # 4. 测试简单 kernel
    if not test_simple_triton_kernel():
        return

    # 5. 测试 MatMul 性能
    test_triton_matmul_performance()

    # Summary
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    if not env_ok:
        print("""
⚠️ 环境变量未完全配置。请在 Docker 容器启动时设置:

docker exec -e TORCH_CUDA_ARCH_LIST=11.0a \\
            -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\
            turbo_pi_eval python /workspace/scripts/test_triton_sm110_config.py

或在容器内设置:

export TORCH_CUDA_ARCH_LIST=11.0a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python /workspace/scripts/test_triton_sm110_config.py
""")


if __name__ == "__main__":
    main()
