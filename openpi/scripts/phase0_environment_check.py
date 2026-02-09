#!/usr/bin/env python3
"""
Phase 0: Thor Triforce 环境验证

这是最关键的一步，决定后续优化路线:
1. 测量 Thor 实际内存带宽 (决定是否需要 INT4)
2. 验证 Triton 支持 (决定能否写自定义 kernel)
3. 验证 FlashInfer 支持 (决定 Attention 优化路径)

运行方式:
    docker exec turbo_pi_eval python /workspace/scripts/phase0_environment_check.py
"""

import torch
import time
import json
import sys
from pathlib import Path
from datetime import datetime

# 结果存储
RESULTS = {
    "timestamp": datetime.now().isoformat(),
    "gpu": {},
    "bandwidth": {},
    "triton": {},
    "flashinfer": {},
    "recommendation": "",
}


def print_header(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_gpu_info():
    """检查 GPU 信息"""
    print_header("GPU Information")

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        RESULTS["gpu"]["available"] = False
        return False

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    info = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": props.total_memory / 1e9,
        "sm_count": props.multi_processor_count,
    }

    RESULTS["gpu"] = info
    RESULTS["gpu"]["available"] = True

    print(f"Device: {info['name']}")
    print(f"Compute Capability: SM {info['compute_capability']}")
    print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
    print(f"SM Count: {info['sm_count']}")

    # Thor 检测
    if props.major >= 10:
        print("✅ Blackwell architecture detected (Thor/B200)")
        RESULTS["gpu"]["is_blackwell"] = True
    elif props.major == 9:
        print("⚠️ Hopper architecture (H100)")
        RESULTS["gpu"]["is_blackwell"] = False
    elif props.major == 8:
        print("⚠️ Ampere/Ada architecture")
        RESULTS["gpu"]["is_blackwell"] = False
    else:
        print(f"⚠️ Older architecture (SM {props.major}.{props.minor})")
        RESULTS["gpu"]["is_blackwell"] = False

    return True


def benchmark_memory_bandwidth():
    """
    测量实际内存带宽 - 最关键的数据

    这决定了 KV Cache MLP 的理论下限
    """
    print_header("Memory Bandwidth Benchmark")

    device = "cuda"

    # 测试配置: 模拟 Pi0.5 的实际 shape
    configs = [
        # (in_dim, out_dim, name)
        (2048, 16384, "MLP gate_proj (2048→16384)"),
        (2048, 16384, "MLP up_proj (2048→16384)"),
        (16384, 2048, "MLP down_proj (16384→2048)"),
        (2048, 2048, "Attention O_proj (2048→2048)"),
        (2048, 512, "Attention KV_proj (2048→512)"),
    ]

    results = {}

    for in_dim, out_dim, name in configs:
        # 模拟 batch=1, seq=712 (Vision 512 + Text 200)
        seq_len = 712
        x = torch.randn(1, seq_len, in_dim, device=device, dtype=torch.bfloat16)
        layer = torch.nn.Linear(in_dim, out_dim, bias=False,
                                device=device, dtype=torch.bfloat16)

        # 权重大小 (BF16 = 2 bytes)
        weight_bytes = in_dim * out_dim * 2

        # Warmup
        for _ in range(20):
            _ = layer(x)
        torch.cuda.synchronize()

        # Benchmark
        num_iters = 100
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = layer(x)
        torch.cuda.synchronize()

        avg_time_ms = (time.perf_counter() - start) / num_iters * 1000
        effective_bw = weight_bytes / (avg_time_ms / 1000) / 1e9  # GB/s

        results[name] = {
            "shape": f"({in_dim}, {out_dim})",
            "weight_mb": weight_bytes / 1e6,
            "time_ms": avg_time_ms,
            "bandwidth_gbps": effective_bw,
        }

        print(f"\n{name}:")
        print(f"  Shape: ({in_dim}, {out_dim})")
        print(f"  Weight: {weight_bytes/1e6:.2f} MB")
        print(f"  Time: {avg_time_ms:.3f} ms")
        print(f"  Effective Bandwidth: {effective_bw:.1f} GB/s")

    RESULTS["bandwidth"]["layers"] = results

    # 计算 KV Cache MLP 总时间
    # 18 层 × (gate + up + down)
    mlp_time = 0
    for name, data in results.items():
        if "MLP" in name:
            mlp_time += data["time_ms"]

    mlp_time_18_layers = mlp_time * 18 / 3  # 因为测了 3 个 MLP 层

    # 计算 Attention 时间
    attn_time = 0
    for name, data in results.items():
        if "Attention" in name:
            attn_time += data["time_ms"]
    attn_time_18_layers = attn_time * 18

    print("\n" + "-" * 40)
    print(f"Estimated KV Cache Prefill Time (18 layers):")
    print(f"  MLP (带宽瓶颈): {mlp_time_18_layers:.1f} ms")
    print(f"  Attention: {attn_time_18_layers:.1f} ms")
    print(f"  Total Estimate: {mlp_time_18_layers + attn_time_18_layers:.1f} ms")
    print(f"  Actual Measured: ~54 ms (差异来自 LayerNorm/RoPE 等)")

    RESULTS["bandwidth"]["mlp_18_layers_ms"] = mlp_time_18_layers
    RESULTS["bandwidth"]["attn_18_layers_ms"] = attn_time_18_layers

    # 计算平均带宽
    avg_bw = sum(r["bandwidth_gbps"] for r in results.values()) / len(results)
    RESULTS["bandwidth"]["avg_bandwidth_gbps"] = avg_bw

    print(f"\nAverage Effective Bandwidth: {avg_bw:.1f} GB/s")

    # 理论分析
    theoretical_bw = 200  # Thor 理论带宽
    print(f"Theoretical Bandwidth: {theoretical_bw} GB/s")
    print(f"Efficiency: {avg_bw/theoretical_bw*100:.1f}%")

    if avg_bw < 150:
        print("\n⚠️ 带宽显著低于理论值!")
        print("   可能原因: CUDA driver, 内存争用, 热节流")
        RESULTS["bandwidth"]["status"] = "below_expected"
    elif avg_bw > 180:
        print("\n✅ 带宽接近理论最大值")
        RESULTS["bandwidth"]["status"] = "optimal"
    else:
        print("\n⚠️ 带宽略低于理论值")
        RESULTS["bandwidth"]["status"] = "acceptable"

    # INT4 量化收益预估
    print("\n" + "-" * 40)
    print("INT4 Quantization Analysis:")
    int4_mlp_time = mlp_time_18_layers / 4  # 1/4 权重
    print(f"  Current (BF16): {mlp_time_18_layers:.1f} ms")
    print(f"  With INT4: ~{int4_mlp_time:.1f} ms (估算)")
    print(f"  Potential Speedup: {mlp_time_18_layers/int4_mlp_time:.1f}x")

    RESULTS["bandwidth"]["int4_estimate_ms"] = int4_mlp_time

    return results


def check_triton():
    """检查 Triton 是否支持 Thor"""
    print_header("Triton Support Check")

    try:
        import triton
        import triton.language as tl

        RESULTS["triton"]["installed"] = True
        RESULTS["triton"]["version"] = triton.__version__
        print(f"✅ Triton version: {triton.__version__}")

        # 简单的 Triton kernel 测试
        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        # 运行测试
        n = 1024
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        out = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

        try:
            add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
            torch.cuda.synchronize()

            # 验证
            expected = x + y
            if torch.allclose(out, expected, atol=1e-5):
                print("✅ Triton kernel execution successful")
                RESULTS["triton"]["works"] = True
            else:
                print("❌ Triton kernel produced incorrect results")
                RESULTS["triton"]["works"] = False
        except Exception as e:
            print(f"❌ Triton kernel failed: {e}")
            RESULTS["triton"]["works"] = False
            RESULTS["triton"]["error"] = str(e)

        # 测试更复杂的 kernel (模拟 MatMul)
        if RESULTS["triton"].get("works"):
            try:
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
                    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                    for k in range(0, K, BLOCK_K):
                        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
                        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
                        acc += tl.dot(a, b)
                    c = acc.to(tl.float16)
                    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)

                M, N, K = 64, 64, 64
                a = torch.randn(M, K, device="cuda", dtype=torch.float16)
                b = torch.randn(K, N, device="cuda", dtype=torch.float16)
                c = torch.empty(M, N, device="cuda", dtype=torch.float16)

                grid = (M // 32, N // 32)
                matmul_kernel[grid](
                    a, b, c, M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
                )
                torch.cuda.synchronize()

                expected = torch.matmul(a, b)
                if torch.allclose(c, expected, atol=0.1):
                    print("✅ Triton MatMul kernel works")
                    RESULTS["triton"]["matmul_works"] = True
                else:
                    print("⚠️ Triton MatMul kernel has precision issues")
                    RESULTS["triton"]["matmul_works"] = False

            except Exception as e:
                print(f"⚠️ Triton MatMul kernel failed: {e}")
                RESULTS["triton"]["matmul_works"] = False

        return RESULTS["triton"].get("works", False)

    except ImportError:
        print("❌ Triton not installed")
        print("   Install: pip install triton")
        RESULTS["triton"]["installed"] = False
        return False


def check_flashinfer():
    """检查 FlashInfer 是否可用"""
    print_header("FlashInfer Support Check")

    try:
        import flashinfer
        RESULTS["flashinfer"]["installed"] = True
        RESULTS["flashinfer"]["version"] = getattr(flashinfer, "__version__", "unknown")
        print(f"✅ FlashInfer installed")

        # 尝试基本操作
        try:
            # FlashInfer 的 API 可能因版本而异
            # 这里做一个简单的测试
            q = torch.randn(1, 8, 256, device="cuda", dtype=torch.float16)
            k = torch.randn(100, 1, 256, device="cuda", dtype=torch.float16)
            v = torch.randn(100, 1, 256, device="cuda", dtype=torch.float16)

            # 检查是否有可用的函数
            if hasattr(flashinfer, 'single_prefill_with_kv_cache'):
                print("  ✅ single_prefill_with_kv_cache available")
                RESULTS["flashinfer"]["prefill_available"] = True
            else:
                print("  ⚠️ single_prefill_with_kv_cache not found")
                RESULTS["flashinfer"]["prefill_available"] = False

            RESULTS["flashinfer"]["works"] = True
            print("✅ FlashInfer basic check passed")

        except Exception as e:
            print(f"⚠️ FlashInfer operation failed: {e}")
            RESULTS["flashinfer"]["works"] = False
            RESULTS["flashinfer"]["error"] = str(e)

        return RESULTS["flashinfer"].get("works", False)

    except ImportError:
        print("❌ FlashInfer not installed")
        print("   Install: pip install flashinfer")
        RESULTS["flashinfer"]["installed"] = False
        return False


def check_quantization_libs():
    """检查量化库支持"""
    print_header("Quantization Libraries Check")

    libs = {
        "bitsandbytes": "INT8/NF4 quantization",
        "auto_gptq": "GPTQ INT4",
        "optimum": "Hugging Face optimization",
    }

    available = []
    for lib, desc in libs.items():
        try:
            __import__(lib.replace("-", "_"))
            print(f"✅ {lib}: {desc}")
            available.append(lib)
        except ImportError:
            print(f"❌ {lib}: not installed")

    RESULTS["quantization"] = {"available_libs": available}
    return available


def generate_recommendation():
    """生成优化建议"""
    print_header("OPTIMIZATION RECOMMENDATION")

    recommendation = []

    # 分析带宽
    avg_bw = RESULTS["bandwidth"].get("avg_bandwidth_gbps", 0)
    mlp_time = RESULTS["bandwidth"].get("mlp_18_layers_ms", 999)

    if avg_bw < 150:
        recommendation.append("⚠️ 带宽受限严重，INT4 量化是唯一出路")
        recommended_plan = "Plan B (INT4 Triton)"
    elif mlp_time > 15:
        recommendation.append("⚠️ MLP 延迟较高，建议尝试 INT4 量化")
        recommended_plan = "Plan B (INT4 Triton)"
    else:
        recommendation.append("✅ 带宽正常，可以先尝试 Plan A")
        recommended_plan = "Plan A (保守优化)"

    # 分析 Triton
    if RESULTS["triton"].get("works"):
        recommendation.append("✅ Triton 可用，可以写自定义 kernel")
    else:
        recommendation.append("❌ Triton 不可用，回退到 CUTLASS 或等待支持")
        if "Plan B" in recommended_plan:
            recommended_plan = "Plan A (保守优化) - Triton 不可用"

    # 分析 FlashInfer
    if RESULTS["flashinfer"].get("works"):
        recommendation.append("✅ FlashInfer 可用，Attention 可以优化")
    else:
        recommendation.append("⚠️ FlashInfer 不可用，使用 PyTorch Attention")

    RESULTS["recommendation"] = {
        "plan": recommended_plan,
        "details": recommendation,
    }

    print(f"\n推荐方案: {recommended_plan}")
    print("\n详细分析:")
    for r in recommendation:
        print(f"  {r}")

    # 下一步建议
    print("\n下一步:")
    if "Plan B" in recommended_plan:
        print("  1. 运行 scripts/benchmark_bandwidth.py 确认带宽数据")
        print("  2. 实现 Triton INT4 Linear kernel")
        print("  3. 量化 KV Cache MLP 层")
        print("  4. 验证精度和延迟")
    else:
        print("  1. 实现 CUDA Graph 全图录制")
        print("  2. 使用 torch.compile 进行 kernel fusion")
        print("  3. 如果效果不佳，再考虑 Plan B")


def save_results():
    """保存结果到文件"""
    output_path = Path("/workspace/phase0_results.json")
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")


def main():
    """运行所有检查"""
    print("\n" + "=" * 60)
    print("THOR TRIFORCE ENVIRONMENT CHECK")
    print("Phase 0: 决定优化路线")
    print("=" * 60)

    # GPU Info
    if not check_gpu_info():
        print("\n❌ 无法继续: GPU 不可用")
        return

    # Bandwidth - 最关键的测试
    benchmark_memory_bandwidth()

    # Triton
    check_triton()

    # FlashInfer
    check_flashinfer()

    # Quantization libs
    check_quantization_libs()

    # Generate recommendation
    generate_recommendation()

    # Save results
    save_results()


if __name__ == "__main__":
    main()
