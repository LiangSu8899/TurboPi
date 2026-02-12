#!/usr/bin/env python3
"""
W4A16 TVM Integration Test

Tests the full integration of W4A16 TVM kernels with the Pi0 inference pipeline.

Usage:
    python scripts/test_w4a16_integration.py --checkpoint /path/to/checkpoint

    # Quick kernel test (no checkpoint needed)
    python scripts/test_w4a16_integration.py --kernel-only

Author: Claude Code
Date: 2026-02-10
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_kernel_standalone():
    """Test W4A16 TVM kernel in isolation."""
    print("=" * 70)
    print("W4A16 TVM Kernel Standalone Test")
    print("=" * 70)

    try:
        import tvm
        print(f"TVM version: {tvm.__version__}")
    except ImportError:
        print("ERROR: TVM not available")
        return False

    try:
        from openpi.models_pytorch.tvm_kernels.w4a16_gemv import (
            create_w4a16_gemv_fast,
            build_kernel,
            quantize_to_nvfp4_packed,
            NVFP4_LUT,
            BLOCK_SIZE,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import TVM kernel: {e}")
        return False

    # Test dimensions
    test_cases = [
        ("gate/up_proj", 16384, 2048),
        ("down_proj", 2048, 16384),
    ]

    all_passed = True

    for name, N, K in test_cases:
        print(f"\n--- Testing {name}: N={N}, K={K} ---")

        # Generate test data
        np.random.seed(42)
        A_np = np.random.randn(1, K).astype(np.float32)
        W_np = np.random.randn(N, K).astype(np.float32) * 0.1

        # Quantize
        W_packed, scales = quantize_to_nvfp4_packed(W_np)
        num_blocks_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Reference computation
        def dequant_and_compute(A, W_packed, scales, K):
            N_out = W_packed.shape[0]
            W_dequant = np.zeros((N_out, K), dtype=np.float32)
            for n in range(N_out):
                for k in range(K):
                    byte_idx = k // 2
                    is_high = k % 2
                    packed = W_packed[n, byte_idx]
                    fp4_idx = ((packed >> 4) & 0xF) if is_high else (packed & 0xF)
                    w_val = NVFP4_LUT[fp4_idx]
                    block_idx = k // BLOCK_SIZE
                    W_dequant[n, k] = w_val * scales[n, block_idx]
            return A @ W_dequant.T

        out_ref = dequant_and_compute(A_np, W_packed, scales, K)

        # Build and run TVM kernel
        try:
            kernel_func = create_w4a16_gemv_fast(N, K)
            mod = build_kernel(kernel_func)
            func = mod["w4a16_gemv_fast"]
        except Exception as e:
            print(f"  Build FAILED: {e}")
            all_passed = False
            continue

        device = tvm.runtime.cuda(0)

        A_tvm = tvm.runtime.empty((1, K), dtype="float32", device=device)
        A_tvm.copyfrom(A_np)

        W_packed_tvm = tvm.runtime.empty((N, K // 2), dtype="uint8", device=device)
        W_packed_tvm.copyfrom(W_packed)

        scales_tvm = tvm.runtime.empty((N, num_blocks_k), dtype="float32", device=device)
        scales_tvm.copyfrom(scales)

        out_tvm = tvm.runtime.empty((1, N), dtype="float32", device=device)

        # Warmup
        for _ in range(50):
            func(A_tvm, W_packed_tvm, scales_tvm, out_tvm)
        device.sync()

        # Benchmark
        runs = 200
        device.sync()
        start = time.time()
        for _ in range(runs):
            func(A_tvm, W_packed_tvm, scales_tvm, out_tvm)
        device.sync()
        elapsed_ms = (time.time() - start) / runs * 1000

        # Verify
        out_np = out_tvm.numpy()
        cos_sim = np.dot(out_np.flatten(), out_ref.flatten()) / (
            np.linalg.norm(out_np) * np.linalg.norm(out_ref) + 1e-8)

        passed = cos_sim > 0.99
        status = "PASS" if passed else "FAIL"

        print(f"  Time:    {elapsed_ms:.4f} ms")
        print(f"  Cos sim: {cos_sim:.6f}")
        print(f"  Status:  {status}")

        if not passed:
            all_passed = False

    print(f"\n{'='*70}")
    print(f"Kernel test result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'='*70}")

    return all_passed


def test_mlp_module():
    """Test W4A16 MLP module."""
    print("\n" + "=" * 70)
    print("W4A16 MLP Module Test")
    print("=" * 70)

    try:
        from openpi.models_pytorch.w4a16_mlp import (
            W4A16MLP,
            W4A16Linear,
            _tvm_available,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import W4A16 MLP: {e}")
        return False

    print(f"TVM available: {_tvm_available}")

    device = torch.device("cuda")
    hidden_size = 2048
    intermediate_size = 16384

    # Create reference BF16 MLP
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

    # Create W4A16 MLP
    w4a16_mlp = W4A16MLP.from_gemma_mlp(bf16_mlp, use_tvm=True).to(device)

    # Test batch=1 (should use TVM)
    print("\n--- Batch=1 (TVM kernel) ---")
    x = torch.randn(1, hidden_size, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(20):
        _ = bf16_mlp(x)
        _ = w4a16_mlp(x)
    torch.cuda.synchronize()

    # Benchmark
    iterations = 100

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = bf16_mlp(x)
    torch.cuda.synchronize()
    bf16_ms = (time.perf_counter() - start) / iterations * 1000

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = w4a16_mlp(x)
    torch.cuda.synchronize()
    w4a16_ms = (time.perf_counter() - start) / iterations * 1000

    # Precision
    with torch.no_grad():
        bf16_out = bf16_mlp(x)
        w4a16_out = w4a16_mlp(x)

    cos_sim = F.cosine_similarity(
        bf16_out.flatten().float().unsqueeze(0),
        w4a16_out.flatten().float().unsqueeze(0)
    ).item()

    print(f"  BF16:    {bf16_ms:.3f} ms")
    print(f"  W4A16:   {w4a16_ms:.3f} ms")
    print(f"  Speedup: {bf16_ms/w4a16_ms:.2f}x")
    print(f"  Cos sim: {cos_sim:.6f}")

    # Test batch=256 (should use fallback)
    print("\n--- Batch=256 (PyTorch fallback) ---")
    x_batch = torch.randn(256, hidden_size, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = bf16_mlp(x_batch)
        _ = w4a16_mlp(x_batch)
    torch.cuda.synchronize()

    # Benchmark
    iterations = 50

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = bf16_mlp(x_batch)
    torch.cuda.synchronize()
    bf16_ms = (time.perf_counter() - start) / iterations * 1000

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = w4a16_mlp(x_batch)
    torch.cuda.synchronize()
    w4a16_ms = (time.perf_counter() - start) / iterations * 1000

    print(f"  BF16:    {bf16_ms:.3f} ms")
    print(f"  W4A16:   {w4a16_ms:.3f} ms")
    print(f"  Speedup: {bf16_ms/w4a16_ms:.2f}x")

    print(f"\n{'='*70}")
    print("MLP module test complete")
    print(f"{'='*70}")

    return True


def test_full_pipeline(checkpoint_dir: str):
    """Test full inference pipeline with W4A16 backend."""
    print("\n" + "=" * 70)
    print("W4A16 Full Pipeline Test")
    print("=" * 70)

    try:
        from openpi.inference import UnifiedPolicy
    except ImportError as e:
        print(f"ERROR: Failed to import UnifiedPolicy: {e}")
        return False

    # Check if W4A16 backend is available
    if "w4a16_tvm" not in UnifiedPolicy.BACKENDS:
        print("ERROR: W4A16 backend not registered")
        return False

    print(f"Checkpoint: {checkpoint_dir}")

    # Create policy with W4A16 backend
    try:
        policy = UnifiedPolicy(
            checkpoint_dir=checkpoint_dir,
            backend="w4a16_tvm",
            num_denoising_steps=3,
        )
    except Exception as e:
        print(f"ERROR: Failed to create policy: {e}")
        return False

    print("Policy created successfully")

    # Create dummy observation
    dummy_obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.randn(8).astype(np.float32),
        "prompt": "pick up the black bowl",
    }

    # Warmup
    print("\nWarming up...")
    policy.warmup(num_iterations=3)

    # Benchmark
    print("\nRunning inference benchmark...")
    num_runs = 10
    times = []

    for i in range(num_runs):
        start = time.perf_counter()
        result = policy.infer(dummy_obs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.1f} ms")

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    hz = 1000 / avg_ms

    print(f"\n{'='*70}")
    print("Pipeline Benchmark Results")
    print(f"{'='*70}")
    print(f"  Average: {avg_ms:.1f} +/- {std_ms:.1f} ms")
    print(f"  Hz:      {hz:.1f}")
    print(f"  Actions shape: {result['actions'].shape}")
    print(f"{'='*70}")

    return True


def main():
    parser = argparse.ArgumentParser(description="W4A16 TVM Integration Test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--kernel-only", action="store_true",
                        help="Run kernel tests only (no checkpoint needed)")
    parser.add_argument("--skip-kernel", action="store_true",
                        help="Skip kernel tests")
    parser.add_argument("--skip-mlp", action="store_true",
                        help="Skip MLP module tests")
    args = parser.parse_args()

    results = {}

    # Kernel test
    if not args.skip_kernel:
        results["kernel"] = test_kernel_standalone()

    # MLP module test
    if not args.skip_mlp:
        results["mlp"] = test_mlp_module()

    # Full pipeline test
    if args.checkpoint and not args.kernel_only:
        results["pipeline"] = test_full_pipeline(args.checkpoint)
    elif not args.kernel_only:
        print("\nSkipping pipeline test (no checkpoint specified)")
        print("Use --checkpoint /path/to/checkpoint to run full pipeline test")

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print("=" * 70)

    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
