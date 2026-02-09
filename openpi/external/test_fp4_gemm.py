import torch
import time

print("=" * 60)
print("Thor SM110 NVFP4 GEMM Test")
print("=" * 60)

# Load TRT-LLM ops
import glob
so_files = glob.glob("/usr/local/lib/python3.12/dist-packages/tensorrt_llm/*.so")
for so in so_files:
    try:
        torch.ops.load_library(so)
    except:
        pass

# Test dimensions (Pi0.5 MLP)
M = 712   # batch
K = 2048  # hidden
N = 16384 # intermediate

print(f"\nGEMM dimensions: [{M}, {K}] x [{K}, {N}]")

# Create test tensors
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

print("\n=== BF16 Baseline ===")
# BF16 baseline
for _ in range(5):
    _ = torch.matmul(x, weight.T)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(100):
    out_bf16 = torch.matmul(x, weight.T)
torch.cuda.synchronize()
bf16_ms = (time.perf_counter() - start) / 100 * 1000
print(f"BF16 GEMM: {bf16_ms:.3f} ms")

print("\n=== NVFP4 GEMM Test ===")
try:
    # First quantize to FP4
    fp4_quantize = torch.ops.trtllm.fp4_quantize
    calc_scale = torch.ops.trtllm.calculate_nvfp4_global_scale
    fp4_gemm = torch.ops.trtllm.fp4_gemm
    
    # Calculate global scale
    global_sf = calc_scale(x, weight)
    print(f"Global scale: {global_sf}")
    
    # Quantize input
    x_fp4, x_sf = fp4_quantize(x)
    print(f"Input FP4 shape: {x_fp4.shape}, scale shape: {x_sf.shape}")
    
    # Quantize weight
    w_fp4, w_sf = fp4_quantize(weight)
    print(f"Weight FP4 shape: {w_fp4.shape}, scale shape: {w_sf.shape}")
    
    # Run FP4 GEMM
    # fp4_gemm(input_fp4, input_sf, weight_fp4, weight_sf, global_sf, output_dtype, sv_vec_size)
    out_fp4 = fp4_gemm(x_fp4, x_sf, w_fp4, w_sf, global_sf, 1, 16)  # 1 = bfloat16
    print(f"FP4 output shape: {out_fp4.shape}")
    
    # Benchmark
    for _ in range(5):
        _ = fp4_gemm(x_fp4, x_sf, w_fp4, w_sf, global_sf, 1, 16)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(100):
        _ = fp4_gemm(x_fp4, x_sf, w_fp4, w_sf, global_sf, 1, 16)
    torch.cuda.synchronize()
    fp4_ms = (time.perf_counter() - start) / 100 * 1000
    print(f"FP4 GEMM: {fp4_ms:.3f} ms")
    
    # Speedup
    speedup = bf16_ms / fp4_ms
    print(f"\nSpeedup: {speedup:.2f}x")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
