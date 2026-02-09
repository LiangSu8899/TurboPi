import torch
import time

print("=" * 60)
print("Thor SM110 NVFP4 GEMM Test v2")
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
    fp4_quantize = torch.ops.trtllm.fp4_quantize
    calc_scale = torch.ops.trtllm.calculate_nvfp4_global_scale
    fp4_gemm = torch.ops.trtllm.fp4_gemm
    
    # Calculate global scale (tokens_per_batch should be None or 1D tensor)
    global_sf = calc_scale(x, None)
    print(f"Global scale shape: {global_sf.shape}")
    
    # Quantize input: fp4_quantize(input, global_scale, sf_vec_size, sf_use_ue8m0, swizzled_layout)
    sf_vec_size = 16
    x_fp4, x_sf = fp4_quantize(x, global_sf, sf_vec_size, False, True)
    print(f"Input FP4: {x_fp4.shape} {x_fp4.dtype}, scale: {x_sf.shape} {x_sf.dtype}")
    
    # Quantize weight
    weight_global_sf = calc_scale(weight, None)
    w_fp4, w_sf = fp4_quantize(weight, weight_global_sf, sf_vec_size, False, True)
    print(f"Weight FP4: {w_fp4.shape} {w_fp4.dtype}, scale: {w_sf.shape} {w_sf.dtype}")
    
    # Run FP4 GEMM: fp4_gemm(mat1, mat2, mat1_scale, mat2_scale, global_scale, sf_use_ue8m0, out_dtype)
    combined_scale = global_sf * weight_global_sf
    out_fp4 = fp4_gemm(x_fp4, w_fp4, x_sf, w_sf, combined_scale, False, torch.bfloat16)
    print(f"FP4 output: {out_fp4.shape} {out_fp4.dtype}")
    
    # Benchmark
    for _ in range(5):
        _ = fp4_gemm(x_fp4, w_fp4, x_sf, w_sf, combined_scale, False, torch.bfloat16)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(100):
        _ = fp4_gemm(x_fp4, w_fp4, x_sf, w_sf, combined_scale, False, torch.bfloat16)
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
