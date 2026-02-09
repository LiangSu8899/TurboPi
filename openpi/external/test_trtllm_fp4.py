import torch

print("Testing TRT-LLM FP4 ops directly...")

# Load TRT-LLM C++ libraries
import glob
so_files = glob.glob("/usr/local/lib/python3.12/dist-packages/tensorrt_llm/*.so")
for so in so_files:
    try:
        torch.ops.load_library(so)
    except:
        pass

# Check trtllm ops
print("\nAvailable trtllm ops:")
try:
    ops = [op for op in dir(torch.ops.trtllm) if not op.startswith("_")]
    for op in sorted(ops)[:30]:
        print(f"  {op}")
    print(f"  ... total {len(ops)} ops")
except Exception as e:
    print(f"Error: {e}")

# Test fp4_gemm
print("\n=== Testing trtllm::fp4_gemm ===")
try:
    fp4_gemm = torch.ops.trtllm.fp4_gemm
    print(f"fp4_gemm op found: {fp4_gemm}")
except Exception as e:
    print(f"Error: {e}")

# Test fp4_quantize
print("\n=== Testing trtllm::fp4_quantize ===")
try:
    fp4_quantize = torch.ops.trtllm.fp4_quantize
    print(f"fp4_quantize op found: {fp4_quantize}")
except Exception as e:
    print(f"Error: {e}")

# Test calculate_nvfp4_global_scale
print("\n=== Testing trtllm::calculate_nvfp4_global_scale ===")
try:
    calc_scale = torch.ops.trtllm.calculate_nvfp4_global_scale
    print(f"calculate_nvfp4_global_scale op found: {calc_scale}")
except Exception as e:
    print(f"Error: {e}")
