#!/bin/bash
# Compile and verify persistent MLP kernel
# Usage: ./compile_and_verify_persistent.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_DIR="/home/heima-thor/suliang/Turbo-Pi/openpi/src/openpi/models_pytorch"

echo "============================================="
echo "Compiling N-Layer Persistent MLP Kernel"
echo "============================================="

cd "$CUDA_DIR"

# Compile for Thor SM110
echo "Compiling for SM110 (Thor)..."
nvcc -O3 -arch=sm_110 \
     -Xptxas="-v" \
     -Xcompiler "-fPIC -Wall" \
     --shared \
     nvfp4_nlayer_persistent.cu \
     -o libnvfp4_persistent.so \
     2>&1 | tee compile.log

# Extract register usage from compile log
echo ""
echo "============================================="
echo "Register Usage Analysis"
echo "============================================="
grep -E "(registers|spill|stack)" compile.log || echo "No register info found"

# Check for local memory spill (BAD!)
if grep -q "spill" compile.log; then
    echo ""
    echo "⚠️  WARNING: Register spill detected!"
    echo "    This will hurt performance significantly."
    echo "    Consider reducing layer count."
fi

echo ""
echo "============================================="
echo "Compilation Complete"
echo "============================================="
ls -lh libnvfp4_persistent.so

echo ""
echo "Next steps:"
echo "1. Create PyTorch C++ extension to load this library"
echo "2. Run benchmark with Nsight Compute:"
echo "   ncu --set full ./benchmark_kernel"
echo ""
echo "Key metrics to check in Nsight:"
echo "  - sm__sass_thread_inst_executed_op_local_load (should be 0)"
echo "  - sm__warps_active.avg.pct_of_peak (target > 40%)"
echo "  - dram__bytes.sum (should be significantly lower than BF16)"
