#!/bin/bash
# Build CUTLASS with SM110a support for Thor
# Run this inside the Docker container: docker exec -it turbo_pi_eval bash

set -e

echo "========================================"
echo "Building CUTLASS with SM110a for Thor"
echo "========================================"

# Check prerequisites
echo "Checking prerequisites..."
nvcc --version | head -3
cmake --version | head -1

# Clone CUTLASS
CUTLASS_DIR="/workspace/external/cutlass"
if [ ! -d "$CUTLASS_DIR" ]; then
    echo "Cloning CUTLASS..."
    mkdir -p /workspace/external
    cd /workspace/external
    git clone https://github.com/NVIDIA/cutlass.git
else
    echo "CUTLASS already exists at $CUTLASS_DIR"
    cd "$CUTLASS_DIR"
    git pull
fi

cd "$CUTLASS_DIR"

# Create build directory
BUILD_DIR="$CUTLASS_DIR/build_sm110a"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with SM110a
echo ""
echo "Configuring CUTLASS for SM110a..."
cmake .. \
    -DCUTLASS_NVCC_ARCHS="110a" \
    -DCUTLASS_LIBRARY_KERNELS=all \
    -DCUTLASS_UNITY_BUILD_ENABLED=ON \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_ENABLE_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local/cutlass_sm110a

# Build profiler
echo ""
echo "Building CUTLASS profiler (this may take a while)..."
make cutlass_profiler -j$(nproc)

# Build Python bindings
echo ""
echo "Building Python bindings..."
cd "$CUTLASS_DIR/python"
pip install .

echo ""
echo "========================================"
echo "CUTLASS SM110a build complete!"
echo "========================================"
echo ""
echo "Profiler location: $BUILD_DIR/tools/profiler/cutlass_profiler"
echo ""
echo "Test NVFP4 GEMM:"
echo "  $BUILD_DIR/tools/profiler/cutlass_profiler \\"
echo "    --operation=gemm --m=712 --n=16384 --k=2048 \\"
echo "    --A=e2m1 --B=e2m1 --C=bf16 --D=bf16 \\"
echo "    --warmup-iterations=10 --profiling-iterations=100"
