#!/bin/bash
# TRT FP8 Full Pipeline Performance Profiling Script
# Run inside Docker container
#
# Usage:
#   # From host - Run optimized baseline (TRT FP16 Vision + TRT FP8 KV + CUDA Graph):
#   docker exec turbo_pi bash /workspace/scripts/run_performance_profile.sh --baseline
#
#   # From host - Run PyTorch baseline (for comparison):
#   docker exec turbo_pi bash /workspace/scripts/run_performance_profile.sh --pytorch
#
#   # From host - Run and compare with baseline:
#   docker exec turbo_pi bash /workspace/scripts/run_performance_profile.sh --compare
#
#   # Inside container:
#   cd /workspace && bash scripts/run_performance_profile.sh [--baseline|--pytorch|--compare] [--tag TAG]

set -e

# Parse arguments
SAVE_BASELINE=false
COMPARE=false
USE_PYTORCH=false
TAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)
            SAVE_BASELINE=true
            shift
            ;;
        --compare)
            COMPARE=true
            shift
            ;;
        --pytorch)
            USE_PYTORCH=true
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--baseline] [--pytorch] [--compare] [--tag TAG]"
            echo ""
            echo "Options:"
            echo "  --baseline   Save results as the new baseline (uses optimized TRT profile)"
            echo "  --pytorch    Use PyTorch BF16 profiler (for comparison)"
            echo "  --compare    Compare with existing baseline"
            echo "  --tag TAG    Tag for this profiling run"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "TRT FP8 Full Pipeline Performance Profiler"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -d "/workspace/src/openpi" ] && [ ! -d "./src/openpi" ]; then
    echo "Error: Please run this script from /workspace or the openpi directory"
    exit 1
fi

# Set working directory
if [ -d "/workspace" ]; then
    cd /workspace
fi

# Check if checkpoint exists
CHECKPOINT_DIR="/root/.cache/openpi/checkpoints/pi05_libero"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_DIR"
    echo "Please download the checkpoint first:"
    echo "  huggingface-cli download liangsu9988/Turbo-Pi0.5-1.1.2 --local-dir $CHECKPOINT_DIR"
    exit 1
fi

echo "Checkpoint found: $CHECKPOINT_DIR"
echo ""

# Choose profiler script
if [ "$USE_PYTORCH" = true ]; then
    PROFILER_SCRIPT="scripts/profile_simple_baseline.py"
    echo "Mode: PyTorch BF16 Baseline"
    if [ -z "$TAG" ]; then
        TAG="pytorch_bf16"
    fi
else
    # Use FullOptimizedPolicy from libero_eval_full_optimized.py
    PROFILER_SCRIPT="scripts/profile_full_optimized.py"
    echo "Mode: FULL OPTIMIZED Pipeline (same as libero_eval_full_optimized.py)"
    echo "  - Vision TRT FP16 (torch_tensorrt.compile)"
    echo "  - KV Cache TRT FP8 MLP (ModelOpt + torch_tensorrt)"
    echo "  - Denoise CUDA Graph (torch.cuda.CUDAGraph)"
    if [ -z "$TAG" ]; then
        TAG="full_optimized_v1.2.0"
    fi
fi

# Build command
CMD="python $PROFILER_SCRIPT \
    --checkpoint $CHECKPOINT_DIR \
    --steps 1 3 10 \
    --iterations 30 \
    --output-dir ./docs"

if [ "$SAVE_BASELINE" = true ]; then
    echo ""
    echo ">> SAVE AS BASELINE"
    CMD="$CMD --save-as-baseline"
fi

if [ -n "$TAG" ]; then
    echo "Tag: $TAG"
    CMD="$CMD --tag $TAG"
fi

echo ""
echo "Running performance profiling for 1, 3, 10 denoising steps..."
echo "This may take 5-15 minutes to complete."
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Profiling Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - ./docs/*_profile_*.json (raw data)"
echo "  - ./docs/trt-fp8-detailed-performance-breakdown.md (report)"
if [ "$SAVE_BASELINE" = true ]; then
    echo "  - ./docs/baseline_profile.json (baseline for comparison)"
fi
echo ""
echo "Next steps:"
echo "  1. Review the markdown report"
echo "  2. After optimization, run with --compare to see improvements"
echo ""
