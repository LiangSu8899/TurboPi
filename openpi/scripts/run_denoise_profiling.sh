#!/bin/bash
# ============================================================================
# Denoise Module Deep Profiling Script
# ============================================================================
#
# Purpose: 精密诊断 Denoise 模块 100ms 延迟的根因
#
# Three Core Diagnostics:
# 1. Kernel Launch Gaps - CPU Launch Overhead
# 2. Memory Bandwidth - HBM Bound Detection
# 3. Stream Synchronization - Implicit Sync Detection
#
# Usage:
#   # Full profiling (in Docker container)
#   ./scripts/run_denoise_profiling.sh
#
#   # Quick profiling (fewer iterations)
#   ./scripts/run_denoise_profiling.sh --quick
#
#   # Analysis only (if you already have .nsys-rep)
#   ./scripts/run_denoise_profiling.sh --analyze-only
#
# Output:
#   - denoise_profile.nsys-rep (Nsight Systems report)
#   - denoise_profile.sqlite (SQLite export)
#   - denoise_profile.analysis.txt (Gap analysis report)
#
# Author: Turbo-Pi Team
# Date: 2026-02-12
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${WORKSPACE_DIR}/profile_output"
PROFILE_NAME="denoise_profile"

# Default parameters
NUM_STEPS=10
WARMUP=3
ITERATIONS=5
QUICK_MODE=false
ANALYZE_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            WARMUP=2
            ITERATIONS=3
            shift
            ;;
        --analyze-only)
            ANALYZE_ONLY=true
            shift
            ;;
        --steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}     DENOISE MODULE DEEP PROFILING${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Denoising Steps: ${NUM_STEPS}"
echo "  Warmup: ${WARMUP}"
echo "  Iterations: ${ITERATIONS}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"
cd "${WORKSPACE_DIR}"

# ============================================================================
# Step 1: Run Profiling with NVTX Markers
# ============================================================================

if [ "$ANALYZE_ONLY" = false ]; then
    echo -e "${YELLOW}Step 1: Running NVTX-instrumented profiling with nsys...${NC}"
    echo ""

    # nsys command with comprehensive tracing
    # Note: Adjust paths if running outside Docker
    NSYS_CMD="nsys profile \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --cuda-memory-usage=true \
        --gpu-metrics-device=all \
        --sample=cpu \
        --cpuctxsw=process-tree \
        --output=${OUTPUT_DIR}/${PROFILE_NAME} \
        --force-overwrite=true \
        --stats=true \
        python scripts/profile_denoise_nsys.py \
            --steps ${NUM_STEPS} \
            --warmup ${WARMUP} \
            --iterations ${ITERATIONS}"

    echo "Command:"
    echo "${NSYS_CMD}"
    echo ""

    # Check if nsys is available
    if ! command -v nsys &> /dev/null; then
        echo -e "${RED}ERROR: nsys not found!${NC}"
        echo "Please install NVIDIA Nsight Systems or run in the Docker container."
        echo ""
        echo "Alternative: Run with Python timing only:"
        echo "  python scripts/profile_denoise_nsys.py --steps ${NUM_STEPS}"
        exit 1
    fi

    # Run profiling
    eval ${NSYS_CMD}

    echo ""
    echo -e "${GREEN}Profiling complete!${NC}"
    echo "Output: ${OUTPUT_DIR}/${PROFILE_NAME}.nsys-rep"
    echo ""
fi

# ============================================================================
# Step 2: Export to SQLite
# ============================================================================

NSYS_REP="${OUTPUT_DIR}/${PROFILE_NAME}.nsys-rep"
SQLITE_FILE="${OUTPUT_DIR}/${PROFILE_NAME}.sqlite"

if [ -f "${NSYS_REP}" ]; then
    echo -e "${YELLOW}Step 2: Exporting to SQLite...${NC}"

    nsys export --type=sqlite --output="${SQLITE_FILE}" "${NSYS_REP}" --force-overwrite

    echo "Exported: ${SQLITE_FILE}"
    echo ""
else
    echo -e "${RED}ERROR: nsys-rep file not found: ${NSYS_REP}${NC}"
    exit 1
fi

# ============================================================================
# Step 3: Run Gap Analysis
# ============================================================================

if [ -f "${SQLITE_FILE}" ]; then
    echo -e "${YELLOW}Step 3: Running gap analysis...${NC}"
    echo ""

    python scripts/analyze_nsys_gaps.py "${SQLITE_FILE}" \
        --output "${OUTPUT_DIR}/${PROFILE_NAME}.analysis.txt" \
        --json

    echo ""
    echo -e "${GREEN}Analysis complete!${NC}"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}     PROFILING COMPLETE${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo "Output Files:"
echo "  - ${OUTPUT_DIR}/${PROFILE_NAME}.nsys-rep    (Nsight Systems report)"
echo "  - ${OUTPUT_DIR}/${PROFILE_NAME}.sqlite      (SQLite export)"
echo "  - ${OUTPUT_DIR}/${PROFILE_NAME}.analysis.txt (Gap analysis)"
echo "  - ${OUTPUT_DIR}/${PROFILE_NAME}.analysis.json (JSON summary)"
echo ""
echo "Next Steps:"
echo ""
echo "  1. Open in Nsight Systems GUI (for visual timeline):"
echo "     nsys-ui ${OUTPUT_DIR}/${PROFILE_NAME}.nsys-rep"
echo ""
echo "  2. Check GPU Metrics for memory bandwidth:"
echo "     - Look for DRAM Read/Write throughput"
echo "     - If > 80% during MLP: Memory Bound confirmed"
echo ""
echo "  3. Review the analysis report:"
echo "     cat ${OUTPUT_DIR}/${PROFILE_NAME}.analysis.txt"
echo ""
echo "============================================================================"
echo ""
echo -e "${YELLOW}DIAGNOSTIC CRITERIA:${NC}"
echo ""
echo "  Gap Analysis:"
echo "    - Gap 5-10us:  ✅ NORMAL (CUDA Graph working)"
echo "    - Gap 50us-1ms: 🚨 SEVERE (CPU Launch Bound)"
echo ""
echo "  Memory Bound Detection:"
echo "    - SM Utilization < 30% + DRAM Bandwidth > 80%"
echo "    - → Need L2 Cache Residency strategy"
echo ""
echo "  Stream Sync:"
echo "    - If cudaStreamSynchronize found: 🚨 CRITICAL"
echo "    - Remove ALL sync points from hot path"
echo ""
