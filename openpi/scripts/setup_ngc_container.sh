#!/bin/bash
# setup_ngc_container.sh - NGC Container Setup for Pi0.5 on Jetson Thor
#
# This script sets up NVIDIA NGC containers for TensorRT-accelerated inference.
# Supports two deployment options:
#   1. jetson-containers (development) - Recommended for quick setup
#   2. NGC L4T TensorRT (production) - Official NVIDIA container
#
# Usage:
#   ./setup_ngc_container.sh [--option jetson|ngc] [--no-pull]
#
# Prerequisites:
#   - NVIDIA Jetson Thor with JetPack 7.1+
#   - Docker with nvidia-container-toolkit
#   - At least 32GB of disk space

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT_DIR="$HOME/.cache/openpi/checkpoints/pi05_libero"

# Container options
JETSON_CONTAINER="dustynv/tensorrt_llm:0.12-r38.0.0"
NGC_CONTAINER="nvcr.io/nvidia/l4t-tensorrt:r38.0.0"
JETSON_CONTAINERS_REPO="https://github.com/dusty-nv/jetson-containers.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
CONTAINER_OPTION="jetson"  # Default to jetson-containers
NO_PULL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --option)
            CONTAINER_OPTION="$2"
            shift 2
            ;;
        --no-pull)
            NO_PULL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--option jetson|ngc] [--no-pull]"
            echo ""
            echo "Options:"
            echo "  --option jetson    Use jetson-containers (default, recommended for dev)"
            echo "  --option ngc       Use NGC L4T TensorRT container"
            echo "  --no-pull          Skip pulling container (use existing)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if running on Jetson
    if [ ! -f /etc/nv_tegra_release ]; then
        log_warn "Not running on NVIDIA Jetson. Some features may not work."
    else
        log_info "Detected Jetson platform:"
        cat /etc/nv_tegra_release
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi
    log_info "Docker: $(docker --version)"

    # Check nvidia-container-toolkit
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        log_warn "NVIDIA container runtime not detected. Installing..."
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
    fi
    log_info "NVIDIA container runtime: OK"

    # Check available disk space (need at least 20GB)
    AVAILABLE_SPACE=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 20 ]; then
        log_warn "Low disk space: ${AVAILABLE_SPACE}GB available (recommend 20GB+)"
    fi
    log_info "Disk space: ${AVAILABLE_SPACE}GB available"

    # Check model checkpoint
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        log_warn "Model checkpoint not found at: $CHECKPOINT_DIR"
        log_warn "You may need to download the model first."
    else
        log_info "Model checkpoint: $CHECKPOINT_DIR"
    fi
}

# Setup jetson-containers
setup_jetson_containers() {
    log_info "Setting up jetson-containers..."

    # Clone if not exists
    JETSON_CONTAINERS_DIR="$HOME/jetson-containers"
    if [ ! -d "$JETSON_CONTAINERS_DIR" ]; then
        log_info "Cloning jetson-containers..."
        git clone "$JETSON_CONTAINERS_REPO" "$JETSON_CONTAINERS_DIR"
    fi

    # Install
    log_info "Installing jetson-containers..."
    cd "$JETSON_CONTAINERS_DIR"
    bash install.sh

    # Pull container
    if [ "$NO_PULL" = false ]; then
        log_info "Pulling container: $JETSON_CONTAINER"
        docker pull "$JETSON_CONTAINER" || {
            log_warn "Pull failed, will try building..."
            jetson-containers build tensorrt_llm
        }
    fi

    log_info "jetson-containers setup complete!"
}

# Setup NGC container
setup_ngc_container() {
    log_info "Setting up NGC L4T TensorRT container..."

    # Login to NGC (optional)
    if [ -n "$NGC_API_KEY" ]; then
        log_info "Logging into NGC..."
        echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
    fi

    # Pull container
    if [ "$NO_PULL" = false ]; then
        log_info "Pulling container: $NGC_CONTAINER"
        docker pull "$NGC_CONTAINER"
    fi

    log_info "NGC container setup complete!"
}

# Create run script
create_run_script() {
    local container_image="$1"
    local script_name="run_pi05_container.sh"
    local script_path="$PROJECT_DIR/scripts/$script_name"

    log_info "Creating run script: $script_path"

    cat > "$script_path" << 'EOFSCRIPT'
#!/bin/bash
# run_pi05_container.sh - Run Pi0.5 in TensorRT container
#
# Usage: ./run_pi05_container.sh [command]
# Examples:
#   ./run_pi05_container.sh                    # Interactive shell
#   ./run_pi05_container.sh python benchmark.py  # Run specific command

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT_DIR="$HOME/.cache/openpi/checkpoints"

EOFSCRIPT

    echo "CONTAINER_IMAGE=\"$container_image\"" >> "$script_path"

    cat >> "$script_path" << 'EOFSCRIPT'

# Docker run command
docker run --runtime nvidia \
    -it --rm \
    --network host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$PROJECT_DIR":/workspace/openpi \
    -v "$CHECKPOINT_DIR":/workspace/checkpoints \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e CUDA_VISIBLE_DEVICES=0 \
    -w /workspace/openpi \
    "$CONTAINER_IMAGE" \
    "${@:-/bin/bash}"
EOFSCRIPT

    chmod +x "$script_path"
    log_info "Run script created: $script_path"
}

# Create container setup script (runs inside container)
create_container_init_script() {
    local script_path="$PROJECT_DIR/scripts/container_init.sh"

    log_info "Creating container init script: $script_path"

    cat > "$script_path" << 'EOFSCRIPT'
#!/bin/bash
# container_init.sh - Initialize environment inside NGC container
#
# Run this after starting the container to set up the Pi0.5 environment.

set -e

echo "=== Pi0.5 Container Initialization ==="

# Install project dependencies
echo "[1/4] Installing openpi package..."
cd /workspace/openpi
pip install -e . --no-deps 2>/dev/null || pip install -e .

# Check TensorRT
echo "[2/4] Checking TensorRT..."
python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')" 2>/dev/null || {
    echo "TensorRT Python bindings not available, checking native..."
    ls -la /usr/lib/aarch64-linux-gnu/libnvinfer* 2>/dev/null || echo "TensorRT not found"
}

# Check PyTorch
echo "[3/4] Checking PyTorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify GPU
echo "[4/4] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""
echo "=== Initialization Complete ==="
echo ""
echo "To run benchmark:"
echo "  cd /workspace/openpi"
echo "  python scripts/benchmark_thor.py --model_path /workspace/checkpoints/pi05_libero"
echo ""
echo "To export TensorRT engines:"
echo "  python scripts/export_tensorrt_modular.py"
EOFSCRIPT

    chmod +x "$script_path"
    log_info "Container init script created: $script_path"
}

# Create modular TensorRT export script
create_trt_export_script() {
    local script_path="$PROJECT_DIR/scripts/export_tensorrt_modular.py"

    log_info "Creating TensorRT export script: $script_path"

    cat > "$script_path" << 'EOFSCRIPT'
#!/usr/bin/env python3
"""
export_tensorrt_modular.py - Export Pi0.5 components to TensorRT engines.

This script exports the Pi0.5 model components (vision encoder, LLM backbone,
action expert) to individual TensorRT engines for optimized inference.

Usage:
    python export_tensorrt_modular.py \
        --model_path /workspace/checkpoints/pi05_libero \
        --output_dir /workspace/trt_engines

Requirements:
    - TensorRT 10.x with NVFP4/FP8 support
    - ONNX 1.14+
    - PyTorch 2.x
"""

import argparse
import logging
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_tensorrt():
    """Check TensorRT availability."""
    try:
        import tensorrt as trt
        logger.info(f"TensorRT version: {trt.__version__}")
        return True
    except ImportError:
        logger.warning("TensorRT Python bindings not available")
        return False


def check_onnx():
    """Check ONNX availability."""
    try:
        import onnx
        logger.info(f"ONNX version: {onnx.__version__}")
        return True
    except ImportError:
        logger.warning("ONNX not available")
        return False


def export_vision_encoder(model, output_dir: Path, precision: str = "fp8"):
    """Export SigLIP vision encoder to ONNX."""
    import torch

    logger.info("Exporting vision encoder...")

    # Get vision model
    vision_model = model.paligemma_with_expert.paligemma.vision_model
    vision_model.eval()

    # Dummy input (batch=1, channels=3, height=224, width=224)
    # SigLIP uses 224x224 or 384x384 depending on variant
    dummy_images = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)

    # Export to ONNX
    onnx_path = output_dir / "siglip_vision.onnx"

    with torch.no_grad():
        torch.onnx.export(
            vision_model,
            dummy_images,
            str(onnx_path),
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["vision_embeddings"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "vision_embeddings": {0: "batch"},
            },
            do_constant_folding=True,
        )

    logger.info(f"Vision encoder exported to: {onnx_path}")
    return onnx_path


def export_action_expert(model, output_dir: Path, precision: str = "fp4"):
    """Export Gemma 300M action expert to ONNX."""
    import torch

    logger.info("Exporting action expert...")

    # Get expert model
    expert_model = model.paligemma_with_expert.expert
    expert_model.eval()

    # Dummy inputs
    batch_size = 1
    seq_len = 512
    hidden_size = expert_model.config.hidden_size

    dummy_inputs = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
    dummy_attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, device="cuda", dtype=torch.bfloat16)

    # Export to ONNX
    onnx_path = output_dir / "gemma_300m_expert.onnx"

    with torch.no_grad():
        torch.onnx.export(
            expert_model,
            (dummy_inputs, dummy_attention_mask),
            str(onnx_path),
            opset_version=17,
            input_names=["hidden_states", "attention_mask"],
            output_names=["output_hidden_states"],
            dynamic_axes={
                "hidden_states": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 2: "seq_len", 3: "seq_len"},
                "output_hidden_states": {0: "batch", 1: "seq_len"},
            },
            do_constant_folding=True,
        )

    logger.info(f"Action expert exported to: {onnx_path}")
    return onnx_path


def build_trt_engine(onnx_path: Path, engine_path: Path, precision: str = "fp8"):
    """Build TensorRT engine from ONNX model."""
    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT not available. Skipping engine build.")
        return None

    logger.info(f"Building TensorRT engine: {onnx_path} -> {engine_path}")
    logger.info(f"Precision: {precision}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(i)}")
            return None

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8GB

    # Set precision
    if precision == "fp8":
        if builder.platform_has_fast_fp8:
            config.set_flag(trt.BuilderFlag.FP8)
            logger.info("FP8 enabled")
        else:
            logger.warning("FP8 not supported, falling back to FP16")
            config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "fp4":
        # FP4 requires ModelOpt quantization first
        config.set_flag(trt.BuilderFlag.FP16)
        logger.warning("FP4 requires pre-quantization. Using FP16.")
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")

    # Build engine
    logger.info("Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return None

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine saved: {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Export Pi0.5 to TensorRT engines")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/checkpoints/pi05_libero",
        help="Path to Pi0.5 checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/trt_engines",
        help="Output directory for TensorRT engines",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp4", "fp8", "fp16"],
        default="fp8",
        help="Target precision for TensorRT engines",
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Only export ONNX, skip TensorRT build",
    )
    args = parser.parse_args()

    # Check dependencies
    if not check_onnx():
        logger.error("ONNX not available. Please install: pip install onnx")
        sys.exit(1)

    has_tensorrt = check_tensorrt()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    try:
        sys.path.insert(0, "/workspace/openpi/src")
        from openpi.models_pytorch.pi0_pytorch import Pi0Pytorch
        from openpi.models_pytorch.model_config import ModelConfig, ModelType

        # Load checkpoint
        import torch
        checkpoint_path = Path(args.model_path)

        # This is a placeholder - actual loading depends on checkpoint format
        logger.info("Loading Pi0.5 model...")
        # model = Pi0Pytorch.from_pretrained(checkpoint_path)

        logger.warning("Model loading not implemented. Skipping ONNX export.")
        logger.info("To complete export, implement model loading for your checkpoint format.")
        return

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Skipping ONNX export.")
        return

    # Export components
    vision_onnx = export_vision_encoder(model, output_dir, args.precision)
    expert_onnx = export_action_expert(model, output_dir, args.precision)

    # Build TensorRT engines
    if not args.export_only and has_tensorrt:
        build_trt_engine(vision_onnx, output_dir / "siglip_vision.engine", args.precision)
        build_trt_engine(expert_onnx, output_dir / "gemma_300m_expert.engine", args.precision)

    logger.info("Export complete!")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
EOFSCRIPT

    chmod +x "$script_path"
    log_info "TensorRT export script created: $script_path"
}

# Print summary
print_summary() {
    local container_image="$1"

    echo ""
    echo "=============================================="
    echo "  NGC Container Setup Complete"
    echo "=============================================="
    echo ""
    echo "Container: $container_image"
    echo ""
    echo "To start the container:"
    echo "  cd $PROJECT_DIR/scripts"
    echo "  ./run_pi05_container.sh"
    echo ""
    echo "Inside the container, run:"
    echo "  ./scripts/container_init.sh"
    echo ""
    echo "To run benchmark:"
    echo "  python scripts/benchmark_thor.py --model_path /workspace/checkpoints/pi05_libero"
    echo ""
    echo "To export TensorRT engines:"
    echo "  python scripts/export_tensorrt_modular.py"
    echo ""
    echo "Documentation: $PROJECT_DIR/../docs/phase4_ngc_deployment.md"
    echo ""
}

# Main execution
main() {
    log_info "NGC Container Setup for Pi0.5 on Jetson Thor"
    log_info "Container option: $CONTAINER_OPTION"

    check_prerequisites

    case "$CONTAINER_OPTION" in
        jetson)
            setup_jetson_containers
            create_run_script "$JETSON_CONTAINER"
            ;;
        ngc)
            setup_ngc_container
            create_run_script "$NGC_CONTAINER"
            ;;
        *)
            log_error "Unknown option: $CONTAINER_OPTION"
            exit 1
            ;;
    esac

    create_container_init_script
    create_trt_export_script

    if [ "$CONTAINER_OPTION" = "jetson" ]; then
        print_summary "$JETSON_CONTAINER"
    else
        print_summary "$NGC_CONTAINER"
    fi
}

main
