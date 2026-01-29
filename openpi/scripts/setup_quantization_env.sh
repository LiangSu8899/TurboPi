#!/bin/bash
# Setup quantization environment for Pi0.5 FP4/FP8 optimization
# Target: NVIDIA Thor (Blackwell SM110)

set -e

echo "=============================================="
echo "Pi0.5 Quantization Environment Setup"
echo "=============================================="

# Check if running in container
if [ ! -f /.dockerenv ]; then
    echo "WARNING: Not running in Docker container"
    echo "Recommended: nvcr.io/nvidia/pytorch:25.12-py3"
fi

# Check CUDA version
echo ""
echo "Checking CUDA version..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Install NVIDIA ModelOpt
echo ""
echo "Installing NVIDIA ModelOpt..."
pip install --quiet nvidia-modelopt[torch]>=0.17.0

# Verify installation
echo ""
echo "Verifying ModelOpt installation..."
python3 -c "import modelopt; print(f'ModelOpt version: {modelopt.__version__}')"

# Check for Blackwell FP4 support
echo ""
echo "Checking GPU capabilities..."
python3 << 'EOF'
import torch

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")

    # SM110 = Blackwell (Thor)
    if props.major >= 10:
        print("FP4 Support: YES (Blackwell SM110+)")
        print("FP8 Support: YES")
    elif props.major >= 9:
        print("FP4 Support: NO (requires SM110+)")
        print("FP8 Support: YES (Hopper SM90+)")
    else:
        print("FP4 Support: NO")
        print("FP8 Support: NO")
else:
    print("CUDA not available!")
EOF

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/quantize_model.py --help"
echo "  2. Or:  python scripts/quantize_model.py \\"
echo "          --model_path ~/.cache/openpi/checkpoints/pi05_libero \\"
echo "          --output_dir ./quantized_models/pi05_fp4fp8"
