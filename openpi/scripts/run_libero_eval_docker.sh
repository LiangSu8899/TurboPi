#!/bin/bash
# Run LIBERO evaluation in NVIDIA container with MuJoCo EGL rendering

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENPI_DIR="$(dirname "$SCRIPT_DIR")"
CONTAINER_IMAGE="nvcr.io/nvidia/vllm:25.09-py3"

# Parse arguments
TASK_SUITE=${1:-"libero_spatial"}
NUM_TRIALS=${2:-50}
PORT=${3:-8000}

echo "=== LIBERO Evaluation Configuration ==="
echo "Task Suite: $TASK_SUITE"
echo "Trials per Task: $NUM_TRIALS"
echo "Server Port: $PORT"
echo "========================================"

# Create LIBERO config
mkdir -p /tmp/libero_eval
cat > /tmp/libero_eval/config.yaml << LIBEROCONF
benchmark_root: /app/third_party/libero/libero/libero
bddl_files: /app/third_party/libero/libero/libero/bddl_files
init_states: /app/third_party/libero/libero/libero/init_files
datasets: /app/third_party/libero/libero/datasets
assets: /app/third_party/libero/libero/libero/assets
LIBEROCONF

# Run evaluation in container
docker run --rm --runtime=nvidia \
    -v "${OPENPI_DIR}:/app" \
    -v "/home/heima-thor/.cache/openpi:/openpi_assets" \
    -v "/tmp/libero_eval:/tmp/libero" \
    -e MUJOCO_GL=egl \
    -e MUJOCO_EGL_DEVICE_ID=0 \
    -e PYOPENGL_PLATFORM=egl \
    -e OPENPI_DATA_HOME=/openpi_assets \
    -e LIBERO_CONFIG_PATH=/tmp/libero \
    --ipc=host \
    --network=host \
    ${CONTAINER_IMAGE} \
    bash -c "
cd /app

echo '=== Installing dependencies ==='
pip install --quiet gymnasium robosuite==1.4.1 mujoco glfw tyro h5py imageio imageio-ffmpeg bddl opencv-python-headless pynput easydict websockets einops transformers safetensors lerobot@git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 gcsfs 2>/dev/null

export PYTHONPATH=/app/src:/app/packages/openpi-client/src:/app/third_party/libero

echo '=== Verifying PyTorch CUDA ==='
python -c 'import torch; print(\"PyTorch:\", torch.__version__); print(\"CUDA available:\", torch.cuda.is_available()); print(\"GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")'

echo '=== Starting Policy Server in background ==='
python scripts/serve_policy.py --env=libero --port=${PORT} &
SERVER_PID=\$!
sleep 30  # Wait for server to start and load model

echo '=== Running LIBERO Evaluation ==='
python examples/libero/main.py \
    --host=localhost \
    --port=${PORT} \
    --task_suite_name=${TASK_SUITE} \
    --num_trials_per_task=${NUM_TRIALS}

echo '=== Evaluation Complete ==='
kill \$SERVER_PID 2>/dev/null || true
"
