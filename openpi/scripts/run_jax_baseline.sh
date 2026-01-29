#!/bin/bash
# Run JAX baseline evaluation to confirm original model works
# This uses the original OpenPi JAX checkpoint (without model.safetensors)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENPI_DIR="$(dirname "$SCRIPT_DIR")"

TASK_SUITE=${1:-"libero_spatial"}
NUM_TRIALS=${2:-10}  # Start with fewer trials for quick test
PORT=8000

echo "============================================="
echo "JAX Baseline Evaluation"
echo "============================================="
echo "Task Suite: $TASK_SUITE"
echo "Trials per Task: $NUM_TRIALS"
echo "Checkpoint: JAX (OpenPi original)"
echo "============================================="

# Run evaluation in Docker container
docker run --rm --runtime=nvidia --ipc=host --network=host \
    -v "${OPENPI_DIR}:/app" \
    -v "/home/heima-thor/.cache/openpi:/openpi_cache" \
    -e MUJOCO_GL=egl \
    -e MUJOCO_EGL_DEVICE_ID=0 \
    -e PYOPENGL_PLATFORM=egl \
    -e OPENPI_DATA_HOME=/openpi_cache \
    libero_eval:latest bash -c "
set -e

# Install gym
pip install --quiet gym==0.26.2 2>/dev/null

# Set PYTHONPATH
export PYTHONPATH=/app/src:/app/packages/openpi-client/src:/app/third_party/libero

# Create LIBERO config
mkdir -p /tmp/libero
cat > /tmp/libero/config.yaml << 'EOF'
benchmark_root: /app/third_party/libero/libero/libero
bddl_files: /app/third_party/libero/libero/libero/bddl_files
init_states: /app/third_party/libero/libero/libero/init_files
datasets: /app/third_party/libero/libero/datasets
assets: /app/third_party/libero/libero/libero/assets
EOF
export LIBERO_CONFIG_PATH=/tmp/libero

echo '=== Verifying environment ==='
python -c 'import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())'
python -c 'import jax; print(\"JAX devices:\", jax.devices())'

echo ''
echo '=== Checking checkpoint type ==='
JAX_CKPT='/openpi_cache/openpi-assets/checkpoints/pi05_libero'
echo \"JAX checkpoint: \$JAX_CKPT\"
ls -la \$JAX_CKPT
echo ''
echo 'Has model.safetensors?'
ls \$JAX_CKPT/model.safetensors 2>/dev/null && echo 'YES' || echo 'NO (will use JAX)'

echo ''
echo '=== Starting JAX Policy Server ==='
# Use the JAX checkpoint path explicitly
# Note: --port must come before the subcommand
python /app/scripts/serve_policy.py \\
    --env=LIBERO \\
    --port=${PORT} \\
    policy:checkpoint \\
    --policy.config=pi05_libero \\
    --policy.dir=\$JAX_CKPT &
SERVER_PID=\$!

# Wait for server to start
echo 'Waiting for JAX model to load (this may take a while)...'
sleep 120

# Check if server is running
if ! kill -0 \$SERVER_PID 2>/dev/null; then
    echo 'ERROR: Server failed to start'
    exit 1
fi
echo 'Server is running!'

echo ''
echo '=== Running LIBERO Evaluation ==='
python /app/examples/libero/main.py \\
    --host=localhost \\
    --port=${PORT} \\
    --task_suite_name=${TASK_SUITE} \\
    --num_trials_per_task=${NUM_TRIALS}

echo ''
echo '=== JAX Baseline Evaluation Complete ==='
kill \$SERVER_PID 2>/dev/null || true
"
