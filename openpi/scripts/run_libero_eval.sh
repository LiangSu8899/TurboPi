#!/bin/bash
# Run LIBERO evaluation in Docker container
# Usage: ./scripts/run_libero_eval.sh [task_suite] [num_trials]
#   task_suite: libero_spatial, libero_object, libero_goal, libero_10, libero_90
#   num_trials: number of trials per task (default: 50)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENPI_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
TASK_SUITE=${1:-"libero_spatial"}
NUM_TRIALS=${2:-50}
PORT=8000

echo "============================================="
echo "LIBERO Evaluation Configuration"
echo "============================================="
echo "Task Suite: $TASK_SUITE"
echo "Trials per Task: $NUM_TRIALS"
echo "Server Port: $PORT"
echo "============================================="

# Run evaluation in Docker container
docker run --rm --runtime=nvidia --ipc=host --network=host \
    -v "${OPENPI_DIR}:/app" \
    -v "/home/heima-thor/.cache/openpi:/openpi_assets" \
    -e MUJOCO_GL=egl \
    -e MUJOCO_EGL_DEVICE_ID=0 \
    -e PYOPENGL_PLATFORM=egl \
    -e OPENPI_DATA_HOME=/openpi_assets \
    libero_eval:latest bash -c "
set -e

# Install gym (needed by LIBERO)
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

echo ''
echo '=== Starting Policy Server ==='
python /app/scripts/serve_policy.py --env=libero --port=${PORT} &
SERVER_PID=\$!

# Wait for server to start
echo 'Waiting for server to load model...'
sleep 60

# Check if server is running
if ! kill -0 \$SERVER_PID 2>/dev/null; then
    echo 'ERROR: Server failed to start'
    exit 1
fi

echo ''
echo '=== Running LIBERO Evaluation ==='
python /app/examples/libero/main.py \
    --host=localhost \
    --port=${PORT} \
    --task_suite_name=${TASK_SUITE} \
    --num_trials_per_task=${NUM_TRIALS} \
    --video_out_path=/app/data/libero/videos_${TASK_SUITE}

echo ''
echo '=== Evaluation Complete ==='
kill \$SERVER_PID 2>/dev/null || true
"
