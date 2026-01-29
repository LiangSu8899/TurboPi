#!/bin/bash
# Run LIBERO evaluation with PyTorch model in Docker container
# Usage: ./scripts/run_libero_pytorch.sh [task_suite] [num_trials]
#   task_suite: libero_spatial, libero_object, libero_goal, libero_10, libero_90
#   num_trials: number of trials per task (default: 50)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENPI_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
TASK_SUITE=${1:-"libero_spatial"}
NUM_TRIALS=${2:-50}
PORT=8000

# Local PyTorch checkpoint path (HuggingFace format)
CHECKPOINT_DIR="/openpi_cache/checkpoints/pi05_libero"

echo "============================================="
echo "LIBERO PyTorch Evaluation Configuration"
echo "============================================="
echo "Task Suite: $TASK_SUITE"
echo "Trials per Task: $NUM_TRIALS"
echo "Server Port: $PORT"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "============================================="

# Run evaluation in Docker container
docker run --rm --runtime=nvidia --ipc=host --network=host \
    -v "${OPENPI_DIR}:/app" \
    -v "/home/heima-thor/.cache/openpi:/openpi_cache" \
    -e MUJOCO_GL=egl \
    -e MUJOCO_EGL_DEVICE_ID=0 \
    -e PYOPENGL_PLATFORM=egl \
    libero_eval:latest bash -c "
set -e

# Set PYTHONPATH
export PYTHONPATH=/app/src:/app/packages/openpi-client/src:/app/third_party/libero

# Create LIBERO config in ~/.libero (required by libero library on import)
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: /app/third_party/libero/libero/libero
bddl_files: /app/third_party/libero/libero/libero/bddl_files
init_states: /app/third_party/libero/libero/libero/init_files
datasets: /app/third_party/libero/libero/datasets
assets: /app/third_party/libero/libero/libero/assets
EOF

echo '=== Verifying environment ==='
python -c 'import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available(), \"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")'

echo ''
echo '=== Starting Policy Server (PyTorch) ==='
python /app/scripts/serve_policy.py \
    --env=LIBERO \
    --port=${PORT} \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=${CHECKPOINT_DIR} &
SERVER_PID=\$!

# Wait for server to start
echo 'Waiting for server to load model...'
for i in {1..120}; do
    if curl -s http://localhost:${PORT}/healthz > /dev/null 2>&1; then
        echo 'Server is ready!'
        break
    fi
    if ! kill -0 \$SERVER_PID 2>/dev/null; then
        echo 'ERROR: Server failed to start'
        exit 1
    fi
    sleep 1
done

# Check if server is running
if ! kill -0 \$SERVER_PID 2>/dev/null; then
    echo 'ERROR: Server failed to start after timeout'
    exit 1
fi

echo ''
echo '=== Running LIBERO Evaluation ==='
python /app/examples/libero/main.py \
    --args.host localhost \
    --args.port ${PORT} \
    --args.task-suite-name ${TASK_SUITE} \
    --args.num-trials-per-task ${NUM_TRIALS} \
    --args.video-out-path /app/data/libero/videos_${TASK_SUITE}

echo ''
echo '=== Evaluation Complete ==='
kill \$SERVER_PID 2>/dev/null || true
"
