# Turbo-Pi0.5 Deployment Container
# For NVIDIA Jetson Thor / GPU systems with CUDA 12.0+

FROM nvcr.io/nvidia/pytorch:25.12-py3

LABEL maintainer="LiangSu8899"
LABEL version="1.1.2"
LABEL description="Turbo-Pi0.5 VLA Model - 19.2x faster inference"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY openpi/ /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install pycuda einops safetensors pillow && \
    pip install -e . && \
    pip install huggingface_hub

# Create cache directory
RUN mkdir -p /root/.cache/openpi/checkpoints

# Download model from HuggingFace (optional - can mount externally)
ARG DOWNLOAD_MODEL=false
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    huggingface-cli download liangsu9988/Turbo-Pi0.5-1.1.2 \
        --local-dir /root/.cache/openpi/checkpoints/pi05_libero; \
    fi

# Expose policy server port
EXPOSE 8000

# Default command: start policy server
CMD ["python", "scripts/serve_policy.py", \
     "--env=LIBERO", \
     "--port=8000", \
     "policy:checkpoint", \
     "--policy.config=pi05_libero", \
     "--policy.dir=/root/.cache/openpi/checkpoints/pi05_libero"]
