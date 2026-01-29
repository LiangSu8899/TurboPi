# Phase 4: NGC Container Deployment for Pi0.5 on Jetson Thor

## Executive Summary

**Objective**: Deploy Pi0.5 VLA model using NVIDIA NGC containers with TensorRT acceleration to achieve 12-15 Hz inference on Jetson Thor.

**Platform**: NVIDIA Jetson Thor (Blackwell architecture, JetPack 7.1, CUDA 13.0)

---

## Available Deployment Options

### Option 1: TensorRT Edge-LLM (Recommended for Production)

**Repository**: [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)

**Key Features**:
- Purpose-built for Jetson Thor and DRIVE AGX Thor
- Native NVFP4 quantization support
- CUDA Graph optimization
- C++ runtime (no Python overhead)
- EAGLE-3 speculative decoding

**Supported VLMs** (as of v0.4.0):
- Qwen2/2.5/3-VL series
- InternVL3 (1B, 2B)
- Phi-4-Multimodal (5.6B)
- *PaliGemma not officially supported*

**Pipeline**:
```
HuggingFace Model → Python Export (ONNX) → Engine Builder (TRT) → C++ Runtime
```

**Pros**:
- Best performance (native TensorRT kernels)
- Production-ready C++ runtime
- Minimal dependencies

**Cons**:
- Requires porting Pi0.5 architecture to supported format
- PaliGemma/Gemma not in supported model list
- Custom model integration requires significant development

---

### Option 2: Jetson-Containers (Recommended for Development)

**Repository**: [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers)

**Key Features**:
- Pre-built containers for JetPack 7.x
- Modular build system
- PyTorch + TensorRT integration
- SGLang, vLLM, transformers support

**Available Containers**:
```bash
# Base PyTorch with TensorRT
dustynv/l4t-pytorch:r38.0.0

# TensorRT-LLM (Jetson optimized)
dustynv/tensorrt_llm:0.12-r38.0.0

# Full ML stack
dustynv/l4t-ml:r38.0.0
```

**Pros**:
- Easy setup (pre-built images)
- Can run existing PyTorch code
- Modular customization

**Cons**:
- Less optimized than native TensorRT
- Larger container size
- Still requires manual TRT engine building

---

### Option 3: NGC L4T TensorRT Container

**NGC URL**: [nvcr.io/nvidia/l4t-tensorrt](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt)

**Features**:
- Official NVIDIA TensorRT container for Jetson
- TensorRT 10.x with NVFP4/FP8 support
- Python and C++ APIs

**Usage**:
```bash
docker pull nvcr.io/nvidia/l4t-tensorrt:r38.0.0
docker run --runtime nvidia -it --rm nvcr.io/nvidia/l4t-tensorrt:r38.0.0
```

---

## Recommended Deployment Strategy

Given that Pi0.5 uses PaliGemma (which is not directly supported by TensorRT Edge-LLM), we recommend a **hybrid approach**:

### Strategy: Modular TensorRT Engine Building

Split the model into components and optimize each separately:

```
┌─────────────────────────────────────────────────────────────┐
│                     Pi0.5 Architecture                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ SigLIP-SO400M│  │ Gemma 2B LLM │  │ Gemma 300M Expert│   │
│  │  (Vision)    │  │  (Language)  │  │   (Action)       │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                 │                    │             │
│         v                 v                    v             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ TRT Engine 1 │  │ TRT Engine 2 │  │  TRT Engine 3    │   │
│  │    (FP8)     │  │    (FP4)     │  │      (FP4)       │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Phase 4a: Container Setup

```bash
# 1. Install jetson-containers
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
bash install.sh

# 2. Pull TensorRT-LLM container
jetson-containers run dustynv/tensorrt_llm:0.12-r38.0.0

# Alternative: NGC TensorRT container
docker pull nvcr.io/nvidia/l4t-tensorrt:r38.0.0
```

### Phase 4b: Model Component Export

Export each model component to ONNX:

```python
# 1. Export SigLIP Vision Encoder
torch.onnx.export(
    model.paligemma_with_expert.paligemma.vision_model,
    dummy_images,
    "siglip_vision.onnx",
    opset_version=17,
    input_names=["pixel_values"],
    output_names=["vision_embeddings"],
    dynamic_axes={"pixel_values": {0: "batch"}},
)

# 2. Export Gemma 2B (Language backbone)
torch.onnx.export(
    model.paligemma_with_expert.paligemma.language_model,
    (dummy_inputs, dummy_attention_mask),
    "gemma_2b.onnx",
    opset_version=17,
)

# 3. Export Gemma 300M (Action Expert)
torch.onnx.export(
    model.paligemma_with_expert.expert,
    (dummy_inputs, dummy_attention_mask),
    "gemma_300m_expert.onnx",
    opset_version=17,
)
```

### Phase 4c: TensorRT Engine Building

```bash
# Build TensorRT engines with NVFP4/FP8
trtexec --onnx=siglip_vision.onnx \
        --saveEngine=siglip_vision.engine \
        --fp8 \
        --builderOptimizationLevel=5

trtexec --onnx=gemma_2b.onnx \
        --saveEngine=gemma_2b.engine \
        --fp4 \
        --builderOptimizationLevel=5

trtexec --onnx=gemma_300m_expert.onnx \
        --saveEngine=gemma_300m_expert.engine \
        --fp4 \
        --builderOptimizationLevel=5
```

### Phase 4d: Runtime Integration

Create a Python wrapper that orchestrates the TensorRT engines:

```python
import tensorrt as trt
import numpy as np

class Pi0TensorRTInference:
    def __init__(self, engine_paths):
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.vision_engine = self._load_engine(engine_paths['vision'])
        self.llm_engine = self._load_engine(engine_paths['llm'])
        self.expert_engine = self._load_engine(engine_paths['expert'])

    def _load_engine(self, path):
        with open(path, 'rb') as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def sample_actions(self, observation, num_steps=10):
        # 1. Run vision encoder
        vision_embs = self._run_vision(observation['images'])

        # 2. Run LLM backbone (prefix)
        prefix_embs, kv_cache = self._run_llm_prefix(vision_embs, observation['prompt'])

        # 3. Run denoising loop with expert
        actions = self._run_denoising(kv_cache, observation['state'], num_steps)

        return actions
```

---

## Expected Performance

| Configuration | Estimated Throughput | Memory |
|--------------|---------------------|--------|
| PyTorch BF16 (baseline) | 3.5 Hz | 7.6 GB |
| Modular TensorRT FP8 | 8-10 Hz | 4.5 GB |
| Modular TensorRT FP4 | 12-15 Hz | 3.0 GB |
| Edge-LLM (if supported) | 15-20 Hz | 2.5 GB |

---

## Quick Start Script

```bash
#!/bin/bash
# phase4_deploy.sh - NGC Container Deployment for Pi0.5

set -e

# Configuration
CONTAINER_IMAGE="dustynv/tensorrt_llm:0.12-r38.0.0"
MODEL_DIR="/home/heima-thor/suliang/Turbo-Pi/openpi"
CHECKPOINT_DIR="$HOME/.cache/openpi/checkpoints/pi05_libero"

# Step 1: Pull container
echo "[1/4] Pulling container..."
docker pull $CONTAINER_IMAGE

# Step 2: Run container with mounted volumes
echo "[2/4] Starting container..."
docker run --runtime nvidia \
    -it --rm \
    --network host \
    -v $MODEL_DIR:/workspace/openpi \
    -v $CHECKPOINT_DIR:/workspace/checkpoint \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    $CONTAINER_IMAGE \
    /bin/bash

# Inside container, run:
# cd /workspace/openpi
# pip install -e .
# python scripts/benchmark_thor.py
```

---

## Alternative: TensorRT Edge-LLM Custom Model

If maximum performance is required, consider porting Pi0.5 to TensorRT Edge-LLM:

### Step 1: Clone Edge-LLM Repository

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
```

### Step 2: Add PaliGemma Support

Create custom model definition in Edge-LLM:
- `src/models/paligemma/` - Model architecture
- `python/export/paligemma.py` - Export script
- `configs/paligemma.yaml` - Model configuration

### Step 3: Build and Deploy

```bash
# Export to ONNX
python python/export/paligemma.py --checkpoint /path/to/pi05

# Build TRT engine
./build_engine --onnx models/paligemma.onnx --output engines/paligemma.engine

# Run inference
./examples/vlm_inference --engine engines/paligemma.engine
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| ONNX export fails for complex ops | High | Use modular export, skip problematic layers |
| TRT engine build OOM | Medium | Reduce batch size, use workspace limits |
| KV cache incompatibility | High | Implement custom KV cache management |
| Performance below target | Medium | Use fewer denoising steps (3-5) |

---

## Next Steps

1. **Verify JetPack Version**: Ensure JetPack 7.1 is installed
   ```bash
   cat /etc/nv_tegra_release
   ```

2. **Install Docker Runtime**: Ensure NVIDIA container runtime is configured
   ```bash
   sudo apt-get install nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Pull Container**: Start with jetson-containers for development

4. **Benchmark Inside Container**: Run existing benchmark scripts

5. **Iteratively Optimize**: Export components to TensorRT engines

---

## References

- [TensorRT Edge-LLM GitHub](https://github.com/NVIDIA/TensorRT-Edge-LLM)
- [TensorRT Edge-LLM Blog Post](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm)
- [Jetson Containers](https://github.com/dusty-nv/jetson-containers)
- [NGC L4T TensorRT](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
