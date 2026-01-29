# Turbo-Pi0.5

**14.7x faster Pi0.5 VLA model optimized for NVIDIA Jetson Thor**

| Metric | Before | After |
|--------|--------|-------|
| Inference Speed | 1.4 Hz | **20.6 Hz** |
| Latency | 714 ms | **48.5 ms** |
| LIBERO Spatial | - | **98%** |

## What is this?

Turbo-Pi0.5 is an optimized version of [Physical Intelligence's Pi0.5](https://www.physicalintelligence.company/blog/pi0) Vision-Language-Action model, designed for real-time robot control on edge devices.

**Key optimizations:**
- KV Cache for efficient denoising
- TensorRT FP16 acceleration
- Reduced denoising steps (10 → 3) with <1% accuracy loss

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull and run
docker pull liangsu9988/turbopi:latest
docker run --runtime=nvidia -it --network=host liangsu9988/turbopi:latest

# Inside container
python scripts/serve_policy.py --env=LIBERO --port=8000 \
    policy:checkpoint --policy.config=pi05_libero
```

### Option 2: Manual Installation

```bash
# Clone
git clone https://github.com/LiangSu8899/TurboPi.git
cd TurboPi/openpi

# Install
pip install -e .

# Download model
huggingface-cli download liangsu9988/Turbo-Pi0.5 \
    --local-dir ~/.cache/openpi/checkpoints/pi05_libero

# Run server
python scripts/serve_policy.py --env=LIBERO --port=8000 \
    policy:checkpoint --policy.config=pi05_libero
```

### Client Usage

```python
from openpi_client import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="localhost", port=8000)
action = policy.get_action({
    "images": {
        "cam_high": high_img,
        "cam_left_wrist": left_img,
        "cam_right_wrist": right_img
    },
    "state": robot_state,
    "prompt": "pick up the black bowl"
})
```

## Model

**HuggingFace:** [liangsu9988/Turbo-Pi0.5](https://huggingface.co/liangsu9988/Turbo-Pi0.5)

**Architecture:**
- Vision: SigLIP-SO400M
- Language: Gemma 2B
- Action: Gemma 300M Expert

## Benchmark Results

### LIBERO Spatial (10 tasks × 10 trials)

| Task | Success |
|------|---------|
| pick_up_black_bowl_between_plate_and_ramekin | 100% |
| pick_up_black_bowl_next_to_ramekin | 100% |
| pick_up_black_bowl_from_table_center | 100% |
| pick_up_black_bowl_next_to_cookie_box | 100% |
| pick_up_black_bowl_in_top_drawer | 100% |
| pick_up_black_bowl_on_ramekin | 80% |
| pick_up_black_bowl_on_cookie_box | 100% |
| pick_up_black_bowl_on_stove | 100% |
| pick_up_black_bowl_next_to_plate | 100% |
| pick_up_black_bowl_on_wooden_cabinet | 100% |
| **Total** | **98%** |

### Inference Speed (Jetson Thor)

| Denoising Steps | Latency | Throughput |
|-----------------|---------|------------|
| 10 | 131.7 ms | 7.6 Hz |
| 5 | 72.3 ms | 13.8 Hz |
| **3 (default)** | **48.5 ms** | **20.6 Hz** |
| 2 | 36.7 ms | 27.3 Hz |

## Hardware Requirements

- **Recommended:** NVIDIA Jetson Thor (JetPack 7.1+)
- **Minimum:** GPU with 8GB+ VRAM, CUDA 12.0+

## Version History

### v1.0.0 (2026-01-29)
- Initial release
- 14.7x speedup over baseline
- LIBERO Spatial: 98% success rate
- KV Cache + TensorRT optimization

## License

Apache License 2.0

## Acknowledgments

Based on [OpenPi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence.
