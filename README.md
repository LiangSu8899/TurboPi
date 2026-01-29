# Turbo-Pi0.5: High-Performance VLA Model on NVIDIA Jetson Thor

**14.7x Speedup** | **20.6 Hz Inference** | **98% LIBERO Spatial Accuracy**

Turbo-Pi0.5 is an optimized implementation of the [Pi0.5 Vision-Language-Action (VLA) model](https://www.physicalintelligence.company/blog/pi0) for NVIDIA Jetson Thor platform, achieving real-time robot control at 20.6 Hz.

## Model

**HuggingFace**: [liangsu9988/Turbo-Pi0.5](https://huggingface.co/liangsu9988/Turbo-Pi0.5)

## Performance Summary

| Metric | Baseline | Turbo-Pi0.5 | Improvement |
|--------|----------|-------------|-------------|
| **Throughput** | 1.4 Hz | **20.6 Hz** | **14.7x** |
| **Latency** | 714 ms | **48.5 ms** | **14.7x** |
| **LIBERO Spatial** | - | **98%** | - |

### Optimization Techniques

1. **KV Cache** - Reuse prefix K,V across denoising steps
2. **TensorRT Export** - ONNX to TensorRT FP16 engines
3. **Dual Stream Pipeline** - Vision-Action parallel execution
4. **Reduced Denoising Steps** - 10 to 3 steps with minimal accuracy loss

## Quick Start

### Prerequisites

- NVIDIA Jetson Thor (JetPack 7.1+)
- Docker with NVIDIA runtime
- Python 3.10+

### Installation

```bash
# Clone repository
git clone https://github.com/LiangSu8899/TurboPi.git
cd TurboPi/openpi

# Install dependencies
pip install -e .

# Download model from HuggingFace
huggingface-cli download liangsu9988/Turbo-Pi0.5 --local-dir ~/.cache/openpi/checkpoints/pi05_libero
```

### Run Inference Server

```bash
# Start policy server (PyTorch)
python scripts/serve_policy.py \
    --env=LIBERO \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=~/.cache/openpi/checkpoints/pi05_libero
```

### Run LIBERO Benchmark

```bash
# Build Docker container for LIBERO evaluation
docker build -f Dockerfile.libero_eval -t libero_eval:latest .

# Run LIBERO Spatial benchmark (10 tasks x 10 trials)
bash scripts/run_libero_pytorch.sh libero_spatial 10
```

### Benchmark Performance

```bash
# Run TensorRT pipeline benchmark
python scripts/benchmark_trt_pipeline.py

# Run denoising steps comparison
python scripts/benchmark_denoising_steps.py
```

## LIBERO Benchmark Results

### libero_spatial (10 tasks, 10 trials each)

| Task | Success Rate |
|------|--------------|
| pick_up_black_bowl_between_plate_and_ramekin | 10/10 (100%) |
| pick_up_black_bowl_next_to_ramekin | 10/10 (100%) |
| pick_up_black_bowl_from_table_center | 10/10 (100%) |
| pick_up_black_bowl_next_to_cookie_box | 10/10 (100%) |
| pick_up_black_bowl_in_top_drawer | 10/10 (100%) |
| pick_up_black_bowl_on_ramekin | 8/10 (80%) |
| pick_up_black_bowl_on_cookie_box | 10/10 (100%) |
| pick_up_black_bowl_on_stove | 10/10 (100%) |
| pick_up_black_bowl_next_to_plate | 10/10 (100%) |
| pick_up_black_bowl_on_wooden_cabinet | 10/10 (100%) |
| **Total** | **98/100 (98%)** |

**Configuration**: PyTorch bfloat16, 10 denoising steps, KV Cache enabled

## TensorRT Performance

| Configuration | Latency | Throughput |
|---------------|---------|------------|
| 10 steps | 131.7 ms | 7.6 Hz |
| 5 steps | 72.3 ms | 13.8 Hz |
| **3 steps** | **48.5 ms** | **20.6 Hz** |
| 2 steps | 36.7 ms | 27.3 Hz |

### Individual Engine Performance

| Engine | Precision | Latency | Throughput |
|--------|-----------|---------|------------|
| SigLIP Vision Encoder | FP16 | 12.61 ms | 79.3 Hz |
| Gemma 300M Expert | FP16 | 14.40 ms | 69.4 Hz |
| Projections | FP16 | ~0.02 ms | ~860 Hz |

## Project Structure

```
TurboPi/
├── openpi/
│   ├── src/openpi/           # Core model implementation
│   │   ├── models_pytorch/   # PyTorch model definitions
│   │   └── inference/        # TensorRT inference pipeline
│   ├── scripts/              # Benchmark and utility scripts
│   ├── examples/             # LIBERO evaluation examples
│   └── docs/                 # Documentation
└── docs/                     # Additional documentation
```

## Version History

### v1.0.0 (2026-01-29)

**Initial Release**

- Pi0.5 VLA model optimized for NVIDIA Jetson Thor
- 14.7x speedup over baseline (1.4 Hz to 20.6 Hz)
- KV Cache implementation for efficient denoising
- TensorRT FP16 engine export
- LIBERO Spatial benchmark: 98% success rate
- Docker container for reproducible evaluation

## Citation

If you use this work, please cite:

```bibtex
@misc{turbopi05,
  title={Turbo-Pi0.5: High-Performance VLA Model on NVIDIA Jetson Thor},
  author={Liang Su},
  year={2026},
  url={https://github.com/LiangSu8899/TurboPi}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](openpi/LICENSE) file for details.

## Acknowledgments

- [Physical Intelligence](https://www.physicalintelligence.company/) for the original Pi0 model
- [OpenPi](https://github.com/Physical-Intelligence/openpi) for the open-source implementation
- NVIDIA for Jetson Thor platform and TensorRT tools
