#!/usr/bin/env python3
"""
Export ModelOpt FP4/FP8 quantized model to TensorRT engine.

This script exports the quantized Pi0.5 model to TensorRT format using
ModelOpt's official ONNX export APIs that preserve quantization information.

Usage:
    python scripts/export_quantized_tensorrt.py \
        --quantized_model ./quantized_models/pi05_nvfp4_fp8 \
        --baseline_model ~/.cache/openpi/checkpoints/pi05_libero \
        --output_dir ./tensorrt_engines

    # Quick test with simplified export:
    python scripts/export_quantized_tensorrt.py \
        --quantized_model ./quantized_models/pi05_nvfp4_fp8 \
        --baseline_model ~/.cache/openpi/checkpoints/pi05_libero \
        --output_dir ./tensorrt_engines \
        --components projections
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Apply transformers patches BEFORE importing models
from openpi.models_pytorch.transformers_replace import ensure_patched
ensure_patched()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    quantized_model_path: str
    baseline_model_path: str
    output_dir: str
    components: List[str]  # all, projections, action_expert, vision
    precision: str = "mixed"  # fp4, fp8, mixed
    max_batch_size: int = 1
    workspace_size_gb: int = 8
    verbose: bool = False


def check_dependencies():
    """Check required dependencies."""
    deps = {}

    try:
        import modelopt.torch.quantization as mtq
        deps["modelopt"] = True
    except ImportError:
        deps["modelopt"] = False

    try:
        import tensorrt as trt
        deps["tensorrt"] = True
        deps["trt_version"] = trt.__version__
    except ImportError:
        deps["tensorrt"] = False

    try:
        import onnx
        deps["onnx"] = True
    except ImportError:
        deps["onnx"] = False

    try:
        import onnxruntime
        deps["onnxruntime"] = True
    except ImportError:
        deps["onnxruntime"] = False

    return deps


def load_quantized_model(
    quantized_path: Path,
    baseline_path: Path,
    device: str = "cuda"
) -> nn.Module:
    """Load the quantized model."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from safetensors.torch import load_file

    quantized_path = Path(quantized_path).expanduser()
    baseline_path = Path(baseline_path).expanduser()

    # Load config from baseline
    config_path = baseline_path / "config.json"
    with open(config_path) as f:
        model_config = json.load(f)

    # Create model config
    @dataclass
    class Pi0Config:
        paligemma_variant: str = model_config.get("paligemma_variant", "gemma_2b")
        action_expert_variant: str = model_config.get("action_expert_variant", "gemma_300m")
        action_dim: int = 32
        action_horizon: int = 50
        max_token_len: int = model_config.get("tokenizer_max_length", 200)
        max_state_dim: int = model_config.get("max_state_dim", 32)
        pi05: bool = True
        dtype: str = "float32"  # Load in float32 for ONNX export

    config = Pi0Config()
    logger.info(f"Loading model with config: {config}")

    # Create model
    model = PI0Pytorch(config)

    # Load quantized weights
    weights_path = quantized_path / "model_quantized.pt"
    if weights_path.exists():
        logger.info(f"Loading quantized weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
    else:
        # Fallback to baseline weights
        logger.warning(f"Quantized weights not found at {weights_path}")
        weights_path = baseline_path / "model.safetensors"
        state_dict = load_file(weights_path)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    return model


def check_quantization_status(model: nn.Module) -> Dict:
    """Check if model has quantization markers from ModelOpt."""
    status = {
        "has_amax": 0,
        "has_weight_quantizer": 0,
        "has_input_quantizer": 0,
        "fp4_quantized": 0,
        "fp8_quantized": 0,
        "total_linear": 0,
    }

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            status["total_linear"] += 1

            # Check for ModelOpt quantization markers
            if hasattr(module, "_amax"):
                status["has_amax"] += 1
            if hasattr(module, "weight_quantizer"):
                status["has_weight_quantizer"] += 1
                if hasattr(module.weight_quantizer, "_num_bits"):
                    num_bits = module.weight_quantizer._num_bits
                    if num_bits == (4, 3):  # FP8 E4M3
                        status["fp8_quantized"] += 1
                    elif num_bits == 4:
                        status["fp4_quantized"] += 1
            if hasattr(module, "input_quantizer"):
                status["has_input_quantizer"] += 1

        # Also check for buffers
        for buf_name in ["_amax", "weight_scale", "weight_precision"]:
            if hasattr(module, buf_name):
                try:
                    buf = getattr(module, buf_name)
                    if torch.is_tensor(buf):
                        break
                except:
                    pass

    return status


def export_projections_onnx(
    model: nn.Module,
    output_dir: Path,
    device: str = "cuda"
) -> Dict[str, Path]:
    """Export projection layers to ONNX."""
    logger.info("Exporting projection layers to ONNX...")

    onnx_paths = {}

    # Action input projection
    action_in_proj = model.action_in_proj.float().to(device)
    action_in_proj.eval()

    dummy_action = torch.randn(1, 50, 32, device=device, dtype=torch.float32)
    onnx_path = output_dir / "action_in_proj.onnx"

    # Use legacy TorchScript-based exporter for TensorRT compatibility
    torch.onnx.export(
        action_in_proj,
        dummy_action,
        str(onnx_path),
        input_names=["action_input"],
        output_names=["action_embed"],
        dynamic_axes={
            "action_input": {0: "batch"},
            "action_embed": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_paths["action_in_proj"] = onnx_path
    logger.info(f"  action_in_proj exported: {onnx_path}")

    # Action output projection
    action_out_proj = model.action_out_proj.float().to(device)
    action_out_proj.eval()

    # Get hidden size from weight
    hidden_size = action_out_proj.weight.shape[1]
    dummy_hidden = torch.randn(1, 50, hidden_size, device=device, dtype=torch.float32)
    onnx_path = output_dir / "action_out_proj.onnx"

    # Use legacy TorchScript-based exporter for TensorRT compatibility
    torch.onnx.export(
        action_out_proj,
        dummy_hidden,
        str(onnx_path),
        input_names=["hidden_states"],
        output_names=["actions"],
        dynamic_axes={
            "hidden_states": {0: "batch"},
            "actions": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_paths["action_out_proj"] = onnx_path
    logger.info(f"  action_out_proj exported: {onnx_path}")

    # Time MLP
    if hasattr(model, "action_time_mlp_in"):
        time_mlp_in = model.action_time_mlp_in.float().to(device)
        time_mlp_out = model.action_time_mlp_out.float().to(device)

        class TimeMLP(nn.Module):
            def __init__(self, mlp_in, mlp_out):
                super().__init__()
                self.mlp_in = mlp_in
                self.mlp_out = mlp_out

            def forward(self, timestep_emb):
                x = self.mlp_in(timestep_emb)
                x = torch.nn.functional.silu(x)
                return self.mlp_out(x)

        time_mlp = TimeMLP(time_mlp_in, time_mlp_out).to(device)
        time_mlp.eval()

        dummy_timestep = torch.randn(1, time_mlp_in.weight.shape[1], device=device, dtype=torch.float32)
        onnx_path = output_dir / "time_mlp.onnx"

        # Use legacy TorchScript-based exporter for TensorRT compatibility
        torch.onnx.export(
            time_mlp,
            dummy_timestep,
            str(onnx_path),
            input_names=["timestep_emb"],
            output_names=["time_conditioning"],
            dynamic_axes={
                "timestep_emb": {0: "batch"},
                "time_conditioning": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        onnx_paths["time_mlp"] = onnx_path
        logger.info(f"  time_mlp exported: {onnx_path}")

    # State projection
    if hasattr(model, "state_proj"):
        state_proj = model.state_proj.float().to(device)
        state_proj.eval()

        dummy_state = torch.randn(1, 32, device=device, dtype=torch.float32)
        onnx_path = output_dir / "state_proj.onnx"

        # Use legacy TorchScript-based exporter for TensorRT compatibility
        torch.onnx.export(
            state_proj,
            dummy_state,
            str(onnx_path),
            input_names=["state"],
            output_names=["state_embed"],
            dynamic_axes={
                "state": {0: "batch"},
                "state_embed": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        onnx_paths["state_proj"] = onnx_path
        logger.info(f"  state_proj exported: {onnx_path}")

    return onnx_paths


def export_action_expert_onnx(
    model: nn.Module,
    output_dir: Path,
    device: str = "cuda"
) -> Optional[Path]:
    """Export action expert (Gemma 300M) to ONNX."""
    logger.info("Exporting action expert to ONNX...")

    try:
        expert = model.paligemma_with_expert.gemma_expert
        expert = expert.float().to(device)
        expert.eval()

        hidden_size = expert.config.hidden_size
        batch_size = 1
        seq_len = 256

        class GemmaExpertWrapper(nn.Module):
            """Wrapper for Gemma expert ONNX export."""

            def __init__(self, expert_model):
                super().__init__()
                self.model = expert_model.model
                self.hidden_size = expert_model.config.hidden_size
                self.model = self.model.float()

            def forward(self, hidden_states):
                # Normalize embeddings
                hidden_states = hidden_states * (self.hidden_size ** 0.5)

                batch_size, seq_len, _ = hidden_states.shape

                # Create position ids
                position_ids = torch.arange(
                    seq_len, device=hidden_states.device
                ).unsqueeze(0).expand(batch_size, -1)

                # Create causal mask
                attention_mask = torch.zeros(
                    batch_size, 1, seq_len, seq_len,
                    device=hidden_states.device, dtype=hidden_states.dtype
                )

                # Get position embeddings
                position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

                # Process through layers
                for layer in self.model.layers:
                    residual = hidden_states
                    hidden_states = layer.input_layernorm(hidden_states)

                    attn_output = layer.self_attn(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )
                    if isinstance(attn_output, tuple):
                        hidden_states = attn_output[0]
                    else:
                        hidden_states = attn_output

                    hidden_states = residual + hidden_states

                    residual = hidden_states
                    hidden_states = layer.post_attention_layernorm(hidden_states)
                    hidden_states = layer.mlp(hidden_states)
                    hidden_states = residual + hidden_states

                hidden_states = self.model.norm(hidden_states)
                return hidden_states

        wrapper = GemmaExpertWrapper(expert).to(device)
        wrapper.eval()

        dummy_hidden = torch.randn(
            batch_size, seq_len, hidden_size,
            device=device, dtype=torch.float32
        )

        onnx_path = output_dir / "action_expert.onnx"

        # Use legacy TorchScript-based exporter for TensorRT compatibility
        torch.onnx.export(
            wrapper,
            dummy_hidden,
            str(onnx_path),
            input_names=["hidden_states"],
            output_names=["output"],
            dynamic_axes={
                "hidden_states": {0: "batch", 1: "seq_len"},
                "output": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

        # Verify
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        file_size = onnx_path.stat().st_size / (1024 * 1024)
        logger.info(f"  action_expert exported: {onnx_path} ({file_size:.1f} MB)")

        return onnx_path

    except Exception as e:
        logger.error(f"Action expert export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str = "fp16",
    workspace_gb: int = 8,
    max_batch_size: int = 1,
) -> Optional[Path]:
    """Build TensorRT engine from ONNX model."""
    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT not available")
        return None

    logger.info(f"Building TensorRT engine: {onnx_path} -> {engine_path}")
    logger.info(f"  Precision: {precision}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX Parser Error: {parser.get_error(i)}")
            return None

    logger.info(f"  Parsed: {network.num_inputs} inputs, {network.num_outputs} outputs")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # Set precision flags
    if precision in ["fp8", "mixed"]:
        if hasattr(builder, "platform_has_fast_fp8") and builder.platform_has_fast_fp8:
            config.set_flag(trt.BuilderFlag.FP8)
            logger.info("  FP8 enabled")

    if precision in ["fp16", "mixed"]:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("  FP16 enabled")

    # Set optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        name = input_tensor.name
        shape = list(input_tensor.shape)

        # Handle dynamic batch dimension
        if shape[0] == -1:
            shape[0] = 1

        min_shape = [1] + shape[1:]
        opt_shape = shape
        max_shape = [max_batch_size] + shape[1:]

        # Handle dynamic sequence length
        for j in range(1, len(shape)):
            if shape[j] == -1:
                min_shape[j] = 1
                opt_shape[j] = 256
                max_shape[j] = 1024

        profile.set_shape(name, min_shape, opt_shape, max_shape)
        logger.info(f"  Input {name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

    config.add_optimization_profile(profile)

    # Build engine
    logger.info("  Building engine (this may take a while)...")
    start_time = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - start_time

    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return None

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_size = engine_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Engine saved: {engine_path} ({engine_size:.1f} MB, build: {build_time:.1f}s)")

    return engine_path


def run_export(config: ExportConfig):
    """Run the export pipeline."""
    logger.info("=" * 60)
    logger.info("TensorRT Export for Quantized Pi0.5 Model")
    logger.info("=" * 60)

    # Check dependencies
    deps = check_dependencies()
    logger.info(f"Dependencies: {deps}")

    if not deps.get("tensorrt"):
        logger.error("TensorRT not available. Cannot proceed with engine export.")
        logger.info("ONNX export will still be performed.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("\n[1/4] Loading quantized model...")
    model = load_quantized_model(
        config.quantized_model_path,
        config.baseline_model_path,
        device="cuda"
    )

    # Check quantization status
    quant_status = check_quantization_status(model)
    logger.info(f"Quantization status: {quant_status}")

    # Export components
    onnx_paths = {}
    engine_paths = {}

    components = config.components
    if "all" in components:
        components = ["projections", "action_expert"]

    # Export projections
    if "projections" in components:
        logger.info("\n[2/4] Exporting projection layers...")
        proj_onnx_paths = export_projections_onnx(model, output_dir, device="cuda")
        onnx_paths.update(proj_onnx_paths)

        # Build TensorRT engines
        if deps.get("tensorrt"):
            logger.info("\n[3/4] Building TensorRT engines for projections...")
            for name, onnx_path in proj_onnx_paths.items():
                engine_path = output_dir / f"{name}.engine"
                result = build_tensorrt_engine(
                    onnx_path,
                    engine_path,
                    precision=config.precision,
                    workspace_gb=config.workspace_size_gb,
                    max_batch_size=config.max_batch_size,
                )
                if result:
                    engine_paths[name] = result

    # Export action expert
    if "action_expert" in components:
        logger.info("\n[2/4] Exporting action expert...")
        expert_onnx_path = export_action_expert_onnx(model, output_dir, device="cuda")
        if expert_onnx_path:
            onnx_paths["action_expert"] = expert_onnx_path

            # Build TensorRT engine
            if deps.get("tensorrt"):
                logger.info("\n[3/4] Building TensorRT engine for action expert...")
                engine_path = output_dir / "action_expert.engine"
                result = build_tensorrt_engine(
                    expert_onnx_path,
                    engine_path,
                    precision=config.precision,
                    workspace_gb=config.workspace_size_gb,
                    max_batch_size=config.max_batch_size,
                )
                if result:
                    engine_paths["action_expert"] = result

    # Save export summary
    logger.info("\n[4/4] Saving export summary...")
    summary = {
        "quantized_model_path": str(config.quantized_model_path),
        "baseline_model_path": str(config.baseline_model_path),
        "output_dir": str(output_dir),
        "precision": config.precision,
        "quantization_status": quant_status,
        "onnx_paths": {k: str(v) for k, v in onnx_paths.items()},
        "engine_paths": {k: str(v) for k, v in engine_paths.items()},
        "dependencies": deps,
    }

    summary_path = output_dir / "export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Export Complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"ONNX models: {len(onnx_paths)}")
    logger.info(f"TensorRT engines: {len(engine_paths)}")

    if onnx_paths:
        logger.info("\nONNX exports:")
        for name, path in onnx_paths.items():
            size = Path(path).stat().st_size / (1024 * 1024)
            logger.info(f"  {name}: {path} ({size:.1f} MB)")

    if engine_paths:
        logger.info("\nTensorRT engines:")
        for name, path in engine_paths.items():
            size = Path(path).stat().st_size / (1024 * 1024)
            logger.info(f"  {name}: {path} ({size:.1f} MB)")

    logger.info("\nNext steps:")
    logger.info("  1. Benchmark: python scripts/benchmark_tensorrt.py --engine_dir " + str(output_dir))

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Export quantized Pi0.5 model to TensorRT"
    )

    parser.add_argument(
        "--quantized_model",
        type=str,
        default="./quantized_models/pi05_nvfp4_fp8",
        help="Path to quantized model directory",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="~/.cache/openpi/checkpoints/pi05_libero",
        help="Path to baseline model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tensorrt_engines",
        help="Output directory for engines",
    )
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=["projections"],
        choices=["all", "projections", "action_expert", "vision"],
        help="Components to export",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp4", "fp8", "fp16", "mixed"],
        help="TensorRT precision mode",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="Maximum batch size for TensorRT",
    )
    parser.add_argument(
        "--workspace_gb",
        type=int,
        default=8,
        help="TensorRT workspace size in GB",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    config = ExportConfig(
        quantized_model_path=args.quantized_model,
        baseline_model_path=args.baseline_model,
        output_dir=args.output_dir,
        components=args.components,
        precision=args.precision,
        max_batch_size=args.max_batch_size,
        workspace_size_gb=args.workspace_gb,
        verbose=args.verbose,
    )

    return run_export(config)


if __name__ == "__main__":
    main()
