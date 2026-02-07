#!/usr/bin/env python3
"""
export_onnx_components.py - Export Pi0.5 model components to ONNX.

This script exports individual components of the Pi0.5 model for TensorRT optimization:
1. Vision Encoder (SigLIP) - Run once per inference
2. Action Expert (Gemma 300M) - Run N times during denoising (high priority)
3. Projection layers - Small but frequently used

Usage:
    python export_onnx_components.py \
        --checkpoint_path ~/.cache/openpi/checkpoints/pi05_libero \
        --output_dir ./onnx_exports \
        --component all

Requirements:
    - PyTorch 2.x with CUDA
    - ONNX 1.14+
    - openpi package installed
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import time

import torch
import torch.nn as nn

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Apply transformers patches BEFORE importing any models
from openpi.models_pytorch.transformers_replace import ensure_patched
ensure_patched()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_onnx():
    """Check ONNX availability."""
    try:
        import onnx
        logger.info(f"ONNX version: {onnx.__version__}")
        return True
    except ImportError:
        logger.error("ONNX not available. Install with: pip install onnx")
        return False


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load Pi0.5 model from checkpoint."""
    import json
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
    from safetensors.torch import load_file

    logger.info(f"Loading model from: {checkpoint_path}")
    start = time.time()

    model_path = Path(checkpoint_path).expanduser()
    weights_path = model_path / "model.safetensors"
    config_path = model_path / "config.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    # Load config
    with open(config_path) as f:
        model_config = json.load(f)

    logger.info(f"Model variant: {model_config.get('paligemma_variant', 'gemma_2b')}")
    logger.info(f"Action expert: {model_config.get('action_expert_variant', 'gemma_300m')}")

    # Create model configuration
    pi0_config = Pi0Config(
        paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
        action_dim=model_config.get("action_dim", 32),
        action_horizon=model_config.get("action_horizon", 50),
        max_token_len=model_config.get("tokenizer_max_length", 200),
        pi05=True,
        dtype="bfloat16",
    )

    # Create model
    model = PI0Pytorch(pi0_config)

    # Load weights
    state_dict = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")

    model = model.to(device=device)
    model.eval()

    logger.info(f"Model loaded in {time.time() - start:.1f}s")
    return model


class VisionEncoderWrapper(nn.Module):
    """Wrapper for SigLIP vision encoder for clean ONNX export."""

    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower

    def forward(self, pixel_values):
        # SigLIP forward pass
        outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        # Return the last hidden state
        return outputs.last_hidden_state


class ActionExpertWrapper(nn.Module):
    """Wrapper for Gemma 300M action expert for clean ONNX export."""

    def __init__(self, expert_model, action_in_proj, action_out_proj, time_mlp_in, time_mlp_out):
        super().__init__()
        self.expert = expert_model
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj
        self.time_mlp_in = time_mlp_in
        self.time_mlp_out = time_mlp_out

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim] - Input embeddings
            attention_mask: [batch, 1, seq_len, seq_len] - 4D attention mask
        """
        outputs = self.expert(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.last_hidden_state


class DenoisingStepModule(nn.Module):
    """Single denoising step for ONNX export - combines expert + projections."""

    def __init__(self, pi0_model):
        super().__init__()
        self.action_in_proj = pi0_model.action_in_proj
        self.action_out_proj = pi0_model.action_out_proj
        self.time_mlp_in = pi0_model.time_mlp_in
        self.time_mlp_out = pi0_model.time_mlp_out
        self.expert = pi0_model.paligemma_with_expert.gemma_expert

    def forward(self, noisy_actions, time_emb, prefix_embs, attention_mask):
        """
        Single denoising step.

        Args:
            noisy_actions: [batch, action_horizon, 32] - Current noisy actions
            time_emb: [batch, hidden_dim] - Time embedding
            prefix_embs: [batch, prefix_len, hidden_dim] - Cached prefix embeddings
            attention_mask: [batch, 1, total_len, total_len] - Attention mask

        Returns:
            velocity: [batch, action_horizon, 32] - Predicted velocity
        """
        batch_size = noisy_actions.shape[0]

        # Project actions to hidden dim
        action_embs = self.action_in_proj(noisy_actions)

        # Add time embedding via MLP
        time_cond = self.time_mlp_out(torch.nn.functional.silu(self.time_mlp_in(time_emb)))
        action_embs = action_embs + time_cond.unsqueeze(1)

        # Concatenate with prefix
        full_embs = torch.cat([prefix_embs, action_embs], dim=1)

        # Run expert
        expert_out = self.expert(
            inputs_embeds=full_embs,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

        # Extract action part and project back
        action_len = noisy_actions.shape[1]
        action_hidden = expert_out[:, -action_len:, :]
        velocity = self.action_out_proj(action_hidden)

        return velocity


def export_vision_encoder(model, output_dir: Path, device: str = "cuda"):
    """Export SigLIP vision encoder to ONNX."""
    logger.info("Exporting Vision Encoder (SigLIP)...")

    vision_tower = model.paligemma_with_expert.paligemma.vision_tower
    wrapper = VisionEncoderWrapper(vision_tower).to(device)
    wrapper.eval()

    # Get image size from config (default SigLIP-SO400M uses 224x224)
    image_size = 224
    batch_size = 1

    # Create dummy input
    dummy_images = torch.randn(
        batch_size, 3, image_size, image_size,
        device=device, dtype=torch.float32
    )

    onnx_path = output_dir / "siglip_vision_encoder.onnx"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_images,
            str(onnx_path),
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["vision_embeddings"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "vision_embeddings": {0: "batch"},
            },
            do_constant_folding=True,
        )

    # Verify
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    file_size = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"Vision encoder exported: {onnx_path} ({file_size:.1f} MB)")
    return onnx_path


def export_action_expert(model, output_dir: Path, device: str = "cuda"):
    """Export Gemma 300M action expert to ONNX."""
    logger.info("Exporting Action Expert (Gemma 300M)...")

    expert = model.paligemma_with_expert.gemma_expert
    expert.eval()

    # Get dimensions from config
    hidden_size = expert.config.hidden_size
    batch_size = 1
    seq_len = 256  # Typical sequence length

    # Create dummy inputs - use float32 for better ONNX compatibility
    dummy_hidden = torch.randn(
        batch_size, seq_len, hidden_size,
        device=device, dtype=torch.float32
    )

    onnx_path = output_dir / "gemma_300m_expert.onnx"

    # Use underlying GemmaModel for clean export with adaRMS support
    class GemmaModelWrapper(nn.Module):
        """Wrapper for clean ONNX export of GemmaModel with adaptive RMSNorm support."""

        def __init__(self, gemma_model):
            super().__init__()
            self.model = gemma_model.model
            self.hidden_size = gemma_model.config.hidden_size
            self.use_adarms = getattr(gemma_model.config, 'use_adarms', False)
            self.adarms_cond_dim = getattr(gemma_model.config, 'adarms_cond_dim', None)
            # Convert to float32 for ONNX export
            self.model = self.model.float()
            logger.info(f"Wrapper: use_adarms={self.use_adarms}, adarms_cond_dim={self.adarms_cond_dim}")

        def _gated_residual(self, x, y, gate):
            """Gated residual connection (no sigmoid, matching JAX)."""
            if gate is None:
                return x + y
            return x + y * gate

        def forward(self, hidden_states, adarms_cond=None):
            """
            Forward pass with optional adaptive RMSNorm conditioning.

            Args:
                hidden_states: [batch, seq_len, hidden_size] - Input embeddings
                adarms_cond: [batch, hidden_size] - Optional conditioning for adaptive RMSNorm
            """
            # Normalize embeddings (same as GemmaModel forward)
            hidden_states = hidden_states * (self.hidden_size ** 0.5)

            batch_size, seq_len, _ = hidden_states.shape

            # Create position ids
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

            # Create causal attention mask (4D: batch, 1, seq, seq)
            attention_mask = torch.zeros(
                batch_size, 1, seq_len, seq_len,
                device=hidden_states.device, dtype=hidden_states.dtype
            )

            # Get position embeddings
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

            # Process through layers
            for layer in self.model.layers:
                residual = hidden_states

                # Input layernorm with optional adaRMS conditioning
                if self.use_adarms and adarms_cond is not None:
                    norm_output = layer.input_layernorm(hidden_states, adarms_cond)
                else:
                    norm_output = layer.input_layernorm(hidden_states)

                if isinstance(norm_output, tuple):
                    hidden_states, gate = norm_output
                else:
                    hidden_states, gate = norm_output, None

                # Self attention with attention mask
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

                hidden_states = self._gated_residual(residual, hidden_states, gate)

                # MLP
                residual = hidden_states

                if self.use_adarms and adarms_cond is not None:
                    norm_output = layer.post_attention_layernorm(hidden_states, adarms_cond)
                else:
                    norm_output = layer.post_attention_layernorm(hidden_states)

                if isinstance(norm_output, tuple):
                    hidden_states, gate = norm_output
                else:
                    hidden_states, gate = norm_output, None

                hidden_states = layer.mlp(hidden_states)
                hidden_states = self._gated_residual(residual, hidden_states, gate)

            # Final norm
            if self.use_adarms and adarms_cond is not None:
                norm_output = self.model.norm(hidden_states, adarms_cond)
            else:
                norm_output = self.model.norm(hidden_states)

            if isinstance(norm_output, tuple):
                hidden_states = norm_output[0]
            else:
                hidden_states = norm_output

            return hidden_states

    # Export with tracing using legacy exporter
    with torch.no_grad():
        try:
            wrapper = GemmaModelWrapper(expert).to(device)
            wrapper.eval()

            # Check if model uses adaptive RMSNorm
            use_adarms = wrapper.use_adarms
            adarms_cond_dim = wrapper.adarms_cond_dim

            if use_adarms and adarms_cond_dim is not None:
                logger.info(f"Exporting with adaRMS support (cond_dim={adarms_cond_dim})")
                # Create dummy adarms conditioning input
                dummy_adarms_cond = torch.randn(
                    batch_size, adarms_cond_dim,
                    device=device, dtype=torch.float32
                )
                onnx_path = output_dir / "gemma_300m_expert_adarms.onnx"

                # Use legacy ONNX exporter with both inputs
                torch.onnx.export(
                    wrapper,
                    (dummy_hidden, dummy_adarms_cond),
                    str(onnx_path),
                    opset_version=17,
                    input_names=["hidden_states", "adarms_cond"],
                    output_names=["output"],
                    dynamic_axes={
                        "hidden_states": {0: "batch", 1: "seq_len"},
                        "adarms_cond": {0: "batch"},
                        "output": {0: "batch", 1: "seq_len"},
                    },
                    do_constant_folding=True,
                    export_params=True,
                    dynamo=False,
                )
            else:
                logger.info("Exporting without adaRMS (standard mode)")
                # Use legacy ONNX exporter (TorchScript-based)
                torch.onnx.export(
                    wrapper,
                    dummy_hidden,
                    str(onnx_path),
                    opset_version=17,
                    input_names=["hidden_states"],
                    output_names=["output"],
                    dynamic_axes={
                        "hidden_states": {0: "batch", 1: "seq_len"},
                        "output": {0: "batch", 1: "seq_len"},
                    },
                    do_constant_folding=True,
                    export_params=True,
                    dynamo=False,
                )

            # Verify
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            file_size = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"Action expert exported: {onnx_path} ({file_size:.1f} MB)")
            return onnx_path

        except Exception as e:
            logger.error(f"Action expert export failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def export_projections(model, output_dir: Path, device: str = "cuda"):
    """Export projection layers to ONNX."""
    logger.info("Exporting Projection Layers...")

    # Action input projection
    action_in = model.action_in_proj.to(device)
    dummy_action = torch.randn(1, 50, 32, device=device, dtype=torch.float32)

    action_in_path = output_dir / "action_in_proj.onnx"
    with torch.no_grad():
        torch.onnx.export(
            action_in,
            dummy_action,
            str(action_in_path),
            opset_version=17,
            input_names=["actions"],
            output_names=["projected"],
            dynamic_axes={
                "actions": {0: "batch", 1: "action_len"},
                "projected": {0: "batch", 1: "action_len"},
            },
        )

    # Action output projection
    action_out = model.action_out_proj.to(device)
    hidden_size = action_out.in_features
    dummy_hidden = torch.randn(1, 50, hidden_size, device=device, dtype=torch.float32)

    action_out_path = output_dir / "action_out_proj.onnx"
    with torch.no_grad():
        torch.onnx.export(
            action_out,
            dummy_hidden,
            str(action_out_path),
            opset_version=17,
            input_names=["hidden"],
            output_names=["actions"],
            dynamic_axes={
                "hidden": {0: "batch", 1: "seq_len"},
                "actions": {0: "batch", 1: "seq_len"},
            },
        )

    # Time MLP
    class TimeMLP(nn.Module):
        def __init__(self, mlp_in, mlp_out):
            super().__init__()
            self.mlp_in = mlp_in
            self.mlp_out = mlp_out

        def forward(self, time_emb):
            return self.mlp_out(torch.nn.functional.silu(self.mlp_in(time_emb)))

    time_mlp = TimeMLP(model.time_mlp_in, model.time_mlp_out).to(device)
    dummy_time = torch.randn(1, hidden_size, device=device, dtype=torch.float32)

    time_mlp_path = output_dir / "time_mlp.onnx"
    with torch.no_grad():
        torch.onnx.export(
            time_mlp,
            dummy_time,
            str(time_mlp_path),
            opset_version=17,
            input_names=["time_embedding"],
            output_names=["conditioned"],
            dynamic_axes={
                "time_embedding": {0: "batch"},
                "conditioned": {0: "batch"},
            },
        )

    logger.info(f"Projection layers exported to {output_dir}")
    return [action_in_path, action_out_path, time_mlp_path]


def export_denoising_step(model, output_dir: Path, device: str = "cuda"):
    """Export complete denoising step module to ONNX."""
    logger.info("Exporting Denoising Step Module...")

    denoising_module = DenoisingStepModule(model).to(device)
    denoising_module.eval()

    # Get dimensions
    hidden_size = model.paligemma_with_expert.gemma_expert.config.hidden_size
    action_horizon = 50  # Typical action horizon
    prefix_len = 512  # Typical prefix length
    total_len = prefix_len + action_horizon
    batch_size = 1

    # Dummy inputs
    dummy_noisy_actions = torch.randn(batch_size, action_horizon, 32, device=device, dtype=torch.float32)
    dummy_time_emb = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
    dummy_prefix_embs = torch.randn(batch_size, prefix_len, hidden_size, device=device, dtype=torch.bfloat16)
    dummy_attention_mask = torch.zeros(batch_size, 1, total_len, total_len, device=device, dtype=torch.bfloat16)

    onnx_path = output_dir / "denoising_step.onnx"

    with torch.no_grad():
        try:
            torch.onnx.export(
                denoising_module,
                (dummy_noisy_actions, dummy_time_emb, dummy_prefix_embs, dummy_attention_mask),
                str(onnx_path),
                opset_version=17,
                input_names=["noisy_actions", "time_embedding", "prefix_embeddings", "attention_mask"],
                output_names=["velocity"],
                dynamic_axes={
                    "noisy_actions": {0: "batch"},
                    "time_embedding": {0: "batch"},
                    "prefix_embeddings": {0: "batch", 1: "prefix_len"},
                    "attention_mask": {0: "batch", 2: "total_len", 3: "total_len"},
                    "velocity": {0: "batch"},
                },
                do_constant_folding=True,
            )

            # Verify
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            file_size = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"Denoising step exported: {onnx_path} ({file_size:.1f} MB)")
            return onnx_path

        except Exception as e:
            logger.error(f"Failed to export denoising step: {e}")
            logger.info("This is expected for complex models. Use component exports instead.")
            return None


def main():
    parser = argparse.ArgumentParser(description="Export Pi0.5 components to ONNX")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero"),
        help="Path to Pi0.5 checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./onnx_exports",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["vision", "expert", "projections", "denoising", "all"],
        default="all",
        help="Component to export",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for export",
    )
    args = parser.parse_args()

    if not check_onnx():
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint_path, args.device)

    exported_files = []

    # Export requested components
    if args.component in ["vision", "all"]:
        try:
            path = export_vision_encoder(model, output_dir, args.device)
            exported_files.append(path)
        except Exception as e:
            logger.error(f"Vision encoder export failed: {e}")

    if args.component in ["expert", "all"]:
        try:
            path = export_action_expert(model, output_dir, args.device)
            exported_files.append(path)
        except Exception as e:
            logger.error(f"Action expert export failed: {e}")

    if args.component in ["projections", "all"]:
        try:
            paths = export_projections(model, output_dir, args.device)
            exported_files.extend(paths)
        except Exception as e:
            logger.error(f"Projections export failed: {e}")

    if args.component in ["denoising", "all"]:
        try:
            path = export_denoising_step(model, output_dir, args.device)
            if path:
                exported_files.append(path)
        except Exception as e:
            logger.error(f"Denoising step export failed: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ONNX Export Summary")
    logger.info("=" * 60)
    for f in exported_files:
        if f and f.exists():
            size = f.stat().st_size / (1024 * 1024)
            logger.info(f"  {f.name}: {size:.1f} MB")

    logger.info("\nNext steps:")
    logger.info("  1. Build TensorRT engines with trtexec:")
    logger.info(f"     /usr/src/tensorrt/bin/trtexec --onnx={output_dir}/gemma_300m_expert.onnx --saveEngine={output_dir}/gemma_300m_expert.engine --fp16")
    logger.info("  2. Benchmark with TensorRT runtime")


if __name__ == "__main__":
    main()
