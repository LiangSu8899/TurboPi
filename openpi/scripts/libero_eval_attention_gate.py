#!/usr/bin/env python3
"""
LIBERO Evaluation with Pro Attention-Weighted Drift Gate.

This script implements the three-layer defense KV reuse strategy:
1. State Guard - Fast joint change detection
2. Anti-Noise - Downsampled pixel diff
3. Dilated Attention - Attention-weighted drift with dilation

Key difference from baseline:
- When gate allows reuse, skip Vision TRT + KV Cache computation
- Directly use previous frame's KV cache for denoising
- Expected speedup: 60-70% of frames can reuse KV cache

Usage:
    python scripts/libero_eval_attention_gate.py --quick
    python scripts/libero_eval_attention_gate.py --task_suite_name libero_spatial
"""

import sys
import os
import collections
import logging
import math
import pathlib
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "openpi-client", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "libero"))

for path in ["/workspace/src", "/workspace/packages/openpi-client/src", "/workspace/third_party/libero"]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

torch.backends.cudnn.enabled = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
SEQ_LEN = 968

MAX_STEPS_DICT = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class ProAttentionGate:
    """
    Pro Attention-Weighted Drift Gate with three-layer defense.

    Layer 1: State Guard - Fastest, checks joint changes
    Layer 2: Anti-Noise - Downsampled pixel diff (224->28)
    Layer 3: Dilated Attention - Attention-weighted drift

    This gate decides whether to reuse the previous frame's KV cache
    or compute a new one.
    """

    def __init__(
        self,
        state_threshold: float = 0.23,   # P90 from analysis
        drift_threshold: float = 1.0,     # P75 from analysis
        dilation_kernel: int = 5,
        device: str = "cuda",
    ):
        self.state_threshold = state_threshold
        self.drift_threshold = drift_threshold
        self.dilation_kernel = dilation_kernel
        self.device = device

        # Cached state
        self.prev_image_small = None   # (B, num_cameras, C, 28, 28)
        self.prev_attention = None     # (B, num_cameras, 16, 16)
        self.prev_state = None         # (B, state_dim)
        self.prev_kv_cache = None      # List of (K, V) tuples
        self.prev_prefix_pad_masks = None

        # Statistics
        self.stats = {
            "total_frames": 0,
            "reused_frames": 0,
            "state_guard_triggers": 0,
            "vision_drift_triggers": 0,
            "first_frame_triggers": 0,
        }

    def reset(self):
        """Reset for new episode."""
        self.prev_image_small = None
        self.prev_attention = None
        self.prev_state = None
        self.prev_kv_cache = None
        self.prev_prefix_pad_masks = None

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_frames": 0,
            "reused_frames": 0,
            "state_guard_triggers": 0,
            "vision_drift_triggers": 0,
            "first_frame_triggers": 0,
        }

    def should_recompute(
        self,
        current_images: Dict[str, torch.Tensor],  # {name: (B, C, H, W)}
        current_state: torch.Tensor,  # (B, state_dim)
    ) -> Tuple[bool, str, float]:
        """
        Three-layer defense to decide if KV cache should be recomputed.

        Returns:
            should_recompute: bool
            reason: str (for debugging/logging)
            drift_value: float
        """
        self.stats["total_frames"] += 1

        # ========== Layer 1: State Guard (fastest) ==========
        if self.prev_state is not None:
            state_diff = torch.abs(current_state - self.prev_state).max().item()
            if state_diff > self.state_threshold:
                self.stats["state_guard_triggers"] += 1
                return True, f"State Guard: {state_diff:.4f}", state_diff

        # ========== First frame check ==========
        if self.prev_image_small is None or self.prev_attention is None:
            self.stats["first_frame_triggers"] += 1
            return True, "First Frame", 0.0

        # ========== Layer 2 & 3: Anti-Noise + Attention Weighting ==========
        # Prepare current images
        camera_names = list(current_images.keys())
        num_cameras = len(camera_names)
        images_list = [current_images[name] for name in camera_names]
        images_stacked = torch.stack(images_list, dim=1)  # (B, num_cameras, C, H, W)

        B, _, C, H, W = images_stacked.shape

        # Downsample to 28x28 for anti-noise
        images_flat = images_stacked.view(B * num_cameras, C, H, W)
        images_small = F.avg_pool2d(images_flat, kernel_size=8, stride=8)  # (B*C, C, 28, 28)
        images_small = images_small.view(B, num_cameras, C, 28, 28)

        # Compute pixel diff
        pixel_diff = torch.abs(images_small - self.prev_image_small).mean(dim=2)  # (B, num_cameras, 28, 28)

        # Resize attention to 28x28
        prev_attn = self.prev_attention  # (B, num_cameras, 16, 16)
        attn_resized = F.interpolate(
            prev_attn.view(B * num_cameras, 1, 16, 16),
            size=(28, 28),
            mode='bilinear',
            align_corners=False
        ).view(B, num_cameras, 28, 28)

        # Dilate attention
        attn_dilated = self._dilate_attention(attn_resized)

        # Weighted drift
        weighted_diff = (pixel_diff * attn_dilated).sum(dim=(2, 3))  # (B, num_cameras)
        total_drift = weighted_diff.sum().item()

        if total_drift > self.drift_threshold:
            self.stats["vision_drift_triggers"] += 1
            return True, f"Vision Drift: {total_drift:.4f}", total_drift

        # All checks passed - can reuse
        self.stats["reused_frames"] += 1
        return False, f"Reuse (drift={total_drift:.4f})", total_drift

    def _dilate_attention(self, attention_map: torch.Tensor) -> torch.Tensor:
        """Dilate attention map using max pooling."""
        B, num_cameras, H, W = attention_map.shape
        attn_flat = attention_map.view(B * num_cameras, 1, H, W)

        padding = self.dilation_kernel // 2
        dilated = F.max_pool2d(
            attn_flat,
            kernel_size=self.dilation_kernel,
            stride=1,
            padding=padding
        )

        return dilated.view(B, num_cameras, H, W)

    def update_cache(
        self,
        current_images: Dict[str, torch.Tensor],
        current_attention: torch.Tensor,  # (B, num_cameras, 16, 16)
        current_state: torch.Tensor,
        kv_cache,
        prefix_pad_masks: torch.Tensor,
    ):
        """Update cached state after computing new KV cache."""
        # Prepare and downsample images
        camera_names = list(current_images.keys())
        num_cameras = len(camera_names)
        images_list = [current_images[name] for name in camera_names]
        images_stacked = torch.stack(images_list, dim=1)

        B, _, C, H, W = images_stacked.shape
        images_flat = images_stacked.view(B * num_cameras, C, H, W)
        images_small = F.avg_pool2d(images_flat, kernel_size=8, stride=8)
        self.prev_image_small = images_small.view(B, num_cameras, C, 28, 28)

        self.prev_attention = current_attention
        self.prev_state = current_state.clone()
        self.prev_kv_cache = kv_cache
        self.prev_prefix_pad_masks = prefix_pad_masks

    def get_cached_kv(self):
        """Get cached KV cache and prefix pad masks."""
        return self.prev_kv_cache, self.prev_prefix_pad_masks

    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics."""
        stats = self.stats.copy()
        total = stats["total_frames"]
        if total > 0:
            stats["reuse_rate"] = stats["reused_frames"] / total
            stats["state_guard_rate"] = stats["state_guard_triggers"] / total
            stats["vision_drift_rate"] = stats["vision_drift_triggers"] / total
            stats["first_frame_rate"] = stats["first_frame_triggers"] / total
        return stats


class VisionWrapper(nn.Module):
    """Wrapper for Vision encoder TRT compilation."""
    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower

    def forward(self, pixel_values):
        outputs = self.vision_tower(pixel_values, output_hidden_states=False)
        return outputs.last_hidden_state


class DenoiseStepWrapper(nn.Module):
    """Wrapper for denoise step for CUDA Graph capture."""

    def __init__(self, pi0_model, prefix_len: int = SEQ_LEN):
        super().__init__()
        self.pi0_model = pi0_model
        self.prefix_len = prefix_len

        self.gemma_expert = pi0_model.paligemma_with_expert.gemma_expert.model
        self.paligemma_lm = pi0_model.paligemma_with_expert.paligemma.language_model
        self.action_in_proj = pi0_model.action_in_proj
        self.action_out_proj = pi0_model.action_out_proj
        self.time_mlp_in = pi0_model.time_mlp_in
        self.time_mlp_out = pi0_model.time_mlp_out

        self.action_horizon = pi0_model.config.action_horizon
        self.action_dim = pi0_model.config.action_dim
        self.num_layers = self.gemma_expert.config.num_hidden_layers
        self.hidden_size = self.gemma_expert.config.hidden_size
        self.head_dim = self.gemma_expert.layers[0].self_attn.head_dim

        self.register_buffer('_suffix_pad_masks', torch.ones(1, self.action_horizon, dtype=torch.bool))

    def forward(
        self,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        from transformers.models.gemma import modeling_gemma

        batch_size = x_t.shape[0]
        device = x_t.device

        adarms_cond = self._compute_time_embedding(timestep, batch_size, device)

        action_embs = self.action_in_proj(x_t.to(self.action_in_proj.weight.dtype))
        suffix_embs = action_embs.to(torch.bfloat16)

        suffix_pad_masks = self._suffix_pad_masks.expand(batch_size, -1)
        suffix_att_masks = torch.zeros(batch_size, self.action_horizon, device=device, dtype=torch.bfloat16)
        suffix_att_masks[:, 0] = 1.0

        suffix_att_2d = self._make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        suffix_to_prefix = prefix_pad_masks[:, None, :].expand(batch_size, self.action_horizon, -1)
        full_att_masks = torch.cat([suffix_to_prefix, suffix_att_2d], dim=2)
        full_att_masks_4d = self._prepare_4d_mask(full_att_masks)

        prefix_offsets = torch.sum(prefix_pad_masks.long(), dim=-1, keepdim=True)
        suffix_position_ids = prefix_offsets + torch.arange(self.action_horizon, device=device)

        hidden_states = suffix_embs

        for layer_idx in range(self.num_layers):
            layer = self.gemma_expert.layers[layer_idx]
            cached_key = prefix_keys[:, layer_idx]
            cached_value = prefix_values[:, layer_idx]

            normed_hidden, gate = layer.input_layernorm(hidden_states, cond=adarms_cond)

            input_shape = normed_hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            query_states = layer.self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)

            dummy = torch.zeros(
                query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                device=device, dtype=query_states.dtype
            )
            cos, sin = self.paligemma_lm.rotary_emb(dummy, suffix_position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            full_key = torch.cat([cached_key, key_states], dim=2)
            full_value = torch.cat([cached_value, value_states], dim=2)

            scaling = layer.self_attn.scaling
            att_output, _ = modeling_gemma.eager_attention_forward(
                layer.self_attn, query_states, full_key, full_value,
                full_att_masks_4d, scaling
            )

            num_heads = layer.self_attn.config.num_attention_heads
            att_output = att_output.reshape(batch_size, -1, num_heads * layer.self_attn.head_dim)

            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output)

            out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gate)
            after_first_residual = out_emb.clone()

            out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond)
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb)

            hidden_states = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)

        hidden_states, _ = self.gemma_expert.norm(hidden_states, cond=adarms_cond)
        suffix_out = hidden_states.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    def _compute_time_embedding(self, timestep, batch_size, device):
        half_dim = self.action_in_proj.out_features // 2
        fraction = torch.linspace(0.0, 1.0, half_dim, dtype=torch.float64, device=device)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * timestep[:, None].to(torch.float64)
        time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).float()
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        return F.silu(time_emb)

    def _make_att_2d_masks(self, pad_masks, att_masks):
        cumsum = torch.cumsum(att_masks.float(), dim=1)
        att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
        return att_2d & pad_2d

    def _prepare_4d_mask(self, mask):
        mask_4d = mask[:, None, :, :]
        return torch.where(mask_4d, 0.0, -2.3819763e38).to(torch.bfloat16)


class CUDAGraphDenoiseLoop:
    """CUDA Graph captured denoising loop."""

    def __init__(self, wrapper: DenoiseStepWrapper, num_steps: int = 10, schedule_type: str = "linear"):
        self.wrapper = wrapper
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.graph = None
        self.static_inputs = {}
        self.static_output = None

    def capture_graph(self, prefix_keys, prefix_values, prefix_pad_masks, device):
        from openpi.models_pytorch.pi0_pytorch import get_noise_schedule

        batch_size = 1

        self.static_inputs = {
            'prefix_keys': prefix_keys.clone(),
            'prefix_values': prefix_values.clone(),
            'prefix_pad_masks': prefix_pad_masks.clone(),
            'x_t': torch.randn(batch_size, self.wrapper.action_horizon, self.wrapper.action_dim,
                              device=device, dtype=torch.bfloat16),
        }

        timesteps = get_noise_schedule(self.num_steps, self.schedule_type, device)

        self.static_timesteps = []
        self.static_dts = []
        for step in range(self.num_steps):
            t_curr = timesteps[step]
            t_next = timesteps[step + 1]
            dt = t_next - t_curr

            self.static_timesteps.append(
                torch.tensor([t_curr.item()], device=device, dtype=torch.float32)
            )
            self.static_dts.append(
                torch.tensor(dt.item(), device=device, dtype=torch.float32)
            )

        # Warmup
        torch.cuda.synchronize()
        for _ in range(3):
            x_t = self.static_inputs['x_t'].clone()
            for step in range(self.num_steps):
                v_t = self.wrapper(
                    self.static_inputs['prefix_keys'],
                    self.static_inputs['prefix_values'],
                    self.static_inputs['prefix_pad_masks'],
                    x_t, self.static_timesteps[step]
                )
                x_t = x_t + self.static_dts[step] * v_t
        torch.cuda.synchronize()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(self.graph, stream=capture_stream):
                x_t = self.static_inputs['x_t']
                for step in range(self.num_steps):
                    v_t = self.wrapper(
                        self.static_inputs['prefix_keys'],
                        self.static_inputs['prefix_values'],
                        self.static_inputs['prefix_pad_masks'],
                        x_t, self.static_timesteps[step]
                    )
                    x_t = x_t + self.static_dts[step] * v_t
                self.static_output = x_t

        torch.cuda.synchronize()
        logger.info(f"CUDA Graph captured for {self.num_steps} denoising steps")

    def infer(self, x_t_init):
        self.static_inputs['x_t'].copy_(x_t_init)
        self.graph.replay()
        return self.static_output.clone()

    def update_kv_cache(self, prefix_keys, prefix_values, prefix_pad_masks):
        """Update static inputs with new KV cache."""
        self.static_inputs['prefix_keys'].copy_(prefix_keys)
        self.static_inputs['prefix_values'].copy_(prefix_values)
        self.static_inputs['prefix_pad_masks'].copy_(prefix_pad_masks)


class AttentionGatePolicy:
    """
    VLA Policy with Pro Attention-Weighted Drift Gate.

    When gate allows reuse:
    - Skip Vision TRT and KV Cache computation
    - Use cached KV from previous frame
    - Only run Denoising step

    Expected speedup: ~1.5x when 50% frames can be reused
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        num_denoising_steps: int = 10,
        prefix_len: int = SEQ_LEN,
        schedule_type: str = "linear",
        state_threshold: float = 0.23,
        drift_threshold: float = 1.0,
    ):
        self.device = device
        self.num_denoising_steps = num_denoising_steps
        self.prefix_len = prefix_len
        self.schedule_type = schedule_type
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser()

        # Latency tracking
        self.component_latencies = {
            'vision': [],
            'kv_cache': [],
            'denoise': [],
            'gate': [],
            'total': [],
        }

        # Load model
        self._load_model()
        self._load_tokenizer()
        self._load_norm_stats()

        # Setup optimized components
        self.use_vision_trt = False
        self._setup_vision_trt()
        self._setup_kv_cache_trt()
        self._setup_denoise_cuda_graph()

        # Setup attention gate
        self.gate = ProAttentionGate(
            state_threshold=state_threshold,
            drift_threshold=drift_threshold,
            device=device,
        )

        logger.info(f"Attention Gate configured: state_threshold={state_threshold}, drift_threshold={drift_threshold}")

    def _load_model(self):
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config
        from safetensors.torch import load_file

        config_path = self.checkpoint_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
        else:
            model_config = {}

        pi0_config = Pi0Config(
            paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
            action_dim=model_config.get("action_dim", 32),
            action_horizon=model_config.get("action_horizon", 50),
            max_token_len=model_config.get("tokenizer_max_length", 200),
            pi05=True,
            dtype="bfloat16",
        )

        self.model = PI0Pytorch(pi0_config)

        weights_path = self.checkpoint_dir / "model.safetensors"
        state_dict = load_file(weights_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device=self.device)
        self.model.eval()

        self.action_dim = pi0_config.action_dim
        self.action_horizon = pi0_config.action_horizon
        self.max_token_len = pi0_config.max_token_len
        self.max_state_dim = 32

        logger.info(f"Loaded model from {self.checkpoint_dir}")

    def _load_tokenizer(self):
        import sentencepiece as spm

        tokenizer_paths = [
            self.checkpoint_dir / "tokenizer.model",
            pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model",
            pathlib.Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model"),
        ]

        for path in tokenizer_paths:
            if path.exists():
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.Load(str(path))
                logger.info(f"Loaded tokenizer from {path}")
                return

        raise FileNotFoundError("Tokenizer not found")

    def _load_norm_stats(self):
        norm_stats_paths = [
            self.checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            self.checkpoint_dir / "norm_stats.json",
        ]

        for path in norm_stats_paths:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                self.norm_stats = data.get("norm_stats", data)
                logger.info(f"Loaded norm stats from {path}")
                return

        logger.warning("No norm_stats.json found")
        self.norm_stats = None

    def _setup_vision_trt(self):
        import torch_tensorrt

        vision_tower = self.model.paligemma_with_expert.paligemma.vision_tower
        self.multi_modal_projector = self.model.paligemma_with_expert.paligemma.model.multi_modal_projector

        wrapper = VisionWrapper(vision_tower).to(self.device).half()
        wrapper.eval()

        self.vision_trt = torch_tensorrt.compile(
            wrapper,
            inputs=[torch_tensorrt.Input(shape=(1, 3, 224, 224), dtype=torch.float16)],
            enabled_precisions={torch.float16},
            workspace_size=4 << 30,
            min_block_size=1,
        )

        # Warmup
        for _ in range(5):
            img_fp16 = torch.randn(1, 3, 224, 224, device=self.device, dtype=torch.float16)
            vision_out = self.vision_trt(img_fp16)
            vision_out_bf16 = vision_out.to(torch.bfloat16)
            _ = self.multi_modal_projector(vision_out_bf16)
        torch.cuda.synchronize()

        self.use_vision_trt = True
        logger.info("Vision TRT ready")

    def _setup_kv_cache_trt(self):
        from openpi.inference.torch_trt_fp8_kv_cache import TorchTRTFP8KVCacheEngine
        self.kv_engine = TorchTRTFP8KVCacheEngine(str(self.checkpoint_dir), self.device, compile_trt=True)
        logger.info(f"KV Cache TRT ready")

    def _setup_denoise_cuda_graph(self):
        wrapper = DenoiseStepWrapper(self.model, prefix_len=self.prefix_len)
        wrapper = wrapper.to(self.device)
        wrapper.eval()

        self.denoise_graph = CUDAGraphDenoiseLoop(
            wrapper,
            num_steps=self.num_denoising_steps,
            schedule_type=self.schedule_type
        )

        # Capture with dummy KV
        num_layers = 18
        num_kv_heads = 1
        head_dim = self.model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.head_dim

        dummy_keys = torch.randn(1, num_layers, num_kv_heads, self.prefix_len, head_dim,
                                 device=self.device, dtype=torch.bfloat16)
        dummy_values = torch.randn(1, num_layers, num_kv_heads, self.prefix_len, head_dim,
                                   device=self.device, dtype=torch.bfloat16)
        dummy_pad_masks = torch.ones(1, self.prefix_len, device=self.device, dtype=torch.bool)

        self.denoise_graph.capture_graph(dummy_keys, dummy_values, dummy_pad_masks, self.device)
        logger.info("Denoising CUDA Graph captured")

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return state

        stats = self.norm_stats.get("state", {})
        q01 = stats.get("q01")
        q99 = stats.get("q99")

        if q01 is not None and q99 is not None:
            q01 = np.array(q01, dtype=np.float32)[:len(state)]
            q99 = np.array(q99, dtype=np.float32)[:len(state)]
            return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return state

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return actions

        stats = self.norm_stats.get("actions", {})
        q01 = stats.get("q01")
        q99 = stats.get("q99")

        if q01 is not None and q99 is not None:
            q01 = np.array(q01, dtype=np.float32)
            q99 = np.array(q99, dtype=np.float32)
            action_dim = min(actions.shape[-1], len(q01))

            unnormalized = actions.copy()
            unnormalized[..., :action_dim] = (
                (actions[..., :action_dim] + 1.0) / 2.0 *
                (q99[:action_dim] - q01[:action_dim] + 1e-6) + q01[:action_dim]
            )
            return unnormalized
        return actions

    def _preprocess(self, observation: Dict[str, Any]):
        from openpi.models_pytorch.pi0_pytorch import Observation

        img = observation.get("observation/image")
        wrist_img = observation.get("observation/wrist_image")
        state = observation.get("observation/state")
        prompt = observation.get("prompt", "")

        if img is not None:
            img_tensor = torch.from_numpy(img).float() / 127.5 - 1.0
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            img_tensor = torch.zeros(1, 3, 224, 224) - 1.0

        if wrist_img is not None:
            wrist_tensor = torch.from_numpy(wrist_img).float() / 127.5 - 1.0
            if wrist_tensor.ndim == 3:
                wrist_tensor = wrist_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            wrist_tensor = torch.zeros(1, 3, 224, 224) - 1.0

        images = {
            "base_0_rgb": img_tensor.to(self.device, dtype=torch.bfloat16),
            "left_wrist_0_rgb": wrist_tensor.to(self.device, dtype=torch.bfloat16),
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, device=self.device, dtype=torch.bfloat16) - 1.0,
        }
        image_masks = {
            "base_0_rgb": torch.ones(1, device=self.device, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, device=self.device, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(1, device=self.device, dtype=torch.bool),
        }

        if state is not None:
            state_np = np.asarray(state, dtype=np.float32)
            state_np = self._normalize_state(state_np)
            state_tensor = torch.from_numpy(state_np).float()
            if state_tensor.shape[-1] < self.max_state_dim:
                padding = torch.zeros(self.max_state_dim - state_tensor.shape[-1])
                state_tensor = torch.cat([state_tensor, padding])
            state_tensor = state_tensor.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        else:
            state_tensor = torch.zeros(1, self.max_state_dim, device=self.device, dtype=torch.bfloat16)

        token_ids = self.tokenizer.Encode(prompt, add_bos=True)
        if len(token_ids) > self.max_token_len:
            token_ids = token_ids[:self.max_token_len]
        padding_len = self.max_token_len - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * padding_len
        token_ids = token_ids + [0] * padding_len

        tokenized_prompt = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        tokenized_prompt_mask = torch.tensor([attention_mask], dtype=torch.bool, device=self.device)

        return Observation(
            images=images,
            image_masks=image_masks,
            state=state_tensor,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=None,
            token_loss_mask=None,
        )

    def infer(self, observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Run inference with attention gate."""
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        obs = self._preprocess(observation)

        with torch.no_grad():
            # Prepare images for gate check
            current_images_for_gate = {
                "base": obs.images["base_0_rgb"],
                "wrist": obs.images["left_wrist_0_rgb"],
            }

            # ========== Gate Check ==========
            torch.cuda.synchronize()
            gate_start = time.perf_counter()

            should_recompute, reason, drift = self.gate.should_recompute(
                current_images_for_gate,
                obs.state,
            )

            torch.cuda.synchronize()
            gate_time = (time.perf_counter() - gate_start) * 1000
            self.component_latencies['gate'].append(gate_time)

            if should_recompute:
                # ========== Full Forward: Vision + KV Cache ==========
                torch.cuda.synchronize()
                vision_start = time.perf_counter()

                images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(
                    obs, train=False
                )

                if self.use_vision_trt:
                    embs = []
                    pad_masks = []
                    att_masks = []

                    for img, img_mask in zip(images, img_masks, strict=True):
                        img_fp16 = img.half()
                        vision_out = self.vision_trt(img_fp16)
                        vision_out_bf16 = vision_out.to(torch.bfloat16)
                        img_emb = self.multi_modal_projector(vision_out_bf16)

                        bsize, num_img_embs = img_emb.shape[:2]
                        embs.append(img_emb)
                        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                        att_masks += [0] * num_img_embs

                    lang_emb = self.model.paligemma_with_expert.embed_language_tokens(lang_tokens)
                    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

                    embs.append(lang_emb)
                    pad_masks.append(lang_masks)
                    num_lang_embs = lang_emb.shape[1]
                    att_masks += [0] * num_lang_embs

                    prefix_embs = torch.cat(embs, dim=1)
                    prefix_pad_masks = torch.cat(pad_masks, dim=1)
                    att_masks = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
                    bsize = prefix_pad_masks.shape[0]
                    prefix_att_masks = att_masks[None, :].expand(bsize, len(att_masks))
                else:
                    prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                        images, img_masks, lang_tokens, lang_masks
                    )

                torch.cuda.synchronize()
                vision_time = (time.perf_counter() - vision_start) * 1000
                self.component_latencies['vision'].append(vision_time)

                # ========== KV Cache ==========
                torch.cuda.synchronize()
                kv_start = time.perf_counter()

                cumsum = torch.cumsum(prefix_att_masks.float(), dim=1)
                att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
                pad_2d = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
                att_2d_masks = att_2d & pad_2d
                att_4d = att_2d_masks[:, None, :, :].to(torch.bfloat16)
                att_4d = torch.where(att_4d.bool(), 0.0, -2.3819763e38).to(torch.bfloat16)
                position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1

                kv_cache = self.kv_engine.infer_list(prefix_embs, position_ids, att_4d)

                keys = torch.stack([kv[0] for kv in kv_cache], dim=1)
                values = torch.stack([kv[1] for kv in kv_cache], dim=1)

                torch.cuda.synchronize()
                kv_time = (time.perf_counter() - kv_start) * 1000
                self.component_latencies['kv_cache'].append(kv_time)

                # Create dummy attention for gate update (we don't have real attention here)
                # In a full implementation, we'd extract attention from the KV computation
                dummy_attention = torch.ones(1, 2, 16, 16, device=self.device) / 256.0

                # Update gate cache
                self.gate.update_cache(
                    current_images_for_gate,
                    dummy_attention,
                    obs.state,
                    (keys, values),
                    prefix_pad_masks,
                )

                # Update CUDA graph
                self.denoise_graph.update_kv_cache(keys, values, prefix_pad_masks)

            else:
                # ========== Reuse cached KV ==========
                self.component_latencies['vision'].append(0.0)
                self.component_latencies['kv_cache'].append(0.0)

                # Get cached KV
                cached_kv, prefix_pad_masks = self.gate.get_cached_kv()
                keys, values = cached_kv

                # Update state for next frame
                self.gate.prev_state = obs.state.clone()

            # ========== Denoise ==========
            torch.cuda.synchronize()
            denoise_start = time.perf_counter()

            x_t = torch.randn(1, self.action_horizon, self.action_dim,
                             device=self.device, dtype=torch.bfloat16)
            actions = self.denoise_graph.infer(x_t)

            torch.cuda.synchronize()
            denoise_time = (time.perf_counter() - denoise_start) * 1000
            self.component_latencies['denoise'].append(denoise_time)

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.component_latencies['total'].append(latency_ms)

        # Post-process
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np, "reason": reason, "drift": drift}

    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics."""
        latencies = np.array(self.component_latencies['total'])
        if len(latencies) == 0:
            return {}

        component_stats = {}
        for name, records in self.component_latencies.items():
            if records:
                arr = np.array(records)
                component_stats[name] = {
                    "mean_ms": float(np.mean(arr)),
                    "std_ms": float(np.std(arr)),
                }

        total_mean = np.mean(latencies)
        breakdown = {}
        for name in ['vision', 'kv_cache', 'denoise', 'gate']:
            if name in component_stats:
                breakdown[name] = {
                    "ms": component_stats[name]["mean_ms"],
                    "pct": 100 * component_stats[name]["mean_ms"] / total_mean,
                }

        return {
            "count": len(latencies),
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "hz": float(1000 / np.mean(latencies)),
            "components": component_stats,
            "breakdown": breakdown,
            "gate_stats": self.gate.get_stats(),
        }

    def reset_episode(self):
        """Reset for new episode."""
        self.gate.reset()


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def resize_with_pad(img, target_h, target_w):
    try:
        from openpi_client import image_tools
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, target_h, target_w)
        )
    except ImportError:
        import cv2
        return cv2.resize(img, (target_w, target_h))


def run_episode(env, policy, task_description, args):
    """Run a single episode."""
    action_plan = collections.deque()
    max_steps = MAX_STEPS_DICT.get(args.task_suite_name, 300)

    t = 0
    obs = env.reset()
    policy.reset_episode()

    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if len(action_plan) == 0:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            img = resize_with_pad(img, args.resize_size, args.resize_size)
            wrist_img = resize_with_pad(wrist_img, args.resize_size, args.resize_size)

            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )).astype(np.float32)

            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": str(task_description),
            }

            result = policy.infer(element)
            action_chunk = result["actions"]
            action_plan.extend(action_chunk[:args.replan_steps])

        action = action_plan.popleft()
        if hasattr(action, 'tolist'):
            action = action.tolist()
        if len(action) > 7:
            action = action[:7]

        obs, reward, done, info = env.step(action)

        if done:
            return True

        t += 1

    return False


def eval_libero(args):
    """Main evaluation function."""
    np.random.seed(args.seed)

    from libero.libero import benchmark
    import tqdm

    logger.info(f"Creating Attention Gate Policy")
    logger.info(f"  state_threshold={args.state_threshold}")
    logger.info(f"  drift_threshold={args.drift_threshold}")
    logger.info(f"  denoising_steps={args.denoising_steps}")

    policy = AttentionGatePolicy(
        checkpoint_dir=args.checkpoint_dir,
        num_denoising_steps=args.denoising_steps,
        schedule_type=args.schedule,
        state_threshold=args.state_threshold,
        drift_threshold=args.drift_threshold,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    logger.info(f"Task suite: {args.task_suite_name}, {num_tasks_in_suite} tasks")

    if args.quick:
        task_start = 0
        task_end = min(3, num_tasks_in_suite)
        num_trials = 3
        logger.info(f"Quick mode: {task_end} tasks, {num_trials} trials each")
    else:
        task_start = args.task_start
        task_end = min(args.task_end, num_tasks_in_suite)
        num_trials = args.num_trials

    task_range = range(task_start, task_end)

    total_episodes = 0
    total_successes = 0
    task_results = []

    for task_id in tqdm.tqdm(task_range, desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes = 0
        task_successes = 0

        for episode_idx in tqdm.tqdm(range(num_trials), desc=f"Task {task_id}", leave=False):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])

            success = run_episode(env, policy, task_description, args)

            if success:
                task_successes += 1
                total_successes += 1

            task_episodes += 1
            total_episodes += 1

        task_result = {
            "task_id": task_id,
            "description": str(task_description),
            "successes": task_successes,
            "episodes": task_episodes,
            "success_rate": 100 * task_successes / task_episodes if task_episodes > 0 else 0,
        }
        task_results.append(task_result)

        print(f"\n>>> Task {task_id}: {task_successes}/{task_episodes} ({task_result['success_rate']:.1f}%)")
        env.close()

    latency_stats = policy.get_latency_stats()
    gate_stats = latency_stats.get("gate_stats", {})

    print("\n" + "=" * 70)
    print("ATTENTION GATE POLICY EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Task Suite: {args.task_suite_name}")
    print(f"  Denoising Steps: {args.denoising_steps}")
    print(f"  State Threshold: {args.state_threshold}")
    print(f"  Drift Threshold: {args.drift_threshold}")

    print(f"\nAccuracy:")
    print(f"  Total: {total_successes}/{total_episodes} ({100*total_successes/total_episodes:.1f}%)")

    print(f"\nGate Statistics:")
    print(f"  Reuse Rate: {gate_stats.get('reuse_rate', 0)*100:.1f}%")
    print(f"  State Guard Triggers: {gate_stats.get('state_guard_rate', 0)*100:.1f}%")
    print(f"  Vision Drift Triggers: {gate_stats.get('vision_drift_rate', 0)*100:.1f}%")
    print(f"  First Frame Triggers: {gate_stats.get('first_frame_rate', 0)*100:.1f}%")

    print(f"\nLatency ({latency_stats.get('count', 0)} inferences):")
    print(f"  Mean: {latency_stats.get('mean_ms', 0):.2f} ms")
    print(f"  P50:  {latency_stats.get('p50_ms', 0):.2f} ms")
    print(f"  Hz:   {latency_stats.get('hz', 0):.1f}")

    breakdown = latency_stats.get('breakdown', {})
    if breakdown:
        print(f"\nComponent Breakdown:")
        print(f"  {'Component':<15} {'Latency (ms)':<15} {'Percentage':<12}")
        print(f"  {'-' * 42}")
        for name in ['gate', 'vision', 'kv_cache', 'denoise']:
            if name in breakdown:
                b = breakdown[name]
                display_name = {
                    'gate': 'Gate Check',
                    'vision': 'Vision TRT',
                    'kv_cache': 'KV Cache TRT',
                    'denoise': 'Denoise CUDA'
                }[name]
                print(f"  {display_name:<15} {b['ms']:>8.2f} ms     {b['pct']:>6.1f}%")

    print("\n" + "=" * 70)

    if args.output_file:
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "task_suite": args.task_suite_name,
                "backend": "attention_gate",
                "denoising_steps": args.denoising_steps,
                "state_threshold": args.state_threshold,
                "drift_threshold": args.drift_threshold,
            },
            "accuracy": {
                "total_episodes": total_episodes,
                "total_successes": total_successes,
                "success_rate": 100 * total_successes / total_episodes,
            },
            "latency": latency_stats,
            "tasks": task_results,
        }
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIBERO evaluation with Attention Gate")

    parser.add_argument("--task_suite_name", default="libero_spatial",
                       choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--task_start", type=int, default=0)
    parser.add_argument("--task_end", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=10)

    parser.add_argument("--checkpoint_dir",
                       default=os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero"))
    parser.add_argument("--denoising_steps", type=int, default=10)
    parser.add_argument("--schedule", type=str, default="linear")

    # Gate thresholds
    parser.add_argument("--state_threshold", type=float, default=0.23,
                       help="State drift threshold (P90 = 0.23)")
    parser.add_argument("--drift_threshold", type=float, default=1.0,
                       help="Attention-weighted drift threshold (P75 = 1.0)")

    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()
    eval_libero(args)
