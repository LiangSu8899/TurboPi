#!/usr/bin/env python3
"""
Step 1: Visualize Attention Importance

This script visualizes where the model's attention is focused during inference.
The goal is to verify that attention concentrates on task-relevant regions (gripper, target object)
rather than being scattered across the image.

Key verification points:
1. Heatmap should follow gripper and target object
2. Background should have low attention
3. Dilated attention should provide reasonable "safety margin"

Usage:
    python scripts/visualize_attention_importance.py --quick
    python scripts/visualize_attention_importance.py --num_frames 50 --output_dir attention_viz
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "openpi-client", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "libero"))

for path in ["/workspace/src", "/workspace/packages/openpi-client/src", "/workspace/third_party/libero"]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Set MuJoCo rendering options
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Disable cuDNN for Jetson compatibility
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


class AttentionExtractor:
    """Extracts and processes attention weights from Pi0.5 model."""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.attention_weights = {}
        self.hooks = []

    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        # Hook into Gemma Expert's attention layers (these are the action-processing layers)
        gemma_expert = self.model.paligemma_with_expert.gemma_expert.model

        for layer_idx, layer in enumerate(gemma_expert.layers):
            def hook_fn(module, input, output, layer_idx=layer_idx):
                # Store the query, key for this layer
                # We'll compute attention scores manually
                self.attention_weights[f"layer_{layer_idx}"] = {
                    "input": input,
                    "output": output,
                }
            self.hooks.append(layer.self_attn.register_forward_hook(hook_fn))

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_weights = {}

    def compute_attention_with_scores(
        self,
        observation,
        num_steps: int = 1,  # Just one step for visualization
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run forward pass and compute attention scores between action and vision tokens.

        Returns:
            actions: Predicted actions
            attention_info: Dict containing attention scores and metadata
        """
        from transformers.models.gemma import modeling_gemma
        from openpi.models_pytorch.pi0_pytorch import Observation, make_att_2d_masks, get_noise_schedule

        with torch.no_grad():
            # Preprocess observation
            images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(
                observation, train=False
            )

            # Embed prefix (vision + language)
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )

            # Compute prefix KV cache
            prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )

            # Get model dimensions
            bsize = state.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            action_horizon = self.model.config.action_horizon
            action_dim = self.model.config.action_dim

            # Start with noise
            x_t = torch.randn(bsize, action_horizon, action_dim, device=self.device)
            x_t = x_t.to(next(self.model.parameters()).dtype)

            # Get timestep for visualization (use t=0.5 for mid-denoising)
            timestep = torch.tensor([0.5], device=self.device, dtype=torch.float32)

            # Compute one denoising step with attention extraction
            # We need to manually run the forward pass to capture attention

            # Embed suffix
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.model.embed_suffix(
                state, x_t, timestep
            )

            gemma_expert = self.model.paligemma_with_expert.gemma_expert.model
            paligemma_lm = self.model.paligemma_with_expert.paligemma.language_model
            num_layers = gemma_expert.config.num_hidden_layers

            batch_size = suffix_pad_masks.shape[0]
            suffix_len = suffix_pad_masks.shape[1]

            # Convert to bfloat16
            if gemma_expert.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

            # Build attention masks
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            suffix_to_prefix_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            full_att_masks = torch.cat([suffix_to_prefix_masks, suffix_att_2d_masks], dim=2)

            # Position IDs
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            # Store attention scores per layer
            layer_attention_scores = []

            hidden_states = suffix_embs

            for layer_idx in range(num_layers):
                layer = gemma_expert.layers[layer_idx]
                cached_key, cached_value = prefix_kv_cache[layer_idx]

                # Input layernorm
                normed_hidden, gate = layer.input_layernorm(hidden_states, cond=adarms_cond)

                # Compute Q, K, V
                input_shape = normed_hidden.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                query_states = layer.self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
                key_states = layer.self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
                value_states = layer.self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)

                # Apply RoPE
                dummy_tensor = torch.zeros(
                    query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                    device=query_states.device, dtype=query_states.dtype
                )
                cos, sin = paligemma_lm.rotary_emb(dummy_tensor, suffix_position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                # Concatenate cached K, V with suffix K, V
                full_key_states = torch.cat([cached_key, key_states], dim=2)
                full_value_states = torch.cat([cached_value, value_states], dim=2)

                # Compute attention scores manually
                # query_states: (B, num_heads, suffix_len, head_dim)
                # full_key_states: (B, num_heads, prefix_len + suffix_len, head_dim)
                scaling = layer.self_attn.scaling

                attn_weights = torch.matmul(query_states, full_key_states.transpose(-2, -1)) * scaling

                # Apply attention mask
                full_att_masks_4d = full_att_masks[:, None, :, :]
                attn_mask_value = torch.where(full_att_masks_4d, 0.0, -2.3819763e38).to(query_states.dtype)
                attn_weights = attn_weights + attn_mask_value

                # Softmax
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                # Store attention weights for this layer
                layer_attention_scores.append(attn_weights.clone())

                # Complete the forward pass
                attn_output = torch.matmul(attn_weights, full_value_states)

                head_dim = layer.self_attn.head_dim
                num_heads = layer.self_attn.config.num_attention_heads
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, suffix_len, num_heads * head_dim)

                if attn_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    attn_output = attn_output.to(layer.self_attn.o_proj.weight.dtype)
                out_emb = layer.self_attn.o_proj(attn_output)

                out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gate)
                after_first_residual = out_emb.clone()

                out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond)
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    out_emb = out_emb.to(dtype=torch.bfloat16)
                out_emb = layer.mlp(out_emb)

                hidden_states = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)

            # Final norm and projection
            hidden_states, _ = gemma_expert.norm(hidden_states, cond=adarms_cond)
            suffix_out = hidden_states[:, -action_horizon:]
            suffix_out = suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
            actions = self.model.action_out_proj(suffix_out)

            # Stack attention scores: (num_layers, B, num_heads, suffix_len, total_len)
            attention_scores = torch.stack(layer_attention_scores, dim=0)

            # Compute vision token importance
            # Vision tokens are the first 256 * num_cameras tokens in prefix
            # Each camera contributes 256 tokens (16x16 patches)
            num_cameras = len(images)  # 3 cameras typically
            num_vision_tokens_per_camera = 256  # 16x16 patches
            num_vision_tokens = num_cameras * num_vision_tokens_per_camera

            attention_info = {
                "attention_scores": attention_scores,  # (L, B, H, suffix_len, total_len)
                "num_vision_tokens": num_vision_tokens,
                "num_vision_tokens_per_camera": num_vision_tokens_per_camera,
                "num_cameras": num_cameras,
                "prefix_len": prefix_len,
                "suffix_len": suffix_len,
                "num_layers": num_layers,
            }

            return actions, attention_info


def extract_patch_importance(attention_info: Dict, layer_range: Tuple[int, int] = (14, 18)) -> torch.Tensor:
    """
    Extract importance score for each vision patch based on action tokens' attention.

    Args:
        attention_info: Dict from AttentionExtractor.compute_attention_with_scores
        layer_range: Which layers to use (default: last few layers)

    Returns:
        importance: (B, num_cameras, 16, 16) importance map for each camera
    """
    attention_scores = attention_info["attention_scores"]  # (L, B, H, suffix_len, total_len)
    num_vision_tokens = attention_info["num_vision_tokens"]
    num_vision_tokens_per_camera = attention_info["num_vision_tokens_per_camera"]
    num_cameras = attention_info["num_cameras"]
    num_layers = attention_info["num_layers"]

    # Adjust layer range if needed
    layer_start = min(layer_range[0], num_layers - 1)
    layer_end = min(layer_range[1], num_layers)

    # Select layers and average
    selected_attn = attention_scores[layer_start:layer_end]  # (L', B, H, suffix_len, total_len)
    avg_attn = selected_attn.mean(dim=(0, 2))  # (B, suffix_len, total_len) - average over layers and heads

    # Extract action tokens' attention to vision tokens
    # Action tokens attend to: [vision tokens | language tokens | action tokens]
    action_to_vision = avg_attn[:, :, :num_vision_tokens]  # (B, suffix_len, num_vision_tokens)

    # Average over all action tokens
    importance = action_to_vision.mean(dim=1)  # (B, num_vision_tokens)

    # Reshape to per-camera 2D maps
    B = importance.shape[0]
    importance = importance.view(B, num_cameras, num_vision_tokens_per_camera)
    importance = importance.view(B, num_cameras, 16, 16)

    # Normalize each camera's importance map
    importance = importance / (importance.sum(dim=(2, 3), keepdim=True) + 1e-8)

    return importance


def dilate_attention(attention_map: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Dilate attention map using max pooling to prevent blind spot invasion.

    Args:
        attention_map: (B, H, W) or (B, C, H, W)
        kernel_size: Dilation kernel size

    Returns:
        Dilated attention map with same shape
    """
    if attention_map.dim() == 3:
        attention_map = attention_map.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False

    padding = kernel_size // 2
    dilated = F.max_pool2d(
        attention_map,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    if squeeze_output:
        dilated = dilated.squeeze(1)

    return dilated


def visualize_frame(
    original_images: Dict[str, np.ndarray],
    importance_maps: torch.Tensor,
    frame_idx: int,
    output_dir: str,
    dilation_kernel: int = 5,
):
    """
    Visualize attention importance overlaid on original images.

    Args:
        original_images: Dict of camera_name -> image array (H, W, 3)
        importance_maps: (B, num_cameras, 16, 16) importance tensor
        frame_idx: Frame index for filename
        output_dir: Output directory
        dilation_kernel: Kernel size for dilation
    """
    camera_names = list(original_images.keys())
    num_cameras = len(camera_names)

    # Create figure with 3 rows: original, attention, dilated
    fig, axes = plt.subplots(3, num_cameras, figsize=(5 * num_cameras, 15))

    if num_cameras == 1:
        axes = axes.reshape(3, 1)

    for cam_idx, cam_name in enumerate(camera_names):
        img = original_images[cam_name]
        importance = importance_maps[0, cam_idx].float().cpu().numpy()  # (16, 16)

        # Compute dilated importance
        importance_tensor = importance_maps[0, cam_idx:cam_idx+1]  # (1, 16, 16)
        dilated_tensor = dilate_attention(importance_tensor, kernel_size=dilation_kernel)
        dilated = dilated_tensor[0].float().cpu().numpy()

        # Resize importance maps to image size
        h, w = img.shape[:2]
        importance_resized = np.array(
            F.interpolate(
                torch.tensor(importance).unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )[0, 0].numpy()
        )
        dilated_resized = np.array(
            F.interpolate(
                torch.tensor(dilated).unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )[0, 0].numpy()
        )

        # Row 0: Original image
        axes[0, cam_idx].imshow(img)
        axes[0, cam_idx].set_title(f"{cam_name}\n(Original)", fontsize=10)
        axes[0, cam_idx].axis('off')

        # Row 1: Attention heatmap overlay
        axes[1, cam_idx].imshow(img)
        heatmap = axes[1, cam_idx].imshow(
            importance_resized,
            alpha=0.6,
            cmap='jet',
            vmin=0,
            vmax=importance_resized.max()
        )
        axes[1, cam_idx].set_title(f"Attention Importance\nmax={importance.max():.4f}", fontsize=10)
        axes[1, cam_idx].axis('off')

        # Row 2: Dilated attention heatmap
        axes[2, cam_idx].imshow(img)
        axes[2, cam_idx].imshow(
            dilated_resized,
            alpha=0.6,
            cmap='jet',
            vmin=0,
            vmax=dilated_resized.max()
        )
        axes[2, cam_idx].set_title(f"Dilated Attention\n(kernel={dilation_kernel})", fontsize=10)
        axes[2, cam_idx].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


class AttentionVisualizationPolicy:
    """Policy wrapper for attention visualization."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
    ):
        self.device = device
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser()

        self._load_model()
        self._load_tokenizer()
        self._load_norm_stats()

        self.extractor = AttentionExtractor(self.model, device)

    def _load_model(self):
        """Load PyTorch model."""
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
        """Load tokenizer."""
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
        """Load normalization statistics."""
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
        """Preprocess observation."""
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

    def infer_with_attention(
        self,
        observation: Dict[str, Any],
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Run inference and extract attention importance maps.

        Returns:
            actions: Predicted actions (unnormalized)
            importance_maps: (B, num_cameras, 16, 16) attention importance
        """
        obs = self._preprocess(observation)

        with torch.no_grad():
            actions, attention_info = self.extractor.compute_attention_with_scores(obs)
            importance_maps = extract_patch_importance(attention_info)

        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return actions_np, importance_maps


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle."""
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
    """Create LIBERO environment."""
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
    """Resize image with padding."""
    try:
        from openpi_client import image_tools
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, target_h, target_w)
        )
    except ImportError:
        import cv2
        return cv2.resize(img, (target_w, target_h))


def run_attention_visualization(args):
    """Main visualization function."""
    np.random.seed(args.seed)

    from libero.libero import benchmark

    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create policy
    logger.info("Loading model for attention visualization...")
    policy = AttentionVisualizationPolicy(
        checkpoint_dir=args.checkpoint_dir,
    )

    # Initialize LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    # Get first task
    task = task_suite.get_task(0)
    initial_states = task_suite.get_task_init_states(0)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    logger.info(f"Task: {task_description}")
    logger.info(f"Output directory: {output_dir}")

    # Run episode and collect frames
    action_plan = collections.deque()
    max_steps = min(args.num_frames, 220)

    t = 0
    frame_idx = 0
    obs = env.reset()
    env.set_init_state(initial_states[0])

    attention_stats = []

    while t < max_steps + 10:  # wait steps
        if t < 10:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if len(action_plan) == 0:
            # Get images
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            img_resized = resize_with_pad(img, args.resize_size, args.resize_size)
            wrist_resized = resize_with_pad(wrist_img, args.resize_size, args.resize_size)

            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )).astype(np.float32)

            element = {
                "observation/image": img_resized,
                "observation/wrist_image": wrist_resized,
                "observation/state": state,
                "prompt": str(task_description),
            }

            # Get actions and attention
            actions, importance_maps = policy.infer_with_attention(element)
            action_plan.extend(actions[:args.replan_steps])

            # Visualize this frame
            original_images = {
                "base_0_rgb": img_resized,
                "left_wrist_0_rgb": wrist_resized,
            }

            output_path = visualize_frame(
                original_images,
                importance_maps,
                frame_idx,
                str(output_dir),
                dilation_kernel=args.dilation_kernel,
            )

            # Collect stats
            importance_np = importance_maps.float().cpu().numpy()
            stats = {
                "frame": frame_idx,
                "max_importance_base": float(importance_np[0, 0].max()),
                "max_importance_wrist": float(importance_np[0, 1].max()),
                "mean_importance_base": float(importance_np[0, 0].mean()),
                "mean_importance_wrist": float(importance_np[0, 1].mean()),
            }
            attention_stats.append(stats)

            logger.info(f"Frame {frame_idx}: base_max={stats['max_importance_base']:.4f}, "
                       f"wrist_max={stats['max_importance_wrist']:.4f}")

            frame_idx += 1

        if frame_idx >= args.num_frames:
            break

        action = action_plan.popleft()
        if hasattr(action, 'tolist'):
            action = action.tolist()
        if len(action) > 7:
            action = action[:7]

        obs, reward, done, info = env.step(action)

        if done:
            logger.info("Episode completed successfully!")
            break

        t += 1

    env.close()

    # Save statistics
    stats_path = output_dir / "attention_stats.json"
    with open(stats_path, 'w') as f:
        json.dump({
            "task": str(task_description),
            "num_frames": frame_idx,
            "dilation_kernel": args.dilation_kernel,
            "frames": attention_stats,
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("ATTENTION VISUALIZATION SUMMARY")
    print("=" * 70)
    print(f"\nTask: {task_description}")
    print(f"Frames visualized: {frame_idx}")
    print(f"Output directory: {output_dir}")
    print(f"\nAttention Statistics:")

    if attention_stats:
        base_maxes = [s['max_importance_base'] for s in attention_stats]
        wrist_maxes = [s['max_importance_wrist'] for s in attention_stats]
        print(f"  Base camera - max importance: {np.mean(base_maxes):.4f} ± {np.std(base_maxes):.4f}")
        print(f"  Wrist camera - max importance: {np.mean(wrist_maxes):.4f} ± {np.std(wrist_maxes):.4f}")

    print("\n" + "=" * 70)
    print(f"View frames at: {output_dir}/frame_*.png")
    print(f"Stats saved to: {stats_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attention importance in Pi0.5 model")

    parser.add_argument("--task_suite_name", default="libero_spatial",
                       choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--checkpoint_dir",
                       default=os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero"))

    parser.add_argument("--num_frames", type=int, default=20,
                       help="Number of frames to visualize")
    parser.add_argument("--output_dir", type=str, default="attention_viz",
                       help="Output directory for visualizations")
    parser.add_argument("--dilation_kernel", type=int, default=5,
                       help="Kernel size for attention dilation")

    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: only 10 frames")

    args = parser.parse_args()

    if args.quick:
        args.num_frames = 10

    run_attention_visualization(args)
