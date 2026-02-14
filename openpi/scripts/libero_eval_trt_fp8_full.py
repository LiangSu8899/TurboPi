#!/usr/bin/env python3
"""
LIBERO Evaluation with Full TRT FP8 Pipeline.

Current optimal configuration:
1. Vision TRT FP16: ~17ms
2. VLM KV Cache MLP-only TRT FP8: ~54ms
3. Denoise TRT FP8 (10-step): ~44ms
Total: ~115ms (~8.7 Hz)

Usage:
    python scripts/libero_eval_trt_fp8_full.py --quick
    python scripts/libero_eval_trt_fp8_full.py --task_suite_name libero_spatial --num_trials 5
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
from typing import Dict, Any, List, Optional

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

MAX_STEPS_DICT = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Denoise TRT constants
NUM_LAYERS = 18
NUM_KV_HEADS = 1
HEAD_DIM = 256
HIDDEN_SIZE = 1024
ACTION_HORIZON = 50
ACTION_DIM = 32
NUM_STEPS = 10


class VisionWrapper(nn.Module):
    """Wrapper for Vision encoder TRT compilation."""
    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower

    def forward(self, pixel_values):
        outputs = self.vision_tower(pixel_values, output_hidden_states=False)
        return outputs.last_hidden_state


class FullTRTFP8Policy:
    """
    Full TRT FP8 optimized VLA policy:
    - Vision TRT FP16
    - VLM KV Cache MLP-only TRT FP8
    - Denoise TRT FP8 (10-step loop)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        denoise_trt_path: str,
        device: str = "cuda",
        num_denoising_steps: int = 10,
        prefix_len: int = SEQ_LEN,
    ):
        self.device = device
        self.num_denoising_steps = num_denoising_steps
        self.prefix_len = prefix_len
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser()
        self.denoise_trt_path = denoise_trt_path

        # Latency tracking
        self.component_latencies = {
            'vision': [],
            'kv_cache': [],
            'denoise': [],
            'total': [],
        }

        # Load model and components
        self._load_model()
        self._load_tokenizer()
        self._load_norm_stats()

        # Setup optimized components
        self.use_vision_trt = False
        self._setup_vision_trt()
        self._setup_kv_cache_trt()
        self._setup_denoise_trt()

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
        """Load SentencePiece tokenizer."""
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

        raise FileNotFoundError(f"Tokenizer not found")

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

    def _setup_vision_trt(self):
        """Setup Vision TRT FP16."""
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
        logger.info("Vision TRT FP16 ready")

    def _setup_kv_cache_trt(self):
        """Setup KV Cache TRT FP8."""
        from openpi.inference.torch_trt_fp8_kv_cache import TorchTRTFP8KVCacheEngine
        self.kv_engine = TorchTRTFP8KVCacheEngine(str(self.checkpoint_dir), self.device, compile_trt=True)
        logger.info(f"KV Cache TRT FP8 ready ({self.kv_engine._trt_compiled_count}/18 layers)")

    def _setup_denoise_trt(self):
        """Setup Denoise TRT FP8."""
        logger.info(f"Loading Denoise TRT FP8 from {self.denoise_trt_path}")
        self.denoise_trt = torch.load(self.denoise_trt_path, weights_only=False)

        # Check if it's a single-step or loop model by looking at the path
        self.use_single_step = "step" in self.denoise_trt_path.lower()
        if self.use_single_step:
            logger.info("Detected single-step TRT model - will run 10 iterations in Python loop")

        # Pre-compute adarms_conds for all timesteps
        self._precompute_adarms_conds()

        # Pre-compute dt for flow matching
        self._dt = -1.0 / self.num_denoising_steps

        # Warmup
        self._warmup_denoise_trt()
        model_type = "single-step" if self.use_single_step else "10-step loop"
        logger.info(f"Denoise TRT FP8 ready ({model_type})")

    def _precompute_adarms_conds(self):
        """Pre-compute time embeddings for all denoise steps."""
        from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding

        model_dtype = self.model.action_in_proj.weight.dtype
        dt = -1.0 / self.num_denoising_steps

        adarms_conds = []
        for i in range(self.num_denoising_steps):
            timestep_val = 1.0 + i * dt
            timestep = torch.tensor([timestep_val], dtype=torch.float32, device=self.device)

            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.model.action_in_proj.out_features,
                min_period=4e-3,
                max_period=4.0,
                device=torch.device(self.device)
            )
            time_emb = time_emb.to(dtype=model_dtype)

            with torch.no_grad():
                x = self.model.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.model.time_mlp_out(x)
                adarms_cond = F.silu(x)

            adarms_conds.append(adarms_cond)

        # Stack: (num_steps, B, hidden_size) and convert to FP16
        self._static_adarms_conds = torch.stack(adarms_conds, dim=0).half()
        # Also keep individual adarms_conds for single-step model
        self._static_adarms_conds_list = [c.half() for c in adarms_conds]

    def _warmup_denoise_trt(self):
        """Warmup Denoise TRT."""
        dtype = torch.float16

        # Create common inputs (6 arguments WITH attention_mask)
        x_t = torch.randn(1, ACTION_HORIZON, ACTION_DIM, dtype=dtype, device=self.device)
        suffix_position_ids = torch.arange(self.prefix_len, self.prefix_len + ACTION_HORIZON,
                                           dtype=torch.long, device=self.device).unsqueeze(0)
        keys = torch.randn(NUM_LAYERS, 1, NUM_KV_HEADS, self.prefix_len, HEAD_DIM, dtype=dtype, device=self.device)
        values = torch.randn(NUM_LAYERS, 1, NUM_KV_HEADS, self.prefix_len, HEAD_DIM, dtype=dtype, device=self.device)
        # Note: No attention mask needed - matching original model (SDPA without mask)

        for _ in range(3):
            with torch.no_grad():
                if self.use_single_step:
                    # Single-step model: run 10 iterations
                    x = x_t.clone()
                    for step in range(self.num_denoising_steps):
                        v_t = self.denoise_trt(
                            x, suffix_position_ids, self._static_adarms_conds_list[step],
                            keys, values
                        )
                        x = x + v_t * self._dt
                else:
                    # Loop model: single call
                    _ = self.denoise_trt(
                        x_t, suffix_position_ids, self._static_adarms_conds,
                        keys, values
                    )
        torch.cuda.synchronize()

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
            unnormalized[..., :action_dim] = (actions[..., :action_dim] + 1.0) / 2.0 * (q99[:action_dim] - q01[:action_dim] + 1e-6) + q01[:action_dim]
            return unnormalized
        return actions

    def _preprocess(self, observation: Dict[str, Any]):
        """Preprocess observation."""
        from openpi.models_pytorch.pi0_pytorch import Observation

        img = observation.get("observation/image")
        wrist_img = observation.get("observation/wrist_image")
        state = observation.get("observation/state")
        prompt = observation.get("prompt", "")

        # Convert images
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

        # Process state
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

        # Tokenize prompt
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
        """Run optimized inference."""
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        obs = self._preprocess(observation)

        with torch.no_grad():
            # ============== 1. Vision (TRT FP16) ==============
            torch.cuda.synchronize()
            vision_start = time.perf_counter()

            images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(
                obs, train=False
            )

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
            att_masks_tensor = torch.tensor(att_masks, dtype=torch.bool, device=self.device)
            bsize = prefix_pad_masks.shape[0]
            prefix_att_masks = att_masks_tensor[None, :].expand(bsize, len(att_masks))

            torch.cuda.synchronize()
            vision_time = (time.perf_counter() - vision_start) * 1000
            self.component_latencies['vision'].append(vision_time)

            # ============== 2. KV Cache (TRT FP8) ==============
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

            # Stack KV cache for Denoise TRT
            keys = torch.stack([kv[0] for kv in kv_cache], dim=0)  # (num_layers, B, num_kv_heads, seq, head_dim)
            values = torch.stack([kv[1] for kv in kv_cache], dim=0)

            torch.cuda.synchronize()
            kv_time = (time.perf_counter() - kv_start) * 1000
            self.component_latencies['kv_cache'].append(kv_time)

            # ============== 3. Denoise (TRT FP8) ==============
            torch.cuda.synchronize()
            denoise_start = time.perf_counter()

            # Compute suffix position IDs
            prefix_offset = torch.sum(prefix_pad_masks.long(), dim=-1, keepdim=True)
            suffix_position_ids = prefix_offset + torch.arange(
                ACTION_HORIZON, device=self.device, dtype=torch.long
            )

            # Note: No attention mask needed - original model uses SDPA without mask
            # The comment says: "Since suffix attention mask is ALL TRUE (bidirectional),
            # we can skip the mask entirely."

            # Initial noise
            x_t = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=self.device, dtype=torch.float16)

            # Run Denoise TRT (5 arguments WITHOUT attention_mask - matching original model)
            if self.use_single_step:
                # Single-step model: run 10 iterations in Python loop
                keys_fp16 = keys.half()
                values_fp16 = values.half()
                for step in range(self.num_denoising_steps):
                    v_t = self.denoise_trt(
                        x_t, suffix_position_ids, self._static_adarms_conds_list[step],
                        keys_fp16, values_fp16
                    )
                    x_t = x_t + v_t * self._dt
                actions = x_t
            else:
                # Loop model: single call (10-step loop compiled)
                actions = self.denoise_trt(
                    x_t,
                    suffix_position_ids,
                    self._static_adarms_conds,
                    keys.half(),
                    values.half(),
                )

            torch.cuda.synchronize()
            denoise_time = (time.perf_counter() - denoise_start) * 1000
            self.component_latencies['denoise'].append(denoise_time)

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.component_latencies['total'].append(latency_ms)

        # Post-process
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics."""
        if not self.component_latencies['total']:
            return {}

        latencies = np.array(self.component_latencies['total'])

        component_stats = {}
        for name, records in self.component_latencies.items():
            if records:
                arr = np.array(records)
                component_stats[name] = {
                    "mean_ms": float(np.mean(arr)),
                    "std_ms": float(np.std(arr)),
                    "min_ms": float(np.min(arr)),
                    "max_ms": float(np.max(arr)),
                }

        total_mean = np.mean(latencies)
        breakdown = {}
        for name in ['vision', 'kv_cache', 'denoise']:
            if name in component_stats:
                breakdown[name] = {
                    "ms": component_stats[name]["mean_ms"],
                    "pct": 100 * component_stats[name]["mean_ms"] / total_mean,
                }

        return {
            "count": len(latencies),
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "hz": float(1000 / np.mean(latencies)),
            "components": component_stats,
            "breakdown": breakdown,
        }


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


def run_episode(env, policy, task_description, args):
    """Run a single episode."""
    action_plan = collections.deque()
    max_steps = MAX_STEPS_DICT.get(args.task_suite_name, 300)

    t = 0
    obs = env.reset()

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

    # Create policy
    logger.info(f"Creating Full TRT FP8 Policy")
    logger.info(f"  Denoise TRT: {args.denoise_trt_path}")

    policy = FullTRTFP8Policy(
        checkpoint_dir=args.checkpoint_dir,
        denoise_trt_path=args.denoise_trt_path,
        num_denoising_steps=args.denoising_steps,
    )

    # Initialize LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    logger.info(f"Task suite: {args.task_suite_name}, {num_tasks_in_suite} tasks")

    # Determine task range
    if args.quick:
        task_start = 0
        task_end = min(3, num_tasks_in_suite)
        num_trials = 5
        logger.info(f"Quick mode: {task_end} tasks, {num_trials} trials each")
    else:
        task_start = args.task_start
        task_end = min(args.task_end, num_tasks_in_suite)
        num_trials = args.num_trials

    task_range = range(task_start, task_end)

    # Evaluation loop
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

    # Get latency stats
    latency_stats = policy.get_latency_stats()

    # Final results
    print("\n" + "=" * 70)
    print(f"FULL TRT FP8 PIPELINE EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Task Suite: {args.task_suite_name}")
    print(f"  Backend: Vision TRT FP16 + VLM TRT FP8 + Denoise TRT FP8")
    print(f"  Denoising Steps: {args.denoising_steps}")
    print(f"  Tasks: {task_end - task_start}, Trials per task: {num_trials}")

    print(f"\nAccuracy:")
    print(f"  Total: {total_successes}/{total_episodes} ({100*total_successes/total_episodes:.1f}%)")

    print(f"\nLatency ({latency_stats.get('count', 0)} inferences):")
    print(f"  Mean: {latency_stats.get('mean_ms', 0):.2f} ms")
    print(f"  Std:  {latency_stats.get('std_ms', 0):.2f} ms")
    print(f"  P50:  {latency_stats.get('p50_ms', 0):.2f} ms")
    print(f"  P95:  {latency_stats.get('p95_ms', 0):.2f} ms")
    print(f"  Hz:   {latency_stats.get('hz', 0):.1f}")

    breakdown = latency_stats.get('breakdown', {})
    if breakdown:
        print(f"\nComponent Breakdown:")
        print(f"  {'Component':<20} {'Latency (ms)':<15} {'Percentage':<12}")
        print(f"  {'-' * 47}")
        for name in ['vision', 'kv_cache', 'denoise']:
            if name in breakdown:
                b = breakdown[name]
                display_name = {'vision': 'Vision TRT FP16', 'kv_cache': 'VLM KV TRT FP8', 'denoise': 'Denoise TRT FP8'}[name]
                print(f"  {display_name:<20} {b['ms']:>8.2f} ms     {b['pct']:>6.1f}%")
        print(f"  {'-' * 47}")
        total_ms = latency_stats.get('mean_ms', 0)
        print(f"  {'Total':<20} {total_ms:>8.2f} ms     {100.0:>6.1f}%")

    print("\n" + "=" * 70)

    # Save results
    if args.output_file:
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "task_suite": args.task_suite_name,
                "backend": "full_trt_fp8",
                "denoising_steps": args.denoising_steps,
                "num_trials": num_trials,
                "seed": args.seed,
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

    return {
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate": 100 * total_successes / total_episodes,
        "latency": latency_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIBERO evaluation with Full TRT FP8 Pipeline")

    parser.add_argument("--task_suite_name", default="libero_spatial",
                       choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--task_start", type=int, default=0)
    parser.add_argument("--task_end", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=5)

    parser.add_argument("--checkpoint_dir",
                       default=os.path.expanduser("~/.cache/openpi/checkpoints/pi05_libero"))
    parser.add_argument("--denoise_trt_path",
                       default="/workspace/denoise_trt_static/denoise_loop_fp8.pt")
    parser.add_argument("--denoising_steps", type=int, default=10)

    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 tasks, 5 trials")
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()
    eval_libero(args)
