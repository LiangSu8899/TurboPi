#!/usr/bin/env python3
"""
Unified Policy Interface for Pi0.5 VLA Model.

Provides a consistent API across different inference backends:
- pytorch: Pure PyTorch (baseline)
- tensorrt: TensorRT engines (optimized)
- tensorrt_pipelined: TensorRT with dual-stream pipelining (fastest)

Usage:
    from openpi.inference.unified_policy import UnifiedPolicy

    # Create policy with desired backend
    policy = UnifiedPolicy(
        checkpoint_dir="/path/to/checkpoint",
        backend="tensorrt_pipelined",  # or "pytorch", "tensorrt"
        num_denoising_steps=3,
    )

    # Warmup (important for TensorRT)
    policy.warmup()

    # Run inference with consistent interface
    result = policy.infer({
        "observation/image": img,           # uint8, H×W×3
        "observation/wrist_image": wrist,   # uint8, H×W×3
        "observation/state": state,         # float32, (8,)
        "prompt": "pick up the black bowl",
    })

    actions = result["actions"]  # (horizon, action_dim)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for unified policy."""
    checkpoint_dir: str
    backend: str = "tensorrt_pipelined"  # pytorch, tensorrt, tensorrt_pipelined
    num_denoising_steps: int = 3
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Model architecture
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    max_state_dim: int = 32

    # TensorRT options
    engine_dir: Optional[str] = None  # If None, use checkpoint_dir/engines


class BaseBackend(ABC):
    """Base class for inference backends."""

    @abstractmethod
    def infer(self, observation: Dict[str, Any], num_steps: int) -> Dict[str, np.ndarray]:
        """Run inference and return actions."""
        pass

    @abstractmethod
    def warmup(self, num_iterations: int = 5):
        """Warmup the backend."""
        pass


class PyTorchBackend(BaseBackend):
    """Pure PyTorch inference backend."""

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.device = config.device
        self.dtype = getattr(torch, config.dtype)

        # Load model
        self._load_model()
        self._load_tokenizer()
        self._load_norm_stats()

    def _load_model(self):
        """Load PyTorch model."""
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Pi0Config

        checkpoint_path = Path(self.config.checkpoint_dir)

        # Load config
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
        else:
            model_config = {}

        # Create model config
        pi0_config = Pi0Config(
            paligemma_variant=model_config.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=model_config.get("action_expert_variant", "gemma_300m"),
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            max_token_len=self.config.max_token_len,
            max_state_dim=self.config.max_state_dim,
            pi05=True,
            dtype=self.config.dtype,
        )

        # Create model
        self.model = PI0Pytorch(pi0_config)

        # Load weights
        weights_path = checkpoint_path / "model.safetensors"
        if weights_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            weights_path = checkpoint_path / "model.pt"
            state_dict = torch.load(weights_path, map_location="cpu")

        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        logger.info(f"Loaded PyTorch model from {checkpoint_path}")

    def _load_tokenizer(self):
        """Load SentencePiece tokenizer."""
        import sentencepiece as spm

        # Try different tokenizer paths
        tokenizer_paths = [
            Path(self.config.checkpoint_dir) / "tokenizer.model",
            Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model",
            Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model"),
        ]

        for path in tokenizer_paths:
            if path.exists():
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.Load(str(path))
                logger.info(f"Loaded tokenizer from {path}")
                return

        raise FileNotFoundError(f"Tokenizer not found in {tokenizer_paths}")

    def _load_norm_stats(self):
        """Load normalization statistics from checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir)

        # Try different norm stats paths
        norm_stats_paths = [
            checkpoint_path / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_path / "norm_stats.json",
        ]

        for path in norm_stats_paths:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                self.norm_stats = data.get("norm_stats", data)
                logger.info(f"Loaded norm stats from {path}")
                return

        # Default norm stats if not found (identity transform)
        logger.warning("No norm_stats.json found, using identity normalization")
        self.norm_stats = None

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using quantile normalization (matches Pi0.5 training)."""
        if self.norm_stats is None:
            return state

        stats = self.norm_stats.get("state", {})

        # Use quantile normalization for Pi0.5
        q01 = stats.get("q01")
        q99 = stats.get("q99")

        if q01 is not None and q99 is not None:
            q01 = np.array(q01, dtype=np.float32)[:len(state)]
            q99 = np.array(q99, dtype=np.float32)[:len(state)]
            # Quantile normalization: map [q01, q99] to [-1, 1]
            return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            # Fallback to z-score if quantile stats not available
            mean = np.array(stats.get("mean", [0.0] * len(state)), dtype=np.float32)[:len(state)]
            std = np.array(stats.get("std", [1.0] * len(state)), dtype=np.float32)[:len(state)]
            std = np.clip(std, 1e-6, None)
            return (state - mean) / std

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Unnormalize actions using quantile normalization (matches Pi0.5 training)."""
        if self.norm_stats is None:
            return actions

        stats = self.norm_stats.get("actions", {})

        # Use quantile normalization for Pi0.5
        q01 = stats.get("q01")
        q99 = stats.get("q99")

        if q01 is not None and q99 is not None:
            q01 = np.array(q01, dtype=np.float32)
            q99 = np.array(q99, dtype=np.float32)
            action_dim = min(actions.shape[-1], len(q01))

            unnormalized = actions.copy()
            # Quantile unnormalization: map [-1, 1] to [q01, q99]
            unnormalized[..., :action_dim] = (actions[..., :action_dim] + 1.0) / 2.0 * (q99[:action_dim] - q01[:action_dim] + 1e-6) + q01[:action_dim]
            return unnormalized
        else:
            # Fallback to z-score if quantile stats not available
            mean = np.array(stats.get("mean", [0.0] * 7), dtype=np.float32)
            std = np.array(stats.get("std", [1.0] * 7), dtype=np.float32)
            action_dim = min(actions.shape[-1], len(mean))
            unnormalized = actions.copy()
            unnormalized[..., :action_dim] = actions[..., :action_dim] * std[:action_dim] + mean[:action_dim]
            return unnormalized

    def _preprocess(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess observation for model input."""
        from openpi.models_pytorch.pi0_pytorch import Observation

        # Get images (expect uint8, H×W×3)
        img = observation.get("observation/image")
        wrist_img = observation.get("observation/wrist_image")
        state = observation.get("observation/state")
        prompt = observation.get("prompt", "")

        # Convert images to [-1, 1] range
        if img is not None:
            img_tensor = torch.from_numpy(img).float() / 127.5 - 1.0
            if img_tensor.ndim == 3:  # H, W, C
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # 1, C, H, W
        else:
            img_tensor = torch.zeros(1, 3, 224, 224) - 1.0

        if wrist_img is not None:
            wrist_tensor = torch.from_numpy(wrist_img).float() / 127.5 - 1.0
            if wrist_tensor.ndim == 3:
                wrist_tensor = wrist_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            wrist_tensor = torch.zeros(1, 3, 224, 224) - 1.0

        # Create image dict
        images = {
            "base_0_rgb": img_tensor.to(self.device, dtype=self.dtype),
            "left_wrist_0_rgb": wrist_tensor.to(self.device, dtype=self.dtype),
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, device=self.device, dtype=self.dtype) - 1.0,
        }
        image_masks = {
            "base_0_rgb": torch.ones(1, device=self.device, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, device=self.device, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(1, device=self.device, dtype=torch.bool),
        }

        # Process state with normalization
        if state is not None:
            state_np = np.asarray(state, dtype=np.float32)
            # Normalize state
            state_np = self._normalize_state(state_np)
            state_tensor = torch.from_numpy(state_np).float()
            # Pad to max_state_dim
            if state_tensor.shape[-1] < self.config.max_state_dim:
                padding = torch.zeros(self.config.max_state_dim - state_tensor.shape[-1])
                state_tensor = torch.cat([state_tensor, padding])
            state_tensor = state_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)
        else:
            state_tensor = torch.zeros(1, self.config.max_state_dim, device=self.device, dtype=self.dtype)

        # Tokenize prompt
        token_ids = self.tokenizer.Encode(prompt, add_bos=True)
        if len(token_ids) > self.config.max_token_len:
            token_ids = token_ids[:self.config.max_token_len]
        padding_len = self.config.max_token_len - len(token_ids)
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

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        obs = self._preprocess(observation)

        with torch.no_grad():
            actions = self.model.sample_actions(
                self.device,
                obs,
                num_steps=num_steps,
                use_kv_cache=True,
            )

        # Post-process: convert to numpy and unnormalize
        actions_np = actions.float().cpu().numpy()[0]

        # Unnormalize actions to original scale
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def warmup(self, num_iterations: int = 5):
        """Warmup the model."""
        logger.info(f"Warming up PyTorch backend ({num_iterations} iterations)")

        dummy_obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "warmup",
        }

        for i in range(num_iterations):
            self.infer(dummy_obs)

        logger.info("Warmup complete")


class PyTorchPipelinedBackend(PyTorchBackend):
    """
    PyTorch backend with async pipelined execution.

    Uses dual CUDA streams to overlap Vision+KV cache computation with
    denoising, achieving ~26Hz throughput on Jetson Thor.
    """

    def __init__(self, config: PolicyConfig):
        # Initialize base PyTorch backend (loads model, tokenizer, etc.)
        super().__init__(config)

        # Create async pipeline wrapper
        from openpi.inference.async_pipeline import AsyncVLAPipeline
        self.pipeline = AsyncVLAPipeline(
            model=self.model,
            device=torch.device(self.device),
            num_denoising_steps=config.num_denoising_steps,
        )

        logger.info(f"Initialized PyTorch Pipelined backend with {config.num_denoising_steps} denoising steps")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run pipelined inference."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        # Preprocess observation (same as base class)
        obs = self._preprocess(observation)

        with torch.no_grad():
            # Use sequential inference for single frame
            # (pipelined batching requires multiple frames)
            actions = self.pipeline.infer_sequential(obs)

        # Post-process
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def infer_batch(self, observations: List[Dict[str, Any]], num_steps: int = None) -> List[Dict[str, np.ndarray]]:
        """Run pipelined batch inference for multiple frames."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        # Preprocess all observations
        obs_list = [self._preprocess(obs) for obs in observations]

        with torch.no_grad():
            # Use pipelined inference for batch
            actions_list, stats = self.pipeline.infer_pipelined_batch(obs_list, return_stats=True)

        # Post-process all
        results = []
        for actions in actions_list:
            actions_np = actions.float().cpu().numpy()[0]
            actions_np = self._unnormalize_actions(actions_np)
            results.append({"actions": actions_np})

        return results

    def warmup(self, num_iterations: int = 5):
        """Warmup the pipelined model."""
        logger.info(f"Warming up PyTorch Pipelined backend ({num_iterations} iterations)")

        dummy_obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "warmup",
        }

        # Warmup with preprocessed observation
        obs = self._preprocess(dummy_obs)
        self.pipeline.warmup(obs, num_iterations=num_iterations)

        logger.info("Warmup complete")


class HybridTensorRTBackend(PyTorchBackend):
    """
    Hybrid TensorRT + PyTorch inference backend.

    Combines PyTorch's complete inference flow with optional TensorRT acceleration.
    This is the recommended backend for 26+ Hz inference with full accuracy.

    Architecture:
    - Uses PyTorch for complete inference flow (embed_prefix, KV cache, denoise)
    - Optional TensorRT acceleration for Vision Encoder
    - Dual CUDA stream pipelining for overlapped execution
    """

    def __init__(self, config: PolicyConfig, pipelined: bool = True):
        # Initialize base PyTorch backend (loads model, tokenizer, norm_stats)
        super().__init__(config)
        self.pipelined = pipelined

        # Create hybrid pipeline
        from openpi.inference.hybrid_trt_pipeline import HybridTensorRTPipeline, HybridPipelineConfig

        pipeline_config = HybridPipelineConfig(
            checkpoint_dir=config.checkpoint_dir,
            num_denoising_steps=config.num_denoising_steps,
            device=config.device,
            dtype=config.dtype,
            use_trt_vision=True,  # Try to use TensorRT for Vision
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            max_token_len=config.max_token_len,
            max_state_dim=config.max_state_dim,
        )

        self.pipeline = HybridTensorRTPipeline(
            config=pipeline_config,
            model=self.model,
            tokenizer=self.tokenizer,
            norm_stats=self.norm_stats,
        )

        logger.info(f"Initialized Hybrid TensorRT backend (pipelined={pipelined})")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run hybrid inference."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        # Preprocess observation (same as base PyTorchBackend)
        obs = self._preprocess(observation)

        with torch.no_grad():
            # Use single-frame inference
            actions = self.pipeline.infer_single(obs)

        # Post-process
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def infer_batch(self, observations: List[Dict[str, Any]], num_steps: int = None) -> List[Dict[str, np.ndarray]]:
        """Run pipelined batch inference for higher throughput."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        # Preprocess all observations
        obs_list = [self._preprocess(obs) for obs in observations]

        with torch.no_grad():
            if self.pipelined:
                actions_list, _ = self.pipeline.infer_batch(obs_list, return_stats=False)
            else:
                actions_list = [self.pipeline.infer_single(obs) for obs in obs_list]

        # Post-process all
        results = []
        for actions in actions_list:
            actions_np = actions.float().cpu().numpy()[0]
            actions_np = self._unnormalize_actions(actions_np)
            results.append({"actions": actions_np})

        return results

    def warmup(self, num_iterations: int = 5):
        """Warmup the hybrid pipeline."""
        logger.info(f"Warming up Hybrid TensorRT backend ({num_iterations} iterations)")

        dummy_obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "warmup",
        }

        obs = self._preprocess(dummy_obs)
        self.pipeline.warmup(obs, num_iterations=num_iterations)

        logger.info("Warmup complete")

    def benchmark(self, num_frames: int = 100):
        """Run performance benchmark."""
        dummy_obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "benchmark test",
        }

        obs = self._preprocess(dummy_obs)
        seq_stats, pipe_stats = self.pipeline.benchmark(obs, num_frames=num_frames)

        from openpi.inference.hybrid_trt_pipeline import print_benchmark_comparison
        print_benchmark_comparison(seq_stats, pipe_stats)

        return seq_stats, pipe_stats


# Keep TensorRTBackend as alias for backward compatibility
TensorRTBackend = HybridTensorRTBackend


class TripleStreamBackend(PyTorchBackend):
    """
    Triple Stream Pipeline backend for 26+ Hz inference.

    Uses three CUDA streams for maximum pipelining:
    - Stream 0: Observation prefetch
    - Stream 1: Vision + KV Cache computation
    - Stream 2: Denoising

    This backend achieves the highest throughput by overlapping:
    - Frame N+1's Vision+KV with Frame N's denoising
    - Frame N+2's prefetch with Frame N+1's Vision+KV
    """

    def __init__(self, config: PolicyConfig, kv_cache_engine_path: Optional[str] = None):
        # Initialize base PyTorch backend (loads model, tokenizer, norm_stats)
        super().__init__(config)

        from openpi.inference.triple_stream_pipeline import TripleStreamPipeline, TripleStreamConfig
        from openpi.inference.kv_cache_trt import create_kv_cache_engine

        # Create KV cache engine
        self.kv_cache_engine = create_kv_cache_engine(
            engine_path=kv_cache_engine_path,
            model=self.model,
            device=config.device,
        )

        # Create pipeline config
        pipeline_config = TripleStreamConfig(
            num_denoising_steps=config.num_denoising_steps,
            action_horizon=config.action_horizon,
            action_dim=config.action_dim,
            device=config.device,
            dtype=config.dtype,
            kv_cache_engine_path=kv_cache_engine_path,
        )

        # Create pipeline
        self.pipeline = TripleStreamPipeline(
            model=self.model,
            config=pipeline_config,
            tokenizer=self.tokenizer,
            norm_stats=self.norm_stats,
            kv_cache_engine=self.kv_cache_engine,
        )

        logger.info(f"Initialized TripleStreamBackend")
        logger.info(f"  - KV Cache engine: {type(self.kv_cache_engine).__name__}")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run triple-stream inference."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        # Preprocess observation
        obs = self._preprocess(observation)

        with torch.no_grad():
            actions = self.pipeline.infer_single(obs)

        # Post-process
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def infer_batch(
        self, observations: List[Dict[str, Any]], num_steps: int = None
    ) -> List[Dict[str, np.ndarray]]:
        """Run pipelined batch inference for maximum throughput."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        # Preprocess all observations
        obs_list = [self._preprocess(obs) for obs in observations]

        with torch.no_grad():
            actions_list, _ = self.pipeline.infer_pipelined_batch(obs_list, return_stats=False)

        # Post-process all
        results = []
        for actions in actions_list:
            actions_np = actions.float().cpu().numpy()[0]
            actions_np = self._unnormalize_actions(actions_np)
            results.append({"actions": actions_np})

        return results

    def warmup(self, num_iterations: int = 5):
        """Warmup the triple-stream pipeline."""
        logger.info(f"Warming up TripleStreamBackend ({num_iterations} iterations)")

        dummy_obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "warmup",
        }

        obs = self._preprocess(dummy_obs)
        self.pipeline.warmup(obs, num_iterations=num_iterations)

        logger.info("Warmup complete")

    def benchmark(self, num_frames: int = 100):
        """Run performance benchmark."""
        dummy_obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "benchmark test",
        }

        obs = self._preprocess(dummy_obs)
        seq_stats, pipe_stats = self.pipeline.benchmark(obs, num_frames=num_frames)

        from openpi.inference.triple_stream_pipeline import print_benchmark_comparison
        print_benchmark_comparison(seq_stats, pipe_stats)

        return seq_stats, pipe_stats


class FullTRTBackend(PyTorchBackend):
    """
    Full TensorRT Backend with KV Cache support.

    Uses:
    - PyTorch for prefix embedding computation
    - TensorRT engine for KV Cache (20+ Hz target)
    - PyTorch for denoising with cached KV

    PRECISION FIX (2026-02-01):
    The SDPA-based TRT engines had precision issues due to TensorRT's SDPA
    implementation differing from PyTorch. This has been fixed by using
    EXPLICIT attention (matmul + softmax) in the ONNX export.

    RECOMMENDED ENGINES (in order of accuracy):
    1. paligemma_kv_cache_explicit_fp32.engine - Best accuracy (FP32 explicit attn)
    2. paligemma_kv_cache_explicit_fp16.engine - Good accuracy (FP16 explicit attn)
    3. paligemma_kv_cache_fp16.engine - SDPA-based, has precision issues

    To build the recommended engine:
        python scripts/export_kv_cache_explicit_attn.py
        trtexec --onnx=paligemma_kv_cache_explicit.onnx \\
            --saveEngine=paligemma_kv_cache_explicit_fp32.engine
    """

    def __init__(self, config: PolicyConfig, kv_cache_engine_path: Optional[str] = None, force_trt: bool = False):
        # Initialize base PyTorch backend (loads model, tokenizer, norm_stats)
        super().__init__(config)

        from openpi.inference.kv_cache_trt import create_kv_cache_engine

        # By default, use PyTorch KV Cache for accuracy
        # Set force_trt=True to enable TRT (for benchmarking)
        if not force_trt:
            logger.warning(
                "FullTRTBackend: Using PyTorch KV Cache by default for accuracy. "
                "TRT KV Cache has known precision issues (see docs). "
                "Set force_trt=True to enable TRT for benchmarking."
            )
            kv_cache_engine_path = None

        # Determine KV Cache engine path if force_trt
        if force_trt and kv_cache_engine_path is None:
            from pathlib import Path
            from openpi.inference.kv_cache_trt import find_best_kv_cache_engine

            engine_dir = config.engine_dir or str(Path(config.checkpoint_dir).parent / "onnx_exports")
            kv_cache_engine_path = find_best_kv_cache_engine(engine_dir)

            if kv_cache_engine_path:
                logger.info(f"Auto-selected KV Cache engine: {kv_cache_engine_path}")
                # Warn if not using explicit attention engine (has precision issues)
                if "explicit" not in kv_cache_engine_path.lower():
                    logger.warning(
                        "SDPA-based engine may have precision issues. "
                        "For best accuracy, use explicit attention engine: "
                        "paligemma_kv_cache_explicit_fp32.engine"
                    )
            else:
                logger.warning("No TRT KV Cache engine found, falling back to PyTorch")

        # Create KV cache engine
        self.kv_cache_engine = create_kv_cache_engine(
            engine_path=kv_cache_engine_path,
            model=self.model,
            device=config.device,
        )

        # Check if using TRT or PyTorch
        self.using_trt = kv_cache_engine_path is not None

        logger.info(f"FullTRTBackend initialized (TRT KV Cache: {self.using_trt})")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference with TRT KV Cache acceleration."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        # Preprocess observation (same as PyTorchBackend)
        obs = self._preprocess(observation)

        with torch.no_grad():
            # Step 1: Embed prefix (uses PyTorch)
            # Convert image dicts to lists as expected by embed_prefix
            images = list(obs.images.values())
            img_masks = list(obs.image_masks.values())

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                images, img_masks,
                obs.tokenized_prompt, obs.tokenized_prompt_mask
            )

            # Step 2: Compute KV Cache using TRT engine
            batch_size, seq_len, _ = prefix_embs.shape

            # Prepare inputs for KV Cache engine
            position_ids = torch.arange(seq_len, device=prefix_embs.device).unsqueeze(0)

            # Create 4D attention mask
            # Note: TRT INT8 engine was calibrated with zeros (valid attention)
            # Using -2.38e38 for masked positions can cause INT8 overflow
            # Use -1e4 instead which works well with INT8 quantization
            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
            att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            att_2d_masks_4d = att_2d_masks[:, None, :, :]
            # Use -1e4 instead of -2.38e38 for TRT INT8 compatibility
            attention_mask_4d = torch.where(att_2d_masks_4d, 0.0, -1e4).to(prefix_embs.dtype)

            # Run TRT KV Cache (returns list of (K, V) tuples)
            prefix_kv_cache = self.kv_cache_engine.infer_list(
                prefix_embs, position_ids, attention_mask_4d
            )

            # Step 3: Denoise with cached KV (uses PyTorch)
            # Initialize with noise
            actions_shape = (batch_size, self.config.action_horizon, self.config.action_dim)
            device = torch.device(self.device)
            model_dtype = next(self.model.parameters()).dtype

            x_t = torch.randn(actions_shape, device=device, dtype=model_dtype)

            dt = -1.0 / num_steps
            dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
            time = torch.tensor(1.0, dtype=torch.float32, device=device)

            while time >= -dt_tensor / 2:
                expanded_time = time.expand(batch_size)
                v_t = self.model.denoise_step_with_cache(
                    obs.state,
                    prefix_kv_cache,
                    prefix_pad_masks,
                    x_t,
                    expanded_time,
                )
                x_t = x_t + dt_tensor * v_t
                time += dt_tensor

            actions = x_t

        # Post-process: convert to numpy and unnormalize
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def warmup(self, num_iterations: int = 5):
        """Warmup the backend including TRT engine."""
        logger.info(f"Warming up FullTRTBackend ({num_iterations} iterations)")

        # Warmup TRT KV Cache engine
        self.kv_cache_engine.warmup(num_iterations=num_iterations)

        # Warmup full inference
        dummy_obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "warmup",
        }

        for _ in range(num_iterations):
            self.infer(dummy_obs)

        logger.info("Warmup complete")


class TensorRTW8A16Backend(PyTorchBackend):
    """
    TensorRT W8A16 Backend for 20+ Hz inference.

    Uses W8A16 quantized KV Cache engine for fast inference:
    - INT8 weights + FP16 activations
    - 42ms KV Cache latency (2x faster than PyTorch 85ms)
    - 0.046% relative error
    - With Freq=2 reuse: 23.5 Hz throughput

    This backend is recommended for production 20Hz inference.
    """

    def __init__(self, config: PolicyConfig, w8a16_engine_path: Optional[str] = None):
        # Initialize base PyTorch backend
        super().__init__(config)

        from openpi.inference.kv_cache_trt import create_kv_cache_engine, find_w8a16_engine

        # Find W8A16 engine
        if w8a16_engine_path is None:
            checkpoint_dir = config.checkpoint_dir
            w8a16_engine_path = find_w8a16_engine(checkpoint_dir)

            if w8a16_engine_path is None:
                from pathlib import Path
                parent_dir = Path(checkpoint_dir).parent
                w8a16_engine_path = find_w8a16_engine(str(parent_dir))

        if w8a16_engine_path is None:
            raise FileNotFoundError(
                "W8A16 KV Cache engine not found. Build it with:\n"
                "  python scripts/calibrate_w8a16_real_data.py\n"
                "  trtexec --onnx=paligemma_kv_cache_w8a16.onnx "
                "--saveEngine=paligemma_kv_cache_w8a16.engine --int8 --fp16"
            )

        self.kv_cache_engine = create_kv_cache_engine(
            engine_path=w8a16_engine_path,
            model=self.model,
            device=config.device,
            prefer_w8a16=True,
        )

        logger.info(f"TensorRTW8A16Backend initialized with engine: {w8a16_engine_path}")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference with W8A16 TRT KV Cache acceleration."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        obs = self._preprocess(observation)

        with torch.no_grad():
            images = list(obs.images.values())
            img_masks = list(obs.image_masks.values())

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                images, img_masks,
                obs.tokenized_prompt, obs.tokenized_prompt_mask
            )

            batch_size, seq_len, _ = prefix_embs.shape
            position_ids = torch.arange(seq_len, device=prefix_embs.device).unsqueeze(0)

            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
            att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            att_2d_masks_4d = att_2d_masks[:, None, :, :]
            attention_mask_4d = torch.where(att_2d_masks_4d, 0.0, -1e4).to(prefix_embs.dtype)

            prefix_kv_cache = self.kv_cache_engine.infer_list(
                prefix_embs, position_ids, attention_mask_4d
            )

            actions_shape = (batch_size, self.config.action_horizon, self.config.action_dim)
            device = torch.device(self.device)
            model_dtype = next(self.model.parameters()).dtype

            x_t = torch.randn(actions_shape, device=device, dtype=model_dtype)

            dt = -1.0 / num_steps
            dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
            time = torch.tensor(1.0, dtype=torch.float32, device=device)

            while time >= -dt_tensor / 2:
                expanded_time = time.expand(batch_size)
                v_t = self.model.denoise_step_with_cache(
                    obs.state,
                    prefix_kv_cache,
                    prefix_pad_masks,
                    x_t,
                    expanded_time,
                )
                x_t = x_t + dt_tensor * v_t
                time += dt_tensor

            actions = x_t

        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def warmup(self, num_iterations: int = 5):
        """Warmup the W8A16 backend."""
        logger.info(f"Warming up TensorRTW8A16Backend ({num_iterations} iterations)")

        self.kv_cache_engine.warmup(num_iterations=num_iterations)

        dummy_obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "warmup",
        }

        for _ in range(num_iterations):
            self.infer(dummy_obs)

        logger.info("Warmup complete")


class TensorRTW8A16KVReuseBackend(TensorRTW8A16Backend):
    """
    W8A16 TRT backend with Vision/KV Cache reuse for 20+ Hz inference.

    Combines:
    - W8A16 quantized KV Cache (42ms)
    - Vision/KV reuse (Freq=2)

    Performance (Freq=2):
    - Full frame: ~70ms (Vision 12ms + KV 42ms + Denoise 15ms)
    - Fast frame: ~15ms (Denoise only)
    - Average: ~42.5ms → 23.5 Hz

    LIBERO validated:
    - Freq=2: 88.9% accuracy
    """

    def __init__(self, config: PolicyConfig, reuse_freq: int = 2, w8a16_engine_path: Optional[str] = None):
        super().__init__(config, w8a16_engine_path)

        self.reuse_freq = reuse_freq
        self.frame_count = 0
        self.cached_kv = None
        self.cached_pad_masks = None

        logger.info(f"TensorRTW8A16KVReuseBackend initialized with reuse_freq={reuse_freq}")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference with W8A16 KV Cache and reuse."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        obs = self._preprocess(observation)
        is_full_frame = (self.frame_count % self.reuse_freq == 0) or (self.cached_kv is None)

        with torch.no_grad():
            if is_full_frame:
                images = list(obs.images.values())
                img_masks = list(obs.image_masks.values())

                prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                    images, img_masks,
                    obs.tokenized_prompt, obs.tokenized_prompt_mask
                )

                batch_size, seq_len, _ = prefix_embs.shape
                position_ids = torch.arange(seq_len, device=prefix_embs.device).unsqueeze(0)

                from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
                att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                att_2d_masks_4d = att_2d_masks[:, None, :, :]
                attention_mask_4d = torch.where(att_2d_masks_4d, 0.0, -1e4).to(prefix_embs.dtype)

                self.cached_kv = self.kv_cache_engine.infer_list(
                    prefix_embs, position_ids, attention_mask_4d
                )
                self.cached_pad_masks = prefix_pad_masks

            batch_size = 1
            device = torch.device(self.device)
            model_dtype = next(self.model.parameters()).dtype

            actions_shape = (batch_size, self.config.action_horizon, self.config.action_dim)
            x_t = torch.randn(actions_shape, device=device, dtype=model_dtype)

            dt = -1.0 / num_steps
            dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
            time = torch.tensor(1.0, dtype=torch.float32, device=device)

            while time >= -dt_tensor / 2:
                expanded_time = time.expand(batch_size)
                v_t = self.model.denoise_step_with_cache(
                    obs.state,
                    self.cached_kv,
                    self.cached_pad_masks,
                    x_t,
                    expanded_time,
                )
                x_t = x_t + dt_tensor * v_t
                time += dt_tensor

            actions = x_t

        self.frame_count += 1

        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def reset_cache(self):
        """Reset the KV cache. Call this when starting a new episode."""
        self.frame_count = 0
        self.cached_kv = None
        self.cached_pad_masks = None


class PyTorchKVReuseBackend(PyTorchBackend):
    """
    PyTorch backend with Vision/KV Cache reuse for higher throughput.

    Strategy:
    - Every N frames (reuse_freq): Full inference (Vision + KV Cache + Denoise)
    - Intermediate frames: Fast inference (Denoise only with cached KV)

    Trade-off:
    - Higher throughput: ~10.6 Hz (reuse_freq=3) vs ~5.7 Hz (no reuse)
    - Visual lag: ~189ms (reuse_freq=3)
    - Suitable for slow manipulation tasks where visual changes are gradual
    """

    def __init__(self, config: PolicyConfig, reuse_freq: int = 3):
        # Initialize base PyTorch backend
        super().__init__(config)

        self.reuse_freq = reuse_freq
        self.frame_count = 0
        self.cached_kv = None
        self.cached_pad_masks = None

        logger.info(f"Initialized PyTorchKVReuseBackend with reuse_freq={reuse_freq}")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference with optional KV cache reuse."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        obs = self._preprocess(observation)

        # Determine if this is a full or fast frame
        is_full_frame = (self.frame_count % self.reuse_freq == 0) or (self.cached_kv is None)

        with torch.no_grad():
            if is_full_frame:
                # Full inference: Vision + KV Cache + Denoise
                images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(obs, train=False)

                # Embed prefix (Vision + Language)
                prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks
                )

                # Compute and cache KV
                self.cached_kv = self.model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)
                self.cached_pad_masks = prefix_pad_masks

                # Denoise with fresh KV
                actions = self.model.sample_actions_with_external_kv(
                    torch.device(self.device),
                    state,
                    self.cached_kv,
                    self.cached_pad_masks,
                    num_steps=num_steps,
                )
            else:
                # Fast inference: Denoise only with cached KV
                # Still need to extract state from observation
                _, _, _, _, state = self.model._preprocess_observation(obs, train=False)

                actions = self.model.sample_actions_with_external_kv(
                    torch.device(self.device),
                    state,
                    self.cached_kv,
                    self.cached_pad_masks,
                    num_steps=num_steps,
                )

        self.frame_count += 1

        # Post-process
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def reset_cache(self):
        """Reset the KV cache. Call this when starting a new episode."""
        self.frame_count = 0
        self.cached_kv = None
        self.cached_pad_masks = None

    def warmup(self, num_iterations: int = 5):
        """Warmup the model."""
        logger.info(f"Warming up PyTorchKVReuseBackend ({num_iterations} iterations)")

        dummy_obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "warmup",
        }

        for i in range(num_iterations):
            self.infer(dummy_obs)

        # Reset cache after warmup
        self.reset_cache()

        logger.info("Warmup complete")


class PyTorchTextCacheBackend(PyTorchBackend):
    """
    PyTorch backend with Text-Only Embedding Caching.

    Optimization:
    - Language embeddings are constant within an episode (same task prompt)
    - Cache language embeddings once, recompute only image embeddings each frame
    - The full KV cache is still computed together to maintain proper cross-attention
    - Saves ~10-15% by avoiding redundant language embedding computation

    Note: This is different from caching the KV cache itself, which would break
    the cross-attention pattern between image and language tokens. Here we cache
    only the embeddings before KV cache computation.

    Prefix structure:
    - Images: 256 tokens × num_images (e.g., 768 tokens for 3 images) - recomputed
    - Language: max_token_len tokens (e.g., 200 tokens) - cached embeddings
    """

    def __init__(self, config: PolicyConfig):
        # Initialize base PyTorch backend
        super().__init__(config)

        # Cache for language embeddings
        self._cached_lang_emb = None
        self._cached_lang_pad_mask = None
        self._cached_prompt = None

        logger.info("Initialized PyTorchTextCacheBackend (Text Embedding Caching)")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference with text embedding caching."""
        import math

        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        obs = self._preprocess(observation)
        prompt = observation.get("prompt", "")

        with torch.no_grad():
            # Extract components from observation
            images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(obs, train=False)

            # Check if language embedding cache is valid
            if self._cached_prompt != prompt or self._cached_lang_emb is None:
                # Cache new language embeddings
                logger.debug(f"Caching language embeddings for prompt: {prompt[:50]}...")

                # Embed language tokens
                lang_emb = self.model.paligemma_with_expert.embed_language_tokens(lang_tokens)
                lang_emb_dim = lang_emb.shape[-1]
                lang_emb = lang_emb * math.sqrt(lang_emb_dim)

                self._cached_lang_emb = lang_emb
                self._cached_lang_pad_mask = lang_masks
                self._cached_prompt = prompt

            # Compute image embeddings (every frame)
            img_embs = []
            img_pad_masks = []
            img_att_masks = []

            for img, img_mask in zip(images, img_masks, strict=True):
                img_emb = self.model.paligemma_with_expert.embed_image(img)
                bsize, num_img_embs = img_emb.shape[:2]
                img_embs.append(img_emb)
                img_pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                # Image tokens can attend to each other (mask=0)
                img_att_masks.extend([0] * num_img_embs)

            # Concatenate embeddings: [images, cached_language]
            prefix_embs = torch.cat(img_embs + [self._cached_lang_emb], dim=1)
            prefix_pad_masks = torch.cat(img_pad_masks + [self._cached_lang_pad_mask], dim=1)

            # Language tokens can also attend to all tokens (mask=0)
            num_lang_tokens = self._cached_lang_emb.shape[1]
            att_masks = img_att_masks + [0] * num_lang_tokens
            prefix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=prefix_embs.device)
            prefix_att_masks = prefix_att_masks[None, :].expand(prefix_embs.shape[0], -1)

            # Compute KV cache (full prefix with proper cross-attention)
            prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )

            # Run denoising with the KV cache
            actions = self.model.sample_actions_with_external_kv(
                torch.device(self.device),
                state,
                prefix_kv_cache,
                prefix_pad_masks,
                num_steps=num_steps,
            )

        # Post-process
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)

        return {"actions": actions_np}

    def reset_cache(self):
        """Reset the language embedding cache. Call this when starting a new episode."""
        self._cached_lang_emb = None
        self._cached_lang_pad_mask = None
        self._cached_prompt = None

    def warmup(self, num_iterations: int = 5):
        """Warmup the model."""
        logger.info(f"Warming up PyTorchTextCacheBackend ({num_iterations} iterations)")

        dummy_obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "warmup",
        }

        for i in range(num_iterations):
            self.infer(dummy_obs)

        # Reset cache after warmup
        self.reset_cache()

        logger.info("Warmup complete")


class TurboTitanBackend(PyTorchBackend):
    """
    Turbo-Pi Titan Backend: High-Performance TRT with Precision Protection.

    Architecture:
    - MLP layers: Aggressive FP16/INT8 TRT optimization
    - Attention layers: FP32 softmax for precision (via precision barriers or plugin)
    - Freq=N KV cache reuse for throughput boost

    Performance (expected):
    - Full Frame: ~62.5ms (Vision 12.5ms + KV Cache 35ms + Denoise 15ms)
    - Fast Frame: ~15ms (Denoise only with cached KV)
    - Average (Freq=2): ~38.75ms → 25.8 Hz @ 100% accuracy

    This is the recommended production backend for 20+ Hz with high accuracy.
    """

    def __init__(self, config: PolicyConfig, kv_reuse_freq: int = 2):
        super().__init__(config)
        self.kv_reuse_freq = kv_reuse_freq
        self.frame_counter = 0

        # KV cache storage
        self._cached_kv = None
        self._cached_prefix_hidden = None

        # Try to load Turbo Titan TRT engine
        self._init_turbo_engine()

        logger.info(f"Initialized TurboTitanBackend (KV Reuse Freq={kv_reuse_freq})")

    def _init_turbo_engine(self):
        """Initialize Turbo Titan TRT engine if available."""
        self.use_turbo_trt = False
        self.turbo_kv_engine = None

        # Search multiple possible engine directories
        engine_dirs = []
        if self.config.engine_dir:
            engine_dirs.append(Path(self.config.engine_dir))
        engine_dirs.extend([
            Path(self.config.checkpoint_dir).parent / "onnx_exports",
            Path(__file__).parent.parent.parent.parent / "onnx_exports",  # openpi/onnx_exports
            Path.cwd() / "onnx_exports",
        ])

        # Look for Turbo Titan engine variants (priority order)
        # FP32 engines have best precision; BF16/FP16 need pycuda stream fix
        engine_candidates = [
            "paligemma_kv_cache_explicit_fp32_new.engine",  # Best: newly built FP32
            "paligemma_kv_cache_explicit_fp32.engine",      # FP32 explicit attention
            "paligemma_kv_cache_surgical_fp16.engine",      # FP32 softmax + FP16 MLP (BF16 engine)
            "paligemma_kv_cache_explicit_fp16.engine",
            "paligemma_kv_cache_fp16.engine",
        ]

        for engine_dir in engine_dirs:
            if not engine_dir.exists():
                continue
            for candidate in engine_candidates:
                engine_path = engine_dir / candidate
                if engine_path.exists():
                    try:
                        from openpi.inference.kv_cache_trt import TensorRTKVCacheEngine
                        self.turbo_kv_engine = TensorRTKVCacheEngine(str(engine_path))
                        self.use_turbo_trt = True
                        logger.info(f"Loaded Turbo Titan KV Cache engine: {engine_path}")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to load {engine_path}: {e}")

        if not self.use_turbo_trt:
            logger.info("Turbo Titan TRT engine not found - using PyTorch KV Cache with Freq reuse")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference with Turbo Titan optimizations."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        obs = self._preprocess(observation)

        # Determine if this is a full or fast frame
        should_compute_full = (self.frame_counter % self.kv_reuse_freq == 0) or \
                              (self._cached_kv is None)

        with torch.no_grad():
            if should_compute_full:
                # Full frame: compute everything
                result = self._full_frame_inference(obs, num_steps)
            else:
                # Fast frame: reuse cached KV
                result = self._fast_frame_inference(obs, num_steps)

        self.frame_counter += 1
        return result

    def _full_frame_inference(self, obs: Dict, num_steps: int) -> Dict[str, np.ndarray]:
        """Full frame inference with KV cache computation."""
        # Preprocess
        images, img_masks, tokens, token_masks, state = self.model._preprocess_observation(obs, train=False)

        # Embed prefix (4 args: images, img_masks, lang_tokens, lang_masks)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, token_masks
        )

        # Compute KV cache (use TRT if available)
        if self.use_turbo_trt and self.turbo_kv_engine is not None:
            # Use Turbo Titan TRT engine
            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

            position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            attention_mask = torch.where(prefix_att_2d[:, None, :, :], 0.0, -2.3819763e38)
            attention_mask = attention_mask.to(prefix_embs.dtype)

            # Use GPU-direct method to avoid CPU roundtrip
            keys, values, hidden = self.turbo_kv_engine.infer_gpu_direct(
                prefix_embs, position_ids, attention_mask
            )
            # Convert to list of tuples format
            prefix_kv_cache = [(keys[:, i], values[:, i]) for i in range(keys.shape[1])]
        else:
            # Use PyTorch
            prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )

        # Cache for fast frames
        self._cached_kv = prefix_kv_cache
        self._cached_prefix_hidden = (prefix_embs, prefix_pad_masks, prefix_att_masks)

        # Run denoising with external KV cache
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        # Convert bfloat16 to float32 for numpy compatibility, then unnormalize
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def _fast_frame_inference(self, obs: Dict, num_steps: int) -> Dict[str, np.ndarray]:
        """Fast frame inference using cached KV."""
        # Get cached values
        prefix_kv_cache = self._cached_kv
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._cached_prefix_hidden

        # Extract state from observation
        _, _, _, _, state = self.model._preprocess_observation(obs, train=False)

        # Run denoising with cached KV
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        # Convert bfloat16 to float32 for numpy compatibility, then unnormalize
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def reset_cache(self):
        """Reset cached KV values (call at episode start)."""
        self._cached_kv = None
        self._cached_prefix_hidden = None
        self.frame_counter = 0

    def warmup(self, num_iterations: int = 5):
        """Warmup the backend."""
        logger.info(f"Warming up TurboTitanBackend ({num_iterations} iterations)...")

        dummy_obs = {
            'observation/image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/wrist_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/state': np.zeros(8, dtype=np.float32),
            'prompt': 'pick up the black bowl',
        }

        for _ in range(num_iterations):
            self.infer(dummy_obs)

        self.reset_cache()
        logger.info("Warmup complete")


class TurboTitanTRTBackend(PyTorchBackend):
    """
    TurboAttention TRT Backend: Uses TRT Python API-built engine with TurboAttention plugin.

    This backend uses the TurboAttention TRT plugin for FP32 softmax accumulation,
    achieving ~58ms KV Cache latency (vs ~165ms for ONNX-based engines).

    Architecture:
    - KV Cache: TRT engine with TurboAttention plugin (FP32 softmax + FP16 MLP)
    - Denoise: PyTorch
    - Freq=N KV cache reuse for throughput boost

    Performance (expected):
    - KV Cache: ~58ms (TurboAttention TRT)
    - Vision+Lang: ~70ms (PyTorch)
    - Denoise (3 steps): ~45ms (PyTorch)
    - Full Frame: ~173ms
    - With Freq=2: ~109ms avg → ~9.2 Hz

    Build the engine first:
        python scripts/build_turbo_kv_cache_engine.py
    """

    def __init__(self, config: PolicyConfig, kv_reuse_freq: int = 2):
        super().__init__(config)
        self.kv_reuse_freq = kv_reuse_freq
        self.frame_counter = 0

        # KV cache storage
        self._cached_kv = None
        self._cached_prefix_hidden = None

        # Initialize TurboTRT engine
        self._init_turbo_trt_engine()

        logger.info(f"Initialized TurboTitanTRTBackend (KV Reuse Freq={kv_reuse_freq})")

    def _init_turbo_trt_engine(self):
        """Initialize TurboAttention TRT engine."""
        self.use_turbo_trt = False
        self.turbo_trt_engine = None

        try:
            from openpi.inference.turbo_trt_engine import TurboTRTEngine, find_turbo_engine, find_turbo_plugin

            # Check for plugin
            plugin_path = find_turbo_plugin()
            if plugin_path is None:
                logger.warning("TurboAttention plugin not found - falling back to PyTorch")
                return

            # Check for engine
            engine_path = find_turbo_engine()

            # Also check engine_dir from config
            if engine_path is None and self.config.engine_dir:
                from pathlib import Path
                candidate = Path(self.config.engine_dir) / "turbo_kv_cache.engine"
                if candidate.exists():
                    engine_path = str(candidate)

            if engine_path is None:
                logger.warning(
                    "TurboAttention TRT engine not found. Build it with:\n"
                    "  python scripts/build_turbo_kv_cache_engine.py\n"
                    "Falling back to PyTorch KV Cache"
                )
                return

            # Load engine
            self.turbo_trt_engine = TurboTRTEngine(
                engine_path=engine_path,
                plugin_path=plugin_path,
                device=self.device,
            )
            self.use_turbo_trt = True
            logger.info(f"Loaded TurboTRT engine: {engine_path}")

        except Exception as e:
            logger.warning(f"Failed to initialize TurboTRT engine: {e}")
            logger.warning("Falling back to PyTorch KV Cache")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference with TurboAttention TRT optimizations."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        obs = self._preprocess(observation)

        # Determine if this is a full or fast frame
        should_compute_full = (self.frame_counter % self.kv_reuse_freq == 0) or \
                              (self._cached_kv is None)

        with torch.no_grad():
            if should_compute_full:
                result = self._full_frame_inference(obs, num_steps)
            else:
                result = self._fast_frame_inference(obs, num_steps)

        self.frame_counter += 1
        return result

    def _full_frame_inference(self, obs, num_steps: int) -> Dict[str, np.ndarray]:
        """Full frame inference with TurboTRT KV cache computation."""
        # Preprocess
        images, img_masks, tokens, token_masks, state = self.model._preprocess_observation(obs, train=False)

        # Embed prefix (Vision + Language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, token_masks
        )

        # Compute KV cache
        if self.use_turbo_trt and self.turbo_trt_engine is not None:
            # Use TurboAttention TRT engine
            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

            position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            attention_mask = torch.where(prefix_att_2d[:, None, :, :], 0.0, -2.3819763e38)
            attention_mask = attention_mask.half()

            # Run TurboTRT engine
            prefix_kv_cache = self.turbo_trt_engine.infer_list(
                prefix_embs.float(), position_ids.int(), attention_mask
            )
        else:
            # Fallback to PyTorch
            prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )

        # Cache for fast frames
        self._cached_kv = prefix_kv_cache
        self._cached_prefix_hidden = (prefix_embs, prefix_pad_masks, prefix_att_masks)

        # Run denoising with external KV cache
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        # Convert and unnormalize
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def _fast_frame_inference(self, obs, num_steps: int) -> Dict[str, np.ndarray]:
        """Fast frame inference using cached KV."""
        prefix_kv_cache = self._cached_kv
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._cached_prefix_hidden

        # Extract state from observation
        _, _, _, _, state = self.model._preprocess_observation(obs, train=False)

        # Run denoising with cached KV
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def reset_cache(self):
        """Reset cached KV values (call at episode start)."""
        self._cached_kv = None
        self._cached_prefix_hidden = None
        self.frame_counter = 0

    def warmup(self, num_iterations: int = 5):
        """Warmup the backend."""
        logger.info(f"Warming up TurboTitanTRTBackend ({num_iterations} iterations)...")

        if self.turbo_trt_engine is not None:
            self.turbo_trt_engine.warmup(num_iterations)

        dummy_obs = {
            'observation/image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/wrist_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/state': np.zeros(8, dtype=np.float32),
            'prompt': 'pick up the black bowl',
        }

        for _ in range(num_iterations):
            self.infer(dummy_obs)

        self.reset_cache()
        logger.info("Warmup complete")


class TorchTRTBackend(PyTorchBackend):
    """
    Torch-TensorRT Backend for KV Cache acceleration.

    Uses torch_tensorrt to compile the KV Cache model, which handles SDPA
    attention natively without triggering Myelin kernel crashes on Jetson Thor.

    Advantages over ONNX-based TRT:
    - Direct PyTorch → TensorRT compilation (no ONNX export)
    - SDPA attention is compiled natively
    - Avoids Myelin fusion issues that cause CUDA error 400
    - 1.45x speedup validated on Thor (13.5ms vs 19ms for 4 layers)

    Architecture:
    - KV Cache: TorchTRT-compiled 18-layer transformer
    - Denoise: PyTorch (using cached KV)
    - Optional KV reuse for higher throughput

    Build the compiled model first:
        python scripts/build_torch_trt_kv_cache.py --save-compiled
    """

    def __init__(self, config: PolicyConfig, kv_reuse_freq: int = 2, compile_on_init: bool = True):
        super().__init__(config)
        self.kv_reuse_freq = kv_reuse_freq
        self.frame_counter = 0

        # KV cache storage
        self._cached_kv = None
        self._cached_prefix_hidden = None

        # Initialize TorchTRT engine
        self._init_torch_trt_engine(compile_on_init)

        logger.info(f"Initialized TorchTRTBackend (KV Reuse Freq={kv_reuse_freq})")

    def _init_torch_trt_engine(self, compile_on_init: bool):
        """Initialize TorchTRT KV Cache engine."""
        self.use_torch_trt = False
        self.torch_trt_engine = None

        try:
            from openpi.inference.torch_trt_kv_cache import (
                TorchTRTKVCacheEngine,
                find_torch_trt_compiled_model,
            )

            # Create engine
            self.torch_trt_engine = TorchTRTKVCacheEngine(
                checkpoint_dir=self.config.checkpoint_dir,
                device=self.config.device,
            )

            # Check for pre-compiled model or compile now
            compiled_path = find_torch_trt_compiled_model(self.config.checkpoint_dir)

            if compiled_path:
                logger.info(f"Found pre-compiled TorchTRT model: {compiled_path}")
                self.torch_trt_engine._load_compiled_model(compiled_path)
                self.use_torch_trt = True
            elif compile_on_init:
                logger.info("Compiling TorchTRT model (this may take a while)...")
                if self.torch_trt_engine.compile():
                    self.use_torch_trt = True
                    logger.info("TorchTRT compilation successful!")
                else:
                    logger.warning("TorchTRT compilation failed, falling back to PyTorch")
            else:
                logger.info("TorchTRT model not compiled yet. Call compile() to compile.")

        except ImportError as e:
            logger.warning(f"torch_tensorrt not available: {e}")
            logger.warning("Falling back to PyTorch KV Cache")
        except Exception as e:
            logger.warning(f"Failed to initialize TorchTRT engine: {e}")
            logger.warning("Falling back to PyTorch KV Cache")

    def compile(self, save_path: Optional[str] = None) -> bool:
        """Compile the TorchTRT model."""
        if self.torch_trt_engine is None:
            logger.error("TorchTRT engine not initialized")
            return False
        return self.torch_trt_engine.compile(save_path)

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference with TorchTRT KV Cache acceleration."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        obs = self._preprocess(observation)

        # Determine if this is a full or fast frame
        should_compute_full = (self.frame_counter % self.kv_reuse_freq == 0) or \
                              (self._cached_kv is None)

        with torch.no_grad():
            if should_compute_full:
                result = self._full_frame_inference(obs, num_steps)
            else:
                result = self._fast_frame_inference(obs, num_steps)

        self.frame_counter += 1
        return result

    def _full_frame_inference(self, obs, num_steps: int) -> Dict[str, np.ndarray]:
        """Full frame inference with TorchTRT KV cache computation."""
        # Preprocess
        images, img_masks, tokens, token_masks, state = self.model._preprocess_observation(obs, train=False)

        # Embed prefix (Vision + Language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, token_masks
        )

        # Compute KV cache
        if self.use_torch_trt and self.torch_trt_engine is not None:
            # Use TorchTRT engine
            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

            position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            attention_mask = torch.where(prefix_att_2d[:, None, :, :], 0.0, -2.3819763e38)
            attention_mask = attention_mask.to(prefix_embs.dtype)

            # Run TorchTRT engine
            prefix_kv_cache = self.torch_trt_engine.infer_list(
                prefix_embs, position_ids, attention_mask
            )
        else:
            # Fallback to PyTorch
            prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )

        # Cache for fast frames
        self._cached_kv = prefix_kv_cache
        self._cached_prefix_hidden = (prefix_embs, prefix_pad_masks, prefix_att_masks)

        # Run denoising with external KV cache
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        # Convert and unnormalize
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def _fast_frame_inference(self, obs, num_steps: int) -> Dict[str, np.ndarray]:
        """Fast frame inference using cached KV."""
        prefix_kv_cache = self._cached_kv
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._cached_prefix_hidden

        # Extract state from observation
        _, _, _, _, state = self.model._preprocess_observation(obs, train=False)

        # Run denoising with cached KV
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def reset_cache(self):
        """Reset cached KV values (call at episode start)."""
        self._cached_kv = None
        self._cached_prefix_hidden = None
        self.frame_counter = 0

    def warmup(self, num_iterations: int = 5):
        """Warmup the backend."""
        logger.info(f"Warming up TorchTRTBackend ({num_iterations} iterations)...")

        if self.torch_trt_engine is not None:
            self.torch_trt_engine.warmup(num_iterations)

        dummy_obs = {
            'observation/image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/wrist_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/state': np.zeros(8, dtype=np.float32),
            'prompt': 'pick up the black bowl',
        }

        for _ in range(num_iterations):
            self.infer(dummy_obs)

        self.reset_cache()
        logger.info("Warmup complete")

    def benchmark(self) -> dict:
        """Run benchmark comparing PyTorch vs TorchTRT."""
        if self.torch_trt_engine is not None:
            return self.torch_trt_engine.benchmark()
        else:
            logger.warning("TorchTRT engine not available for benchmark")
            return {}


class FP8MLPBackend(PyTorchBackend):
    """
    FP8 MLP Backend: FP16 Attention + FP8 MLP for optimized inference.

    Uses torch._scaled_mm for FP8 Tensor Core acceleration on MLP layers
    while keeping attention in FP16 for precision.

    Architecture:
    - Attention: FP16 (preserves softmax precision)
    - MLP: FP8 using torch._scaled_mm (1.75x speedup per GEMM)
    - KV Reuse: Optional (Freq=N) for higher throughput

    Performance:
    - FP16 baseline: ~87 ms (18 layers)
    - FP8 MLP: ~70 ms expected (1.24x speedup)
    - With Freq=2 reuse: ~42.5 ms avg → ~23.5 Hz

    This backend provides the best balance of speed and accuracy on Thor.
    """

    def __init__(self, config: PolicyConfig, kv_reuse_freq: int = 2, use_static_scale: bool = True):
        super().__init__(config)
        self.kv_reuse_freq = kv_reuse_freq
        self.use_static_scale = use_static_scale
        self.frame_counter = 0

        # KV cache storage
        self._cached_kv = None
        self._cached_prefix_hidden = None

        # Initialize FP8 KV Cache engine
        self._init_fp8_engine()

        logger.info(f"Initialized FP8MLPBackend (KV Reuse Freq={kv_reuse_freq}, Static Scale={use_static_scale})")

    def _init_fp8_engine(self):
        """Initialize FP8 KV Cache engine."""
        self.use_fp8 = False
        self.fp8_kv_engine = None

        try:
            from openpi.inference.fp8_kv_cache import FP8KVCacheEngine

            self.fp8_kv_engine = FP8KVCacheEngine(
                checkpoint_dir=self.config.checkpoint_dir,
                device=self.config.device,
                use_static_scale=self.use_static_scale,
            )
            self.use_fp8 = True
            logger.info("FP8 KV Cache Engine initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize FP8 KV Cache Engine: {e}")
            logger.warning("Falling back to PyTorch FP16 KV Cache")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference with FP8 MLP acceleration."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        obs = self._preprocess(observation)

        # Determine if this is a full or fast frame
        should_compute_full = (self.frame_counter % self.kv_reuse_freq == 0) or \
                              (self._cached_kv is None)

        with torch.no_grad():
            if should_compute_full:
                result = self._full_frame_inference(obs, num_steps)
            else:
                result = self._fast_frame_inference(obs, num_steps)

        self.frame_counter += 1
        return result

    def _full_frame_inference(self, obs, num_steps: int) -> Dict[str, np.ndarray]:
        """Full frame inference with FP8 KV cache computation."""
        # Preprocess
        images, img_masks, tokens, token_masks, state = self.model._preprocess_observation(obs, train=False)

        # Embed prefix (Vision + Language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, token_masks
        )

        # Compute KV cache
        if self.use_fp8 and self.fp8_kv_engine is not None:
            # Use FP8 KV Cache Engine
            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

            position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            attention_mask = torch.where(prefix_att_2d[:, None, :, :], 0.0, -2.3819763e38)
            attention_mask = attention_mask.to(prefix_embs.dtype)

            # Run FP8 engine
            prefix_kv_cache = self.fp8_kv_engine.infer_list(
                prefix_embs, position_ids, attention_mask
            )
        else:
            # Fallback to PyTorch
            prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )

        # Cache for fast frames
        self._cached_kv = prefix_kv_cache
        self._cached_prefix_hidden = (prefix_embs, prefix_pad_masks, prefix_att_masks)

        # Run denoising with external KV cache
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        # Convert and unnormalize
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def _fast_frame_inference(self, obs, num_steps: int) -> Dict[str, np.ndarray]:
        """Fast frame inference using cached KV."""
        prefix_kv_cache = self._cached_kv
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._cached_prefix_hidden

        # Extract state from observation
        _, _, _, _, state = self.model._preprocess_observation(obs, train=False)

        # Run denoising with cached KV
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def reset_cache(self):
        """Reset cached KV values (call at episode start)."""
        self._cached_kv = None
        self._cached_prefix_hidden = None
        self.frame_counter = 0

    def warmup(self, num_iterations: int = 5):
        """Warmup the backend."""
        logger.info(f"Warming up FP8MLPBackend ({num_iterations} iterations)...")

        if self.fp8_kv_engine is not None:
            self.fp8_kv_engine.warmup(num_iterations)

        dummy_obs = {
            'observation/image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/wrist_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/state': np.zeros(8, dtype=np.float32),
            'prompt': 'pick up the black bowl',
        }

        for _ in range(num_iterations):
            self.infer(dummy_obs)

        self.reset_cache()
        logger.info("Warmup complete")

    def benchmark(self) -> dict:
        """Run benchmark for FP8 KV Cache."""
        if self.fp8_kv_engine is not None:
            return self.fp8_kv_engine.benchmark()
        else:
            logger.warning("FP8 KV Cache engine not available for benchmark")
            return {}


class FlashFP8Backend(PyTorchBackend):
    """
    Flash Attention + FP8 Hybrid MLP Backend: FASTEST KV Cache implementation.

    Combines:
    - Flash Attention: 2.13x speedup over manual attention
    - FP8 Hybrid MLP: 1.12x speedup over FP16 MLP

    Architecture:
    - Attention: Flash Attention (FP16 with FP32 softmax accumulation)
    - MLP: FP8 Hybrid (FP8 gate/up, FP16 down) to avoid quantization overhead
    - KV Reuse: Optional (Freq=N) for higher throughput

    Performance (Thor, seq=970, 18 layers):
    - Manual Attention + FP16 MLP: ~87 ms (11.5 Hz)
    - Flash Attention + FP8 MLP: ~75 ms (13.3 Hz) → 1.15x speedup

    This backend provides the BEST performance on Thor.
    """

    def __init__(self, config: PolicyConfig, kv_reuse_freq: int = 2, use_fp8_mlp: bool = True):
        super().__init__(config)
        self.kv_reuse_freq = kv_reuse_freq
        self.use_fp8_mlp = use_fp8_mlp
        self.frame_counter = 0

        # KV cache storage
        self._cached_kv = None
        self._cached_prefix_hidden = None

        # Initialize Flash + FP8 KV Cache engine
        self._init_flash_fp8_engine()

        logger.info(f"Initialized FlashFP8Backend (KV Reuse Freq={kv_reuse_freq}, FP8 MLP={use_fp8_mlp})")

    def _init_flash_fp8_engine(self):
        """Initialize Flash Attention + FP8 KV Cache engine."""
        self.use_flash_fp8 = False
        self.flash_fp8_engine = None

        try:
            from openpi.inference.flash_fp8_kv_cache import FlashFP8KVCacheEngine

            self.flash_fp8_engine = FlashFP8KVCacheEngine(
                checkpoint_dir=self.config.checkpoint_dir,
                device=self.config.device,
                use_fp8_mlp=self.use_fp8_mlp,
            )
            self.use_flash_fp8 = True
            logger.info("Flash + FP8 KV Cache Engine initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize Flash + FP8 KV Cache Engine: {e}")
            logger.warning("Falling back to PyTorch KV Cache")
            import traceback
            traceback.print_exc()

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference with Flash Attention + FP8 MLP acceleration."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        obs = self._preprocess(observation)

        # Determine if this is a full or fast frame
        should_compute_full = (self.frame_counter % self.kv_reuse_freq == 0) or \
                              (self._cached_kv is None)

        with torch.no_grad():
            if should_compute_full:
                result = self._full_frame_inference(obs, num_steps)
            else:
                result = self._fast_frame_inference(obs, num_steps)

        self.frame_counter += 1
        return result

    def _full_frame_inference(self, obs, num_steps: int) -> Dict[str, np.ndarray]:
        """Full frame inference with Flash + FP8 KV cache computation."""
        # Preprocess
        images, img_masks, tokens, token_masks, state = self.model._preprocess_observation(obs, train=False)

        # Embed prefix (Vision + Language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, token_masks
        )

        # Compute KV cache
        if self.use_flash_fp8 and self.flash_fp8_engine is not None:
            # Use Flash + FP8 KV Cache Engine
            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

            position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            attention_mask = torch.where(prefix_att_2d[:, None, :, :], 0.0, -2.3819763e38)
            attention_mask = attention_mask.to(prefix_embs.dtype)

            # Run Flash + FP8 engine
            prefix_kv_cache = self.flash_fp8_engine.infer_list(
                prefix_embs, position_ids, attention_mask
            )
        else:
            # Fallback to PyTorch
            prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )

        # Cache for fast frames
        self._cached_kv = prefix_kv_cache
        self._cached_prefix_hidden = (prefix_embs, prefix_pad_masks, prefix_att_masks)

        # Run denoising with external KV cache
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        # Convert and unnormalize
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def _fast_frame_inference(self, obs, num_steps: int) -> Dict[str, np.ndarray]:
        """Fast frame inference using cached KV."""
        prefix_kv_cache = self._cached_kv
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._cached_prefix_hidden

        # Extract state from observation
        _, _, _, _, state = self.model._preprocess_observation(obs, train=False)

        # Run denoising with cached KV
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def reset_cache(self):
        """Reset cached KV values (call at episode start)."""
        self._cached_kv = None
        self._cached_prefix_hidden = None
        self.frame_counter = 0

    def warmup(self, num_iterations: int = 5):
        """Warmup the backend."""
        logger.info(f"Warming up FlashFP8Backend ({num_iterations} iterations)...")

        if self.flash_fp8_engine is not None:
            self.flash_fp8_engine.warmup(num_iterations)

        dummy_obs = {
            'observation/image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/wrist_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/state': np.zeros(8, dtype=np.float32),
            'prompt': 'pick up the black bowl',
        }

        for _ in range(num_iterations):
            self.infer(dummy_obs)

        self.reset_cache()
        logger.info("Warmup complete")

    def benchmark(self) -> dict:
        """Run benchmark for Flash + FP8 KV Cache."""
        if self.flash_fp8_engine is not None:
            return self.flash_fp8_engine.benchmark()
        else:
            logger.warning("Flash + FP8 KV Cache engine not available for benchmark")
            return {}


class TorchTRTFP8Backend(PyTorchBackend):
    """
    Torch-TRT FP8 Backend: Uses ModelOpt + Torch-TensorRT compiled FP8 MLP.

    This is the CORRECT way to use FP8 for acceleration (2.94x speedup).
    PyTorch native FP8 (torch._scaled_mm) has NO speedup due to quantization overhead.

    Performance (Thor, seq=970, 18 layers MLP):
    - PyTorch FP16:          59.89 ms (baseline)
    - PyTorch native FP8:    ~60 ms (1.00x - NO speedup!)
    - Torch-TRT FP8:         20.39 ms (2.94x speedup)

    Expected Full Pipeline:
    - Current (PyTorch FP8): 180 ms (5.5 Hz)
    - With Torch-TRT FP8:    ~140 ms (7.1 Hz)
    """

    def __init__(self, config: PolicyConfig, kv_reuse_freq: int = 1):
        super().__init__(config)
        self.kv_reuse_freq = kv_reuse_freq
        self.frame_counter = 0

        # KV cache storage
        self._cached_kv = None
        self._cached_prefix_hidden = None

        # Initialize Torch-TRT FP8 KV Cache engine
        self._init_torch_trt_fp8_engine()

        logger.info(f"Initialized TorchTRTFP8Backend (KV Reuse Freq={kv_reuse_freq})")

    def _init_torch_trt_fp8_engine(self):
        """Initialize Torch-TRT FP8 KV Cache engine."""
        self.use_trt_fp8 = False
        self.trt_fp8_engine = None

        try:
            from openpi.inference.torch_trt_fp8_kv_cache import TorchTRTFP8KVCacheEngine

            self.trt_fp8_engine = TorchTRTFP8KVCacheEngine(
                checkpoint_dir=self.config.checkpoint_dir,
                device=self.config.device,
                compile_trt=True,
            )
            self.use_trt_fp8 = True
            logger.info("Torch-TRT FP8 KV Cache Engine initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize Torch-TRT FP8 KV Cache Engine: {e}")
            logger.warning("Falling back to PyTorch KV Cache")
            import traceback
            traceback.print_exc()

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """Run inference with Torch-TRT FP8 MLP acceleration."""
        if num_steps is None:
            num_steps = self.config.num_denoising_steps

        obs = self._preprocess(observation)

        # Determine if this is a full or fast frame
        should_compute_full = (self.frame_counter % self.kv_reuse_freq == 0) or \
                              (self._cached_kv is None)

        with torch.no_grad():
            if should_compute_full:
                result = self._full_frame_inference(obs, num_steps)
            else:
                result = self._fast_frame_inference(obs, num_steps)

        self.frame_counter += 1
        return result

    def _full_frame_inference(self, obs, num_steps: int) -> Dict[str, np.ndarray]:
        """Full frame inference with Torch-TRT FP8 KV cache computation."""
        # Preprocess
        images, img_masks, tokens, token_masks, state = self.model._preprocess_observation(obs, train=False)

        # Embed prefix (Vision + Language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, token_masks
        )

        # Compute KV cache
        if self.use_trt_fp8 and self.trt_fp8_engine is not None:
            # Use Torch-TRT FP8 KV Cache Engine
            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

            position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            attention_mask = torch.where(prefix_att_2d[:, None, :, :], 0.0, -2.3819763e38)
            attention_mask = attention_mask.to(prefix_embs.dtype)

            # Run Torch-TRT FP8 engine
            prefix_kv_cache = self.trt_fp8_engine.infer_list(
                prefix_embs, position_ids, attention_mask
            )
        else:
            # Fallback to PyTorch
            prefix_kv_cache = self.model.compute_prefix_kv_cache(
                prefix_embs, prefix_pad_masks, prefix_att_masks
            )

        # Cache for fast frames
        self._cached_kv = prefix_kv_cache
        self._cached_prefix_hidden = (prefix_embs, prefix_pad_masks, prefix_att_masks)

        # Run denoising with external KV cache
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        # Convert and unnormalize
        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def _fast_frame_inference(self, obs, num_steps: int) -> Dict[str, np.ndarray]:
        """Fast frame inference using cached KV."""
        prefix_kv_cache = self._cached_kv
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._cached_prefix_hidden

        # Extract state from observation
        _, _, _, _, state = self.model._preprocess_observation(obs, train=False)

        # Run denoising with cached KV
        actions = self.model.sample_actions_with_external_kv(
            device=torch.device(self.device),
            state=state,
            prefix_kv_cache=prefix_kv_cache,
            prefix_pad_masks=prefix_pad_masks,
            num_steps=num_steps
        )

        actions_np = actions.float().cpu().numpy()[0]
        actions_np = self._unnormalize_actions(actions_np)
        return {"actions": actions_np}

    def reset_cache(self):
        """Reset cached KV values (call at episode start)."""
        self._cached_kv = None
        self._cached_prefix_hidden = None
        self.frame_counter = 0

    def warmup(self, num_iterations: int = 5):
        """Warmup the backend."""
        logger.info(f"Warming up TorchTRTFP8Backend ({num_iterations} iterations)...")

        if self.trt_fp8_engine is not None:
            self.trt_fp8_engine.warmup(num_iterations)

        dummy_obs = {
            'observation/image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/wrist_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'observation/state': np.zeros(8, dtype=np.float32),
            'prompt': 'pick up the black bowl',
        }

        for _ in range(num_iterations):
            self.infer(dummy_obs)

        self.reset_cache()
        logger.info("Warmup complete")

    def benchmark(self) -> dict:
        """Run benchmark for Torch-TRT FP8 KV Cache."""
        if self.trt_fp8_engine is not None:
            return self.trt_fp8_engine.benchmark()
        else:
            logger.warning("Torch-TRT FP8 KV Cache engine not available for benchmark")
            return {}


class UnifiedPolicy:
    """
    Unified policy interface for Pi0.5 VLA model.

    Provides consistent API across different inference backends.
    """

    BACKENDS = {
        # PyTorch backends (baseline)
        "pytorch": PyTorchBackend,
        "pytorch_pipelined": PyTorchPipelinedBackend,

        # PyTorch KV Reuse backends (trades visual latency for speed)
        "pytorch_kv_reuse": lambda cfg: PyTorchKVReuseBackend(cfg, reuse_freq=3),
        "pytorch_kv_reuse_2": lambda cfg: PyTorchKVReuseBackend(cfg, reuse_freq=2),
        "pytorch_kv_reuse_4": lambda cfg: PyTorchKVReuseBackend(cfg, reuse_freq=4),

        # Text-Only Prefix Caching (caches language KV, recomputes image KV each frame)
        "pytorch_text_cache": PyTorchTextCacheBackend,

        # Turbo-Pi Titan backends (uses ONNX-exported TRT engine)
        "turbo_titan": TurboTitanBackend,  # Freq=2 default
        "turbo_titan_freq1": lambda cfg: TurboTitanBackend(cfg, kv_reuse_freq=1),  # No KV reuse, highest accuracy
        "turbo_titan_freq2": lambda cfg: TurboTitanBackend(cfg, kv_reuse_freq=2),  # ~25.8 Hz
        "turbo_titan_freq3": lambda cfg: TurboTitanBackend(cfg, kv_reuse_freq=3),  # ~28.5 Hz

        # TurboAttention TRT backends (uses TRT Python API with TurboAttention plugin)
        # Build engine first: python scripts/build_turbo_kv_cache_engine.py
        "turbo_trt": TurboTitanTRTBackend,  # Freq=2 default, ~58ms KV Cache
        "turbo_trt_freq1": lambda cfg: TurboTitanTRTBackend(cfg, kv_reuse_freq=1),  # No KV reuse
        "turbo_trt_freq2": lambda cfg: TurboTitanTRTBackend(cfg, kv_reuse_freq=2),  # Default

        # Torch-TensorRT backends (RECOMMENDED for Thor - avoids Myelin crashes)
        # Build model first: python scripts/build_torch_trt_kv_cache.py --save-compiled
        "torch_trt": TorchTRTBackend,  # Freq=2 default, 1.45x speedup
        "torch_trt_freq1": lambda cfg: TorchTRTBackend(cfg, kv_reuse_freq=1),  # No KV reuse
        "torch_trt_freq2": lambda cfg: TorchTRTBackend(cfg, kv_reuse_freq=2),  # Default
        "torch_trt_freq3": lambda cfg: TorchTRTBackend(cfg, kv_reuse_freq=3),  # Higher throughput

        # FP8 MLP backends (FP16 Attention + FP8 MLP using torch._scaled_mm)
        # Best balance of speed and accuracy on Thor
        "fp8_mlp": FP8MLPBackend,  # Freq=2 default, ~1.24x speedup
        "fp8_mlp_freq1": lambda cfg: FP8MLPBackend(cfg, kv_reuse_freq=1),  # No KV reuse, highest accuracy
        "fp8_mlp_freq2": lambda cfg: FP8MLPBackend(cfg, kv_reuse_freq=2),  # Default
        "fp8_mlp_freq3": lambda cfg: FP8MLPBackend(cfg, kv_reuse_freq=3),  # Higher throughput
        "fp8_mlp_dynamic": lambda cfg: FP8MLPBackend(cfg, kv_reuse_freq=2, use_static_scale=False),  # Dynamic scaling

        # Flash Attention + FP8 Hybrid MLP backends (uses PyTorch native FP8 - NO speedup!)
        # Warning: PyTorch native FP8 (torch._scaled_mm) has hidden quantization overhead
        # Use torch_trt_fp8 instead for real 2.94x MLP speedup
        "flash_fp8": FlashFP8Backend,  # Freq=2 default, ~180ms (NO speedup!)
        "flash_fp8_freq1": lambda cfg: FlashFP8Backend(cfg, kv_reuse_freq=1),  # No KV reuse
        "flash_fp8_freq2": lambda cfg: FlashFP8Backend(cfg, kv_reuse_freq=2),  # Default
        "flash_fp8_freq3": lambda cfg: FlashFP8Backend(cfg, kv_reuse_freq=3),  # Higher throughput
        "flash_fp16": lambda cfg: FlashFP8Backend(cfg, kv_reuse_freq=2, use_fp8_mlp=False),  # Flash only, no FP8
        "flash_fp16_freq1": lambda cfg: FlashFP8Backend(cfg, kv_reuse_freq=1, use_fp8_mlp=False),  # No KV reuse

        # Torch-TRT FP8 backends (RECOMMENDED - uses ModelOpt + Torch-TensorRT for REAL FP8 speedup)
        # MLP speedup: 2.94x (vs 1.0x for PyTorch native FP8)
        # Expected pipeline: ~140ms (7.1 Hz)
        "torch_trt_fp8": TorchTRTFP8Backend,  # No KV reuse, highest accuracy, ~140ms
        "torch_trt_fp8_freq1": lambda cfg: TorchTRTFP8Backend(cfg, kv_reuse_freq=1),  # Same as above
        "torch_trt_fp8_freq2": lambda cfg: TorchTRTFP8Backend(cfg, kv_reuse_freq=2),  # KV reuse

        # W8A16 TensorRT backends (has precision issues - use turbo_titan instead)
        "tensorrt_w8a16": TensorRTW8A16Backend,  # W8A16 KV Cache, 42ms, 0.046% error
        "tensorrt_w8a16_reuse": lambda cfg: TensorRTW8A16KVReuseBackend(cfg, reuse_freq=2),  # 23.5 Hz
        "tensorrt_w8a16_reuse_3": lambda cfg: TensorRTW8A16KVReuseBackend(cfg, reuse_freq=3),  # Higher Hz but lower accuracy

        # Hybrid TensorRT backends (Vision TRT + PyTorch KV Cache)
        "tensorrt": lambda cfg: HybridTensorRTBackend(cfg, pipelined=False),
        "tensorrt_pipelined": lambda cfg: HybridTensorRTBackend(cfg, pipelined=True),

        # Triple-stream pipelined backends
        "triple_pipelined": TripleStreamBackend,
        "tensorrt_triple_pipelined": lambda cfg: TripleStreamBackend(cfg, kv_cache_engine_path=cfg.engine_dir),

        # Full TRT backends (default: PyTorch KV Cache for accuracy)
        "full_trt": FullTRTBackend,

        # Full TRT with forced TRT KV Cache (for benchmarking only - has precision issues)
        "full_trt_benchmark": lambda cfg: FullTRTBackend(cfg, kv_cache_engine_path=cfg.engine_dir, force_trt=True),

        # Legacy aliases
        "full_trt_int8": FullTRTBackend,
    }

    def __init__(
        self,
        checkpoint_dir: str,
        backend: str = "pytorch",
        num_denoising_steps: int = 3,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Initialize unified policy.

        Args:
            checkpoint_dir: Path to model checkpoint
            backend: One of "pytorch", "tensorrt", "tensorrt_pipelined"
            num_denoising_steps: Number of denoising steps
            device: Device to run on
            **kwargs: Additional configuration options
        """
        self.config = PolicyConfig(
            checkpoint_dir=checkpoint_dir,
            backend=backend,
            num_denoising_steps=num_denoising_steps,
            device=device,
            **{k: v for k, v in kwargs.items() if hasattr(PolicyConfig, k)},
        )

        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Available: {list(self.BACKENDS.keys())}")

        # Create backend
        backend_factory = self.BACKENDS[backend]
        if callable(backend_factory) and not isinstance(backend_factory, type):
            self.backend = backend_factory(self.config)
        else:
            self.backend = backend_factory(self.config)

        logger.info(f"Created UnifiedPolicy with backend={backend}, steps={num_denoising_steps}")

    def infer(self, observation: Dict[str, Any], num_steps: int = None) -> Dict[str, np.ndarray]:
        """
        Run inference on observation.

        Args:
            observation: Dict with keys:
                - "observation/image": uint8 array (H, W, 3)
                - "observation/wrist_image": uint8 array (H, W, 3)
                - "observation/state": float32 array (state_dim,)
                - "prompt": str
            num_steps: Optional override for denoising steps

        Returns:
            Dict with "actions" key containing (horizon, action_dim) array
        """
        return self.backend.infer(observation, num_steps)

    def warmup(self, num_iterations: int = 5):
        """Warmup the policy."""
        self.backend.warmup(num_iterations)

    @property
    def num_denoising_steps(self) -> int:
        return self.config.num_denoising_steps

    @property
    def backend_name(self) -> str:
        return self.config.backend
