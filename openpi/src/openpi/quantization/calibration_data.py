"""Calibration data loader for Pi0.5 quantization.

Provides calibration samples for PTQ (Post-Training Quantization).
Supports both synthetic data and real LIBERO dataset.
"""

import torch
from dataclasses import dataclass
from typing import Iterator


@dataclass
class CalibrationSample:
    """A single calibration sample for Pi0.5."""

    images: dict[str, torch.Tensor]  # {camera_name: (1, 3, 224, 224)}
    image_masks: dict[str, torch.Tensor]  # {camera_name: (1,)}
    state: torch.Tensor  # (1, state_dim)
    tokenized_prompt: torch.Tensor  # (1, max_token_len)
    tokenized_prompt_mask: torch.Tensor  # (1, max_token_len)


class SyntheticCalibrationDataset:
    """Generate synthetic calibration data for quantization.

    This is useful when real data is not available or for quick testing.
    The synthetic data covers the expected input distribution.
    """

    def __init__(
        self,
        num_samples: int = 512,
        image_size: int = 224,
        state_dim: int = 32,
        max_token_len: int = 200,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize synthetic calibration dataset.

        Args:
            num_samples: Number of calibration samples to generate
            image_size: Image resolution (assumes square)
            state_dim: Robot state dimension
            max_token_len: Maximum token sequence length
            device: Device to create tensors on
            dtype: Data type for floating point tensors
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.state_dim = state_dim
        self.max_token_len = max_token_len
        self.device = device
        self.dtype = dtype

        # Camera configuration (Pi0.5 LIBERO uses 3 cameras)
        self.camera_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[CalibrationSample]:
        """Yield calibration samples."""
        for i in range(self.num_samples):
            yield self._generate_sample(seed=i)

    def _generate_sample(self, seed: int) -> CalibrationSample:
        """Generate a single calibration sample.

        Uses different random distributions to cover the expected input space:
        - Images: Normal distribution (ImageNet-like normalized)
        - State: Uniform in [-1, 1] (typical robot state range)
        - Tokens: Random integers in vocab range
        """
        torch.manual_seed(seed)

        # Generate images with ImageNet-like distribution
        images = {}
        image_masks = {}
        for idx, name in enumerate(self.camera_names):
            # First two cameras active, third optional
            is_active = idx < 2 or torch.rand(1).item() > 0.5

            if is_active:
                # Normal distribution, mean ~0, std ~1 (normalized images)
                img = torch.randn(1, 3, self.image_size, self.image_size,
                                  device=self.device, dtype=self.dtype)
                mask = torch.ones(1, device=self.device, dtype=torch.bool)
            else:
                # Inactive camera: zeros
                img = torch.zeros(1, 3, self.image_size, self.image_size,
                                  device=self.device, dtype=self.dtype)
                mask = torch.zeros(1, device=self.device, dtype=torch.bool)

            images[name] = img
            image_masks[name] = mask

        # Robot state: uniform in [-1, 1]
        state = torch.rand(1, self.state_dim, device=self.device, dtype=self.dtype) * 2 - 1

        # Language tokens: mix of padding and actual tokens
        # Simulate variable-length prompts (20-150 tokens)
        prompt_len = torch.randint(20, 150, (1,)).item()
        tokenized_prompt = torch.zeros(1, self.max_token_len, device=self.device, dtype=torch.long)
        # Random token IDs (Gemma vocab size is ~256k, but most are <50k)
        tokenized_prompt[0, :prompt_len] = torch.randint(1, 50000, (prompt_len,))

        tokenized_prompt_mask = torch.zeros(1, self.max_token_len, device=self.device, dtype=torch.bool)
        tokenized_prompt_mask[0, :prompt_len] = True

        return CalibrationSample(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )


def create_observation_from_sample(sample: CalibrationSample):
    """Convert CalibrationSample to Observation for model inference.

    Args:
        sample: CalibrationSample instance

    Returns:
        Observation dataclass compatible with PI0Pytorch.sample_actions()
    """
    from openpi.models_pytorch.pi0_pytorch import Observation

    return Observation(
        images=sample.images,
        image_masks=sample.image_masks,
        state=sample.state,
        tokenized_prompt=sample.tokenized_prompt,
        tokenized_prompt_mask=sample.tokenized_prompt_mask,
    )


def calibration_forward_loop(model, dataset: SyntheticCalibrationDataset, num_samples: int | None = None):
    """Run forward passes for calibration data collection.

    This function is used by ModelOpt to collect activation statistics
    for quantization calibration.

    Args:
        model: PI0Pytorch model instance
        dataset: Calibration dataset
        num_samples: Number of samples to use (default: all)
    """
    device = next(model.parameters()).device
    num_samples = num_samples or len(dataset)

    print(f"Running calibration forward passes ({num_samples} samples)...")

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break

            observation = create_observation_from_sample(sample)

            # Run inference (single denoising step for efficiency)
            _ = model.sample_actions(device, observation, num_steps=1, use_kv_cache=True)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_samples} samples")

    print("Calibration complete.")
