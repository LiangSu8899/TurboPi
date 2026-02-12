"""Utility modules for OpenPI."""

from .w4a16_packer import W4A16Packer, W4A16PackerFast, pack_linear_weight

# model_patcher imports are done lazily to avoid circular dependency
# Users should import directly: from openpi.utils.model_patcher import ...

__all__ = [
    "W4A16Packer",
    "W4A16PackerFast",
    "pack_linear_weight",
]
