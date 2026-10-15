"""
Hook interfaces for baseline-specific behavior.

This module defines the abstract base classes for all hooks that baselines must implement.
Each baseline provides concrete implementations of these hooks to customize:
- Preprocessing: batch conditioning and masking
- Inference: TrajNet/PoseNet execution and trajectory injection
- Postprocessing: output reconstruction and world-frame recovery
- Metrics: baseline-specific evaluation metrics
"""

from .base import (
    PreprocessingHook,
    InferenceHook,
    PostprocessingHook,
    MetricsHook,
)

__all__ = [
    'PreprocessingHook',
    'InferenceHook',
    'PostprocessingHook',
    'MetricsHook',
]
