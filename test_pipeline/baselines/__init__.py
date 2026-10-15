"""
Baseline registry system for comparing different human motion reconstruction methods.

This module provides a registry-based architecture for:
- Defining baseline method specifications
- Implementing baseline-specific preprocessing, inference, and postprocessing hooks
- Computing baseline-specific evaluation metrics

All methods (including "ours") are implemented as baselines with full hook implementations.
"""

from .registry import BaselineRegistry
from .spec import BaselineSpec
from .context import InferenceContext, create_inference_context
from .hooks import (
    PreprocessingHook,
    InferenceHook,
    PostprocessingHook,
    MetricsHook,
)

# Import implementations to trigger registration
from . import implementations


def get_baseline(name: str) -> BaselineSpec:
    """
    Convenience function to get a baseline spec by name.
    
    Args:
        name: Baseline identifier (case-insensitive)
        
    Returns:
        BaselineSpec instance
        
    Raises:
        KeyError: If baseline not found
    """
    return BaselineRegistry.get(name)


__all__ = [
    'BaselineRegistry',
    'BaselineSpec',
    'InferenceContext',
    'create_inference_context',
    'PreprocessingHook',
    'InferenceHook',
    'PostprocessingHook',
    'MetricsHook',
    'get_baseline',
]
