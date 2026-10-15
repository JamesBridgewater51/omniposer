"""
Modular test pipeline for RoHM motion reconstruction.

This package provides a clean, extensible architecture for:
- Batch inference with sliding window support
- Dataset-specific visualization
- Metrics computation and analysis
- Parallel processing orchestration
- Baseline comparison via registry pattern
"""

from .runner import TestRunner
from .config import TestConfig

# Baseline system exports
from .baselines import (
    BaselineRegistry,
    BaselineSpec,
    InferenceContext,
    create_inference_context,
    get_baseline,
)

__all__ = [
    'TestRunner',
    'TestConfig',
    'BaselineRegistry',
    'BaselineSpec',
    'InferenceContext',
    'create_inference_context',
    'get_baseline',
]
