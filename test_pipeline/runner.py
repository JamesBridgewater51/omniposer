"""
Main test runner that orchestrates the evaluation pipeline.

This module provides TestRunner which uses the hook-based baseline registry
system for flexible baseline comparison and evaluation.
"""

from .runner_hooks import TestRunner

__all__ = ['TestRunner']
