# rohm/argparsers.py
"""
Argument parsing utilities for RoHM.

This module provides OmegaConf-based configuration loading for both training and testing.
Training uses Hydra's decorator pattern, while testing loads configs from training logdirs.

For training:
    python main_train.py                                        # Use Hydra defaults
    python main_train.py experiment=discriminative_toy          # Use specific experiment

For testing:
    Config is loaded from the training run's logdir (config.yaml or params.json).
"""
from __future__ import annotations
import os
import torch
import inspect
from typing import Any, Dict, Set, List, Optional, Union, cast
from omegaconf import OmegaConf, DictConfig, open_dict
import random
import numpy as np


# =============================================================================
# OmegaConf Utilities
# =============================================================================

def load_omegaconf(config_path: str) -> DictConfig:
    """
    Load an OmegaConf configuration from a YAML file.
    
    This follows the IMM/dnnlib pattern where configs are structured
    with class_name and constructor parameters.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        DictConfig: OmegaConf configuration object
    """
    config = OmegaConf.load(config_path)
    # Resolve any interpolations and ensure we return DictConfig
    resolved_yaml = OmegaConf.to_yaml(config, resolve=True)
    result = OmegaConf.create(resolved_yaml)
    assert isinstance(result, DictConfig), "Expected DictConfig from config file"
    return result


def get_config_section(omegaconf_config: DictConfig, section: str) -> Any:
    """
    Extract a configuration section or value from OmegaConf config.
    
    Args:
        omegaconf_config: Full OmegaConf configuration.
        section: Section name (e.g., 'trajnet', 'posenet', 'task').
    
    Returns:
        Configuration value (dict, list, or primitive).
    """
    return OmegaConf.to_container(omegaconf_config[section], resolve=True)


def get_model_type_from_class_name(class_name: str) -> str:
    """
    Infer model type from class name for determining preparation function.
    
    Returns:
        - Generative types: 'diffusion', 'meanflow', 'shortcut', 'consistency', 
          'improved_consistency', 'flow_matching'
        - Discriminative types: 'toy', 'discriminative'
        - 'unknown' if not recognized
    """
    class_name_lower = class_name.lower()
    
    # Generative models
    if 'diffusion' in class_name_lower:
        return 'diffusion'
    elif 'meanflow' in class_name_lower:
        return 'meanflow'
    elif 'shortcut' in class_name_lower:
        return 'shortcut'
    elif 'improvedconsistency' in class_name_lower:
        return 'improved_consistency'
    elif 'consistency' in class_name_lower:
        return 'consistency'
    elif 'flowmatching' in class_name_lower or '.fm.' in class_name_lower:
        return 'flow_matching'
    # Discriminative models
    elif 'toy' in class_name_lower:
        return 'toy'
    elif 'discriminative' in class_name_lower:
        return 'discriminative'
    else:
        return 'unknown'

def fixseed(seed: int = 42, dbg=False):
    """
    Sets the seed for generating random numbers in PyTorch, numpy and core python.
    
    Args:
        seed (int): The integer value seed for global random state.
    """
    # 1. Python & Numpy (The Basics)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # disable Python hash randomization
    np.random.seed(seed)

    # 2. PyTorch CPU & GPU (The Core)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 3. CuDNN (The Black Box)
    # A trade-off: determinism usually costs performance.
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    
    if dbg:
        # 4. Low-level Linear Algebra (The Hidden Killer)
        # Some CUDA operators (e.g. scatter_add_) are non-deterministic at the atomic level:
        # even with a fixed seed, differing float addition orders accumulate tiny errors.
        # This forces PyTorch to use deterministic algorithms and errors out if none exist.
        torch.use_deterministic_algorithms(True)
        
        # 5. Environment Variables for CUDA
        # Some matmul algorithms need an explicit workspace setting to be deterministic.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    print(f"[INFO] Global Seed set to {seed}. Determinism engaged.")

# =============================================================================
# Test Configuration Loading (Hydra/OmegaConf-based)
# =============================================================================
def apply_mpi_config(config: DictConfig, rank: int, world_size: int) -> DictConfig:
    """
    Apply MPI-related configuration based on process rank and world size.
    
    This handles test_mask_schemes and inference_steps_list distribution
    across MPI processes.
    
    Args:
        config: Test configuration.
        rank: MPI process rank.
        world_size: Total number of MPI processes.
    
    Returns:
        DictConfig: Configuration with MPI-specific values applied.
    """
    with open_dict(config):
        config.dist_rank = rank
        config.dist_world_size = world_size
        
        # Distribute test_mask_schemes across ranks
        if 'test_mask_schemes' in config and config.test_mask_schemes is not None:
            schemes_container = OmegaConf.to_container(config.test_mask_schemes, resolve=True)
            assert isinstance(schemes_container, list), "test_mask_schemes must be a list"
            schemes: List[str] = schemes_container
            config.test_mask_scheme = schemes[rank % len(schemes)]
        
        # Distribute inference_steps_list across ranks
        if 'inference_steps_list' in config and config.inference_steps_list is not None:
            steps_container = OmegaConf.to_container(config.inference_steps_list, resolve=True)
            assert isinstance(steps_container, list), "inference_steps_list must be a list"
            steps_list: List[int] = steps_container
            config.num_inference_steps = steps_list[rank % len(steps_list)]
    
    return config


def validate_test_config(config: DictConfig) -> None:
    """
    Validate test configuration and raise assertion errors for invalid combinations.
    
    Args:
        config: Test configuration to validate.
    
    Raises:
        AssertionError: If configuration is invalid.
    """
    # Check for mutually exclusive evaluation modes
    eval_trajnet_only = config.get('eval_on_trajnet_only', False)
    eval_posenet_only = config.get('eval_on_posenet_only', False)
    assert not (eval_trajnet_only and eval_posenet_only), \
        "Cannot evaluate **only** on both TrajNet and PoseNet at the same time."
    
    # Validate per-sequence metrics requirements
    if config.get('compute_metrics_per_seq', False):
        assert config.get('slice_strategy') == 'full_sequence', \
            "Per-sequence metrics can only be computed when slice_strategy is 'full_sequence'."
        assert config.get('with_gt', False), \
            "Ground truth data is required for per-sequence metrics."
    
    # Validate multi-worker constraints
    num_workers = config.get('num_parallel_inference_workers', 0)
    if num_workers > 0:
        if 'test_mask_schemes' in config and config.test_mask_schemes is not None:
            schemes_container = OmegaConf.to_container(config.test_mask_schemes, resolve=True)
            assert isinstance(schemes_container, list), "test_mask_schemes must be a list"
            assert len(schemes_container) == 1, \
                "When using multiple workers, only one mask scheme should be specified."
        if 'inference_steps_list' in config and config.inference_steps_list is not None:
            steps_container = OmegaConf.to_container(config.inference_steps_list, resolve=True)
            assert isinstance(steps_container, list), "inference_steps_list must be a list"
            assert len(steps_container) == 1, \
                "When using multiple workers, only one inference step should be specified."


def filter_kwargs_for_class(cls: type, config: DictConfig, exclude: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Filter config values to only include valid constructor arguments for a class.
    
    This replaces the legacy _filter_dataloader_args function.
    
    If the class constructor accepts **kwargs, all non-excluded parameters are passed through.
    Otherwise, only parameters matching the constructor signature are included.
    
    Args:
        cls: The class to filter arguments for.
        config: OmegaConf configuration containing potential kwargs.
        exclude: Set of keys to exclude from filtering.
    
    Returns:
        Dict containing only valid kwargs for the class constructor.
    """
    exclude_set: Set[str] = exclude if exclude is not None else set()
    
    # Get the class constructor signature
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    
    # Check if the class accepts **kwargs (VAR_KEYWORD)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD 
        for p in sig.parameters.values()
    )
    
    # Convert config to dict and filter
    config_container = OmegaConf.to_container(config, resolve=True)
    assert isinstance(config_container, dict), "config must be a DictConfig"
    config_dict = cast(Dict[str, Any], config_container)
    
    filtered = cast(Dict[str, Any], {})
    for key, value in config_dict.items():
        if key in exclude_set:
            continue
        # If class accepts **kwargs, pass all non-excluded parameters
        # Otherwise, only pass parameters in the signature
        if accepts_kwargs or key in valid_params:
            filtered[key] = value
    
    return filtered


# =============================================================================
# Convenience Functions
# =============================================================================

def save_test_config(save_root: str, config: DictConfig) -> None:
    """
    Save test configuration to the results directory.
    
    Args:
        save_root: Path to save directory.
        config: OmegaConf DictConfig to save.
    """
    from utils.other_utils import save_omegaconf_config
    save_omegaconf_config(save_root, config)
