"""
Refactored test script for RoHM motion reconstruction model.

This is a clean entry point that uses the modular test_pipeline package.
The original test.py is preserved for backward compatibility.

Usage:
    python test_new.py logdir=runs/both_example_run model_path=path/to/ckpt.safetensors
    python test_new.py logdir=runs/both_example_run dataset=debug
"""
import os
import sys
import torch
import torch.multiprocessing as mp

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.fixseed import fixseed
from utils.other_utils import ipdb_safety_net

from test_pipeline import TestRunner
from test_pipeline.config import apply_mpi_config, validate_test_config, save_test_config


# Environment setup
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["HYDRA_FULL_ERROR"] = "1"  # Enable full error tracebacks from Hydra


def setup_dataset_globals(dataset_name: str):
    """Initialize dataset-specific global variables."""
    # Lazy imports for dataset-specific modules
    if dataset_name.lower() == 'ego_exo':
        try:
            import third_party.ego_exo4d_egopose.bodypose.utils.utils_option as option
            from data_sets.dataset_egoexo import DatasetEgoExo
            from utils.egoexo_vis_utils import EgoExoFrameSynchronizer
        except ImportError as e:
            print(f"[Warning] EgoExo dependencies not available: {e}")


@hydra.main(version_base=None, config_path="cfg_files/conf/test", config_name="config")
def main(test_config: DictConfig) -> None:
    """
    Hydra entry point for testing.
    
    Args:
        test_config: Test configuration from Hydra.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        rank = 0
        world_size = 1
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
    
    # Load training config from logdir
    logdir = test_config.logdir
    train_config = OmegaConf.load(os.path.join(logdir, "config.yaml"))
    
    # Apply MPI configuration
    test_config = apply_mpi_config(test_config, rank, world_size)
    
    print(f"[INFO] Rank {test_config.dist_rank}/{test_config.dist_world_size}")
    
    # Setup multiprocessing for CUDA
    if test_config.get('num_parallel_inference_workers', 0) > 0 and torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
    
    # Validate configuration
    validate_test_config(test_config)
    
    # Set random seed
    fixseed(test_config.seed)
    
    # Initialize dataset-specific globals
    setup_dataset_globals(test_config.dataset_name)
    
    # Create output directory and save config
    os.makedirs(test_config.save_root, exist_ok=True)
    save_test_config(test_config.save_root, test_config)
    
    if test_config.dist_rank == 0:
        print(f"[*] Config saved to: {test_config.save_root}/test_config.yaml")
    
    # Run test pipeline with explicit exception handling for ipdb
    # Note: Hydra catches exceptions before sys.excepthook, so we need manual try-except
    runner = TestRunner(train_config, test_config)
    metrics = runner.run()
    print("Test finished.")


if __name__ == "__main__":
    ipdb_safety_net()
    main()
