# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/GuyTevet/motion-diffusion-model

"""
Helpers for distributed training.
"""

import socket
import datetime

import torch as th
import torch.distributed as dist
import os

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

used_device = 0

def setup(rank, world_size):
    th.cuda.set_device(rank % GPUS_PER_NODE)
    dist.init_process_group('nccl', init_method='env://', rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3600))

def setup_dist():
    # Check if we're using SLURM
    using_slurm = False
    
    if using_slurm:
        # SLURM environment variables
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        # Non-SLURM (torchrun or direct launch)
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    th.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def dev(rank=0):
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{rank }")
    return th.device("cpu")