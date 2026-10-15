from torch.utils import data
from tqdm import tqdm
import glob
import smplx
from data_sets.motion_representation import *
import pickle as pkl
from utils.representation_registry import RepresentationRegistry
import torch.distributed as dist
from typing import Literal
import hashlib
import json
import os
from pathlib import Path
import numpy as np

from data_sets.dataset_rich import DatasetRICH

class DatasetHPS(DatasetRICH):
    homogenous_transform_align_ = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
    def __init__(self,
                *args,
                repr_registry: RepresentationRegistry = None,
                mask_scheme_registry=None,
                **kwargs):
        self.num_joints = 22
        super().__init__(*args,
                         repr_registry=repr_registry,
                         mask_scheme_registry=mask_scheme_registry,
                         **kwargs)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    from omegaconf import OmegaConf
    from utils.registry_builders import build_mask_scheme_registry, build_representation_registry
    hps_dataset_root= "datasets/HPS_preprocessed"
    from torch.utils.data import DataLoader
    from ipdb_safety_net import ipdb_safety_net
    ipdb_safety_net()
    repo_root = Path(__file__).resolve().parents[1]
    repr_cfg = OmegaConf.load(repo_root / "cfg_files/conf/representation_registry/xyz_only.yaml")
    mask_cfg = OmegaConf.load(repo_root / "cfg_files/conf/mask_registry/custom_body_parts_xyz.yaml")
    repr_registry = build_representation_registry(OmegaConf.to_container(repr_cfg, resolve=True))
    mask_scheme_registry = build_mask_scheme_registry(OmegaConf.to_container(mask_cfg, resolve=True))
    train_dataset = DatasetHPS(preprocessed_dataset_root=hps_dataset_root, split='train',
                                    datasets=None,
                                    cano_jts_pos=False,
                                    body_model_path="data/body_models/smplx_model",
                                    input_noise=True,
                                    noise_std_smplx_global_rot=2.0,
                                    noise_std_smplx_body_rot=2.0,
                                    noise_std_smplx_trans=0.03,
                                    noise_std_smplx_betas=0.00,
                                    slice_strategy='random_uniform_len',
                                    clip_len=145,
                                    mean_std=True,
                                    norm_strategy="meanstd",
                                    joints_num=22,
                                    disable_cache=True,
                                    logdir=tempfile.mkdtemp(),
                                    device=dist_util.dev(),
                                    dtype=torch.float32,
                                    use_ddp=False,
                                    dist_rank=0,
                                    repr_registry=repr_registry,
                                    mask_scheme_registry=mask_scheme_registry)
    
    train_dl = DataLoader(train_dataset, batch_size=64, sampler=None, drop_last=True, num_workers=0)
    for batch in train_dl:
        pass
