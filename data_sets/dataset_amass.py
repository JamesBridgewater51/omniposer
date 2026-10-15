from .dataset_amass_like import DatasetAMASSLike
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json
from utils import dist_util
from typing import Dict, Iterator, Optional, Tuple, Any

class DatasetAMASS(DatasetAMASSLike):
    def __init__(self,
                 *args, **kwargs
                 ):

        super(DatasetAMASS,self).__init__(*args, **kwargs)

@hydra.main(version_base=None, config_path="../cfg_files/conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    """
    from utils.construct import construct_from_config,  get_obj_by_name, construct_class_by_name
    from utils.registry_builders import build_mask_scheme_registry, build_representation_registry

    # Build registries (explicit; no defaults)
    repr_registry_cfg = OmegaConf.to_container(config.representation_registry, resolve=True)
    mask_registry_cfg = OmegaConf.to_container(config.mask_registry, resolve=True)
    repr_registry = build_representation_registry(repr_registry_cfg)
    mask_scheme_registry = build_mask_scheme_registry(mask_registry_cfg)

    save_dir = config.training['save_dir']
    run_id = f"{config.task}_{config.training.run_name}"
    logdir = os.path.join(save_dir, str(run_id))

    def _construct_dataset(dataset_cfg: Dict[str, Any]):
        """Instantiate dataset with shared registries injected when supported."""
        ds_kwargs = dict(dataset_cfg['dataset'])
        ds_kwargs.update({
            "logdir": logdir,
            "dtype": torch.float32,
            "device": torch.device('cuda'),
            "dist_rank": 0,
        })
        cls = get_obj_by_name(ds_kwargs["class_name"])
        # Always inject registries - they are required by DatasetBase
        # Note: We can't rely on inspect.signature for classes using **kwargs
        ds_kwargs["repr_registry"] = repr_registry
        ds_kwargs["mask_scheme_registry"] = mask_scheme_registry
        return construct_class_by_name(**ds_kwargs)

    ds = _construct_dataset(dataset_cfg=config.train_datasets[0])
    for index in range(len(ds)):
        if index > 10:
            break
        ds.__getitem__(index)


if __name__ == '__main__':
    from ipdb_safety_net import ipdb_safety_net
    ipdb_safety_net()
    main()

