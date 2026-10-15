"""
Main training script using Hydra configuration pattern.

All configuration is specified in Hydra YAML config groups following a modular pattern.
Config groups can be composed and overridden via command-line.

Usage:
    python main_train.py                                        # Use all defaults
    python main_train.py experiment=discriminative_toy          # Use specific experiment
    python main_train.py training.num_steps=100000              # Override specific value
    python main_train.py ddp=enabled training.num_steps=50000   # Multiple overrides
"""

import pickle as pkl
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from utils import dist_util
from utils.wandb_writer import NullWandbWriter, create_wandb_writer
from train.training_loop import TrainLoop
from train.training_loop_mask_per_rank import TrainLoopMaskPerRank

# Hydra and OmegaConf for config management
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.arg_util import get_config_section, get_model_type_from_class_name
from utils.arg_util import fixseed
from utils.model_util import create_model_from_config, prepare_diffusion_config
from model.generative.diffusion import gaussian_diffusion
from utils.other_utils import get_logger, save_omegaconf_config
from ipdb_safety_net import ipdb_safety_net
from utils.construct import construct_class_by_name, get_obj_by_name
from utils.registry_builders import build_mask_scheme_registry, build_representation_registry

from torch.utils.data.distributed import DistributedSampler

import os
import torch
import torch.nn as nn
import sys
import inspect
from cprint import cprint

# Environment setup
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import cv2
cv2.setNumThreads(0)

def _compact_config_name(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value)
    if not text:
        return "none"
    return Path(text).name.split(".")[-1]


def _dataset_summary(datasets: Any) -> str:
    names = []
    for item in list(datasets or []):
        dataset_cfg = item.get("dataset", {}) if hasattr(item, "get") else {}
        raw_name = item.get("name") if hasattr(item, "get") else None
        if raw_name:
            names.append(str(raw_name))
            continue
        dataset_names = dataset_cfg.get("datasets") if hasattr(dataset_cfg, "get") else None
        if dataset_names:
            if isinstance(dataset_names, (list, tuple)) or OmegaConf.is_list(dataset_names):
                names.extend(str(name) for name in list(dataset_names)[:3])
            else:
                names.append(str(dataset_names))
        else:
            class_name = dataset_cfg.get("class_name", "dataset") if hasattr(dataset_cfg, "get") else "dataset"
            names.append(str(class_name).split(".")[-1])
    if not names:
        return "no-dataset"
    unique_names = []
    for name in names:
        if name not in unique_names:
            unique_names.append(name)
    suffix = "" if len(unique_names) <= 3 else f"+{len(unique_names) - 3}"
    return "_".join(unique_names[:3]) + suffix


def _safe_wandb_label(value: Any) -> str:
    text = str(value or "").strip()
    for char in " /\\:#?%[]{}(),=+":
        text = text.replace(char, "-")
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-_.") or "unset"


def _make_wandb_identity(config: DictConfig, run_id: str) -> tuple[str, str, str, list[str], str]:
    training = config.training
    traj_arch = _compact_config_name(config.trajnet.network_config.class_name if config.get("trajnet") else None)
    pose_arch = _compact_config_name(config.posenet.network_config.class_name if config.get("posenet") else None)
    train_data = _dataset_summary(config.get("train_datasets", []))
    mask_policy = config.masking.get("schedule_per_step", "mask") if config.get("masking") else "mask"
    repr_name = config.get("representation", {}).get("name", "repr") if hasattr(config.get("representation", {}), "get") else "repr"
    generative_type = str(training.get("generative_model_type", "none")) if training.model_type != "discriminative" else None
    parts = [
        config.task,
        training.model_type,
        generative_type,
        f"traj-{traj_arch}",
        f"pose-{pose_arch}",
        f"mask-{mask_policy}",
        f"data-{train_data}",
        f"seed{training.seed}",
        str(training.run_name),
    ]
    display_name = "__".join(_safe_wandb_label(part) for part in parts if part)
    stable_id = _safe_wandb_label(display_name)
    group = "__".join(
        _safe_wandb_label(part)
        for part in [config.task, training.model_type, generative_type, f"data-{train_data}"]
        if part
    )
    tags = [
        _safe_wandb_label(config.task),
        _safe_wandb_label(training.model_type),
        *([_safe_wandb_label(generative_type)] if generative_type else []),
        f"traj-{_safe_wandb_label(traj_arch)}",
        f"pose-{_safe_wandb_label(pose_arch)}",
        f"mask-{_safe_wandb_label(mask_policy)}",
        f"repr-{_safe_wandb_label(repr_name)}",
    ]
    model_note = f"model_type={training.model_type}"
    if generative_type:
        model_note += f"; generative_model_type={generative_type}"
    notes = (
        f"task={config.task}; {model_note}; "
        f"trajnet={traj_arch}; posenet={pose_arch}; train_data={train_data}; "
        f"run_name={training.run_name}; seed={training.seed}"
    )
    return display_name[:512], stable_id[:128], group[:128], tags, notes


def _create_wandb_writer(config: DictConfig, logdir: str, run_id: str, is_main_process: bool):
    """Create the single W&B run for the main process; other ranks get a no-op writer."""
    wandb_cfg = config.get("logging", {}).get("wandb", {}) if hasattr(config, "get") else {}
    if not is_main_process:
        return NullWandbWriter()
    default_name, default_id, default_group, default_tags, default_notes = _make_wandb_identity(config, run_id)
    return create_wandb_writer(
        logdir,
        project=wandb_cfg.get("project", "RoHM"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name") or default_name,
        group=wandb_cfg.get("group") or default_group,
        tags=list(wandb_cfg.get("tags", []) or []) or default_tags,
        notes=wandb_cfg.get("notes") or default_notes,
        config=OmegaConf.to_container(config, resolve=True),
        mode=wandb_cfg.get("mode", os.environ.get("WANDB_MODE", "offline")),
        enabled=bool(wandb_cfg.get("enabled", True)),
        is_main_process=is_main_process,
        resume=wandb_cfg.get("resume"),
        run_id=wandb_cfg.get("id") or default_id,
        log_code=bool(wandb_cfg.get("log_code", False)),
    )


class MotionConcatDataset(ConcatDataset):
    """
    ConcatDataset variant that uses global normalization parameters.
    """
    def __init__(self, datasets: Sequence[Dataset], global_norm_params: Optional[Dict[str, Any]] = None):
        if not datasets:
            raise ValueError("MotionConcatDataset requires at least one dataset.")
        super().__init__(list(datasets))
        self._primary_dataset = self.datasets[0]
        self.norm_params = global_norm_params
        
        # Propagate global norm params to all sub-datasets
        if self.norm_params is not None:
            for ds in self.datasets:
                ds.norm_params = self.norm_params
                if hasattr(ds, "_meanstd_tensor_cache"):
                    ds._meanstd_tensor_cache = {}

    def _normalize_full_tensor(self, full_tensor: torch.Tensor, device, dtype=torch.float32,
                               store_sample_norms=False):
        return self._primary_dataset._normalize_full_tensor(
            full_tensor, device, dtype, store_sample_norms
        )

    def _unnormalize_full_tensor(self, full_tensor: torch.Tensor, device, dtype=torch.float32,
                                 sample_norms=None):
        return self._primary_dataset._unnormalize_full_tensor(
            full_tensor, device, dtype, sample_norms
        )

    def __getattr__(self, item):
        primary_dataset = object.__getattribute__(self, "_primary_dataset")
        return getattr(primary_dataset, item)


def _create_dataloader(config, dataset, is_train: bool, ddp_enabled: bool, ddp_config: Dict):
    """
    Create DataLoader from Hydra config.
    
    Args:
        config: Full Hydra DictConfig configuration.
        dataset: Dataset to wrap.
        is_train: Whether this is for training (affects shuffle, sampler).
        ddp_enabled: Whether DDP is enabled.
        ddp_config: DDP configuration dict.
        global_norm_params: Global normalization parameters for test dataset.
    """
    loader_section = 'train_dataloader' if is_train else 'test_dataloader'
    loader_config = config[loader_section]
    
    batch_size = loader_config.batch_size
    num_workers = loader_config.num_workers
    drop_last = loader_config.drop_last
    pin_memory = loader_config.get('pin_memory', True)
    
    sampler = None
    shuffle = loader_config.shuffle if is_train else False
    
    # Check if dataset uses high_motion sampling strategy
    use_weighted_sampler = False
    if is_train and hasattr(dataset, '_slice_strategy'):
        if dataset._slice_strategy in ('high_motion_weighted', 'high_motion_focused'):
            use_weighted_sampler = True
    
    if ddp_enabled:
        sampler_config = config.distributed_sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_config['world_size'],
            rank=ddp_config['rank'],
            shuffle=sampler_config['shuffle'] if is_train else False,
            drop_last=sampler_config['drop_last'] if is_train else False,
        )
        shuffle = False  # Sampler handles shuffling
        if use_weighted_sampler:
            cprint.err("[ERROR] high_motion_weighted/focused not fully supported with DDP yet. Using DistributedSampler.")
    elif use_weighted_sampler:
        # Use weighted sampler for high_motion strategies (non-DDP)
        sampler = dataset.get_weighted_sampler()
        shuffle = False  # Sampler handles shuffling
        cprint.info("[INFO] Using WeightedRandomSampler for high_motion sampling strategy.")
    
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'drop_last': drop_last,
        'pin_memory': pin_memory,
    }
    
    if sampler is not None:
        loader_kwargs['sampler'] = sampler
        loader_kwargs.pop('shuffle')
    
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = loader_config.get('persistent_workers', True)
        loader_kwargs['prefetch_factor'] = loader_config.get('prefetch_factor', 2)
    
    return DataLoader(dataset, **loader_kwargs)

def _make_dataset_label(dataset_cfg: Dict[str, Any], dataset: Dataset, index: int) -> str:
    """
    Build a readable, unique label for a dataset from its config.
    """
    raw_name = None
    if hasattr(dataset_cfg, "get"):
        raw_name = dataset_cfg.get("name") or dataset_cfg.get("dataset", {}).get("name")
    if raw_name:
        label = str(raw_name)
    else:
        dataset_cfg_inner = dataset_cfg.get("dataset", {}) if hasattr(dataset_cfg, "get") else {}
        class_name = dataset_cfg_inner.get("class_name", dataset.__class__.__name__)
        base_name = class_name.split(".")[-1]
        split = dataset_cfg_inner.get("split", getattr(dataset, "split", None))
        datasets = dataset_cfg_inner.get("datasets", getattr(dataset, "datasets", None))
        parts = [base_name]
        if split:
            parts.append(str(split))
        if datasets:
            if isinstance(datasets, (list, tuple)) or OmegaConf.is_list(datasets):
                parts.append("_".join(str(item) for item in list(datasets)))
            else:
                parts.append(str(datasets))
        label = "_".join(parts)
    label = label.replace(" ", "_").replace("-", "_").replace("[", "").replace("]", "").replace(",", "")
    return f"{index:02d}_{label}"

def run_training(config: DictConfig, writer, logdir, logger):
    """Main training function."""
    cprint.info("creating data loader...")
    
    # Get DDP config
    ddp_config = get_config_section(config, 'ddp')
    ddp_enabled = ddp_config['enabled']
    
    if ddp_enabled:
        dataset_device = dist_util.dev(rank=ddp_config['local_rank'])
    else:
        dataset_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build registries (explicit; no defaults)
    repr_registry_cfg = OmegaConf.to_container(config.representation_registry, resolve=True)
    mask_registry_cfg = OmegaConf.to_container(config.mask_registry, resolve=True)
    repr_registry = build_representation_registry(repr_registry_cfg)
    mask_scheme_registry = build_mask_scheme_registry(mask_registry_cfg)

    def _construct_dataset(dataset_cfg: Dict[str, Any]):
        """Instantiate dataset with shared registries injected when supported."""
        ds_kwargs = dict(dataset_cfg['dataset'])
        ds_kwargs.update({
            "logdir": logdir,
            "dtype": torch.float32,
            "device": dataset_device,
            "dist_rank": ddp_config['rank'] if ddp_enabled else 0,
        })
        cls = get_obj_by_name(ds_kwargs["class_name"])
        # Always inject registries - they are required by DatasetBase
        # Note: We can't rely on inspect.signature for classes using **kwargs
        ds_kwargs["repr_registry"] = repr_registry
        ds_kwargs["mask_scheme_registry"] = mask_scheme_registry
        return construct_class_by_name(**ds_kwargs)

    # --- Train Datasets ---
    train_dataset_components = []
    for train_dataset_config in config.train_datasets:
        train_dataset_components.append(_construct_dataset(train_dataset_config))
    
    # --- Compute Global Normalization Parameters ---
    # Each dataset already has norm_params computed during initialization.
    # We merge them using Welford's parallel algorithm for statistically correct combination.
    cprint.info("[INFO] Merging normalization parameters across all datasets...")
    if train_dataset_components:
        from data_sets.running_stats import merge_norm_params
        
        global_norm_params = merge_norm_params(
            train_dataset_components,
            mean_std=train_dataset_components[0].mean_std,
            norm_strategy=train_dataset_components[0].norm_strategy
        )
        
        # Apply merged norm_params to all datasets
        for dataset in train_dataset_components:
            dataset.norm_params = global_norm_params
            if hasattr(dataset, "_meanstd_tensor_cache"):
                dataset._meanstd_tensor_cache = {}

        global_norm_params_path = os.path.join(logdir, "global_norm_params.pkl")
        with open(global_norm_params_path, "wb") as result_file:
            pkl.dump(global_norm_params, result_file, protocol=2)
            
        cprint.info("[INFO] Global normalization parameters merged and applied.")
    else:
        global_norm_params = None

    if len(train_dataset_components) == 1:
        train_dataset = train_dataset_components[0]
    else:
        train_dataset = MotionConcatDataset(train_dataset_components, global_norm_params=global_norm_params)
        cprint.info(f"[INFO] Using MotionConcatDataset with {len(train_dataset_components)} components.")

    # --- Create Train DataLoader ---
    train_dataloader = _create_dataloader(config, train_dataset, is_train=True, ddp_enabled=ddp_enabled, ddp_config=ddp_config)

    # --- Test Datasets ---
    test_sets = []
    for idx, test_dataset_config in enumerate(config.test_datasets):
        test_dataset = _construct_dataset(test_dataset_config)
        dataset_label = _make_dataset_label(test_dataset_config, test_dataset, idx)
        setattr(test_dataset, "dataset_label", dataset_label)
        test_sets.append({
            "name": dataset_label,
            "dataset": test_dataset,
            "dataloader": _create_dataloader(
                config, test_dataset, is_train=False, ddp_enabled=ddp_enabled, ddp_config=ddp_config
            ),
        })

    if len(test_sets) == 1:
        test_dataset = test_sets[0]["dataset"]
        test_dataloader = test_sets[0]["dataloader"]
    else:
        test_dataset = test_sets[0]["dataset"]
        test_dataloader = test_sets
        labels = ", ".join(test_set["name"] for test_set in test_sets)
        cprint.info(f"[INFO] Using separate test datasets: {labels}")

    if (ddp_enabled and ddp_config['rank'] == 0) or not ddp_enabled:
        cprint.info("creating model and diffusion...")

    # --- Model Instantiation ---
    device = dataset_device
    timestep_respacing_eval = config.training.timestep_respacing_eval
    
    models = []
    model_trains = []
    model_evals = []
    
    task = config.task
    
    # --- TrajNet ---
    if task == 'traj' or task == 'both':
        trajnet = construct_class_by_name(**config.trajnet, dataset=train_dataset, device=device)
        trajnet.set_test_dataset(test_dataset)  # Set initial test dataset; updated per eval dataset in TrainLoop
        
        traj_train = construct_class_by_name(**config.traj_training_model, 
            dataloader=train_dataloader,
            device=device,
            timestep_respacing=''
        )
        traj_eval = construct_class_by_name(**config.traj_training_model, 
            dataloader=test_sets[0]["dataloader"] if isinstance(test_dataloader, list) else test_dataloader,
            device=device,
            timestep_respacing=timestep_respacing_eval
        )
        
        models.append(trajnet)
        model_trains.append(traj_train)
        model_evals.append(traj_eval)

    # --- PoseNet ---
    if task == 'pose' or task == 'both':
        posenet = construct_class_by_name(**config.posenet, dataset=train_dataset, device=device)
        posenet.set_test_dataset(test_dataset)  # Set initial test dataset; updated per eval dataset in TrainLoop
        
        pose_train = construct_class_by_name(**config.pose_training_model, 
            dataloader=train_dataloader,
            device=device,
            timestep_respacing=''
        )
        pose_eval = construct_class_by_name(**config.pose_training_model, 
            dataloader=test_sets[0]["dataloader"] if isinstance(test_dataloader, list) else test_dataloader,
            device=device,
            timestep_respacing=timestep_respacing_eval
        )

        models.append(posenet)
        model_trains.append(pose_train)
        model_evals.append(pose_eval)

    if (ddp_enabled and ddp_config['rank'] == 0) or not ddp_enabled:
        cprint.info(f"Training for task: {task}...")

    if ddp_enabled:
        if task == 'both':
            models = [nn.SyncBatchNorm.convert_sync_batchnorm(m) for m in models]
        else:
            models[0] = nn.SyncBatchNorm.convert_sync_batchnorm(models[0])

    loop_mode = config.training.get("loop_mode", "default")
    loop_cls = TrainLoopMaskPerRank if loop_mode == "mask_per_rank" else TrainLoop
    loop_cls(
        config=config,
        writer=writer,
        models=models,
        model_trains=model_trains,
        model_evals=model_evals,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        logdir=logdir,
        logger=logger,
        device=str(device)
    ).run_loop()


@hydra.main(version_base=None, config_path="cfg_files/conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    Hydra entry point for training.
    
    Args:
        config: Automatically composed DictConfig from Hydra.
    """
    # Extract key config values
    training_config = config.training
    ddp_config = config.ddp
    task = config.task
    
    run_name = training_config['run_name']
    run_id = f"{task}_{run_name}"
    
    # Setup DDP if enabled
    if ddp_config['enabled']:
        rank, local_rank, world_size = dist_util.setup_dist()
        # Update config with runtime DDP values (modify OmegaConf directly)
        from omegaconf import open_dict
        with open_dict(config):
            config.ddp.rank = rank
            config.ddp.local_rank = local_rank
            config.ddp.world_size = world_size
            # Adjust batch size for DDP
            config.train_dataloader.batch_size = config.train_dataloader.batch_size // world_size
        torch.cuda.set_device(local_rank)
        
        # Refresh ddp_config after updates
        ddp_config = get_config_section(config, 'ddp')
    
    save_dir = training_config['save_dir']
    logdir = os.path.join(save_dir, str(run_id))

    is_main_process = not ddp_config['enabled'] or ddp_config['rank'] == 0
    writer = NullWandbWriter()
    logger = None
    if is_main_process:
        os.makedirs(logdir, exist_ok=True)
        writer = _create_wandb_writer(config, logdir, run_id, is_main_process=True)
        cprint.info('RUNDIR: {}'.format(logdir))
        sys.stdout.flush()
        logger = get_logger(logdir)
        logger.info('Let the games begin')
        save_omegaconf_config(logdir, config)

        from pathlib import Path
        from utils.other_utils import get_git_commit_hash, get_git_diff
        exp_dir = Path(logdir)
        (exp_dir / 'git_commit.txt').write_text(get_git_commit_hash())
        (exp_dir / 'git_diff.txt').write_text(get_git_diff())
    else:
        writer = _create_wandb_writer(config, logdir, run_id, is_main_process=False)

    seed = training_config['seed']
    if ddp_config['enabled']:
        seed = seed + ddp_config['rank']
    fixseed(seed)
    ipdb_safety_net()
    try:
        run_training(config, writer, logdir, logger)
    finally:
        if is_main_process:
            writer.close()


if __name__ == "__main__":
    main()
