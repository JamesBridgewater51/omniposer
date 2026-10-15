import json
import numpy as np
import math
import time
import blobfile as bf
import torch
import smplx
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from copy import deepcopy
from contextlib import ExitStack, nullcontext
from typing import Dict, List, Optional, Any

from utils import dist_util
from utils.fp16_util import MixedPrecisionTrainer
from model.generative.diffusion.resample import create_named_schedule_sampler
from utils.other_utils import TRANS_ROT_MASK_RATIO
from utils.batch_util import process_batch_cond
from utils.batch_util import _apply_masking
from utils.batch_util import _apply_egoexo_like_masking
from utils.feature_accessor import select_modalities, get_modality_dim, insert_modalities_into_cond
from utils.masking_actions import (
    apply_mask_action,
    apply_mask_action_plan,
    build_mask_action_pool,
    materialize_mask_action,
    materialize_per_sample_action_plan,
    resolve_random_visible_ids,
    sample_action_names_from_weights,
    validate_batch_policy,
    validate_masking_config,
    visible_ids_from_action,
)


from safetensors.torch import save_file, load_file
import os
from safetensors.torch import load_file
from safetensors import safe_open
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import models needed for consistency training teacher instantiation
from model.trajnet import TrajNet
from model.posenet import PoseNet

VIS_BATCH_DBG = False


def _tensor_shape_or_none(tensor):
    try:
        return tuple(tensor.shape)
    except RuntimeError:
        return None


def _sync_materialized_lazy_state(ema_model, source_model) -> None:
    """Materialize EMA lazy parameters after the source model's first routed forward."""
    ema_state = ema_model.state_dict()
    source_state = source_model.state_dict()
    updates = {}
    for name, ema_tensor in ema_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            continue
        source_shape = _tensor_shape_or_none(source_tensor)
        if source_shape is None:
            continue
        ema_shape = _tensor_shape_or_none(ema_tensor)
        if ema_shape != source_shape:
            updates[name] = source_tensor.detach().clone()
    if updates:
        ema_model.load_state_dict(updates, strict=False)


def save_checkpoint(
    model,
    step: int,
    opt,
    scheduler,
    task: str,
    save_dir: str,
    is_best_for_stage: bool = False,
    stage: int = None,
    is_latest: bool = False,
    filename_prefix: str = "",
    target_model = None,
    ema_model = None,
) -> tuple:
    """
    Save a checkpoint for a single model.
    
    Args:
        model: The model to save (should already be unwrapped from DDP if applicable)
        step: Current training step
        opt: Optimizer
        scheduler: Learning rate scheduler (can be None)
        task: Task name ('traj' or 'pose')
        save_dir: Directory to save checkpoints
        is_best_for_stage: Whether this is the best model for the current stage
        stage: Current training stage (required if is_best_for_stage is True)
        is_latest: Whether this is the latest checkpoint
        filename_prefix: Prefix for the checkpoint filename (e.g., "trajnet_" or "posenet_")
        target_model: Optional target model for consistency training
        ema_model: Optional EMA model whose weights match in-training evaluation
        
    Returns:
        Tuple of (safetensors_path, optimizer_path)
    """
    # Prepare tensors for safetensors
    tensors_dict = {}
    for k, v in model.state_dict().items():
        if 'smplx_model' in k:
            continue  # Skip smplx model weights
        if not v.is_contiguous():
            v = v.contiguous()
        tensors_dict[f"weight.{k}"] = v
    
    # Add target model weights if provided (for consistency training)
    if target_model is not None:
        for k, v in target_model.state_dict().items():
            if not v.is_contiguous():
                v = v.contiguous()
            tensors_dict[f"target_model_weight.{k}"] = v

    if ema_model is not None:
        for k, v in ema_model.state_dict().items():
            if 'smplx_model' in k:
                continue
            if not v.is_contiguous():
                v = v.contiguous()
            tensors_dict[f"ema_weight.{k}"] = v

    # Prepare metadata
    metadata = {
        'step': str(step),
        'task': task,
        'has_ema': str(ema_model is not None).lower(),
    }
    
    # Construct filename
    if is_best_for_stage:
        filename = f"{filename_prefix}best_model_stage_{stage}"
    elif is_latest:
        filename = f"{filename_prefix}latest_model"
    else:
        filename = f"{filename_prefix}model_{step:09d}"
    
    # Save model weights with safetensors
    safetensors_path = os.path.join(save_dir, f"{filename}.safetensors")
    save_file(tensors_dict, safetensors_path, metadata=metadata)
    
    # Save optimizer and scheduler states with torch.save
    optimizer_path = os.path.join(save_dir, f"{filename}_optimizers.pt")
    optimizer_state = {
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(optimizer_state, optimizer_path)
    
    return safetensors_path, optimizer_path


def load_checkpoint(
    checkpoint_path: str,
    model,
    opt = None,
    scheduler = None,
    target_model = None,
    device = None,
    use_ema: bool = False,
    strict_ema: bool = True,
) -> int:
    """
    Load a checkpoint for a single model.
    
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        model: The model to load weights into (should already be unwrapped from DDP if applicable)
        opt: Optimizer (can be None during evaluation)
        scheduler: Learning rate scheduler (can be None)
        target_model: Optional target model for consistency training
        device: Device to load optimizer state to (e.g., 'cuda:0'). Required when loading optimizer state.
        use_ema: Load EMA weights if they are present in the checkpoint.
        strict_ema: Raise when EMA is requested but unavailable. When false,
            fall back to raw weights for older checkpoints.
        
    Returns:
        step: The training step from the checkpoint metadata
    """
    # Determine paths for optimizer state file
    base_path = checkpoint_path.replace('.safetensors', '')
    optimizer_path = f"{base_path}_optimizers.pt"
    
    # Load model weights with safetensors
    tensors_dict = load_file(checkpoint_path)
    
    # Get metadata
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    
    # Extract model weights
    if use_ema:
        weights = {k[len("ema_weight."):]: v for k, v in tensors_dict.items() if k.startswith("ema_weight.")}
        if not weights:
            if strict_ema:
                raise ValueError(
                    f"Requested EMA weights, but checkpoint does not contain ema_weight.* tensors: {checkpoint_path}"
                )
            weights = {k[len("weight."):]: v for k, v in tensors_dict.items() if k.startswith("weight.")}
    else:
        weights = {k[len("weight."):]: v for k, v in tensors_dict.items() if k.startswith("weight.")}

    if weights:
        model.load_state_dict(weights, strict=False)
    
    # Load target model weights if provided (for consistency training)
    if target_model is not None:
        target_weights = {k[len("target_model_weight."):]: v for k, v in tensors_dict.items() if k.startswith("target_model_weight.")}
        if target_weights:
            target_model.load_state_dict(target_weights, strict=False)
    
    # Load optimizer and scheduler states if opt is provided and file exists
    if opt is not None and os.path.exists(optimizer_path):
        # Load optimizer state to the correct device to avoid device mismatch during opt.step()
        optimizer_state = torch.load(optimizer_path, map_location=device)
        if 'optimizer' in optimizer_state:
            opt.load_state_dict(optimizer_state['optimizer'])
            
            # PyTorch's load_state_dict may not preserve device for all state tensors.
            # Explicitly move all optimizer state tensors to the parameter device.
            for param, state in opt.state.items():
                target_device = param.device
                for key in state:
                    if isinstance(state[key], torch.Tensor):
                        if state[key].device != target_device:
                            state[key] = state[key].to(target_device)
        if scheduler is not None and 'scheduler' in optimizer_state and optimizer_state['scheduler'] is not None:
            scheduler.load_state_dict(optimizer_state['scheduler'])
    
    step = int(metadata['step']) if 'step' in metadata else 0
    return step


def lr_lambda(step, num_warmup_steps, num_training_steps):
    if step < num_warmup_steps:
        # Linear warmup from 0 → 1
        return float(step) / float(max(1, num_warmup_steps))
    # Cosine decay from 1 → 0
    progress = (step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

class TrainLoop:
    def __init__(self, config, writer, models, model_trains, model_evals,
                 train_dataloader, test_dataloader, logdir, logger, device='cpu'):
        """
        Initialize TrainLoop from OmegaConf configuration.
        
        Args:
            config: OmegaConf DictConfig containing all training configuration.
            writer: W&B writer adapter.
            models: List of model(s) to train.
            model_trains: List of training model wrapper(s).
            model_evals: List of evaluation model wrapper(s).
            train_dataloader: Training DataLoader.
            test_dataloader: Test DataLoader.
            logdir: Log directory path.
            logger: Logger instance.
            device: Device string.
        """
        self.config = config
        
        # --- Extract all config values upfront ---
        self._extract_config_values(config)
        
        self.writer = writer

        if self.matmul_precision:
            torch.set_float32_matmul_precision(self.matmul_precision)
        if self.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.task == 'both':
            self.trajnet, self.posenet = models
            self.model_trajnet_train, self.model_posenet_train = model_trains
            self.model_trajnet_eval, self.model_posenet_eval = model_evals
            self.num_joints = self.trajnet.num_joints
        elif self.task == 'traj' or self.task == 'pose':
            assert len(models) == 1 and len(model_trains) == 1 and len(model_evals) == 1, "For single task training, models, model_trains, and model_evals should each contain one model."
            self.model = models[0]
            self.model_train = model_trains[0]
            self.model_eval = model_evals[0]
            self.num_joints = self.model.num_joints

        self.train_dl = train_dataloader
        if hasattr(self.train_dl.dataset, "datasets"):
            self.train_ds = self.train_dl.dataset.datasets[0]
        else:
            self.train_ds = self.train_dl.dataset

        self.test_sets = self._build_test_sets(test_dataloader)
        self.has_multiple_test_sets = len(self.test_sets) > 1
        self.test_dl = self.test_sets[0]["dataloader"] if self.test_sets else None
        self.test_ds = self.test_sets[0]["dataset"] if self.test_sets else None
        validate_masking_config(self.masking_config, self.masking_action_pool, getattr(self.train_ds, "mask_scheme_registry", None))

        # Derive representation flags from dataset's representation registry
        # repr_abs_only: Whether velocity representations are present in the full representation
        self.repr_abs_only = 'local_vel' not in self.train_ds._repr_list
        
        # For Purpose 1 (output representation control), check predicted_modalities for each task
        # This determines whether we decode R_world_joints ↔ smplx_body_pose_6d from full representations
        if self.task == 'both':
            # Separate flags for trajectory and pose predictions
            traj_predicted = self.test_ds.repr_registry.predicted_modalities.get('traj', [])
            pose_predicted = self.test_ds.repr_registry.predicted_modalities.get('pose', [])
            self.traj_outputs_global_rotations = 'R_world_joints' in traj_predicted or 'R_root_joints' in traj_predicted
            self.pose_outputs_global_rotations = 'R_world_joints' in pose_predicted or 'R_root_joints' in pose_predicted
        else:
            # Single task (traj or pose): use predicted_modalities for that task
            task_predicted = self.test_ds.repr_registry.predicted_modalities.get(self.task, [])
            self.outputs_global_rotations = 'R_world_joints' in task_predicted or 'R_root_joints' in task_predicted

        # --- Device Setup ---
        self.device = device

        self.smplx_neutral = smplx.create(model_path=self.body_model_path, model_type="smplx",
                                          gender='neutral', flat_hand_mean=True, use_pca=False).to(self.device)

        self.step = 0

        self.batches_per_epoch_per_gpu = len(self.train_dl)
        batches_per_opt_step = self._batches_per_optimizer_step(step=0)
        global_steps_per_epoch = self.batches_per_epoch_per_gpu / batches_per_opt_step
        self.num_epochs = int(np.ceil(self.num_steps / global_steps_per_epoch)) if global_steps_per_epoch > 0 else 1

        self.save_dir = logdir
        self.logger = logger
        self._write_masking_metadata()

        # --- Model to Device, DDP, EMA, MixedPrecisionTrainer, Optimizer, Scheduler, and Checkpoint Loading ---
        # 
        # CRITICAL: The initialization order matters due to dependencies:
        # 1. Move model to device FIRST: When you call model.to(device), PyTorch creates NEW 
        #    Parameter tensors on the target device. If the optimizer is created before this,
        #    it will reference the old (CPU) Parameter tensors and optimizer state will break.
        # 2. Apply any pretrained backbone / freezing before DDP (if enabled).
        # 3. Wrap in DDP BEFORE creating MixedPrecisionTrainer/optimizers (standard order).
        # 4. MixedPrecisionTrainer: stores references to model.parameters() as master_params
        # 5. Optimizers: need mp_trainer.master_params (which are the Parameter tensors)
        # 6. Schedulers: need optimizers
        # 7. load_checkpoint last: 
        #    - model.load_state_dict() updates parameter values IN-PLACE within the same tensors
        #    - optimizer.load_state_dict() restores optimizer states
        #    - scheduler.load_state_dict() restores scheduler states
        #
        # Key insight: Since load_state_dict updates values in-place, the optimizer's 
        # references to Parameter tensors remain valid and will "see" the pretrained weights.
        
        from utils.construct import construct_class_by_name
        
        # Step 1: Move model to device FIRST (before creating optimizer)
        # EMA models will be created AFTER checkpoint loading to avoid parameter count mismatch
        if self.task == 'both':
            self.trajnet, self.posenet = self.trajnet.to(self.device), self.posenet.to(self.device)
        elif self.task == 'traj' or self.task == 'pose':
            self.model = self.model.to(self.device)
        if (not self.use_ddp or self.rank == 0) and self.writer is not None:
            if self.task == 'both':
                self.writer.watch(self.trajnet, log=None, log_freq=max(1, self.log_interval))
                self.writer.watch(self.posenet, log=None, log_freq=max(1, self.log_interval))
            elif self.task == 'traj' or self.task == 'pose':
                self.writer.watch(self.model, log=None, log_freq=max(1, self.log_interval))

        # Step 2: Apply optional neural-submodule compilation and TrajControl weights/freezing before DDP.
        self._compile_neural_submodules()

        if (self.task == 'traj' or self.task == 'both') and self.trajcontrol:
            trajnet_model = self.trajnet if self.task == 'both' else self.model
            if self.load_pretrained_backbone:
                # This block is specific to TrajNet fine-tuning
                if self.load_pretrained_model:
                    if (not self.use_ddp or (self.use_ddp and self.rank == 0)):
                        print('[ERROR] for TrajControl, cannot set both load_pretrained_backbone and load_pretrained_model!')
                    exit()
                weights_backbone = torch.load(self.pretrained_backbone_path, map_location='cpu')
                trajnet_model.load_state_dict(weights_backbone, strict=False)
                if (not self.use_ddp or (self.use_ddp and self.rank == 0)):
                    print(f'loaded pretrained backbone from {self.pretrained_backbone_path}')
                weights_backbone_copy = {'state_dict': {}}
                for key in weights_backbone.keys():
                    if key.split('.')[0].split('_')[0] == 'diff':
                        weights_backbone_copy['state_dict']['controlnet.control' + key[4:]] = weights_backbone[key]
                trajnet_model.load_state_dict(weights_backbone_copy['state_dict'], strict=False)

            for name, param in trajnet_model.named_parameters():
                param.requires_grad = name.split('.')[0].split('_')[0] == 'controlnet'
            for name, layer in trajnet_model.named_modules():
                if name.split('.')[0].split('_')[0] in ['cond', 'diff', 'time']:
                    layer.eval()

        # Step 3: Wrap in DDP before creating optimizers (standard order)
        if self.use_ddp:
            dist.barrier()
            # find_unused_parameters=True is important as some logic is conditional
            if self.task == 'both':
                self.trajnet = DDP(
                    self.trajnet,
                    device_ids=[self.local_rank],
                    find_unused_parameters=True,
                    broadcast_buffers=True,
                )
                self.posenet = DDP(
                    self.posenet,
                    device_ids=[self.local_rank],
                    find_unused_parameters=True,
                    broadcast_buffers=True,
                )
            elif self.task == 'traj' or self.task == 'pose':
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    find_unused_parameters=True,
                    broadcast_buffers=True,
                )

        # Step 4: Create MixedPrecisionTrainer (stores references to model.parameters() which are now on device)
        if self.task == 'both':
            self.traj_mp_trainer = MixedPrecisionTrainer(
                model=self.trajnet,
                use_fp16=self.use_fp16,
                fp16_scale_growth=self.fp16_scale_growth,
                amp_dtype=self.amp_dtype,
            )
            self.pose_mp_trainer = MixedPrecisionTrainer(
                model=self.posenet,
                use_fp16=self.use_fp16,
                fp16_scale_growth=self.fp16_scale_growth,
                amp_dtype=self.amp_dtype,
            )
        elif self.task == 'traj' or self.task == 'pose':
            self.mp_trainer = MixedPrecisionTrainer(
                model=self.model,
                use_fp16=self.use_fp16,
                fp16_scale_growth=self.fp16_scale_growth,
                amp_dtype=self.amp_dtype,
            )
        
        # Step 5: Instantiate optimizers from config (using mp_trainer.master_params which reference device params)
        if self.task == 'both':
            self.traj_opt = construct_class_by_name(
                self.traj_mp_trainer.master_params,
                **self.traj_opt_config
            )
            self.pose_opt = construct_class_by_name(
                self.pose_mp_trainer.master_params,
                **self.pose_opt_config
            )
        elif self.task == 'traj' or self.task == 'pose':
            opt_config = self.traj_opt_config if self.task == 'traj' else self.pose_opt_config
            self.opt = construct_class_by_name(
                self.mp_trainer.master_params,
                **opt_config
            )
        
        # Step 6: Instantiate schedulers from config
        if self.task == 'both':
            self.traj_scheduler, self.pose_scheduler = self._create_schedulers(
                self.traj_opt, self.pose_opt
            )
        elif self.task == 'traj' or self.task == 'pose':
            self.scheduler = self._create_scheduler(self.opt)

        # Step 7: Load checkpoint if available (updates model weights in-place, loads opt/scheduler states)
        if self.load_pretrained_model:
            self.load_checkpoint_instance(self.pretrained_model_path)

        # Step 8: Create EMA models AFTER checkpoint loading
        # EMA models must be created after loading checkpoint so they get the pretrained weights
        # NOTE: Store EMA models on self (not on model object) to avoid including them in model.parameters()
        if self.task == 'both':
            if self.ema:
                traj_source_model = self.trajnet.module if isinstance(self.trajnet, DDP) else self.trajnet
                pose_source_model = self.posenet.module if isinstance(self.posenet, DDP) else self.posenet
                self.trajnet_ema_model = deepcopy(traj_source_model)
                self.posenet_ema_model = deepcopy(pose_source_model)
                for param in self.trajnet_ema_model.parameters(): param.requires_grad = False
                for param in self.posenet_ema_model.parameters(): param.requires_grad = False
                self.trajnet_ema_model.eval()
                self.posenet_ema_model.eval()
        elif self.task == 'traj' or self.task == 'pose':
            if self.ema:
                source_model = self.model.module if isinstance(self.model, DDP) else self.model
                self.model_ema_model = deepcopy(source_model)
                for param in self.model_ema_model.parameters(): param.requires_grad = False
                self.model_ema_model.eval()

        # grad_clip_val already set in _extract_config_values
        self.disable_tqdm = self.use_ddp and self.rank != 0

        self.masking_num_stages = self._count_masking_stages()
        self.best_loss_per_stage = {stage: float('inf') for stage in range(self.masking_num_stages)}
        self._current_micro_steps = self.grad_accumulate_steps

        # --- Unified Consistency Model Setup ---
        self.teacher_model = None
        self.target_model = None
        self.ema_scale_fn = None

        if self.model_type == 'consistency':
            raise DeprecationWarning("Consistency training is deprecated. Use 'shortcut' instead.")

        if self.model_type == 'diffusion':
            self.schedule_sampler_type = 'uniform'
            if self.task == 'both':
                self.traj_scheduler_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.model_trajnet_train)
                self.pose_scheduler_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.model_posenet_train)
            elif self.task == 'traj' or self.task == 'pose':
                self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.model_train)

    def _compile_neural_submodules(self):
        if not self.torch_compile:
            return
        if not hasattr(torch, "compile"):
            if not self.use_ddp or self.rank == 0:
                self.logger.warning("training.torch_compile=true requested, but torch.compile is unavailable. Skipping.")
            return

        compile_kwargs = {
            "mode": self.torch_compile_mode,
            "dynamic": self.torch_compile_dynamic,
            "fullgraph": False,
        }

        def _compile_attr(module, attr_name, label):
            target = getattr(module, attr_name, None)
            if target is None:
                return
            if getattr(target, "_rohm_compiled", False):
                return
            target._rohm_eager_forward = target.forward
            target.forward = torch.compile(target.forward, **compile_kwargs)
            target._rohm_compiled = True
            if not self.use_ddp or self.rank == 0:
                self.logger.info(f"[compile] Compiled {label}.forward with mode={self.torch_compile_mode}, dynamic={self.torch_compile_dynamic}.")

        if self.task == 'both':
            _compile_attr(self.trajnet, "backbone", "trajnet.backbone")
            _compile_attr(self.posenet, "architecture", "posenet.architecture")
        elif self.task == 'traj':
            _compile_attr(self.model, "backbone", "trajnet.backbone")
        elif self.task == 'pose':
            _compile_attr(self.model, "architecture", "posenet.architecture")

    def _profile_sync(self):
        if self.profile_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _profile_now(self):
        self._profile_sync()
        return time.perf_counter()

    def _extract_config_values(self, config):
        """Extract all configuration values from OmegaConf config into instance attributes."""
        from omegaconf import OmegaConf
        
        # Helper to safely get nested config values
        def get_section(section_name):
            return OmegaConf.to_container(config[section_name], resolve=True)
        
        # Task
        self.task = config['task']
        
        # Training config
        training = get_section('training')
        self.num_steps = training['num_steps']
        self.grad_accumulate_steps = training['grad_accumulate_steps']
        self.grad_clip_val = training['grad_clip_val']
        self.use_fp16 = training['use_fp16']
        self.fp16_scale_growth = training['fp16_scale_growth']
        self.ema = training['ema']
        self.ema_decay = training['ema_decay']
        self.log_interval = training['log_interval']
        self.eval_interval = training['eval_interval']
        self.save_interval = training['save_interval']
        self.timestep_respacing_eval = training['timestep_respacing_eval']
        self.model_type = training['model_type']
        self.generative_model_type = training['generative_model_type']
        self.enable_smpl_loss = training['enable_smpl_loss']
        self.cascade = training['cascade']
        self.body_model_path = training['body_model_path']
        self.profile_timing = bool(training.get('profile_timing', False))
        self.profile_cuda_sync = bool(training.get('profile_cuda_sync', False))
        self.profile_interval = max(1, int(training.get('profile_interval', 20)))
        self.torch_compile = bool(training.get('torch_compile', False))
        self.torch_compile_mode = training.get('torch_compile_mode', 'reduce-overhead')
        self.torch_compile_dynamic = bool(training.get('torch_compile_dynamic', False))
        self.skip_initial_eval = bool(training.get('skip_initial_eval', False))
        self.matmul_precision = training.get('matmul_precision', None)
        self.allow_tf32 = bool(training.get('allow_tf32', False))

        # DDP config
        ddp = get_section('ddp')
        self.use_ddp = ddp['enabled']
        if self.use_ddp:
            self.rank = ddp['rank']
            self.local_rank = ddp['local_rank']
            self.world_size = ddp['world_size']
        else:
            # Default values when DDP is disabled
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1

        # AMP dtype: prefer BF16 when AMP is enabled, fallback to FP16 if unsupported.
        self.amp_dtype = torch.bfloat16
        if self.use_fp16 and hasattr(torch.cuda, "is_bf16_supported"):
            if not torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.float16
                if not self.use_ddp or self.rank == 0:
                    print("[WARN] BF16 not supported on this GPU; falling back to FP16.")
        
        # DataLoader config
        train_dl = get_section('train_dataloader')
        self.batch_size = train_dl['batch_size']
        
        
        # Masking config
        masking = get_section('masking')
        self.masking_config = masking
        self.masking_batch_policy = masking.get('batch_policy', 'whole_batch')
        validate_batch_policy(self.masking_batch_policy)
        self.same_batch_for_mask_schedules = masking.get('same_batch_for_schedules', True)
        self.masking_schedule_per_step = masking.get('schedule_per_step', 'mixture')
        self.masking_action_pool = build_mask_action_pool(masking)
        self.masking_mixture = self._build_mask_mixture(masking)
        self.masking_seed = int(training.get('seed', 0))
        if self.masking_batch_policy == 'per_sample' and self.masking_schedule_per_step == 'all':
            raise ValueError("masking.batch_policy=per_sample requires schedule_per_step='mixture'.")

        # Representation config
        repr_config = get_section('representation')
        # NOTE: repr_abs_only and include_global_joint_rotations are now derived from
        # train_ds._repr_list after dataset initialization (see below line ~238)
        self.input_noise = repr_config['input_noise']
        
        # TrajControl config
        trajcontrol = get_section('trajcontrol')
        self.trajcontrol = trajcontrol['enabled']
        self.load_pretrained_backbone = trajcontrol['load_pretrained_backbone']
        self.pretrained_backbone_path = trajcontrol['pretrained_backbone_path']
        
        # Pretrained model config
        pretrained = get_section('pretrained')
        self.load_pretrained_model = pretrained['load_model']
        self.pretrained_model_path = pretrained['model_path']
        
        # Consistency model config (if applicable)
        if 'consistency' in config:
            consistency = get_section('consistency')
            self.cm_training_mode = consistency['training_mode']
            self.teacher_model_path = consistency['teacher_model_path']
            self.cm_target_ema_mode = consistency['target_ema_mode']
            self.cm_start_ema = consistency['start_ema']
            self.cm_scale_mode = consistency['scale_mode']
            self.cm_start_scales = consistency['start_scales']
            self.cm_end_scales = consistency['end_scales']
        
        # Optimizer configs (for construct_class_by_name)
        optimizers = get_section('optimizers')
        self.traj_opt_config = dict(optimizers['traj_opt'])
        self.pose_opt_config = dict(optimizers['pose_opt'])
        
        # LR Scheduler config
        lr_sched = get_section('lr_schedulers')
        self.lr_scheduler_type = lr_sched['type']
        self.lr_warmup_steps = lr_sched['warmup_steps']
        self.eta_min = lr_sched['eta_min']
        self.lr_decay_steps = lr_sched['decay_steps']
        self.lr_decay_gamma = lr_sched['decay_gamma']

    def _default_dataset_label(self, dataset, index: int) -> str:
        label = getattr(dataset, "dataset_label", None)
        if label:
            return str(label)
        base_name = dataset.__class__.__name__ if dataset is not None else "test"
        split = getattr(dataset, "split", None)
        if split:
            base_name = f"{base_name}_{split}"
        base_name = base_name.replace(" ", "_").replace("-", "_")
        return f"{index:02d}_{base_name}"

    def _build_test_sets(self, test_dataloader):
        if isinstance(test_dataloader, (list, tuple)):
            items = list(test_dataloader)
        else:
            items = [test_dataloader]

        test_sets = []
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                dl = item.get("dataloader")
                ds = item.get("dataset") or (dl.dataset if dl is not None else None)
                name = item.get("name")
            else:
                dl = item
                ds = dl.dataset if dl is not None else None
                name = None

            if name is None:
                name = self._default_dataset_label(ds, idx)
            test_sets.append({"name": name, "dataloader": dl, "dataset": ds})

        seen = {}
        for entry in test_sets:
            name = entry["name"]
            if name in seen:
                seen[name] += 1
                entry["name"] = f"{name}_{seen[name]}"
            else:
                seen[name] = 0

        return test_sets

    def _set_model_test_dataset(self, model, ds):
        if model is None or ds is None:
            return
        target = model.module if isinstance(model, DDP) else model
        if hasattr(target, "set_test_dataset"):
            target.set_test_dataset(ds)

    def _update_eval_wrapper_dataset(self, model_eval, ds, task: str):
        if model_eval is None or ds is None:
            return
        if hasattr(model_eval, "ds"):
            model_eval.ds = ds
        if hasattr(model_eval, "predicted_modalities") and hasattr(ds, "repr_registry"):
            predicted = ds.repr_registry.predicted_modalities.get(task)
            if predicted is not None:
                model_eval.predicted_modalities = predicted

    def _set_eval_dataset_context(self, ds):
        if ds is None:
            return
        if self.task == 'both':
            self._set_model_test_dataset(self.trajnet, ds)
            self._set_model_test_dataset(self.posenet, ds)
            if self.ema:
                self._set_model_test_dataset(self.trajnet_ema_model, ds)
                self._set_model_test_dataset(self.posenet_ema_model, ds)
            self._update_eval_wrapper_dataset(self.model_trajnet_eval, ds, 'traj')
            self._update_eval_wrapper_dataset(self.model_posenet_eval, ds, 'pose')
        else:
            self._set_model_test_dataset(self.model, ds)
            if self.ema:
                self._set_model_test_dataset(self.model_ema_model, ds)
            self._update_eval_wrapper_dataset(self.model_eval, ds, self.task)

    def _visible_ids_for_scheme(self, scheme_name: str):
        """
        Resolve visible joint ids for a mask scheme using registry if available, otherwise fallback to legacy dict.
        """
        reg = getattr(self.train_ds, "mask_scheme_registry", None)
        if reg is None:
            raise ValueError("mask_scheme_registry must be provided on dataset.")
        return set(reg.visible_ids(scheme_name))

    def _build_mask_action_pool(self, masking_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return build_mask_action_pool(masking_cfg)

    def _build_mask_mixture(self, masking_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        mixture = masking_cfg.get("mixture")
        if not mixture:
            raise ValueError("masking.mixture must be defined and non-empty.")
        return list(mixture)

    def _resolve_mixture_weights(self, step: int) -> Dict[str, float]:
        for entry in self.masking_mixture:
            until_step = entry.get("until_step")
            if until_step is None or step < int(until_step):
                weights = entry.get("weights")
                if not weights:
                    raise ValueError(f"masking.mixture entry missing weights: {entry}")
                return dict(weights)
        return dict(self.masking_mixture[-1].get("weights", {}))

    def _count_masking_stages(self) -> int:
        return sum(1 for entry in self.masking_mixture if entry.get("until_step") is not None) + 1

    def _masking_generator(self, offset: int = 0) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        base_seed = int(self.masking_seed)
        generator.manual_seed(base_seed + int(self.step) * 1000003 + int(self.rank) * 9176 + offset)
        return generator

    def _sample_action_names(self, weights: Dict[str, float], count: int, offset: int = 0) -> List[str]:
        return sample_action_names_from_weights(weights, count, rng=self._masking_generator(offset=offset))

    def _action_names_for_step(self) -> List[str]:
        weights = self._resolve_mixture_weights(self.step)
        if self.masking_schedule_per_step == "all":
            return list(weights.keys())
        return self._sample_action_names(weights, self.grad_accumulate_steps)

    def _resolve_random_visible_ids(self, action: Dict[str, Any]) -> List[int]:
        total_joints = getattr(self, "num_joints", 22)
        return resolve_random_visible_ids(action, total_joints)

    def _materialize_mask_action(self, action_name: str, offset: int = 0) -> Dict[str, Any]:
        total_joints = getattr(self, "num_joints", 22)
        return materialize_mask_action(action_name, self.masking_action_pool, total_joints, rng=self._masking_generator(offset=offset))

    def _materialize_per_sample_plan(self, batch_size: int, offset: int = 0) -> List[Dict[str, Any]]:
        weights = self._resolve_mixture_weights(self.step)
        action_names = self._sample_action_names(weights, batch_size, offset=offset)
        return materialize_per_sample_action_plan(
            action_names,
            self.masking_action_pool,
            getattr(self, "num_joints", 22),
            batch_size,
            rng=self._masking_generator(offset=offset + 4096),
        )

    def _materialize_action_instance(self, action: Dict[str, Any]) -> Dict[str, Any]:
        instance = dict(action)
        if instance.get("type") == "random_joint_visibility" and instance.get("visible_ids") is None:
            instance["visible_ids"] = self._resolve_random_visible_ids(instance)
        return instance

    def _apply_mask_action(self, batch, ds, task: str, action: Dict[str, Any]) -> None:
        total_joints = getattr(self, "num_joints", 22)
        return apply_mask_action(batch, ds, task, action, total_joints)

    def _apply_mask_action_plan(self, batch, ds, task: str, action_or_plan) -> None:
        total_joints = getattr(self, "num_joints", 22)
        return apply_mask_action_plan(batch, ds, task, action_or_plan, total_joints, self.masking_batch_policy)

    def _visible_ids_from_action(self, action: Dict[str, Any]) -> Optional[List[int]]:
        total_joints = getattr(self, "num_joints", 22)
        return visible_ids_from_action(action, self.train_ds.mask_scheme_registry, total_joints)

    def _build_eval_actions(self) -> List[Dict[str, Any]]:
        actions = []
        for name, action in self.masking_action_pool.items():
            if action.get("eval", True):
                eval_action = dict(action)
                eval_action["name"] = name
                actions.append(eval_action)
        return actions
    
    def _create_scheduler(self, optimizer):
        """Create a single LR scheduler from config."""
        if self.lr_scheduler_type == 'warmup_cosine':
            warmup_steps = self.lr_warmup_steps
            num_steps = self.num_steps
            return LambdaLR(optimizer, lambda step: lr_lambda(step, warmup_steps, num_steps))
        elif self.lr_scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=self.num_steps, eta_min=self.eta_min)
        elif self.lr_scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_steps, gamma=self.lr_decay_gamma)
        else:
            return None
    
    def _create_schedulers(self, traj_opt, pose_opt):
        """Create LR schedulers for both traj and pose optimizers."""
        traj_scheduler = self._create_scheduler(traj_opt)
        pose_scheduler = self._create_scheduler(pose_opt)
        return traj_scheduler, pose_scheduler

    def _get_masking_schedule(self):
        """
        Generate masking schedule for training.

        Returns:
            List of masking actions for each micro-batch step.
        """
        action_names = self._action_names_for_step()
        if self.masking_schedule_per_step == "all":
            action_names = list(dict.fromkeys(action_names))
        return [self._materialize_mask_action(name, offset=i) for i, name in enumerate(action_names)]

    def _micro_steps_for_step(self, step: int) -> int:
        if self.masking_batch_policy == "per_sample":
            return 1
        if self.masking_schedule_per_step == "all":
            weights = self._resolve_mixture_weights(step)
            return len(weights)
        return self.grad_accumulate_steps

    def _batches_per_optimizer_step(self, step: int) -> int:
        if self.same_batch_for_mask_schedules:
            return 1
        return self._micro_steps_for_step(step)

    def run_loop(self):
        start_epoch = 0
        batches_per_opt_step = self._batches_per_optimizer_step(self.step)
        if self.batches_per_epoch_per_gpu > 0 and batches_per_opt_step > 0:
            num_optimizer_steps_per_epoch = self.batches_per_epoch_per_gpu // batches_per_opt_step
            if num_optimizer_steps_per_epoch > 0: start_epoch = int(self.step // num_optimizer_steps_per_epoch)

        for epoch in range(start_epoch, self.num_epochs):
            if self.task == 'both':
                self.trajnet.train(); self.posenet.train()
            elif self.task == 'traj' or self.task == 'pose':
                self.model.train()
            if self.use_ddp: self.train_dl.sampler.set_epoch(epoch)
            
            data_iterator = iter(self.train_dl)
            pbar = tqdm(range(num_optimizer_steps_per_epoch), disable=self.disable_tqdm)
            
            data_wait_start = self._profile_now() if self.profile_timing else None
            for _ in pbar:
                if self.step >= self.num_steps: return

                step_profile = self.profile_timing and self.step % self.profile_interval == 0
                profile_times = {} if step_profile else None
                profile_step_start = self._profile_now() if step_profile else None

                if self.step % self.eval_interval == 0 and not (self.skip_initial_eval and self.step == 0):
                    eval_start = self._profile_now() if step_profile else None
                    if self.task == 'both':
                        # Run standard evaluation (for logging metrics)
                        self.evaluate(epoch, save_best_model=False)
                        # Run pipeline evaluation and use its metrics for model saving
                        pipeline_metric = self._evaluate_pipeline(epoch)
                        current_stage = self.get_current_stage(self.step)
                        if not hasattr(self, 'best_pipeline_loss_per_stage'):
                            self.best_pipeline_loss_per_stage = {stage: float('inf') for stage in range(self.masking_num_stages)}
                        
                        if pipeline_metric.item() < self.best_pipeline_loss_per_stage.get(current_stage, float('inf')):
                            self.best_pipeline_loss_per_stage[current_stage] = pipeline_metric.item()
                            if not self.use_ddp or (self.use_ddp and self.rank == 0):
                                self.logger.info(f"[*] New best model for STAGE {current_stage} found with Pipeline MPJPE {pipeline_metric.item():.4f} mm. Saving checkpoint.")
                            self.save(is_best_for_stage=True, stage=current_stage)
                    else:
                        # Standard evaluation with model saving for single-task training
                        self.evaluate(epoch)
                    if step_profile:
                        profile_times['eval'] = self._profile_now() - eval_start

                masking_schedule = self._get_masking_schedule()
                micro_steps = self._micro_steps_for_step(self.step)
                self._current_micro_steps = micro_steps
                assert len(masking_schedule) == micro_steps

                if self.task == 'both':
                    self.traj_mp_trainer.zero_grad()
                    self.pose_mp_trainer.zero_grad()
                else:
                    self.mp_trainer.zero_grad()
                    
                accumulated_losses = {}
                accumulated_counts = {}
                log_this_step = self.step % self.log_interval == 0

                def _accumulate(prefix, losses):
                    for key, val in losses.items():
                        full_key = f"{prefix}{key}"
                        accumulated_losses[full_key] = accumulated_losses.get(full_key, 0.0) + val
                        accumulated_counts[full_key] = accumulated_counts.get(full_key, 0) + 1

                def _maybe_no_sync(task_name, is_last_microbatch):
                    if not self.use_ddp or is_last_microbatch:
                        return nullcontext()
                    if self.task == 'both':
                        if task_name == 'traj':
                            return self.trajnet.no_sync()
                        if task_name == 'pose':
                            return self.posenet.no_sync()
                        if task_name == 'both':
                            stack = ExitStack()
                            stack.enter_context(self.trajnet.no_sync())
                            stack.enter_context(self.posenet.no_sync())
                            return stack
                    return self.model.no_sync()
                
                batch_source = self.same_batch_for_mask_schedules
                can_reuse_shared_batch = batch_source and micro_steps == 1
                if batch_source:
                    try:
                        batch = next(data_iterator)
                    except StopIteration:
                        break
                    if step_profile and data_wait_start is not None:
                        profile_times['dataloader_wait'] = profile_times.get('dataloader_wait', 0.0) + (self._profile_now() - data_wait_start)

                for i in range(micro_steps):
                    is_last_microbatch = i == micro_steps - 1
                    compute_metrics = log_this_step and i == 0
                    if batch_source:
                        if can_reuse_shared_batch:
                            micro_batch = batch
                        else:
                            micro_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    else:
                        data_wait_start_micro = self._profile_now() if step_profile else None
                        try:
                            micro_batch = next(data_iterator)
                        except StopIteration:
                            break
                        if step_profile:
                            profile_times['dataloader_wait'] = profile_times.get('dataloader_wait', 0.0) + (self._profile_now() - data_wait_start_micro)

                    if self.task == 'both':
                        if self.cascade:
                            with _maybe_no_sync('both', is_last_microbatch):
                                losses_log = self.run_micro_batch(
                                    micro_batch,
                                    masking_schedule[i],
                                    epoch,
                                    task='both',
                                    compute_metrics=compute_metrics,
                                    profile_times=profile_times,
                                )
                            _accumulate('both/', losses_log)
                        else:
                            if can_reuse_shared_batch:
                                traj_micro_batch = dict(micro_batch)
                                pose_micro_batch = dict(micro_batch)
                            else:
                                traj_micro_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()}
                                pose_micro_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()}
                            with _maybe_no_sync('traj', is_last_microbatch):
                                traj_losses_log = self.run_micro_batch(
                                    traj_micro_batch,
                                    masking_schedule[i],
                                    epoch,
                                    task='traj',
                                    compute_metrics=compute_metrics,
                                    profile_times=profile_times,
                                )
                            with _maybe_no_sync('pose', is_last_microbatch):
                                pose_losses_log = self.run_micro_batch(
                                    pose_micro_batch,
                                    masking_schedule[i],
                                    epoch,
                                    task='pose',
                                    compute_metrics=compute_metrics,
                                    profile_times=profile_times,
                                )

                            # Accumulate losses from both tasks
                            _accumulate('traj/', traj_losses_log)
                            _accumulate('pose/', pose_losses_log)
 
                    elif self.task == 'traj' or self.task == 'pose':
                        with _maybe_no_sync(self.task, is_last_microbatch):
                            losses_log = self.run_micro_batch(
                                micro_batch,
                                masking_schedule[i],
                                epoch,
                                task=self.task,
                                compute_metrics=compute_metrics,
                                profile_times=profile_times,
                            )
                        _accumulate(f'{self.task}/', losses_log)

                opt_start = self._profile_now() if step_profile else None
                if self.task == 'both':
                    self.traj_mp_trainer.optimize(self.traj_opt, max_norm=self.grad_clip_val)
                    self.pose_mp_trainer.optimize(self.pose_opt, max_norm=self.grad_clip_val)
                else:
                    self.mp_trainer.optimize(self.opt, max_norm=self.grad_clip_val)
                if step_profile:
                    profile_times['optimizer'] = profile_times.get('optimizer', 0.0) + (self._profile_now() - opt_start)

                ema_start = self._profile_now() if step_profile else None
                self._update_ema()
                if self.model_type == 'consistency': self._update_target_ema()
                if step_profile:
                    profile_times['ema'] = profile_times.get('ema', 0.0) + (self._profile_now() - ema_start)

                scheduler_start = self._profile_now() if step_profile else None
                if self.task == 'both':
                    self.traj_scheduler.step()
                    self.pose_scheduler.step()
                else:
                    self.scheduler.step()
                if step_profile:
                    profile_times['scheduler'] = profile_times.get('scheduler', 0.0) + (self._profile_now() - scheduler_start)

                if step_profile:
                    profile_times['total_step'] = self._profile_now() - profile_step_start
                    if not self.use_ddp or (self.use_ddp and self.rank == 0):
                        ordered = [
                            'dataloader_wait', 'h2d_transfer', 'process_batch_cond', 'masking',
                            'traj_loss_backward', 'pose_loss_backward', 'both_loss_backward',
                            'optimizer', 'ema', 'scheduler', 'eval', 'total_step'
                        ]
                        timing_parts = [f"{name}={profile_times[name]:.4f}s" for name in ordered if name in profile_times]
                        self.logger.info(f"[Step {self.step}/{self.num_steps}] [timing] " + ", ".join(timing_parts))

                if self.step % self.log_interval == 0:
                    avg_losses = {
                        k: v / max(1, accumulated_counts.get(k, 1))
                        for k, v in accumulated_losses.items()
                    }
                    if not self.use_ddp or (self.use_ddp and self.rank == 0):
                        for key, val in avg_losses.items():
                            self.writer.add_scalar(f'train/{key}', val, self.step)
                            self.logger.info(f'[Step {self.step}/{self.num_steps}] [train] {key}: {val:.10f}')
                        
                        if self.task == 'both':
                            self.writer.add_scalar('train/traj_lr', self.traj_opt.param_groups[0]['lr'], self.step)
                            self.writer.add_scalar('train/pose_lr', self.pose_opt.param_groups[0]['lr'], self.step)
                        else:
                            self.writer.add_scalar('train/lr', self.opt.param_groups[0]['lr'], self.step)


                if self.step > 0 and self.step % self.save_interval == 0: self.save()
                self.step += 1
                data_wait_start = self._profile_now() if self.profile_timing else None

        if self.use_ddp: dist_util.cleanup()

    def _masking_plan_for_batch(self, batch_size: int, fallback_action, offset: int = 0):
        if self.masking_batch_policy == "per_sample":
            return self._materialize_per_sample_plan(batch_size, offset=offset)
        return fallback_action

    def _write_masking_metadata(self) -> None:
        if self.use_ddp and self.rank != 0:
            return
        from omegaconf import OmegaConf
        mixture = (
            OmegaConf.to_container(self.masking_mixture, resolve=True)
            if OmegaConf.is_config(self.masking_mixture)
            else self.masking_mixture
        )
        metadata = {
            "seed": int(self.masking_seed),
            "batch_policy": self.masking_batch_policy,
            "schedule_per_step": self.masking_schedule_per_step,
            "same_batch_for_schedules": bool(self.same_batch_for_mask_schedules),
            "actions": sorted(self.masking_action_pool.keys()),
            "mixture": mixture,
        }
        path = os.path.join(self.save_dir, "masking_metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _log_masking_counts(self, batch) -> None:
        if self.use_ddp and self.rank != 0:
            return
        names = batch.get("cond_pattern_names")
        if not names:
            return
        counts = {name: names.count(name) for name in sorted(set(names))}
        for name, count in counts.items():
            self.writer.add_scalar(f"train/masking_count/{name}", count, self.step)
        self.logger.info(f"[Step {self.step}/{self.num_steps}] [masking] pattern_counts={counts}")

    def run_micro_batch(self, batch, masking_action, epoch, task, compute_metrics: bool = True, profile_times: dict | None = None):
        """
        Run a single micro-batch training step.

        Args:
            batch: Input batch dict
            masking_action: Masking action dict
            epoch: Current epoch
            task: 'traj', 'pose', or 'both'
        """
        accum_scale = 1.0 / max(1, self._current_micro_steps)
        # NOTE: task means 'traj' or 'pose', is a local var for clarity, and is different from self.task
        h2d_start = self._profile_now() if profile_times is not None else None
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, non_blocking=True)
        if profile_times is not None:
            profile_times['h2d_transfer'] = profile_times.get('h2d_transfer', 0.0) + (self._profile_now() - h2d_start)
        bs, clip_len = batch['motion_repr_clean'].shape[:2]
        traj_feat_dim = get_modality_dim(self.train_ds.traj_repr_name_list, self.train_ds._repr_indices)

        VIS_BATCH_DBG = False
        if VIS_BATCH_DBG:
            from utils.repr_adapter import decode_smplx_from_full_repr
            from utils.smplx_vis_frontend import SmplxVisConfig, visualize_smplx_sequence
            from data_sets.motion_representation import SMPLH_KINTREE

            device, dtype = batch['motion_repr_clean'].device, batch['motion_repr_clean'].dtype
            sample_norms = batch['extras/sample_norms'] if self.train_ds.norm_strategy == 'l2' else None
            
            if self.train_ds.norm_strategy == 'l2':
                sample_norms = {k[len("extras/sample_norms/"):]: v for k, v in batch.items() if k.startswith('extras/sample_norms')}
            else:
                sample_norms = None

            batch_motion_repr_clean = self.train_ds._unnormalize_full_tensor(batch['motion_repr_clean'], device, sample_norms=sample_norms)

            if self.input_noise:
                batch_motion_repr_noisy = self.train_ds._unnormalize_full_tensor(batch['motion_repr_noisy'], device, sample_norms=sample_norms)
            output_rot_flag = 'R_world_joints' in self.train_ds._repr_list
            _, joints_dbg, verts_dbg = decode_smplx_from_full_repr(
                full_repr=batch_motion_repr_clean,
                repr_list=self.train_ds._repr_list,
                repr_indices=self.train_ds._repr_indices,
                smplx_model=self.smplx_neutral,
                output_global_rotations=output_rot_flag,
                cano_jts_pos=self.train_ds.cano_jts_pos,
                return_verts=True,
            )

            cfg = SmplxVisConfig(
                window_title="Train Batch Debug",
                num_repeats=1,
                fixed_cam=False,
                camera_follow=True,
                show_mesh=True,
                show_skeleton=True,
            )
            for i in range(bs):
                visualize_smplx_sequence(
                    joints=joints_dbg,
                    vertices=verts_dbg,
                    faces=self.smplx_neutral.faces,
                    parent_indices=SMPLH_KINTREE,
                    batch_idx=i,
                    cfg=cfg,
                )
        
        if task == 'both':
            # Process batch conditioning to get separate traj and pose batches
            process_start = self._profile_now() if profile_times is not None else None
            traj_batch, pose_batch = process_batch_cond(
                batch=batch,
                ds=self.train_ds,
                mask_scheme=None,
                task='both',
                input_noise=self.input_noise,
                num_joints=self.num_joints
            )
            if profile_times is not None:
                profile_times['process_batch_cond'] = profile_times.get('process_batch_cond', 0.0) + (self._profile_now() - process_start)

            bs, clip_len = traj_batch['motion_repr_clean'].shape[:2]
            masking_action = self._masking_plan_for_batch(bs, masking_action)

            mask_start = self._profile_now() if profile_times is not None else None
            self._apply_mask_action_plan(traj_batch, self.train_ds, "traj", masking_action)
            self._apply_mask_action_plan(pose_batch, self.train_ds, "pose", masking_action)
            if profile_times is not None:
                profile_times['masking'] = profile_times.get('masking', 0.0) + (self._profile_now() - mask_start)
            if compute_metrics:
                self._log_masking_counts(pose_batch)

            common_loss_args = {'model': None, 'batch': None, 'smplx_model': self.smplx_neutral, 'epoch': epoch, "enable_smpl_loss": self.enable_smpl_loss, "store_pred": True}
            t, weights = None, None
            loss_dict = {}
            
            # Step 1: TrajNet training.
            traj_train_start = self._profile_now() if profile_times is not None else None
            common_loss_args['model'] = self.trajnet
            common_loss_args['batch'] = traj_batch
            model_train = self.model_trajnet_train
            with torch.amp.autocast(self.device.type, enabled=self.use_fp16, dtype=self.amp_dtype):
                if self.model_type == 'diffusion':
                    schedule_sampler = self.traj_scheduler_sampler
                    t, weights = schedule_sampler.sample(bs, self.device)
                    current_losses = model_train.training_losses(**common_loss_args, t=t, noise=None)
                    loss_for_backward = (current_losses["loss"] * weights).mean()
                elif self.model_type == 'improved_consistency':
                    current_losses = model_train.training_losses(**common_loss_args, current_step=self.step, total_steps=self.num_steps)
                    loss_for_backward = current_losses["loss"]
                elif self.model_type == 'consistency':
                    if self.ema_scale_fn is not None:
                        _, num_scales = self.ema_scale_fn(self.step)
                        current_losses = model_train.training_losses(**common_loss_args, target_model=self.target_model, teacher_model=self.teacher_model, num_scales=num_scales)
                        loss_for_backward = current_losses["loss"]
                    else:
                        # Fallback if ema_scale_fn is None (shouldn't happen but handle gracefully)
                        current_losses = model_train.training_losses(**common_loss_args, target_model=self.target_model, teacher_model=self.teacher_model, num_scales=1)
                        loss_for_backward = current_losses["loss"]
                else: # meanflow, shortcut, flow_matching
                    current_losses = model_train.training_losses(**common_loss_args, t=None, current_step=self.step)
                    loss_for_backward = current_losses["loss"]
            loss_dict = {**loss_dict, **current_losses}
            self.traj_mp_trainer.backward(loss_for_backward * accum_scale)
            if profile_times is not None:
                profile_times['traj_loss_backward'] = profile_times.get('traj_loss_backward', 0.0) + (self._profile_now() - traj_train_start)

            # Step 2: PoseNet training
            if self.cascade:
                traj_output = traj_batch.get('extras/pred')
                if traj_output is None:
                    raise RuntimeError("Missing TrajNet outputs (extras/pred) for cascade PoseNet conditioning.")
                pose_batch['cond'] = insert_modalities_into_cond(
                    pose_batch['cond'],
                    traj_output.detach(),
                    self.train_ds.traj_repr_name_list,
                    self.train_ds,
                    task='pose',
                )
            
            # Step 2: PoseNet training.
            pose_train_start = self._profile_now() if profile_times is not None else None
            common_loss_args['model'] = self.posenet
            common_loss_args['batch'] = pose_batch
            model_train = self.model_posenet_train
            with torch.amp.autocast(self.device.type, enabled=self.use_fp16, dtype=self.amp_dtype):
                if self.model_type == 'diffusion':
                    schedule_sampler = self.pose_scheduler_sampler
                    t, weights = schedule_sampler.sample(bs, self.device)
                    current_losses = model_train.training_losses(**common_loss_args, t=t, noise=None)
                    loss_for_backward = (current_losses["loss"] * weights).mean()
                elif self.model_type == 'improved_consistency':
                    current_losses = model_train.training_losses(**common_loss_args, current_step=self.step, total_steps=self.num_steps)
                    loss_for_backward = current_losses["loss"]
                elif self.model_type == 'consistency':
                    if self.ema_scale_fn is not None:
                        _, num_scales = self.ema_scale_fn(self.step)
                        current_losses = model_train.training_losses(**common_loss_args, target_model=self.target_model, teacher_model=self.teacher_model, num_scales=num_scales)
                        loss_for_backward = current_losses["loss"]
                    else:
                        # Fallback if ema_scale_fn is None (shouldn't happen but handle gracefully)
                        current_losses = model_train.training_losses(**common_loss_args, target_model=self.target_model, teacher_model=self.teacher_model, num_scales=1)
                        loss_for_backward = current_losses["loss"]
                else: # meanflow, shortcut, flow_matching
                    current_losses = model_train.training_losses(**common_loss_args, t=None, current_step=self.step)
                    loss_for_backward = current_losses["loss"]
            loss_dict = {**loss_dict, **current_losses}
            self.pose_mp_trainer.backward(loss_for_backward * accum_scale)
            if profile_times is not None:
                profile_times['pose_loss_backward'] = profile_times.get('pose_loss_backward', 0.0) + (self._profile_now() - pose_train_start)

            if not compute_metrics:
                out = {}
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        out[k] = v.item()
                    else:
                        out[k] = v
                return out

            # Step 4: Recover joint positions and compute pipeline evaluation metrics
            # Import needed for evaluation
            from utils.evaluation import MotionEvaluator
            from utils.repr_adapter import decode_smplx_from_full_repr, maybe_unnormalize_full_repr
            
            # Initialize motion evaluator
            if not hasattr(self, '_motion_evaluator'):
                self._motion_evaluator = MotionEvaluator(fps=30)
            
            # Get dataset normalization statistics
            # All tensors are now in 3D format [bs, T, dim]
            clean_repr_norm = pose_batch['motion_repr_clean']
            pipeline_repr_norm = pose_batch['extras/pred']

            if self.train_ds.norm_strategy == 'l2':
                sample_norms = {k[len("extras/sample_norms/"):]: v for k, v in pose_batch.items() if k.startswith('extras/sample_norms')}
            else:
                sample_norms = None
            
            clean_repr_unnorm = maybe_unnormalize_full_repr(
                clean_repr_norm, self.train_ds, sample_norms=sample_norms
            )
            pipeline_repr_unnorm = maybe_unnormalize_full_repr(
                pipeline_repr_norm, self.train_ds, sample_norms=sample_norms
            )

            output_rot_flag = self.pose_outputs_global_rotations if self.task == 'both' else self.outputs_global_rotations
            clean_repr_dict, joint_pos_clean, _ = decode_smplx_from_full_repr(
                full_repr=clean_repr_unnorm,
                repr_list=self.train_ds._repr_list,
                repr_indices=self.train_ds._repr_indices,
                smplx_model=self.smplx_neutral,
                output_global_rotations=output_rot_flag,
                cano_jts_pos=self.train_ds.cano_jts_pos,
                return_verts=False,
            )
            pipeline_repr_dict, joint_pos_pipeline, _ = decode_smplx_from_full_repr(
                full_repr=pipeline_repr_unnorm,
                repr_list=self.train_ds._repr_list,
                repr_indices=self.train_ds._repr_indices,
                smplx_model=self.smplx_neutral,
                output_global_rotations=output_rot_flag,
                cano_jts_pos=self.train_ds.cano_jts_pos,
                return_verts=False,
            )
                
            if VIS_BATCH_DBG:
                from data_sets.motion_representation import SMPLH_KINTREE
                from utils.smplx_vis_frontend import SmplxVisConfig, visualize_smplx_sequence
                from utils.repr_adapter import decode_smplx_from_full_repr
                bs, clip_len, num_joints, _ = joint_pos_clean.shape

                _, _, verts_pipeline = decode_smplx_from_full_repr(
                    full_repr=pipeline_repr_unnorm,
                    repr_list=self.train_ds._repr_list,
                    repr_indices=self.train_ds._repr_indices,
                    smplx_model=self.smplx_neutral,
                    output_global_rotations=output_rot_flag,
                    cano_jts_pos=self.train_ds.cano_jts_pos,
                    return_verts=True,
                )
                cfg = SmplxVisConfig(
                    window_title="Pose Pipeline Debug",
                    num_repeats=10,
                    fixed_cam=False,
                    camera_follow=True,
                    show_mesh=True,
                    show_skeleton=True,
                )
                for i in range(bs):
                    visualize_smplx_sequence(
                        joints=joint_pos_pipeline,
                        vertices=verts_pipeline,
                        faces=self.smplx_neutral.faces,
                        parent_indices=SMPLH_KINTREE,
                        batch_idx=i,
                        cfg=cfg,
                    )

            # Convert to numpy for evaluation 
            joint_pos_clean_np = joint_pos_clean.detach().cpu().numpy()     # [bs, clip_len, self.num_joints, 3]
            joint_pos_pipeline_np = joint_pos_pipeline.detach().cpu().numpy() # [bs, clip_len, self.num_joints, 3]
            
            vis_joint_mask = pose_batch['cond_joint_mask'][..., 0].detach().cpu().numpy().astype(bool)

            # Compute evaluation metrics using MotionEvaluator
            eval_metrics = self._motion_evaluator.compute_comparison_metrics(
                pred_joints=joint_pos_pipeline_np,
                gt_joints=joint_pos_clean_np,
                repr_dict_rec=pipeline_repr_dict,
                repr_dict_clean=clean_repr_dict,
                vis_joint_ids=None,
                occ_joint_ids=None,
                vis_joint_mask=vis_joint_mask,
                occ_joint_mask=~vis_joint_mask,
            )
            loss_dict = {**loss_dict, **eval_metrics}

            out = {}
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor): out[k] = v.item()
                else: out[k] = v
            return out
        
        else:
            process_start = self._profile_now() if profile_times is not None else None
            batch = process_batch_cond(
                batch=batch,
                ds=self.train_ds,
                mask_scheme=None,
                task=task,
                input_noise=self.input_noise,
                num_joints=self.num_joints
            )
            if profile_times is not None:
                profile_times['process_batch_cond'] = profile_times.get('process_batch_cond', 0.0) + (self._profile_now() - process_start)
            bs, _ = batch['motion_repr_clean'].shape[:2]
            masking_action = self._masking_plan_for_batch(bs, masking_action)
            mask_start = self._profile_now() if profile_times is not None else None
            self._apply_mask_action_plan(batch, self.train_ds, task, masking_action)
            if profile_times is not None:
                profile_times['masking'] = profile_times.get('masking', 0.0) + (self._profile_now() - mask_start)

            # All tensors remain in 3D format [bs, T, dim] - no conversion needed

            # --- Forward and Backward Pass ---
            t, weights = None, None
            
            # Select the correct model and components based on task
            if self.task == 'both':
                if task == 'traj':
                    model = self.trajnet
                    model_train = self.model_trajnet_train
                    if self.model_type == 'diffusion':
                        schedule_sampler = self.traj_scheduler_sampler
                    mp_trainer = self.traj_mp_trainer
                else:  # task == 'pose'
                    model = self.posenet
                    model_train = self.model_posenet_train
                    if self.model_type == 'diffusion':
                        schedule_sampler = self.pose_scheduler_sampler
                    mp_trainer = self.pose_mp_trainer
            else:
                model = self.model
                model_train = self.model_train
                if self.model_type == 'diffusion':
                    schedule_sampler = self.schedule_sampler
                mp_trainer = self.mp_trainer
                
            common_loss_args = {'model': model, 'batch': batch, 'smplx_model': self.smplx_neutral, 'epoch': epoch, "enable_smpl_loss": self.enable_smpl_loss}

            train_start = self._profile_now() if profile_times is not None else None
            with torch.amp.autocast(self.device if isinstance(self.device, str) else self.device.type, enabled=self.use_fp16, dtype=self.amp_dtype):
                if self.model_type == 'diffusion':
                    t, weights = schedule_sampler.sample(bs, self.device)
                    current_losses = model_train.training_losses(**common_loss_args, t=t, noise=None)
                    loss_for_backward = (current_losses["loss"] * weights).mean()
                elif self.model_type == 'improved_consistency':
                    current_losses = model_train.training_losses(**common_loss_args, current_step=self.step, total_steps=self.num_steps)
                    loss_for_backward = current_losses["loss"]
                elif self.model_type == 'consistency':
                    if self.ema_scale_fn is not None:
                        _, num_scales = self.ema_scale_fn(self.step)
                        current_losses = model_train.training_losses(**common_loss_args, target_model=self.target_model, teacher_model=self.teacher_model, num_scales=num_scales)
                        loss_for_backward = current_losses["loss"]
                    else:
                        # Fallback if ema_scale_fn is None (shouldn't happen but handle gracefully)
                        current_losses = model_train.training_losses(**common_loss_args, target_model=self.target_model, teacher_model=self.teacher_model, num_scales=1)
                        loss_for_backward = current_losses["loss"]
                else: # meanflow, shortcut, flow_matching
                    current_losses = model_train.training_losses(**common_loss_args, t=t, current_step=self.step)
                    loss_for_backward = current_losses["loss"]

            mp_trainer.backward(loss_for_backward * accum_scale)
            if profile_times is not None:
                key = f'{task}_loss_backward' if task in ('traj', 'pose') else 'both_loss_backward'
                profile_times[key] = profile_times.get(key, 0.0) + (self._profile_now() - train_start)
            return {k: v.item() for k, v in current_losses.items()}

    def get_current_stage(self, step):
        boundaries = []
        for entry in self.masking_mixture:
            until_step = entry.get("until_step")
            if until_step is None:
                break
            boundaries.append(int(until_step))
        for idx, boundary in enumerate(boundaries):
            if step < boundary:
                return idx
        return len(boundaries)
        
    def _evaluate_pipeline_single_dataset(self, epoch, test_dl, ds, dataset_label):
        """
        Evaluate the pipeline with alternating TrajNet -> PoseNet inference.
        This method tests how trajectory and pose features are coupled together.
        Only runs when task == 'both'.
        """
        if self.task != 'both':
            if not self.use_ddp or (self.use_ddp and self.rank == 0):
                self.logger.warning("Pipeline evaluation is only available when task == 'both'. Skipping.")
            return

        traj_feat_dim = get_modality_dim(ds.traj_repr_name_list, ds._repr_indices)
        
        # Initialize models
        traj_source_model = self.trajnet.module if isinstance(self.trajnet, DDP) else self.trajnet
        pose_source_model = self.posenet.module if isinstance(self.posenet, DDP) else self.posenet
        if self.ema:
            traj_ema_model = self.trajnet_ema_model
            pose_ema_model = self.posenet_ema_model
            traj_ema_model.eval()
            pose_ema_model.eval()
        else:
            traj_ema_model = traj_source_model
            pose_ema_model = pose_source_model
            traj_ema_model.eval()
            pose_ema_model.eval()
    
        if self.masking_batch_policy == "per_sample":
            eval_schemes = {"pipeline_mixed_per_sample": {}}
        else:
            eval_schemes = {}
            for action in self._build_eval_actions():
                action_name = action.get("name", "mask_action")
                scheme_key = f"pipeline_{action_name.replace(' ', '_').replace('-', '_')}"
                eval_schemes[scheme_key] = self._materialize_action_instance(action)

        all_schemes_aggregated_losses = {}
        num_batches_total = len(test_dl)
        for scheme_name, scheme_params in eval_schemes.items():
            if not self.use_ddp or (self.use_ddp and self.rank == 0): 
                self.logger.info(f"Evaluating pipeline scheme: {scheme_name} (dataset={dataset_label})") 
            current_scheme_losses_sum, num_batches_processed = {}, 0
            
            for test_step, test_batch_original in tqdm(enumerate(test_dl), desc=f"Pipeline Eval [{dataset_label}]: {scheme_name}", disable=self.disable_tqdm, leave=False):
                num_batches_processed += 1
                
                # Convert to device and process batch for conditioning
                test_batch = {}
                for k, v in test_batch_original.items():
                    if isinstance(v, torch.Tensor):
                        test_batch[k] = v.to(self.device)
                    else:
                        test_batch[k] = v
                
                # Process batch conditioning to get separate traj and pose batches
                traj_batch, pose_batch = process_batch_cond(
                    batch=test_batch,
                    ds=ds,
                    mask_scheme=None,
                    task='both',
                    input_noise=self.input_noise,
                    num_joints=self.num_joints
                )

                bs_eval, clip_len_eval = traj_batch['motion_repr_clean'].shape[:2]
                masking_action = self._masking_plan_for_batch(bs_eval, scheme_params, offset=test_step)

                self._apply_mask_action_plan(traj_batch, ds, "traj", masking_action)
                self._apply_mask_action_plan(pose_batch, ds, "pose", masking_action)

                # Step 1: TrajNet inference
                traj_shape = [bs_eval, clip_len_eval, traj_feat_dim]
                with torch.no_grad():
                    traj_output = self.model_trajnet_eval.p_sample_loop(
                        model=traj_ema_model, 
                        batch=traj_batch, 
                        shape=traj_shape,
                        num_inference_steps=1, # num_inference_steps is fixed to 1; only the shortcut model uses this argument.
                        progress=False
                    )
                
                # Step 2: Assign trajectory output to pose batch conditioning
                # pose_batch['cond'] structure: [joint-based interleaved] | [global including traj]
                # We need to replace the trajectory part with TrajNet output
                pose_batch['cond'] = insert_modalities_into_cond(pose_batch['cond'], traj_output, ds.traj_repr_name_list, ds, task='pose')
                
                # Step 3: PoseNet inference with metrics computation
                # All tensors remain in 3D format [bs, T, dim] - no conversion needed

                # Determine pose output shape based on predicted modalities
                # Shape is [bs, T, pose_feat_dim] in 3D format
                pose_predicted = ds.repr_registry.predicted_modalities.get('pose', [])
                if not pose_predicted:
                    raise ValueError("predicted_modalities['pose'] is empty; cannot infer pose output shape.")
                pose_feat_dim = get_modality_dim(pose_predicted, ds._repr_indices)
                pose_shape = [bs_eval, clip_len_eval, pose_feat_dim]
                
                with torch.no_grad():
                    model_output_full = self.model_posenet_eval.p_sample_loop(
                        model=pose_ema_model,
                        batch=pose_batch,
                        shape=pose_shape,
                        num_inference_steps=1,
                        progress=False
                    )
                
                # Step 4: Recover joint positions and compute pipeline evaluation metrics
                # Import needed for evaluation
                from utils.evaluation import MotionEvaluator
                from utils.repr_adapter import decode_smplx_from_full_repr, maybe_unnormalize_full_repr
                
                # Initialize motion evaluator
                if not hasattr(self, '_motion_evaluator'):
                    self._motion_evaluator = MotionEvaluator(fps=30)
                
                # All tensors are now in 3D format [bs, T, dim]
                clean_repr_norm = pose_batch['motion_repr_clean']
                pipeline_repr_norm = model_output_full

                if self.train_ds.norm_strategy == 'l2':
                    sample_norms = {k[len("extras/sample_norms/"):]: v for k, v in pose_batch.items() if k.startswith('extras/sample_norms')}
                else:
                    sample_norms = None
                
                clean_repr_unnorm = maybe_unnormalize_full_repr(
                    clean_repr_norm, ds, sample_norms=sample_norms
                )
                pipeline_repr_unnorm = maybe_unnormalize_full_repr(
                    pipeline_repr_norm, ds, sample_norms=sample_norms
                )

                output_rot_flag = self.pose_outputs_global_rotations
                clean_repr_dict, joint_pos_clean, _ = decode_smplx_from_full_repr(
                    full_repr=clean_repr_unnorm,
                    repr_list=ds._repr_list,
                    repr_indices=ds._repr_indices,
                    smplx_model=self.smplx_neutral,
                    output_global_rotations=output_rot_flag,
                    cano_jts_pos=ds.cano_jts_pos,
                    return_verts=False,
                )
                pipeline_repr_dict, joint_pos_pipeline, _ = decode_smplx_from_full_repr(
                    full_repr=pipeline_repr_unnorm,
                    repr_list=ds._repr_list,
                    repr_indices=ds._repr_indices,
                    smplx_model=self.smplx_neutral,
                    output_global_rotations=output_rot_flag,
                    cano_jts_pos=ds.cano_jts_pos,
                    return_verts=False,
                )
                   
                VIS_BATCH_DBG = False
                if VIS_BATCH_DBG:
                    from data_sets.motion_representation import SMPLH_KINTREE
                    from utils.smplx_vis_frontend import SmplxVisConfig, visualize_smplx_sequence
                    from utils.repr_adapter import decode_smplx_from_full_repr
                    bs, clip_len, num_joints, _ = joint_pos_clean.shape

                    _, _, verts_pipeline = decode_smplx_from_full_repr(
                        full_repr=pipeline_repr_unnorm,
                        repr_list=ds._repr_list,
                        repr_indices=ds._repr_indices,
                        smplx_model=self.smplx_neutral,
                        output_global_rotations=output_rot_flag,
                        cano_jts_pos=ds.cano_jts_pos,
                        return_verts=True,
                    )
                    cfg = SmplxVisConfig(
                        window_title="Pipeline Eval Debug",
                        num_repeats=10,
                        fixed_cam=False,
                        camera_follow=True,
                        show_mesh=True,
                        show_skeleton=True,
                    )
                    for i in range(bs):
                        visualize_smplx_sequence(
                            joints=joint_pos_pipeline,
                            vertices=verts_pipeline,
                            faces=self.smplx_neutral.faces,
                            parent_indices=SMPLH_KINTREE,
                            batch_idx=i,
                            cfg=cfg,
                        )

                # Convert to numpy for evaluation 
                joint_pos_clean_np = joint_pos_clean.detach().cpu().numpy()     # [bs, clip_len, self.num_joints, 3]
                joint_pos_pipeline_np = joint_pos_pipeline.detach().cpu().numpy() # [bs, clip_len, self.num_joints, 3]
                
                vis_joint_mask = pose_batch.get('cond_joint_mask')
                vis_joint_mask_np = None
                if vis_joint_mask is not None:
                    vis_joint_mask_np = vis_joint_mask[..., 0].detach().cpu().numpy().astype(bool)
                vis_joint_ids = None
                occ_joint_ids = None
                if vis_joint_mask_np is None:
                    vis_joint_ids = self._visible_ids_from_action(scheme_params)
                    if vis_joint_ids is None:
                        vis_joint_ids = list(range(self.num_joints))
                    occ_joint_ids = [i for i in range(self.num_joints) if i not in vis_joint_ids] or None

                # Compute evaluation metrics using MotionEvaluator
                eval_metrics = self._motion_evaluator.compute_comparison_metrics(
                    pred_joints=joint_pos_pipeline_np,
                    gt_joints=joint_pos_clean_np,
                    repr_dict_rec=pipeline_repr_dict,
                    repr_dict_clean=clean_repr_dict,
                    vis_joint_ids=vis_joint_ids,
                    occ_joint_ids=occ_joint_ids,
                    vis_joint_mask=vis_joint_mask_np,
                    occ_joint_mask=(~vis_joint_mask_np) if vis_joint_mask_np is not None else None,
                )
                
                # Accumulate metrics
                for metric_name, metric_value in eval_metrics.items():
                    current_scheme_losses_sum[f"pipeline_{metric_name}"] = current_scheme_losses_sum.get(f"pipeline_{metric_name}", 0.0) + metric_value
                
                # Clean up GPU memory
                del traj_output, model_output_full, clean_repr_unnorm, pipeline_repr_unnorm
                torch.cuda.empty_cache()
            
            if num_batches_processed > 0:
                all_schemes_aggregated_losses[scheme_name] = {k: v / num_batches_processed for k, v in current_scheme_losses_sum.items()}

        final_avg_pipeline_metric = self._log_pipeline_metrics(all_schemes_aggregated_losses, dataset_label)

        # Restore training mode
        self.trajnet.train()
        self.posenet.train()

        return final_avg_pipeline_metric, num_batches_total

    
    def _log_pipeline_metrics(self, all_schemes_aggregated_losses, dataset_label=None):
        """Log and aggregate pipeline evaluation metrics."""
        log_prefix = f"eval/{dataset_label}" if dataset_label else "eval"
        dataset_tag = f"dataset={dataset_label}" if dataset_label else "aggregate"
        # Extract pipeline-specific metrics
        fixed_scheme_mpjpe_metrics = []
        for scheme_name, averaged_losses in all_schemes_aggregated_losses.items():
            if scheme_name.startswith("pipeline_"):
                metric_val = averaged_losses.get('pipeline_mpjpe_global_mm')
                if metric_val is not None:
                    fixed_scheme_mpjpe_metrics.append(metric_val)
        
        # Calculate average MPJPE across fixed schemes
        avg_pipeline_mpjpe_metric = torch.tensor(float('inf'), device=self.device)
        if fixed_scheme_mpjpe_metrics:
            valid_metrics = torch.tensor([m for m in fixed_scheme_mpjpe_metrics]).float()
            if valid_metrics.numel() > 0: 
                avg_pipeline_mpjpe_metric = torch.mean(valid_metrics).to(self.device)

        # DDP gathering for pipeline metrics
        if self.use_ddp:
            gathered_metrics = [torch.zeros_like(avg_pipeline_mpjpe_metric, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(gathered_metrics, avg_pipeline_mpjpe_metric)
            if self.rank == 0: 
                final_avg_pipeline_metric = torch.mean(torch.stack(gathered_metrics))
            else:
                final_avg_pipeline_metric = avg_pipeline_mpjpe_metric
        else:
            final_avg_pipeline_metric = avg_pipeline_mpjpe_metric
        
        # Log pipeline metrics
        if not self.use_ddp or (self.use_ddp and self.rank == 0):
            pipeline_metric_log_str = f"Pipeline Avg Masking MPJPE: {final_avg_pipeline_metric.item():.4f} mm"
            self.writer.add_scalar(f"{log_prefix}/pipeline_avg_masking_mpjpe_mm", final_avg_pipeline_metric.item(), self.step)
            self.logger.info(f"[Step {self.step}] [pipeline_test] [{dataset_tag}] {pipeline_metric_log_str}")
        
        if self.use_ddp: 
            dist.barrier()
        
        # Log all individual pipeline losses
        for scheme_name, averaged_losses in all_schemes_aggregated_losses.items():
            for loss_key, loss_val in averaged_losses.items():
                log_name = f"{log_prefix}/{scheme_name}/{loss_key}"
                _loss_val = torch.tensor(loss_val, device=self.device)
                if self.use_ddp:
                    gathered_losses = [torch.zeros_like(_loss_val, device=self.device) for _ in range(self.world_size)]
                    dist.all_gather(gathered_losses, _loss_val)
                    if self.rank == 0:
                        final_loss = torch.mean(torch.stack(gathered_losses))
                        self.writer.add_scalar(log_name, final_loss.item(), self.step)
                        self.logger.info(f"[Step {self.step}] [pipeline_test] [{dataset_tag}] [{scheme_name}] {loss_key}: {final_loss.item():.6f}")
                else:
                    self.writer.add_scalar(log_name, loss_val, self.step)
                    self.logger.info(f"[Step {self.step}] [pipeline_test] [{dataset_tag}] [{scheme_name}] {loss_key}: {loss_val:.6f}")
        
        if self.use_ddp: 
            dist.barrier()
        
        return final_avg_pipeline_metric

    def _evaluate_pipeline(self, epoch):
        """
        Evaluate the pipeline per test dataset and return an aggregate metric.
        """
        if self.task != 'both':
            if not self.use_ddp or (self.use_ddp and self.rank == 0):
                self.logger.warning("Pipeline evaluation is only available when task == 'both'. Skipping.")
            return

        if not self.test_sets:
            return torch.tensor(float('inf'), device=self.device)

        dataset_results = []
        for test_set in self.test_sets:
            ds = test_set["dataset"]
            test_dl = test_set["dataloader"]
            dataset_label = test_set["name"]
            self._set_eval_dataset_context(ds)
            dataset_metric, num_batches = self._evaluate_pipeline_single_dataset(epoch, test_dl, ds, dataset_label)
            if num_batches > 0:
                dataset_results.append((dataset_metric, num_batches, dataset_label))

        if not dataset_results:
            return torch.tensor(float('inf'), device=self.device)

        total_batches = sum(num_batches for _, num_batches, _ in dataset_results)
        aggregate_metric = dataset_results[0][0]
        if total_batches > 0:
            weighted_sum = sum(metric * num_batches for metric, num_batches, _ in dataset_results)
            aggregate_metric = weighted_sum / total_batches

        if self.has_multiple_test_sets and (not self.use_ddp or (self.use_ddp and self.rank == 0)):
            metric_log_str = f"Pipeline Avg Masking MPJPE: {aggregate_metric.item():.4f} mm"
            self.writer.add_scalar('eval/aggregate/pipeline_avg_masking_mpjpe_mm', aggregate_metric.item(), self.step)
            self.logger.info(f"[Step {self.step}] [pipeline_test] [aggregate] {metric_log_str}")

        return aggregate_metric

    def _evaluate_single_dataset(self, epoch, test_dl, ds, dataset_label, save_best_model=True):
        traj_feat_dim = get_modality_dim(ds.traj_repr_name_list, ds._repr_indices)
        
        # Initialize models based on task
        traj_ema_model = None
        pose_ema_model = None
        ema_model = None
        
        if self.task == 'both':
            traj_source_model = self.trajnet.module if isinstance(self.trajnet, DDP) else self.trajnet
            pose_source_model = self.posenet.module if isinstance(self.posenet, DDP) else self.posenet
            if self.ema:
                traj_ema_model = self.trajnet_ema_model
                pose_ema_model = self.posenet_ema_model
            else:
                traj_ema_model = traj_source_model
                pose_ema_model = pose_source_model
            traj_ema_model.eval()
            pose_ema_model.eval()
        else:
            source_model = self.model.module if isinstance(self.model, DDP) else self.model
            if self.ema:
                ema_model = self.model_ema_model
            else:
                ema_model = source_model
            ema_model.eval()

        if self.masking_batch_policy == "per_sample":
            eval_schemes = {"eval_mixed_per_sample": {}}
        else:
            eval_schemes = {}
            for action in self._build_eval_actions():
                action_name = action.get("name", "mask_action")
                scheme_key = f"eval_{action_name.replace(' ', '_').replace('-', '_')}"
                eval_schemes[scheme_key] = action

        all_schemes_aggregated_losses = {}
        num_batches_total = len(test_dl)
        for scheme_name, scheme_params in eval_schemes.items():
            if not self.use_ddp or (self.use_ddp and self.rank == 0): 
                self.logger.info(f"Evaluating scheme: {scheme_name} (dataset={dataset_label})")
            
            current_scheme_losses_sum = {}
            if self.task == 'both':
                current_scheme_traj_losses_sum = {}
                current_scheme_pose_losses_sum = {}
            
            num_batches_processed = 0
            
            for test_step, test_batch in tqdm(enumerate(test_dl), desc=f"Eval [{dataset_label}]: {scheme_name}", disable=self.disable_tqdm, leave=False):
                num_batches_processed += 1
                bs_eval, clip_len_eval = test_batch['motion_repr_clean'].shape[:2]
                test_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in test_batch.items()}
                
                # Apply masking
                if self.task != 'both':
                    test_batch = process_batch_cond(
                        batch=test_batch,
                        ds=ds,
                        mask_scheme=None,
                        task=self.task,
                        input_noise=self.input_noise,
                        num_joints=self.num_joints)
                    masking_action = self._masking_plan_for_batch(bs_eval, scheme_params, offset=test_step)
                    self._apply_mask_action_plan(test_batch, ds, self.task, masking_action)

                if self.task == 'both':
                    # Evaluate TrajNet
                    traj_test_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in test_batch.items()}
                    traj_test_batch = process_batch_cond(
                        batch=traj_test_batch,
                        ds=ds,
                        mask_scheme=None,
                        task='traj',
                        input_noise=self.input_noise,
                        num_joints=self.num_joints)

                    masking_action = self._masking_plan_for_batch(bs_eval, scheme_params, offset=test_step)

                    # apply masking
                    self._apply_mask_action_plan(traj_test_batch, ds, "traj", masking_action)

                    # Extract trajectory features for shape calculation
                    traj_features = select_modalities(traj_test_batch['motion_repr_clean'], ds.traj_repr_name_list, ds._repr_indices)
                    traj_shape = list(traj_features.shape)
                    
                    traj_eval_losses, _ = self.model_trajnet_eval.eval_losses(
                        model=traj_ema_model, 
                        batch=traj_test_batch, 
                        shape=traj_shape, 
                        smplx_model=self.smplx_neutral, 
                        epoch=epoch, 
                        timestep_respacing=self.timestep_respacing_eval, 
                        compute_eval_metrics=True
                    )
                    
                    # Accumulate TrajNet losses with 'traj/' prefix
                    for loss_key, loss_val in traj_eval_losses.items():
                        prefixed_key = f"traj/{loss_key}"
                        current_scheme_traj_losses_sum[prefixed_key] = current_scheme_traj_losses_sum.get(prefixed_key, 0.0) + loss_val.detach().clone()
                    
                    # Evaluate PoseNet
                    pose_test_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in test_batch.items()}
                    pose_test_batch = process_batch_cond(
                        batch=pose_test_batch,
                        ds=ds,
                        mask_scheme=None,
                        task='pose',
                        input_noise=self.input_noise,
                        num_joints=self.num_joints)

                    self._apply_mask_action_plan(pose_test_batch, ds, "pose", masking_action)

                    # All tensors remain in 3D format [bs, T, dim] - no conversion needed

                    # Determine shape for PoseNet in 3D format [bs, T, pose_feat_dim]
                    pose_predicted = ds.repr_registry.predicted_modalities.get('pose', [])
                    if not pose_predicted:
                        raise ValueError("predicted_modalities['pose'] is empty; cannot infer pose output shape.")
                    pose_feat_dim = get_modality_dim(pose_predicted, ds._repr_indices)
                    pose_shape = [bs_eval, clip_len_eval, pose_feat_dim]
                    
                    pose_eval_losses, _ = self.model_posenet_eval.eval_losses(
                        model=pose_ema_model, 
                        batch=pose_test_batch, 
                        shape=pose_shape, 
                        smplx_model=self.smplx_neutral, 
                        epoch=epoch, 
                        timestep_respacing=self.timestep_respacing_eval, 
                        compute_eval_metrics=True
                    )
                    
                    # Accumulate PoseNet losses with 'pose/' prefix
                    for loss_key, loss_val in pose_eval_losses.items():
                        prefixed_key = f"pose/{loss_key}"
                        current_scheme_pose_losses_sum[prefixed_key] = current_scheme_pose_losses_sum.get(prefixed_key, 0.0) + loss_val.detach().clone()
                    
                elif self.task == 'pose':
                    # All tensors remain in 3D format [bs, T, dim] - no conversion needed

                    # Determine shape for PoseNet in 3D format [bs, T, pose_feat_dim]
                    pose_predicted = ds.repr_registry.predicted_modalities.get('pose', [])
                    if not pose_predicted:
                        raise ValueError("predicted_modalities['pose'] is empty; cannot infer pose output shape.")
                    pose_feat_dim = get_modality_dim(pose_predicted, ds._repr_indices)
                    shape = [bs_eval, clip_len_eval, pose_feat_dim]
                else: # traj
                    # Extract trajectory features for shape calculation
                    traj_features = select_modalities(test_batch['motion_repr_clean'], ds.traj_repr_name_list, ds._repr_indices)
                    shape = list(traj_features.shape)

                if self.task != 'both':
                    eval_losses_cur_batch, _ = self.model_eval.eval_losses(model=ema_model, batch=test_batch, shape=shape, smplx_model=self.smplx_neutral, epoch=epoch, timestep_respacing=self.timestep_respacing_eval, compute_eval_metrics=True)
                    for loss_key, loss_val in eval_losses_cur_batch.items():
                        current_scheme_losses_sum[loss_key] = current_scheme_losses_sum.get(loss_key, 0.0) + loss_val.detach().clone()
            
            # Average losses based on the number of batches processed
            if num_batches_processed > 0:
                if self.task == 'both':
                    all_schemes_aggregated_losses[scheme_name] = {
                        **{k: v / num_batches_processed for k, v in current_scheme_traj_losses_sum.items()},
                        **{k: v / num_batches_processed for k, v in current_scheme_pose_losses_sum.items()}
                    }
                else:
                    all_schemes_aggregated_losses[scheme_name] = {k: v / num_batches_processed for k, v in current_scheme_losses_sum.items()}

        # Extract and process MPJPE metrics
        if self.task == 'both':
            # Track separate metrics for TrajNet and PoseNet
            traj_fixed_scheme_mpjpe_metrics = []
            pose_fixed_scheme_mpjpe_metrics = []
            
            for scheme_name, averaged_losses in all_schemes_aggregated_losses.items():
                if scheme_name.startswith("eval_"):
                    if 'traj/eval_mpjpe_global_mm' in averaged_losses:
                        traj_fixed_scheme_mpjpe_metrics.append(averaged_losses['traj/eval_mpjpe_global_mm'])
                    if 'pose/eval_mpjpe_global_mm' in averaged_losses:
                        pose_fixed_scheme_mpjpe_metrics.append(averaged_losses['pose/eval_mpjpe_global_mm'])
            
            traj_avg_mpjpe_metric = torch.tensor(float('inf'), device=self.device)
            pose_avg_mpjpe_metric = torch.tensor(float('inf'), device=self.device)
            
            if traj_fixed_scheme_mpjpe_metrics:
                valid_metrics = [m for m in traj_fixed_scheme_mpjpe_metrics if torch.is_tensor(m)]
                if valid_metrics: traj_avg_mpjpe_metric = torch.mean(torch.stack(valid_metrics)).to(self.device)
            
            if pose_fixed_scheme_mpjpe_metrics:
                valid_metrics = [m for m in pose_fixed_scheme_mpjpe_metrics if torch.is_tensor(m)]
                if valid_metrics: pose_avg_mpjpe_metric = torch.mean(torch.stack(valid_metrics)).to(self.device)
                
            # DDP gathering for both metrics
            if self.use_ddp:
                traj_gathered = [torch.zeros_like(traj_avg_mpjpe_metric, device=self.device) for _ in range(self.world_size)]
                pose_gathered = [torch.zeros_like(pose_avg_mpjpe_metric, device=self.device) for _ in range(self.world_size)]
                dist.all_gather(traj_gathered, traj_avg_mpjpe_metric)
                dist.all_gather(pose_gathered, pose_avg_mpjpe_metric)
                if self.rank == 0:
                    traj_final_avg_metric = torch.mean(torch.stack(traj_gathered))
                    pose_final_avg_metric = torch.mean(torch.stack(pose_gathered))
                else:
                    traj_final_avg_metric = traj_avg_mpjpe_metric
                    pose_final_avg_metric = pose_avg_mpjpe_metric
            else:
                traj_final_avg_metric = traj_avg_mpjpe_metric
                pose_final_avg_metric = pose_avg_mpjpe_metric

            combined_metric = (traj_final_avg_metric + pose_final_avg_metric) / 2.0
            
            # Log metrics and determine best model
            if not self.use_ddp or (self.use_ddp and self.rank == 0):
                self.writer.add_scalar(f"eval/{dataset_label}/traj_avg_masking_mpjpe_mm", traj_final_avg_metric.item(), self.step)
                self.writer.add_scalar(f"eval/{dataset_label}/pose_avg_masking_mpjpe_mm", pose_final_avg_metric.item(), self.step)
                self.logger.info(f"[Step {self.step}] [test] [dataset={dataset_label}] Traj Avg Masking MPJPE: {traj_final_avg_metric.item():.4f} mm")
                self.logger.info(f"[Step {self.step}] [test] [dataset={dataset_label}] Pose Avg Masking MPJPE: {pose_final_avg_metric.item():.4f} mm")
                
                # Use combined metric for best model determination
                self.writer.add_scalar(f"eval/{dataset_label}/combined_avg_masking_mpjpe_mm", combined_metric.item(), self.step)
                
                current_stage = self.get_current_stage(self.step)
                if (not self.has_multiple_test_sets) and combined_metric.item() < self.best_loss_per_stage[current_stage]:
                    self.best_loss_per_stage[current_stage] = combined_metric.item()
                    if save_best_model:
                        self.logger.info(f"[*] New best model for STAGE {current_stage} found with Combined MPJPE {combined_metric.item():.4f} mm. Saving checkpoint.")
                        self.save(is_best_for_stage=True, stage=current_stage)
        else:
            fixed_scheme_mpjpe_metrics = []
            for scheme_name, averaged_losses in all_schemes_aggregated_losses.items():
                if scheme_name.startswith("eval_"):
                    if 'eval_mpjpe_global_mm' in averaged_losses:
                        fixed_scheme_mpjpe_metrics.append(averaged_losses['eval_mpjpe_global_mm'])
                    elif not self.use_ddp or (self.use_ddp and self.rank == 0):
                        self.logger.warning(f"Metric 'eval_mpjpe_global_mm' not found for scheme '{scheme_name}' (dataset={dataset_label}).")
            
            avg_mpjpe_metric = torch.tensor(float('inf'), device=self.device)
            if fixed_scheme_mpjpe_metrics:
                valid_metrics = [m for m in fixed_scheme_mpjpe_metrics if torch.is_tensor(m)]
                if valid_metrics: avg_mpjpe_metric = torch.mean(torch.stack(valid_metrics)).to(self.device)
            
            # DDP gathering and best model check
            if self.use_ddp:
                gathered_metrics = [torch.zeros_like(avg_mpjpe_metric, device=self.device) for _ in range(self.world_size)]
                dist.all_gather(gathered_metrics, avg_mpjpe_metric)
                if self.rank == 0: 
                    final_avg_metric = torch.mean(torch.stack(gathered_metrics))
                else:
                    final_avg_metric = avg_mpjpe_metric
            else:
                final_avg_metric = avg_mpjpe_metric
            
            if not self.use_ddp or (self.use_ddp and self.rank == 0):
                metric_log_str = f"Avg Masking MPJPE: {final_avg_metric.item():.4f} mm"
                self.writer.add_scalar(f"eval/{dataset_label}/avg_masking_mpjpe_mm", final_avg_metric.item(), self.step)
                self.logger.info(f"[Step {self.step}] [test] [dataset={dataset_label}] {metric_log_str}")
                current_stage = self.get_current_stage(self.step)
                if (not self.has_multiple_test_sets) and final_avg_metric.item() < self.best_loss_per_stage[current_stage]:
                    self.best_loss_per_stage[current_stage] = final_avg_metric.item()
                    if save_best_model:
                        self.logger.info(f"[*] New best model for STAGE {current_stage} found with MPJPE {final_avg_metric.item():.4f} mm. Saving checkpoint.")
                        self.save(is_best_for_stage=True, stage=current_stage)

        if self.use_ddp: dist.barrier()
        
        # Log all individual losses
        for scheme_name, averaged_losses in all_schemes_aggregated_losses.items():
            for loss_key, loss_val in averaged_losses.items():
                log_name = f"eval/{dataset_label}/{scheme_name}/{loss_key}"
                # _loss_val = loss_val.clone().detach()
                _loss_val = torch.tensor(loss_val, device=self.device)
                if self.use_ddp:
                    gathered_losses = [torch.zeros_like(_loss_val) for _ in range(self.world_size)]
                    dist.all_gather(gathered_losses, _loss_val)
                    if self.rank == 0: 
                        final_loss = torch.mean(torch.stack(gathered_losses))
                        self.writer.add_scalar(log_name, final_loss.item(), self.step)
                        self.logger.info(f"[Step {self.step}] [test] [dataset={dataset_label}] [{scheme_name}] {loss_key}: {final_loss.item():.6f}")
                else: 
                    self.writer.add_scalar(log_name, loss_val.item(), self.step)
                    self.logger.info(f"[Step {self.step}] [test] [dataset={dataset_label}] [{scheme_name}] {loss_key}: {loss_val.item():.6f}")
        
        if self.use_ddp: dist.barrier()
        
        # Restore training mode
        if self.task == 'both':
            self.trajnet.train()
            self.posenet.train()
        else:
            self.model.train()

        if self.task == 'both':
            return {
                "num_batches": num_batches_total,
                "traj_metric": traj_final_avg_metric,
                "pose_metric": pose_final_avg_metric,
                "combined_metric": combined_metric,
            }

        return {
            "num_batches": num_batches_total,
            "metric": final_avg_metric,
        }

    def evaluate(self, epoch, save_best_model=True):
        if not self.test_sets:
            return

        if not self.has_multiple_test_sets:
            test_set = self.test_sets[0]
            ds = test_set["dataset"]
            test_dl = test_set["dataloader"]
            dataset_label = test_set["name"]
            self._set_eval_dataset_context(ds)
            self._evaluate_single_dataset(epoch, test_dl, ds, dataset_label, save_best_model=save_best_model)
            return

        dataset_metrics = []
        for test_set in self.test_sets:
            ds = test_set["dataset"]
            test_dl = test_set["dataloader"]
            dataset_label = test_set["name"]
            self._set_eval_dataset_context(ds)
            metrics = self._evaluate_single_dataset(epoch, test_dl, ds, dataset_label, save_best_model=save_best_model)
            if metrics and metrics["num_batches"] > 0:
                dataset_metrics.append(metrics)

        if not dataset_metrics:
            return

        total_batches = sum(m["num_batches"] for m in dataset_metrics)
        if total_batches <= 0:
            return

        if self.task == 'both':
            traj_weighted = torch.zeros_like(dataset_metrics[0]["traj_metric"])
            pose_weighted = torch.zeros_like(dataset_metrics[0]["pose_metric"])
            for metrics in dataset_metrics:
                traj_weighted += metrics["traj_metric"] * metrics["num_batches"]
                pose_weighted += metrics["pose_metric"] * metrics["num_batches"]
            traj_avg_metric = traj_weighted / total_batches
            pose_avg_metric = pose_weighted / total_batches
            combined_metric = (traj_avg_metric + pose_avg_metric) / 2.0

            if not self.use_ddp or (self.use_ddp and self.rank == 0):
                self.writer.add_scalar('eval/aggregate/traj_avg_masking_mpjpe_mm', traj_avg_metric.item(), self.step)
                self.writer.add_scalar('eval/aggregate/pose_avg_masking_mpjpe_mm', pose_avg_metric.item(), self.step)
                self.writer.add_scalar('eval/aggregate/combined_avg_masking_mpjpe_mm', combined_metric.item(), self.step)
                self.logger.info(f"[Step {self.step}] [test] [aggregate] Traj Avg Masking MPJPE: {traj_avg_metric.item():.4f} mm")
                self.logger.info(f"[Step {self.step}] [test] [aggregate] Pose Avg Masking MPJPE: {pose_avg_metric.item():.4f} mm")
                if save_best_model:
                    current_stage = self.get_current_stage(self.step)
                    if combined_metric.item() < self.best_loss_per_stage[current_stage]:
                        self.best_loss_per_stage[current_stage] = combined_metric.item()
                        self.logger.info(f"[*] New best model for STAGE {current_stage} found with Combined MPJPE {combined_metric.item():.4f} mm. Saving checkpoint.")
                        self.save(is_best_for_stage=True, stage=current_stage)
        else:
            metric_weighted = torch.zeros_like(dataset_metrics[0]["metric"])
            for metrics in dataset_metrics:
                metric_weighted += metrics["metric"] * metrics["num_batches"]
            avg_metric = metric_weighted / total_batches

            if not self.use_ddp or (self.use_ddp and self.rank == 0):
                self.writer.add_scalar('eval/aggregate/avg_masking_mpjpe_mm', avg_metric.item(), self.step)
                self.logger.info(f"[Step {self.step}] [test] [aggregate] Avg Masking MPJPE: {avg_metric.item():.4f} mm")
                if save_best_model:
                    current_stage = self.get_current_stage(self.step)
                    if avg_metric.item() < self.best_loss_per_stage[current_stage]:
                        self.best_loss_per_stage[current_stage] = avg_metric.item()
                        self.logger.info(f"[*] New best model for STAGE {current_stage} found with MPJPE {avg_metric.item():.4f} mm. Saving checkpoint.")
                        self.save(is_best_for_stage=True, stage=current_stage)

    @torch.no_grad()
    def _update_target_ema(self):
        raise DeprecationWarning("Consistency training is deprecated. Use 'shortcut' instead.")
        if self.target_model is None or self.ema_scale_fn is None: return
        target_ema, _ = self.ema_scale_fn(self.step)
        source_model = self.model.module if isinstance(self.model, DDP) else self.model
        for target_p, source_p in zip(self.target_model.parameters(), source_model.parameters()): target_p.data.lerp_(source_p.data, 1.0 - target_ema)
        for target_b, source_b in zip(self.target_model.buffers(), source_model.buffers()): target_b.data.copy_(source_b.data)

    @torch.no_grad()
    def _update_ema(self):
        if not self.ema:
            return
        if self.task == 'both':
            traj_source_model = self.trajnet.module if isinstance(self.trajnet, DDP) else self.trajnet
            pose_source_model = self.posenet.module if isinstance(self.posenet, DDP) else self.posenet
            
            _sync_materialized_lazy_state(self.trajnet_ema_model, traj_source_model)
            _sync_materialized_lazy_state(self.posenet_ema_model, pose_source_model)

            # Update trajnet EMA
            for ema_p, source_p in zip(self.trajnet_ema_model.parameters(), traj_source_model.parameters()):
                ema_p.data.lerp_(source_p.data, 1.0 - self.ema_decay)
            for ema_b, source_b in zip(self.trajnet_ema_model.buffers(), traj_source_model.buffers()):
                ema_b.data.copy_(source_b.data)

            # Update posenet EMA
            for ema_p, source_p in zip(self.posenet_ema_model.parameters(), pose_source_model.parameters()):
                ema_p.data.lerp_(source_p.data, 1.0 - self.ema_decay)
            for ema_b, source_b in zip(self.posenet_ema_model.buffers(), pose_source_model.buffers()): 
                ema_b.data.copy_(source_b.data)
        else:
            source_model = self.model.module if isinstance(self.model, DDP) else self.model
            _sync_materialized_lazy_state(self.model_ema_model, source_model)
            for ema_p, source_p in zip(self.model_ema_model.parameters(), source_model.parameters()):
                ema_p.data.lerp_(source_p.data, 1.0 - self.ema_decay)
            for ema_b, source_b in zip(self.model_ema_model.buffers(), source_model.buffers()):
                ema_b.data.copy_(source_b.data)

    def save(self, is_best_for_stage=False, stage=None, is_latest=False):
        
        if self.use_ddp and self.rank != 0: return

        if self.task == 'both':
            traj_model_to_save = self.trajnet.module if isinstance(self.trajnet, DDP) else self.trajnet
            pose_model_to_save = self.posenet.module if isinstance(self.posenet, DDP) else self.posenet

            traj_ema_model_to_save = self.trajnet_ema_model if self.ema else None
            pose_ema_model_to_save = self.posenet_ema_model if self.ema else None

            # Save TrajNet separately
            traj_safetensors_path, traj_optimizer_path = save_checkpoint(
                model=traj_model_to_save,
                step=self.step,
                opt=self.traj_opt,
                scheduler=self.traj_scheduler,
                task='traj',
                save_dir=self.save_dir,
                is_best_for_stage=is_best_for_stage,
                stage=stage,
                is_latest=is_latest,
                filename_prefix="trajnet_",
                ema_model=traj_ema_model_to_save,
            )

            # Save PoseNet separately
            pose_safetensors_path, pose_optimizer_path = save_checkpoint(
                model=pose_model_to_save,
                step=self.step,
                opt=self.pose_opt,
                scheduler=self.pose_scheduler,
                task='pose',
                save_dir=self.save_dir,
                is_best_for_stage=is_best_for_stage,
                stage=stage,
                is_latest=is_latest,
                filename_prefix="posenet_",
                ema_model=pose_ema_model_to_save,
            )
            
            if self.logger:
                self.logger.info(f'[*] TrajNet weights saved to {traj_safetensors_path}')
                self.logger.info(f'[*] TrajNet optimizer states saved to {traj_optimizer_path}')
                self.logger.info(f'[*] PoseNet weights saved to {pose_safetensors_path}')
                self.logger.info(f'[*] PoseNet optimizer states saved to {pose_optimizer_path}\n')
        else:
            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            
            # Get target model for consistency training if applicable
            target_model = self.target_model if self.model_type == 'consistency' else None
            ema_model_to_save = self.model_ema_model if self.ema else None

            safetensors_path, optimizer_path = save_checkpoint(
                model=model_to_save,
                step=self.step,
                opt=self.opt,
                scheduler=self.scheduler,
                task=self.task,
                save_dir=self.save_dir,
                is_best_for_stage=is_best_for_stage,
                stage=stage,
                is_latest=is_latest,
                target_model=target_model,
                ema_model=ema_model_to_save,
            )

            if self.logger: 
                self.logger.info(f'[*] Model weights saved to {safetensors_path}')
                self.logger.info(f'[*] Optimizer states saved to {optimizer_path}\n')

    def load_checkpoint_instance(self, checkpoint_path):
        """
        Instance method wrapper that calls the global load_checkpoint function.
        Handles task == 'both' by loading TrajNet and PoseNet from separate checkpoint files.
        
        For task == 'both', the checkpoint_path should be the base path without prefix.
        The method will derive trajnet and posenet paths by prepending prefixes to the filename.
        """
        if not self.use_ddp or (self.use_ddp and self.rank == 0): 
            self.logger.info(f"[*] Attempting to load checkpoint from {checkpoint_path}")

        if self.task == 'both':
            # Derive separate checkpoint paths for trajnet and posenet
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkpoint_filename = os.path.basename(checkpoint_path)
            
            trajnet_checkpoint_path = os.path.join(checkpoint_dir, f"trajnet_{checkpoint_filename}")
            posenet_checkpoint_path = os.path.join(checkpoint_dir, f"posenet_{checkpoint_filename}")
            
            trajnet_model_to_load = self.trajnet.module if isinstance(self.trajnet, DDP) else self.trajnet
            posenet_model_to_load = self.posenet.module if isinstance(self.posenet, DDP) else self.posenet
            
            # Load TrajNet checkpoint
            traj_step = load_checkpoint(
                checkpoint_path=trajnet_checkpoint_path,
                model=trajnet_model_to_load,
                opt=self.traj_opt,
                scheduler=self.traj_scheduler,
                device=self.device,
            )
            
            # Load PoseNet checkpoint
            pose_step = load_checkpoint(
                checkpoint_path=posenet_checkpoint_path,
                model=posenet_model_to_load,
                opt=self.pose_opt,
                scheduler=self.pose_scheduler,
                device=self.device,
            )
            
            # Use the step from trajnet (both should be the same)
            self.step = traj_step
            
            if not self.use_ddp or (self.use_ddp and self.rank == 0):
                self.logger.info(f"[*] Successfully loaded TrajNet from {trajnet_checkpoint_path}")
                self.logger.info(f"[*] Successfully loaded PoseNet from {posenet_checkpoint_path}")
                self.logger.info(f"[*] Resumed from step {self.step}.")
        else:
            model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
            
            # Get target model for consistency training if applicable
            target_model = self.target_model if self.model_type == 'consistency' else None
            
            self.step = load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model_to_load,
                opt=self.opt,
                scheduler=self.scheduler,
                target_model=target_model,
                device=self.device,
            )
            
            if not self.use_ddp or (self.use_ddp and self.rank == 0):
                if self.model_type == 'consistency' and target_model is not None:
                    self.logger.info("[*] Loaded target model from checkpoint.")
                self.logger.info(f"[*] Successfully loaded checkpoint. Resumed from step {self.step}.")
