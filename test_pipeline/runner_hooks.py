"""
Main test runner that orchestrates the evaluation pipeline.

This module provides TestRunner which coordinates:
- Dataset and model loading
- Batch processing via baseline hooks
- Visualization
- Metrics computation and saving

The runner supports the baseline registry system, allowing different
methods to be evaluated using their specific hooks.
"""
import os
import json
import hashlib
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import inspect

from utils import dist_util
from utils.fixseed import fixseed
from utils.construct import construct_class_by_name, get_obj_by_name
from utils.registry_builders import build_mask_scheme_registry, build_representation_registry
from utils.masking_actions import build_mask_action_pool, materialize_mask_action
from data_sets.motion_representation import recover_from_repr_smpl
from train.training_loop import load_checkpoint

from .config import validate_test_config, save_test_config
from .inference import BatchProcessorWithHooks, ParallelExecutor, get_seq_name_from_batch
from .inference.normalization import create_normalization_stats
from .visualization import get_visualizer, VisualizationData
from .baselines import BaselineRegistry, get_baseline, InferenceContext, create_inference_context


class TestRunner:
    """
    Main orchestrator for the test pipeline.
    
    Responsibilities:
    - Load dataset and models
    - Coordinate batch processing (single/multi-process)
    - Handle visualization
    - Compute and save metrics
    
    The runner uses the baseline registry system to delegate preprocessing,
    inference, postprocessing, and metrics computation to baseline-specific hooks.
    """
    
    def __init__(self, train_config: DictConfig, test_config: DictConfig):
        self.train_config = train_config
        self.test_config = test_config
        self.device = dist_util.dev()
        
        # Build registries from train_config
        self.repr_registry = build_representation_registry(
            OmegaConf.to_container(self.train_config.representation_registry, resolve=True)
        )
        self.mask_scheme_registry = build_mask_scheme_registry(
            OmegaConf.to_container(self.train_config.mask_registry, resolve=True)
        )
        
        # Get baseline spec (defaults to 'ours')
        baseline_name = getattr(test_config, 'baseline', 'ours')
        self._baseline_spec = get_baseline(baseline_name)
        self._baseline_name = baseline_name
        
        # Validate baseline compatibility
        self._validate_baseline_compatibility()
        
        # Components (initialized lazily)
        self._dataset = None
        self._dataloader = None
        self._models = None
        self._diffusions = None
        self._smplx_model = None
        self._fk_model = None
        self._fk_model_type = None
        self._inference_context = None
        self._batch_processor = None
        self._visualizer = None
        self._metrics_hook = None
    
    def _validate_baseline_compatibility(self):
        """Validate baseline is compatible with test configuration."""
        spec = self._baseline_spec
        action_names = []
        if getattr(self.test_config, "test_mask_action", None):
            action_names.append(self.test_config.test_mask_action)
        if getattr(self.test_config, "test_mask_actions", None):
            action_names.extend(list(self.test_config.test_mask_actions))
        action_names = list(dict.fromkeys(action_names))
        if not action_names:
            raise ValueError("test_mask_action or test_mask_actions must be specified.")

        for test_scheme in action_names:
            assert test_scheme in spec.compatible_mask_schemes, \
                f"Baseline '{spec.name}' is not compatible with masking action '{test_scheme}'. " \
                f"Compatible schemes: {spec.compatible_mask_schemes}"

        action_pool = build_mask_action_pool(
            OmegaConf.to_container(self.train_config.masking, resolve=True)
        )
        action = materialize_mask_action(
            action_names[0],
            action_pool,
            self.repr_registry.joint_count_for_topology(
                next(iter(self.repr_registry.topologies.keys()))
            ),
        )
        if action.get("type") == "registry":
            scheme = action.get("scheme")
            if scheme and self.mask_scheme_registry.has(scheme):
                scheme_spec = self.mask_scheme_registry.get(scheme)
                if scheme_spec.representation_config == "xyz_rot":
                    has_rot = any(
                        mod.name in {"R_world_joints", "R_root_joints"}
                        for mod in self.repr_registry.modalities
                    )
                    if not has_rot:
                        raise ValueError(
                            "Mask scheme requires xyz_rot representation, but "
                            "train_config.representation_registry is xyz_only."
                        )
        
        if self.test_config.dist_rank == 0:
            print(f"[TestRunner] Using baseline: {spec.name}")
            print(f"[TestRunner] Mask actions: {action_names}")
    
    def _get_clip_len(self) -> int:
        """Extract clip_len from train_config."""
        if 'train_datasets' in self.train_config and self.train_config.train_datasets:
            train_ds_list = self.train_config.train_datasets
            if isinstance(train_ds_list, list) and len(train_ds_list) > 0:
                first_ds = train_ds_list[0]
                if 'dataset' in first_ds:
                    return first_ds['dataset'].get('clip_len', 33)
        return 33
    
    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset, self._dataloader = self._load_dataset()
        return self._dataset
    
    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataset, self._dataloader = self._load_dataset()
        return self._dataloader
    
    @property
    def models(self):
        if self._models is None:
            self._models = self._create_models()
        return self._models
    
    @property
    def diffusions(self):
        if self._diffusions is None:
            self._diffusions = self._create_diffusions()
        return self._diffusions
    
    @property
    def smplx_model(self):
        if self._smplx_model is None:
            self._smplx_model = self.models[0].smplx_model
        return self._smplx_model
    
    @property
    def fk_model(self):
        if self._fk_model is None:
            from .fk_model_factory import build_fk_model, resolve_fk_model_type
            self._fk_model, self._fk_model_type = build_fk_model(
                dataset_name=self.test_config.dataset_name,
                test_config=self.test_config,
                base_smplx_model=self.smplx_model,
                device=self.device,
            )
            if self.test_config.dist_rank == 0:
                resolved_type = self._fk_model_type or resolve_fk_model_type(
                    self.test_config.dataset_name, getattr(self.test_config, 'fk_model_type', 'auto')
                )
                print(f"[TestRunner] Using FK model: {resolved_type}")
        return self._fk_model
    
    @property
    def inference_context(self) -> InferenceContext:
        """Create InferenceContext for hook-based processing."""
        if self._inference_context is None:
            clip_len = self._get_clip_len()
            norm_stats = create_normalization_stats(self.dataset, self.device)
            
            self._inference_context = create_inference_context(
                train_config=self.train_config,
                test_config=self.test_config,
                dataset=self.dataset,
                repr_registry=self.repr_registry,
                mask_scheme_registry=self.mask_scheme_registry,
                models=self.models,
                diffusions=self.diffusions,
                smplx_model=self.smplx_model,
                fk_model=self.fk_model,
                device=self.device,
                clip_len=clip_len,
                norm_stats=norm_stats,
            )
        return self._inference_context
    
    @property
    def batch_processor(self):
        """Create batch processor using baseline hooks."""
        if self._batch_processor is None:
            self._batch_processor = BatchProcessorWithHooks(
                ctx=self.inference_context,
                baseline_spec=self._baseline_spec,
            )
        return self._batch_processor
    
    @property
    def visualizer(self):
        if self._visualizer is None and self.test_config.visualize:
            self._visualizer = get_visualizer(
                self.test_config.dataset_name,
                self.train_config,
                self.test_config,
                self.fk_model,
            )
            if self._visualizer:
                self._visualizer.setup()
        return self._visualizer
    
    @property
    def metrics_hook(self):
        """Get metrics hook from baseline spec."""
        if self._metrics_hook is None:
            self._metrics_hook = self._baseline_spec.metrics_hook_cls()
        return self._metrics_hook
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the full test pipeline.

        Returns:
            Dictionary of evaluation metrics
        """
        fixseed(self.test_config.seed)

        if self.test_config.dist_rank == 0:
            print(f"[TestRunner] Starting evaluation on {self.test_config.dataset_name}")

        if getattr(self.test_config, "test_batch_policy", "whole_batch") == "sweep":
            metrics = self._compute_sweep_metrics()
        else:
            # Process all batches
            results = self._process_all_batches()

            # Compute metrics using baseline's metrics hook
            metrics = self._compute_metrics(results)

        # Save results
        self._save_results(metrics)

        # Cleanup
        if self._visualizer:
            self._visualizer.cleanup()

        return metrics

    def _compute_sweep_metrics(self) -> Dict[str, Any]:
        action_names = list(getattr(self.test_config, "test_mask_actions", None) or [])
        if not action_names:
            raise ValueError("test_batch_policy='sweep' requires test_mask_actions.")

        original_policy = self.test_config.test_batch_policy
        original_action = getattr(self.test_config, "test_mask_action", None)
        per_pattern = {}
        pattern_counts = {}
        weighted_sums = {}
        total_weight = 0
        try:
            self.test_config.test_batch_policy = "whole_batch"
            for action_name in action_names:
                self.test_config.test_mask_action = action_name
                results = self._process_all_batches()
                metrics = self._compute_metrics(results)
                per_pattern[action_name] = metrics
                count = self._count_processed_patterns(results).get(action_name, len(results['repr_dict_rec']))
                pattern_counts[action_name] = int(count)
                total_weight += int(count)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        weighted_sums[key] = weighted_sums.get(key, 0.0) + float(value) * int(count)
        finally:
            self.test_config.test_batch_policy = original_policy
            self.test_config.test_mask_action = original_action

        aggregate = {key: value / max(total_weight, 1) for key, value in weighted_sums.items()}
        return {
            "aggregate": aggregate,
            "per_pattern": per_pattern,
            "metadata": self._metrics_metadata(pattern_counts=pattern_counts),
        }
    
    def _load_dataset(self) -> Tuple[Any, DataLoader]:
        """Load test dataset and create dataloader."""
        if self.test_config.dist_rank == 0:
            print("Creating test dataset...")
        
        test_datasets_list = OmegaConf.to_container(self.test_config.test_datasets, resolve=True)
        test_dataset_config = test_datasets_list[0]['dataset']
        ds_kwargs = dict(test_dataset_config)
        ds_kwargs.update({
            "logdir": self.test_config.logdir,
            "dtype": torch.float32,
            "device": self.device,
            "dist_rank": self.test_config.dist_rank,
        })
        cls = get_obj_by_name(ds_kwargs["class_name"])
        sig = inspect.signature(cls.__init__)
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if "repr_registry" in sig.parameters or has_var_kwargs:
            ds_kwargs["repr_registry"] = self.repr_registry
        if "mask_scheme_registry" in sig.parameters or has_var_kwargs:
            ds_kwargs["mask_scheme_registry"] = self.mask_scheme_registry
        self._apply_eval_norm_params_policy(ds_kwargs, cls)

        dataset = construct_class_by_name(**ds_kwargs)
        
        loader_config = OmegaConf.to_container(self.test_config.test_dataloader, resolve=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=loader_config['batch_size'],
            shuffle=loader_config['shuffle'],
            num_workers=loader_config['num_workers'],
            drop_last=loader_config['drop_last'],
            pin_memory=loader_config['pin_memory'],
        )
        
        return dataset, dataloader

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _is_subpath(path: str, root: str) -> bool:
        try:
            common = os.path.commonpath([os.path.abspath(path), os.path.abspath(root)])
        except ValueError:
            return False
        return common == os.path.abspath(root)

    def _apply_eval_norm_params_policy(self, ds_kwargs: Dict[str, Any], cls: type) -> None:
        policy = str(getattr(self.test_config, "eval_norm_params_policy", "prefer_global"))
        if policy == "provided":
            return

        logdir = getattr(self.test_config, "logdir", "")
        if not logdir:
            return
        global_norm_path = os.path.join(logdir, "global_norm_params.pkl")
        if not os.path.exists(global_norm_path):
            if policy == "strict_global":
                raise FileNotFoundError(
                    "eval_norm_params_policy=strict_global requires "
                    f"{global_norm_path}, but it does not exist."
                )
            return

        provided_path = ds_kwargs.get("norm_params_path")
        if provided_path:
            requested_path = os.path.abspath(os.path.expanduser(str(provided_path)))
        else:
            requested_path = os.path.abspath(
                os.path.join(logdir, f"{cls.__name__}_train_norm_params.pkl")
            )

        global_abs = os.path.abspath(global_norm_path)
        requested_in_logdir = self._is_subpath(requested_path, logdir)
        should_use_global = (not provided_path) or requested_in_logdir
        if not should_use_global:
            return

        if requested_path != global_abs and os.path.exists(requested_path):
            requested_sha = self._file_sha256(requested_path)
            global_sha = self._file_sha256(global_abs)
            if requested_sha != global_sha:
                message = (
                    "[WARN] Evaluation norm params differ from logdir/global_norm_params.pkl. "
                    f"requested={requested_path} sha256={requested_sha[:12]}..., "
                    f"global={global_abs} sha256={global_sha[:12]}..."
                )
                if policy == "strict_global":
                    raise ValueError(
                        message + " Set eval_norm_params_policy=prefer_global to use the "
                        "training global norm params automatically, or provided to reproduce "
                        "the requested file exactly."
                    )
                if self.test_config.dist_rank == 0:
                    print(message + " Using global norm params for this evaluation.")
        elif requested_path != global_abs and self.test_config.dist_rank == 0:
            print(
                "[INFO] Using logdir/global_norm_params.pkl for evaluation norm params: "
                f"{global_abs}"
            )

        ds_kwargs["norm_params_path"] = global_abs
    
    def _create_models(self) -> Tuple[Any, Any, Any]:
        """Create and load model checkpoints."""
        if self.test_config.dist_rank == 0:
            print("Creating models...")
        
        # TrajNet
        model_trajnet = construct_class_by_name(
            **self.train_config.trajnet,
            dataset=self.dataset,
            device=self.device,
        ).to(self.device)
        
        # PoseNet
        model_posenet = construct_class_by_name(
            **self.train_config.posenet,
            dataset=self.dataset,
            device=self.device,
        ).to(self.device)

        use_ema_checkpoint = self.test_config.use_ema_checkpoint
        if use_ema_checkpoint is None:
            use_ema_checkpoint = bool(self.train_config.get('training', {}).get('ema', False))
        strict_ema_checkpoint = bool(self.test_config.strict_ema_checkpoint)

        # Load checkpoints
        if self.test_config.dist_rank == 0:
            print(f"Loading TrajNet from: {self.test_config.model_path_trajnet}")
            print(
                "Checkpoint weights: "
                f"{'EMA' if use_ema_checkpoint else 'raw'}"
                f"{' (strict)' if strict_ema_checkpoint else ''}"
            )
        load_checkpoint(
            self.test_config.model_path_trajnet,
            model_trajnet,
            device=self.device,
            use_ema=use_ema_checkpoint,
            strict_ema=strict_ema_checkpoint,
        )
        
        if self.test_config.dist_rank == 0:
            print(f"Loading PoseNet from: {self.test_config.model_path_posenet}")
        load_checkpoint(
            self.test_config.model_path_posenet,
            model_posenet,
            device=self.device,
            use_ema=use_ema_checkpoint,
            strict_ema=strict_ema_checkpoint,
        )
        
        model_trajnet.eval()
        model_posenet.eval()
        
        return (model_posenet, model_trajnet, None)
    
    def _create_diffusions(self) -> Tuple[Any, Any, Any]:
        """Create diffusion/training model wrappers."""
        timestep_respacing = self.train_config.training.get('timestep_respacing_eval', '')
        
        trajnet_eval = construct_class_by_name(
            **self.train_config.traj_training_model,
            dataloader=self.dataloader,
            device=self.device,
            timestep_respacing=timestep_respacing,
        )
        
        posenet_eval = construct_class_by_name(
            **self.train_config.pose_training_model,
            dataloader=self.dataloader,
            device=self.device,
            timestep_respacing='',
        )
        
        return (posenet_eval, trajnet_eval, None)
    
    def _process_all_batches(self) -> Dict[str, Any]:
        """Process all batches and collect results."""
        executor = ParallelExecutor(
            num_workers=self.test_config.num_parallel_inference_workers,
            device=self.device,
        )
        
        def process_fn(batch):
            result = self.batch_processor.process_batch(batch)
            
            # Handle visualization if enabled
            if self.visualizer and result is not None:
                self._visualize_result(result, batch)
            
            return result
        
        def get_seq_name(batch, batch_idx):
            return get_seq_name_from_batch(
                batch, batch_idx,
                self.test_config.slice_strategy,
                self.test_config.dataset_name,
            )
        
        raw_results = executor.execute(
            self.dataloader,
            process_fn=process_fn,
            process_fn_kwargs={},
            get_seq_name_fn=get_seq_name,
        )
        
        # Organize results
        repr_dict_rec = {}
        repr_dict_clean = {}
        mask_joint_ids = {}
        visible_joint_masks = {}
        pattern_names = {}

        for seq_name, result in raw_results.items():
            if result is not None:
                repr_dict_rec[seq_name] = result.repr_dict_rec
                mask_joint_ids[seq_name] = result.mask_joint_ids
                if result.visible_joint_mask is not None:
                    visible_joint_masks[seq_name] = result.visible_joint_mask
                if result.cond_pattern_names is not None:
                    pattern_names[seq_name] = result.cond_pattern_names
                if result.repr_dict_clean is not None:
                    repr_dict_clean[seq_name] = result.repr_dict_clean
        
        return {
            'repr_dict_rec': repr_dict_rec,
            'repr_dict_clean': repr_dict_clean if repr_dict_clean else None,
            'mask_joint_ids': mask_joint_ids,
            'visible_joint_masks': visible_joint_masks,
            'pattern_names': pattern_names,
        }
    
    def _visualize_result(self, result, batch: Dict[str, Any]):
        """Visualize a single result."""
        if not self.visualizer:
            return
        
        # Recover joints and vertices using postprocessing hook
        jts_rec, verts_rec = self.batch_processor.postprocessing.recover_world_frame(
            result.repr_dict_rec, batch, self.inference_context
        )
        
        jts_clean, verts_clean = None, None
        if result.repr_dict_clean is not None:
            jts_clean, verts_clean = self.batch_processor.postprocessing.recover_world_frame(
                result.repr_dict_clean, batch, self.inference_context
            )

        vis_data = self._build_visualization_data(
            result=result,
            batch=batch,
            jts_rec=jts_rec,
            verts_rec=verts_rec,
            jts_clean=jts_clean,
            verts_clean=verts_clean,
        )
        self.visualizer.visualize(vis_data)

    def _build_visualization_data(
        self,
        result: Any,
        batch: Dict[str, Any],
        jts_rec: torch.Tensor,
        verts_rec: torch.Tensor,
        jts_clean: Optional[torch.Tensor],
        verts_clean: Optional[torch.Tensor],
    ) -> VisualizationData:
        """Build visualization payload and enforce occlusion shape contract."""
        occluded_joint_ids = np.asarray(result.occluded_joint_ids, dtype=np.int64)
        if occluded_joint_ids.ndim != 1:
            raise ValueError(
                "result.occluded_joint_ids must be a 1-D index array, "
                f"got shape {occluded_joint_ids.shape}."
            )

        return VisualizationData(
            repr_dict_rec_tries=[result.repr_dict_rec],
            jts_rec_tries=[jts_rec],
            verts_rec_tries=[verts_rec],
            occluded_joint_ids_tries=[occluded_joint_ids],
            color_vis_tries=[(0.35, 0.53, 0.97)],
            color_occ_tries=[(0.92, 0.73, 0.23)],
            repr_dict_clean=result.repr_dict_clean,
            jts_clean=jts_clean,
            verts_clean=verts_clean,
            batch=batch,
        )
    
    def _count_processed_patterns(self, results: Dict[str, Any]) -> Dict[str, int]:
        counts = {}
        pattern_names = results.get('pattern_names') or {}
        if not pattern_names:
            action = getattr(self.test_config, "test_mask_action", None)
            if action is not None:
                counts[action] = len(results.get('repr_dict_rec') or {})
            return counts
        for names in pattern_names.values():
            for name in names:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def _metrics_metadata(self, pattern_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        enabled_patterns = list(
            getattr(self.test_config, "test_mask_actions", None)
            or [getattr(self.test_config, "test_mask_action", None)]
        )
        enabled_patterns = [p for p in enabled_patterns if p is not None]
        weights = []
        masking_cfg = OmegaConf.to_container(self.train_config.masking, resolve=True)
        for entry in masking_cfg.get("mixture", []) or []:
            weights.append(dict(entry.get("weights") or {}))
        return {
            "batch_policy": getattr(self.test_config, "test_batch_policy", "whole_batch"),
            "seed": int(self.test_config.seed),
            "dataset": self.test_config.dataset_name,
            "enabled_patterns": enabled_patterns,
            "weights": weights,
            "pattern_counts": pattern_counts or {},
        }

    def _compute_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute evaluation metrics using baseline's metrics hook."""
        repr_dict_rec = results['repr_dict_rec']
        repr_dict_clean = results.get('repr_dict_clean')
        visible_joint_masks = results.get('visible_joint_masks') or {}

        # Aggregate joint positions and repr_dicts
        joint_pos_rec_list = []
        joint_pos_clean_list = []
        vert_pos_rec_list = []
        vert_pos_clean_list = []
        repr_dict_rec_all = {}
        repr_dict_clean_all = {}
        visible_mask_list = []

        cat_axis = 1 if self.test_config.slice_strategy == 'full_sequence' else 0
        metrics_baseline = getattr(self.test_config, "metrics_baseline", None) or self._baseline_name
        from utils.evaluation import get_metric_names_for_baseline
        needs_mesh_error = "mesh_error_cm" in get_metric_names_for_baseline(metrics_baseline)

        if not repr_dict_rec:
            raise RuntimeError("No inference results were produced; cannot compute metrics.")

        if needs_mesh_error or self.test_config.slice_strategy == 'full_sequence':
            return self._compute_metrics_streaming(
                results=results,
                metrics_baseline=metrics_baseline,
                cat_axis=cat_axis,
            )

        for seq_name in repr_dict_rec.keys():
            # Recover joints using postprocessing hook
            jts_rec, verts_rec = self.batch_processor.postprocessing.recover_world_frame(
                repr_dict_rec[seq_name], {}, self.inference_context
            )
            joint_pos_rec_list.append(jts_rec.numpy(force=True))
            if seq_name in visible_joint_masks:
                visible_mask_list.append(visible_joint_masks[seq_name].numpy(force=True).astype(bool))
            if needs_mesh_error:
                vert_pos_rec_list.append(verts_rec.numpy(force=True))

            # Aggregate repr_dict
            for k, v in repr_dict_rec[seq_name].items():
                if k not in repr_dict_rec_all:
                    repr_dict_rec_all[k] = [v.cpu()]
                else:
                    repr_dict_rec_all[k].append(v.cpu())

            if repr_dict_clean is not None and seq_name in repr_dict_clean:
                jts_clean, verts_clean = self.batch_processor.postprocessing.recover_world_frame(
                    repr_dict_clean[seq_name], {}, self.inference_context
                )
                joint_pos_clean_list.append(jts_clean.numpy(force=True))
                if needs_mesh_error:
                    vert_pos_clean_list.append(verts_clean.numpy(force=True))

                for k, v in repr_dict_clean[seq_name].items():
                    if k not in repr_dict_clean_all:
                        repr_dict_clean_all[k] = [v.cpu()]
                    else:
                        repr_dict_clean_all[k].append(v.cpu())
                        
        for k in repr_dict_rec_all:
            repr_dict_rec_all[k] = torch.cat(repr_dict_rec_all[k], dim=cat_axis)
        for k in repr_dict_clean_all:
            repr_dict_clean_all[k] = torch.cat(repr_dict_clean_all[k], dim=cat_axis)
        
        # Concatenate
        joint_pos_rec = np.concatenate(joint_pos_rec_list, axis=cat_axis)
        pred_verts = None
        gt_verts = None
        if needs_mesh_error:
            pred_verts = np.concatenate(vert_pos_rec_list, axis=cat_axis)

        if joint_pos_clean_list:
            joint_pos_clean = np.concatenate(joint_pos_clean_list, axis=cat_axis)
            if needs_mesh_error:
                gt_verts = np.concatenate(vert_pos_clean_list, axis=cat_axis)
            from utils.evaluation import compute_metrics_for_baseline

            realized_vis_mask = np.concatenate(visible_mask_list, axis=cat_axis) if visible_mask_list else None
            if realized_vis_mask is not None:
                vis_ids = None
                occ_ids = None
            else:
                vis_ids = self.inference_context.get_visible_joint_ids()
                occ_ids = self.inference_context.get_occluded_joint_ids()
            metrics = compute_metrics_for_baseline(
                baseline_name=metrics_baseline,
                pred_joints=joint_pos_rec,
                gt_joints=joint_pos_clean,
                pred_verts=pred_verts,
                gt_verts=gt_verts,
                vis_joint_ids=vis_ids,
                occ_joint_ids=occ_ids,
                vis_joint_mask=realized_vis_mask,
                occ_joint_mask=(~realized_vis_mask) if realized_vis_mask is not None else None,
                repr_dict_rec=repr_dict_rec_all,
                repr_dict_clean=repr_dict_clean_all,
                fps=30,
            )
        else:
            from utils.evaluation import compute_metrics_for_baseline
            metrics = compute_metrics_for_baseline(
                baseline_name=metrics_baseline,
                pred_joints=joint_pos_rec,
                gt_joints=None,
                pred_verts=None,
                gt_verts=None,
                vis_joint_ids=[],
                occ_joint_ids=[],
                repr_dict_rec=None,
                repr_dict_clean=None,
                fps=30,
            )
        
        policy = getattr(self.test_config, "test_batch_policy", "whole_batch")
        if policy != "whole_batch":
            metrics = {
                "aggregate": metrics,
                "per_pattern": self._compute_per_pattern_metrics(
                    results=results,
                    joint_pos_rec=joint_pos_rec,
                    joint_pos_clean=joint_pos_clean if joint_pos_clean_list else None,
                    repr_dict_rec_all=repr_dict_rec_all,
                    repr_dict_clean_all=repr_dict_clean_all,
                    metrics_baseline=metrics_baseline,
                    cat_axis=cat_axis,
                ),
                "metadata": self._metrics_metadata(pattern_counts=self._count_processed_patterns(results)),
            }
        return metrics

    def _compute_per_pattern_metrics(
        self,
        results: Dict[str, Any],
        joint_pos_rec: np.ndarray,
        joint_pos_clean: Optional[np.ndarray],
        repr_dict_rec_all: Dict[str, torch.Tensor],
        repr_dict_clean_all: Dict[str, torch.Tensor],
        metrics_baseline: str,
        cat_axis: int,
    ) -> Dict[str, Dict[str, float]]:
        del joint_pos_rec, joint_pos_clean, repr_dict_rec_all, repr_dict_clean_all
        pattern_names = results.get('pattern_names') or {}
        visible_joint_masks = results.get('visible_joint_masks') or {}
        repr_dict_rec = results.get('repr_dict_rec') or {}
        repr_dict_clean = results.get('repr_dict_clean') or {}
        if not pattern_names or not repr_dict_clean:
            return {}

        grouped = {}
        for seq_name, rec_dict in repr_dict_rec.items():
            if seq_name not in repr_dict_clean or seq_name not in pattern_names or seq_name not in visible_joint_masks:
                return {}
            names = list(pattern_names[seq_name])
            vis = visible_joint_masks[seq_name].numpy(force=True).astype(bool)
            if len(names) != vis.shape[0]:
                return {}

            jts_rec, _ = self.batch_processor.postprocessing.recover_world_frame(
                rec_dict, {}, self.inference_context
            )
            jts_clean, _ = self.batch_processor.postprocessing.recover_world_frame(
                repr_dict_clean[seq_name], {}, self.inference_context
            )
            jts_rec = jts_rec.numpy(force=True)
            jts_clean = jts_clean.numpy(force=True)
            for sample_idx, name in enumerate(names):
                entry = grouped.setdefault(name, {
                    'pred_joints': [],
                    'gt_joints': [],
                    'visible_masks': [],
                    'repr_rec': {},
                    'repr_clean': {},
                })
                entry['pred_joints'].append(jts_rec[sample_idx:sample_idx + 1])
                entry['gt_joints'].append(jts_clean[sample_idx:sample_idx + 1])
                entry['visible_masks'].append(vis[sample_idx:sample_idx + 1])
                for key, value in rec_dict.items():
                    entry['repr_rec'].setdefault(key, []).append(value[sample_idx:sample_idx + 1].cpu())
                for key, value in repr_dict_clean[seq_name].items():
                    entry['repr_clean'].setdefault(key, []).append(value[sample_idx:sample_idx + 1].cpu())

        from utils.evaluation import compute_metrics_for_baseline

        per_pattern = {}
        for name, entry in grouped.items():
            try:
                pred_joints = np.concatenate(entry['pred_joints'], axis=cat_axis)
                gt_joints = np.concatenate(entry['gt_joints'], axis=cat_axis)
                vis_mask = np.concatenate(entry['visible_masks'], axis=cat_axis)
                repr_rec = {
                    key: torch.cat(values, dim=cat_axis)
                    for key, values in entry['repr_rec'].items()
                }
                repr_clean = {
                    key: torch.cat(values, dim=cat_axis)
                    for key, values in entry['repr_clean'].items()
                }
                per_pattern[name] = compute_metrics_for_baseline(
                    baseline_name=metrics_baseline,
                    pred_joints=pred_joints,
                    gt_joints=gt_joints,
                    vis_joint_ids=None,
                    occ_joint_ids=None,
                    vis_joint_mask=vis_mask,
                    occ_joint_mask=~vis_mask,
                    repr_dict_rec=repr_rec,
                    repr_dict_clean=repr_clean,
                    fps=30,
                )
            except Exception as exc:
                if self.test_config.dist_rank == 0:
                    print(f"[Warning] Failed to compute per-pattern metrics for {name}: {exc}")
        return per_pattern

    def _compute_metrics_streaming(
        self,
        results: Dict[str, Any],
        metrics_baseline: str,
        cat_axis: int,
    ) -> Dict[str, Any]:
        """Compute memory-heavy metrics without retaining all recovered vertices."""
        del cat_axis
        if getattr(self.test_config, "test_batch_policy", "whole_batch") != "whole_batch":
            raise RuntimeError(
                "Streaming mesh metrics currently support whole_batch policy only."
            )

        repr_dict_rec = results['repr_dict_rec']
        repr_dict_clean = results.get('repr_dict_clean')
        visible_joint_masks = results.get('visible_joint_masks') or {}
        if repr_dict_clean is None:
            repr_dict_clean = {}

        from utils.evaluation import compute_metrics_for_baseline

        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        iterator = repr_dict_rec.items()
        if self.test_config.dist_rank == 0:
            iterator = tqdm(
                iterator,
                total=len(repr_dict_rec),
                desc="Computing streaming metrics",
                leave=False,
            )

        for seq_idx, (seq_name, rec_dict) in enumerate(iterator):
            jts_rec, verts_rec = self.batch_processor.postprocessing.recover_world_frame(
                rec_dict, {}, self.inference_context
            )
            pred_joints = jts_rec.numpy(force=True)
            pred_verts = verts_rec.numpy(force=True)

            gt_joints = None
            gt_verts = None
            clean_dict = None
            if seq_name in repr_dict_clean:
                clean_dict = repr_dict_clean[seq_name]
                jts_clean, verts_clean = self.batch_processor.postprocessing.recover_world_frame(
                    clean_dict, {}, self.inference_context
                )
                gt_joints = jts_clean.numpy(force=True)
                gt_verts = verts_clean.numpy(force=True)

            vis_mask = None
            if seq_name in visible_joint_masks:
                vis_mask = visible_joint_masks[seq_name].numpy(force=True).astype(bool)

            vis_ids = None if vis_mask is not None else self.inference_context.get_visible_joint_ids()
            occ_ids = None if vis_mask is not None else self.inference_context.get_occluded_joint_ids()
            metrics = compute_metrics_for_baseline(
                baseline_name=metrics_baseline,
                pred_joints=pred_joints,
                gt_joints=gt_joints,
                pred_verts=pred_verts,
                gt_verts=gt_verts,
                vis_joint_ids=vis_ids,
                occ_joint_ids=occ_ids,
                vis_joint_mask=vis_mask,
                occ_joint_mask=(~vis_mask) if vis_mask is not None else None,
                repr_dict_rec=rec_dict,
                repr_dict_clean=clean_dict,
                fps=30,
            )

            for name, value in metrics.items():
                if not np.isfinite(value):
                    continue
                count = self._metric_sample_count(
                    name=name,
                    pred_joints=pred_joints,
                    pred_verts=pred_verts,
                    vis_mask=vis_mask,
                    repr_dict_rec=rec_dict,
                    vis_joint_ids=vis_ids,
                    occ_joint_ids=occ_ids,
                )
                if count <= 0:
                    continue
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value) * count
                metric_counts[name] = metric_counts.get(name, 0) + count

            del jts_rec, verts_rec, pred_joints, pred_verts
            if gt_joints is not None:
                del jts_clean, verts_clean, gt_joints, gt_verts
            if (seq_idx + 1) % 25 == 0:
                torch.cuda.empty_cache()

        if not metric_sums:
            raise RuntimeError("No streaming metrics were produced.")

        return {
            name: metric_sums[name] / metric_counts[name]
            for name in sorted(metric_sums)
        }

    def _metric_sample_count(
        self,
        name: str,
        pred_joints: np.ndarray,
        pred_verts: Optional[np.ndarray] = None,
        vis_mask: Optional[np.ndarray] = None,
        repr_dict_rec: Optional[Dict[str, torch.Tensor]] = None,
        vis_joint_ids: Optional[Any] = None,
        occ_joint_ids: Optional[Any] = None,
    ) -> int:
        """Return the averaging denominator used by registry metric functions."""
        if name in {
            "mpjpe_global_mm",
            "pa_mpjpe_global_mm",
            "mpjpe_cm",
            "pa_mpjpe_cm",
            "positional_error_cm",
        }:
            return int(np.prod(pred_joints.shape[:-1]))
        if name in {"mpjpe_visible_mm", "pa_mpjpe_visible_mm"}:
            if vis_mask is not None:
                return int(np.asarray(vis_mask, dtype=bool).sum())
            return int(np.prod(pred_joints.shape[:-2]) * max(len(vis_joint_ids or []), 1))
        if name in {"mpjpe_occluded_mm", "pa_mpjpe_occluded_mm"}:
            if vis_mask is not None:
                return int((~np.asarray(vis_mask, dtype=bool)).sum())
            return int(np.prod(pred_joints.shape[:-2]) * max(len(occ_joint_ids or []), 1))
        if name in {"root_position_error_cm", "root_position_error_mm"}:
            return int(np.prod(pred_joints.shape[:-2]))
        if name == "mesh_error_cm" and pred_verts is not None:
            return int(np.prod(pred_verts.shape[:-1]))
        if name == "angle_error_deg" and repr_dict_rec and "R_world_joints" in repr_dict_rec:
            return int(repr_dict_rec["R_world_joints"].numel() // 6)
        if name == "sip_error_deg" and repr_dict_rec and "R_world_joints" in repr_dict_rec:
            return int((repr_dict_rec["R_world_joints"].numel() // (22 * 6)) * 4)
        if name in {"acceleration_error_ms2", "jitter"}:
            offset = 2 if name == "acceleration_error_ms2" else 3
            seq_count = int(np.prod(pred_joints.shape[:-3])) if len(pred_joints.shape) > 3 else 1
            return max(pred_joints.shape[-3] - offset, 0) * pred_joints.shape[-2] * seq_count
        if name in {"skating_ratio_rec", "skating_ratio_rec_gt_floor", "skating_ratio_gt_floor"}:
            seq_count = int(np.prod(pred_joints.shape[:-3])) if len(pred_joints.shape) > 3 else 1
            return max(pred_joints.shape[-3] - 1, 0) * 4 * seq_count
        if name in {"ground_penetration_freq_percent", "ground_penetration_dist_mm"}:
            seq_count = int(np.prod(pred_joints.shape[:-3])) if len(pred_joints.shape) > 3 else 1
            return pred_joints.shape[-3] * 2 * seq_count
        if name in {"head_pos_error_mm", "head_rot_error_deg"}:
            return int(np.prod(pred_joints.shape[:-2]))
        if name == "contact_label_accuracy" and repr_dict_rec and "foot_contact" in repr_dict_rec:
            return int(repr_dict_rec["foot_contact"].numel())
        if name == "gnd_percent":
            return int(np.prod(pred_joints.shape[:-3])) if len(pred_joints.shape) > 3 else 1
        if name == "foot_movement_speed_mm_s":
            seq_count = int(np.prod(pred_joints.shape[:-3])) if len(pred_joints.shape) > 3 else 1
            return max(pred_joints.shape[-3] - 1, 0) * 4 * seq_count
        return 1

    def _save_results(self, metrics: Dict[str, Any]):
        """Save evaluation results."""
        os.makedirs(self.test_config.save_root, exist_ok=True)
        
        # Build filename with baseline name
        filename = f'test_{self.test_config.dataset_name}'
        filename += f'_baseline_{self._baseline_name}'
        mask_label = self.test_config.test_mask_action or getattr(self.test_config, 'test_batch_policy', 'mask')
        filename += f'_mask_{mask_label}'
        filename += f'_steps_{self.test_config.num_inference_steps}'
        filename += f'_seed_{self.test_config.seed}.json'
        
        json_path = os.path.join(self.test_config.save_root, filename)
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        if self.test_config.dist_rank == 0:
            print(f"\n--- Evaluation Results ({self._baseline_name}) ---")
            printable_metrics = metrics.get("aggregate", metrics) if isinstance(metrics, dict) else metrics
            for key, value in sorted(printable_metrics.items()):
                print(f"{key}: {value:.4f}")
            print("-" * 40)
            print(f"Results saved to: {json_path}")
