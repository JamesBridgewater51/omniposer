"""
Test configuration handling and validation.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig, OmegaConf


@dataclass
class TestConfig:
    """Unified test configuration container."""
    # Paths
    logdir: str = ""
    save_root: str = ""
    model_path_trajnet: str = ""
    model_path_posenet: str = ""
    
    # Dataset
    dataset_name: str = "amass"
    slice_strategy: str = "full_sequence"
    test_mask_action: Optional[str] = None
    test_mask_actions: Optional[List[str]] = None
    test_batch_policy: str = "whole_batch"

    # Inference
    num_inference_steps: int = 1
    sample_iter: int = 1
    sample_aggregation: str = "last"
    overlap_size: int = 8
    seed: int = 42
    use_ema_checkpoint: Optional[bool] = None
    strict_ema_checkpoint: bool = False
    
    # Evaluation
    with_gt: bool = True
    compute_metrics_per_seq: bool = False
    eval_on_trajnet_only: bool = False
    eval_on_posenet_only: bool = False
    metrics_baseline: Optional[str] = None
    
    # Visualization
    visualize: bool = False
    vis_interval: int = 1
    visualize_sampling_progress: bool = False
    sampling_progress_stride: int = 1
    sampling_progress_max_steps: int = 10
    
    # Processing
    num_parallel_inference_workers: int = 0
    use_root_init_xy_in_sliding_window_preprocessing: bool = True
    root_init_xy_policy: str = "visible_safe"
    root_init_rot_policy: str = "disabled"
    root_init_rot_visible_rotation_modality: str = "R_world_joints"
    root_init_rot_visible_rotation_axis: str = "y"
    root_init_rot_visible_rotation_joint_ids: Optional[List[int]] = None
    eval_norm_params_policy: str = "prefer_global"
    cond_fn_with_grad: bool = False
    postprocess_rot6d_normalize: bool = False
    postprocess_betas_sequence_average: bool = False
    postprocess_temporal_smoothing: bool = False
    postprocess_temporal_smoothing_kernel: int = 5
    postprocess_temporal_smoothing_strength: float = 0.5
    postprocess_temporal_smoothing_modalities: Optional[List[str]] = None
    postprocess_contact_median: bool = False
    postprocess_contact_median_kernel: int = 3
    postprocess_contact_threshold: float = 0.5
    postprocess_contact_binarize: bool = False
    postprocess_contact_kinematic: bool = False
    postprocess_contact_kinematic_mode: str = "replace"
    postprocess_contact_kinematic_floor_mode: str = "predicted_min"
    postprocess_contact_kinematic_vel_thresh: float = 5e-5
    postprocess_contact_kinematic_height_thresholds: Optional[List[float]] = None
    postprocess_contact_kinematic_joint_ids: Optional[List[int]] = None
    postprocess_contact_kinematic_last_frame: str = "repeat"
    postprocess_contact_kinematic_blend_weight: float = 1.0
    postprocess_foot_lock: bool = False
    postprocess_foot_lock_strength: float = 1.0
    postprocess_foot_lock_contact_threshold: float = 0.5
    postprocess_foot_lock_max_shift_per_frame: float = 0.03
    postprocess_foot_lock_max_total_shift: float = 0.20
    postprocess_foot_lock_no_contact_decay: float = 1.0
    postprocess_foot_lock_joint_ids: Optional[List[int]] = None
    postprocess_observed_joint_anchor: bool = False
    postprocess_observed_joint_anchor_strength: float = 1.0
    postprocess_observed_joint_anchor_min_joints: int = 1
    postprocess_observed_joint_anchor_max_shift: float = 0.25
    postprocess_observed_joint_anchor_smooth_delta: bool = False
    postprocess_observed_joint_anchor_smooth_kernel: int = 3
    postprocess_observed_joint_anchor_smooth_strength: float = 0.5
    postprocess_observed_joint_anchor_dims: Optional[List[int]] = None
    postprocess_observed_joint_anchor_joint_ids: Optional[List[int]] = None
    postprocess_floor_lift: bool = False
    postprocess_floor_lift_mode: str = "sequence"
    postprocess_floor_lift_target_z: float = 0.0
    postprocess_floor_lift_tolerance: float = 0.01
    postprocess_floor_lift_max_shift: float = 0.10
    postprocess_floor_lift_smooth_kernel: int = 1
    postprocess_floor_lift_smooth_strength: float = 0.0
    postprocess_floor_lift_joint_ids: Optional[List[int]] = None
    postprocess_fk_chunk_size: int = 512
    
    # FK model selection
    fk_model_type: str = "auto"  # auto / smplx / articulate
    fk_model_path: Optional[str] = None
    fk_model_use_pose_blendshape: bool = True
    fk_missing_joint_fill_strategy: str = "identity"  # identity / copy_wrists
    
    # MPI/Distributed
    dist_rank: int = 0
    dist_world_size: int = 1
    
    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'TestConfig':
        """Create TestConfig from Hydra DictConfig."""
        return cls(**{
            k: OmegaConf.to_container(v, resolve=True) if isinstance(v, DictConfig) else v
            for k, v in OmegaConf.to_container(cfg, resolve=True).items()
            if k in cls.__dataclass_fields__
        })


def apply_mpi_config(test_config: DictConfig, rank: int, world_size: int) -> DictConfig:
    """Apply MPI rank/world_size to test configuration."""
    from omegaconf import open_dict
    with open_dict(test_config):
        test_config.dist_rank = rank
        test_config.dist_world_size = world_size
        
        # Apply MPI-based parameter sweeps if configured
        if test_config.get('test_mask_actions') and world_size > 1 and test_config.get('test_batch_policy', 'whole_batch') != 'per_sample':
            actions = test_config.test_mask_actions
            test_config.test_mask_action = actions[rank % len(actions)]
        
        if test_config.get('inference_steps_list') and world_size > 1:
            steps_list = test_config.inference_steps_list
            test_config.num_inference_steps = steps_list[rank % len(steps_list)]
    
    return test_config


def validate_test_config(test_config: DictConfig) -> None:
    """Validate test configuration."""
    assert test_config.logdir, "logdir must be specified"
    assert os.path.exists(test_config.logdir), f"logdir does not exist: {test_config.logdir}"
    
    if test_config.model_path_trajnet:
        assert os.path.exists(test_config.model_path_trajnet), \
            f"TrajNet checkpoint not found: {test_config.model_path_trajnet}"
    
    if test_config.model_path_posenet:
        assert os.path.exists(test_config.model_path_posenet), \
            f"PoseNet checkpoint not found: {test_config.model_path_posenet}"
    
    batch_policy = getattr(test_config, 'test_batch_policy', 'whole_batch')
    assert batch_policy in {'whole_batch', 'per_sample', 'sweep'}, \
        "test_batch_policy must be one of {'whole_batch', 'per_sample', 'sweep'}"
    if batch_policy == 'per_sample':
        assert getattr(test_config, 'test_mask_actions', None) or getattr(test_config, 'test_mask_action', None), \
            "per_sample test_batch_policy requires test_mask_actions or test_mask_action"
    elif batch_policy == 'sweep':
        assert getattr(test_config, 'test_mask_actions', None), \
            "sweep test_batch_policy requires test_mask_actions"
    elif batch_policy == 'whole_batch':
        if not getattr(test_config, 'test_mask_action', None):
            train_cfg = OmegaConf.load(os.path.join(test_config.logdir, "config.yaml"))
            action_names = list((OmegaConf.to_container(train_cfg.masking, resolve=True).get('actions') or [{}])[0:1])
            if action_names and action_names[0].get('name'):
                test_config.test_mask_action = action_names[0]['name']
            else:
                raise AssertionError("whole_batch test_batch_policy requires test_mask_action")

    sample_aggregation = getattr(test_config, 'sample_aggregation', 'mean')
    assert sample_aggregation in {'last', 'mean', 'median'}, \
        "sample_aggregation must be one of {'last', 'mean', 'median'}"

    root_init_xy_policy = getattr(test_config, 'root_init_xy_policy', 'visible_safe')
    assert root_init_xy_policy in {'visible_safe', 'visible_mean', 'legacy_root', 'zero'}, \
        "root_init_xy_policy must be one of {'visible_safe', 'visible_mean', 'legacy_root', 'zero'}"
    root_init_rot_policy = getattr(test_config, 'root_init_rot_policy', 'disabled')
    assert root_init_rot_policy in {'disabled', 'visible_safe', 'visible_rotation', 'legacy_gt'}, \
        "root_init_rot_policy must be one of {'disabled', 'visible_safe', 'visible_rotation', 'legacy_gt'}"
    visible_rotation_axis = getattr(test_config, 'root_init_rot_visible_rotation_axis', 'y')
    assert visible_rotation_axis in {'x', 'y', 'z', '-x', '-y', '-z'}, \
        "root_init_rot_visible_rotation_axis must be one of {'x', 'y', 'z', '-x', '-y', '-z'}"
    eval_norm_params_policy = getattr(test_config, 'eval_norm_params_policy', 'prefer_global')
    assert eval_norm_params_policy in {'provided', 'prefer_global', 'strict_global'}, \
        "eval_norm_params_policy must be one of {'provided', 'prefer_global', 'strict_global'}"

    contact_kinematic_mode = getattr(test_config, 'postprocess_contact_kinematic_mode', 'replace')
    assert contact_kinematic_mode in {'replace', 'or', 'and', 'blend'}, \
        "postprocess_contact_kinematic_mode must be one of {'replace', 'or', 'and', 'blend'}"
    contact_floor_mode = getattr(test_config, 'postprocess_contact_kinematic_floor_mode', 'predicted_min')
    assert contact_floor_mode in {'predicted_min', 'absolute', 'target_z'}, \
        "postprocess_contact_kinematic_floor_mode must be one of {'predicted_min', 'absolute', 'target_z'}"
    contact_last_frame = getattr(test_config, 'postprocess_contact_kinematic_last_frame', 'repeat')
    assert contact_last_frame in {'repeat', 'zero'}, \
        "postprocess_contact_kinematic_last_frame must be one of {'repeat', 'zero'}"
    floor_lift_mode = getattr(test_config, 'postprocess_floor_lift_mode', 'sequence')
    assert floor_lift_mode in {'sequence', 'frame'}, \
        "postprocess_floor_lift_mode must be one of {'sequence', 'frame'}"

    fk_type = getattr(test_config, 'fk_model_type', 'auto')
    dataset_name = getattr(test_config, 'dataset_name', '')
    if fk_type == 'auto':
        fk_type = 'articulate' if dataset_name.lower() in {'aist', 'totalcapture', '3dpw'} else 'smplx'
    if fk_type == 'articulate':
        fk_model_path = getattr(test_config, 'fk_model_path', None) or \
            os.path.join('data', 'body_models', 'smpl_model', 'models', 'SMPL_male.pkl')
        assert os.path.exists(fk_model_path), f"FK model path not found: {fk_model_path}"
        assert getattr(test_config, 'fk_missing_joint_fill_strategy', 'identity') in {'identity', 'copy_wrists'}, \
            "fk_missing_joint_fill_strategy must be one of {'identity', 'copy_wrists'}"


def save_test_config(save_root: str, test_config: DictConfig) -> None:
    """Save test configuration to YAML file."""
    os.makedirs(save_root, exist_ok=True)
    config_path = os.path.join(save_root, "test_config.yaml")
    OmegaConf.save(test_config, config_path)
