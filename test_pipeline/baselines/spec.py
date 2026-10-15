"""
Baseline specification dataclass.

Defines the declarative configuration for a baseline method, including:
- Compatible mask schemes and representation requirements
- Inference characteristics (sliding window, variable length support)
- Hook class references for preprocessing, inference, postprocessing, and metrics
"""
from dataclasses import dataclass, field
from typing import List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .hooks import PreprocessingHook, InferenceHook, PostprocessingHook, MetricsHook


@dataclass(frozen=True)
class BaselineSpec:
    """
    Immutable specification for a baseline method.
    
    This dataclass defines all characteristics of a baseline method in a declarative way.
    Each baseline must provide implementations for all four hook types.
    
    Attributes:
        name: Unique identifier for the baseline (e.g., 'ours', 'egoego', 'pip')
        
        compatible_mask_schemes: List of masking action names this baseline can work with.
            Must match actions defined in cfg_files/conf/masking/*.yaml
            
        default_mask_scheme: The default masking action to use if none specified.
            Must be in compatible_mask_schemes.
            
        required_representation_config: The representation config this baseline requires.
            Either 'xyz_only' (positions only) or 'xyz_rot' (positions + rotations).
            Must match the train_config.representation_registry setting.
            
        requires_rotation_input: Whether the baseline requires rotation data in conditioning.
            If True, the mask scheme must provide rotation modalities.
            
        uses_sliding_window: Whether the baseline uses sliding window for long sequences.
            If False, sequences are processed in a single pass (may require padding/truncation).
            
        supports_variable_length: Whether the baseline can handle variable-length sequences.
            If False, all sequences must be padded/truncated to a fixed length.
            
        preprocessing_hook_cls: Class implementing PreprocessingHook for this baseline.
            Handles batch conditioning preparation and masking application.
            
        inference_hook_cls: Class implementing InferenceHook for this baseline.
            Handles TrajNet/PoseNet inference and trajectory injection.
            
        postprocessing_hook_cls: Class implementing PostprocessingHook for this baseline.
            Handles output reconstruction and world-frame recovery.
            
        metrics_hook_cls: Class implementing MetricsHook for this baseline.
            Handles baseline-specific evaluation metrics computation.
            
        metrics: List of metric names to compute for this baseline.
            Standard metrics: 'mpjpe_global_mm', 'pa_mpjpe_global_mm', 'acceleration_error_ms2'
            Baseline-specific metrics can be added (e.g., 'sip_error_deg' for IMU baselines).
    """
    name: str
    
    # Mask scheme compatibility
    compatible_mask_schemes: List[str]
    default_mask_scheme: str
    
    # Representation requirements
    required_representation_config: str  # 'xyz_only' or 'xyz_rot'
    requires_rotation_input: bool
    
    # Inference characteristics
    uses_sliding_window: bool
    supports_variable_length: bool
    
    # Hook implementations (all required)
    preprocessing_hook_cls: Type['PreprocessingHook']
    inference_hook_cls: Type['InferenceHook']
    postprocessing_hook_cls: Type['PostprocessingHook']
    metrics_hook_cls: Type['MetricsHook']
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        'mpjpe_global_mm',
        'pa_mpjpe_global_mm',
        'acceleration_error_ms2',
    ])
    
    def __post_init__(self):
        """Validate spec consistency."""
        assert self.default_mask_scheme in self.compatible_mask_schemes, \
            f"default_mask_scheme '{self.default_mask_scheme}' not in compatible_mask_schemes"
        assert self.required_representation_config in ('xyz_only', 'xyz_rot'), \
            f"required_representation_config must be 'xyz_only' or 'xyz_rot', got '{self.required_representation_config}'"
