"""
Base visualizer interface and common data structures.
"""
import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from omegaconf import DictConfig


@dataclass
class VisualizationData:
    """Container for visualization data."""
    # Reconstructed data (list for multiple tries/settings)
    repr_dict_rec_tries: List[Dict[str, torch.Tensor]]
    jts_rec_tries: List[torch.Tensor]
    verts_rec_tries: List[torch.Tensor]
    occluded_joint_ids_tries: List[np.ndarray]
    color_vis_tries: List[Tuple[float, float, float]]
    color_occ_tries: List[Tuple[float, float, float]]
    
    # Ground truth data (optional)
    repr_dict_clean: Optional[Dict[str, torch.Tensor]] = None
    jts_clean: Optional[torch.Tensor] = None
    verts_clean: Optional[torch.Tensor] = None
    
    # Noisy input data (optional)
    repr_dict_noisy: Optional[Dict[str, torch.Tensor]] = None
    jts_noisy: Optional[torch.Tensor] = None
    verts_noisy: Optional[torch.Tensor] = None
    
    # Sampling progress (optional)
    sampling_progress_entries: List[Dict] = field(default_factory=list)
    
    # Batch metadata
    batch: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate visualization contract with fail-fast semantics."""
        n_tries = len(self.repr_dict_rec_tries)
        if n_tries == 0:
            raise ValueError("VisualizationData requires at least one reconstruction try.")

        lengths = {
            "jts_rec_tries": len(self.jts_rec_tries),
            "verts_rec_tries": len(self.verts_rec_tries),
            "occluded_joint_ids_tries": len(self.occluded_joint_ids_tries),
            "color_vis_tries": len(self.color_vis_tries),
            "color_occ_tries": len(self.color_occ_tries),
        }
        for field_name, field_length in lengths.items():
            if field_length != n_tries:
                raise ValueError(
                    f"{field_name} length mismatch: expected {n_tries}, got {field_length}."
                )

        for idx, occluded_ids in enumerate(self.occluded_joint_ids_tries):
            if not isinstance(occluded_ids, np.ndarray):
                raise TypeError(f"occluded_joint_ids_tries[{idx}] must be np.ndarray.")
            if occluded_ids.ndim != 1:
                raise ValueError(f"occluded_joint_ids_tries[{idx}] must be 1-D.")
            if not np.issubdtype(occluded_ids.dtype, np.integer):
                raise TypeError(f"occluded_joint_ids_tries[{idx}] must be integer array.")


class BaseVisualizer(ABC):
    """
    Abstract base class for dataset-specific visualizers.
    
    Subclasses implement dataset-specific rendering logic while
    sharing common infrastructure for video saving, frame management, etc.
    """
    
    def __init__(
        self,
        train_config: DictConfig,
        test_config: DictConfig,
        fk_model: Any,
    ):
        self.train_config = train_config
        self.test_config = test_config
        # Keep both attribute names for backward compatibility
        self.fk_model = fk_model
        self.smplx_model = fk_model
        self._is_initialized = False
    
    @abstractmethod
    def setup(self) -> None:
        """Initialize visualization resources (windows, renderers, etc.)."""
        pass
    
    @abstractmethod
    def visualize(self, data: VisualizationData) -> None:
        """
        Render visualization for the given data.
        
        Args:
            data: VisualizationData containing all necessary tensors
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Release visualization resources."""
        pass
    
    def should_visualize(self) -> bool:
        """Check if visualization should be performed."""
        return (
            self.test_config.visualize and
            self.test_config.dist_rank == 0
        )
    
    @staticmethod
    def generate_progress_colors(num_steps: int) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Generate color gradient for sampling progress visualization."""
        if num_steps <= 1:
            return [([90/255, 135/255, 247/255], [60/255, 90/255, 180/255])]
        
        start_color = np.array([72, 149, 239], dtype=np.float32) / 255.0
        end_color = np.array([235, 99, 132], dtype=np.float32) / 255.0
        
        colors = []
        for idx in range(num_steps):
            alpha = idx / max(1, num_steps - 1)
            color_vis = (1 - alpha) * start_color + alpha * end_color
            color_occ = np.clip(color_vis * 0.6, 0.0, 1.0)
            colors.append((color_vis.tolist(), color_occ.tolist()))
        
        return colors


# Standard colors for visualization
COLOR_GT = (1.0, 0.0, 0.0)  # Red for ground truth
COLOR_VIS = (0.35, 0.53, 0.97)  # Blue for visible parts
COLOR_OCC = (0.92, 0.73, 0.23)  # Yellow for occluded parts

# SMPL skeleton connections
SMPL_SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21)
]
