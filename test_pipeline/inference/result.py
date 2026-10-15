"""
Data structures for inference results.
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """Container for inference results."""
    repr_dict_rec: Dict[str, torch.Tensor]
    masking_spec: tuple  # (mask_action, visible_ids_override)
    repr_dict_clean: Optional[Dict[str, torch.Tensor]] = None
    repr_dict_noisy: Optional[Dict[str, torch.Tensor]] = None
    sampling_progress_entries: Optional[List[Dict]] = None
    visible_joint_mask: Optional[torch.Tensor] = None
    cond_joint_modality_mask: Optional[torch.Tensor] = None
    cond_joint_modality_names: Optional[List[str]] = None
    cond_pattern_names: Optional[List[str]] = None
    cond_mask_policy: Optional[str] = None
    metrics_metadata: Optional[Dict[str, Any]] = None

    @property
    def mask_joint_ids(self) -> np.ndarray:
        """Compute mask_joint_ids from masking_spec for backward compatibility."""
        mask_action, visible_ids_override = self.masking_spec
        if visible_ids_override is not None:
            return np.array([i for i in range(22) if i not in visible_ids_override], dtype=int)
        visible_ids = None
        if mask_action is not None:
            visible_ids = mask_action.get("visible_ids")
        if visible_ids is not None:
            return np.array([i for i in range(22) if i not in visible_ids], dtype=int)
        if self.visible_joint_mask is not None:
            mask = self.visible_joint_mask.detach().cpu().numpy().astype(bool)
            if mask.size:
                invisible = np.where(~mask.reshape(-1, mask.shape[-1]).any(axis=0))[0]
                return invisible.astype(int)
        return np.array([], dtype=int)
