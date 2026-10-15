"""
Debug analysis utilities for motion reconstruction.
"""
import os
import json
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class SequenceAnalysis:
    """Analysis results for a single sequence."""
    batch_idx: Any
    seq_len: int
    mpjpe_mm: float
    mpjpe_median_mm: float
    mpjpe_std_mm: float
    mpjpe_min_mm: float
    mpjpe_max_mm: float
    gt_mean_velocity_ms: float
    gt_max_velocity_ms: float
    gt_mean_acceleration: float
    gt_max_acceleration: float
    boundary_mpjpe_mm: float
    center_mpjpe_mm: float
    outlier_count: int
    outlier_ratio: float
    num_clusters: int
    hierarchy_errors: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DebugAnalyzer:
    """
    Comprehensive debug analysis for motion reconstruction results.
    
    Analyzes:
    - Per-frame and per-joint errors
    - Motion dynamics (velocity, acceleration, jerk)
    - Sliding window boundary effects
    - Joint hierarchy error distribution
    - Outlier clustering
    """
    
    def __init__(self, save_root: str, window_size: int = 32, overlap_size: int = 8):
        self.save_root = save_root
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.stride = window_size - overlap_size
        
        self.dump_dir = os.path.join(save_root, 'debug_analysis')
        os.makedirs(self.dump_dir, exist_ok=True)
        
        # Joint hierarchy for SMPL
        self.joint_hierarchy = {
            'root': [0],
            'level1': [1, 2, 3],
            'level2': [4, 5, 6],
            'level3': [7, 8, 9, 12, 13, 14],
            'extremities': [10, 11, 15, 16, 17, 18, 19, 20, 21],
        }
    
    def analyze_sequence(
        self,
        seq_name: Any,
        jts_rec: np.ndarray,
        jts_clean: np.ndarray,
        fps: float = 30.0,
    ) -> SequenceAnalysis:
        """
        Perform comprehensive analysis on a single sequence.
        
        Args:
            seq_name: Sequence identifier
            jts_rec: Reconstructed joints [1, T, 22, 3]
            jts_clean: Ground truth joints [1, T, 22, 3]
            fps: Frame rate
            
        Returns:
            SequenceAnalysis with all computed metrics
        """
        seq_len = jts_rec.shape[1]
        
        # Basic error metrics
        per_joint_errors = np.linalg.norm(jts_rec - jts_clean, axis=-1)[0]  # [T, 22]
        per_frame_mpjpe = per_joint_errors.mean(axis=-1)  # [T]
        
        # Motion dynamics
        gt_velocity = np.linalg.norm(jts_clean[:, 1:] - jts_clean[:, :-1], axis=-1)[0]
        gt_accel = np.linalg.norm(
            jts_clean[:, 2:] - 2 * jts_clean[:, 1:-1] + jts_clean[:, :-2], axis=-1
        )[0] if seq_len > 2 else np.zeros((1, 22))
        
        # Sliding window boundary analysis
        boundary_frames, center_frames = self._get_boundary_frames(seq_len)
        boundary_mpjpe = per_frame_mpjpe[list(boundary_frames)].mean() if boundary_frames else 0
        center_mpjpe = per_frame_mpjpe[list(center_frames)].mean() if center_frames else 0
        
        # Joint hierarchy errors
        hierarchy_errors = {}
        for level, joints in self.joint_hierarchy.items():
            hierarchy_errors[level] = float(per_joint_errors[:, joints].mean() * 1000)
        
        # Outlier analysis
        outlier_threshold = per_frame_mpjpe.mean() + 2 * per_frame_mpjpe.std()
        outlier_frames = np.where(per_frame_mpjpe > outlier_threshold)[0]
        clusters = self._cluster_outliers(outlier_frames)
        
        analysis = SequenceAnalysis(
            batch_idx=seq_name,
            seq_len=seq_len,
            mpjpe_mm=float(per_frame_mpjpe.mean() * 1000),
            mpjpe_median_mm=float(np.median(per_frame_mpjpe) * 1000),
            mpjpe_std_mm=float(per_frame_mpjpe.std() * 1000),
            mpjpe_min_mm=float(per_frame_mpjpe.min() * 1000),
            mpjpe_max_mm=float(per_frame_mpjpe.max() * 1000),
            gt_mean_velocity_ms=float(gt_velocity.mean() * fps),
            gt_max_velocity_ms=float(gt_velocity.max() * fps),
            gt_mean_acceleration=float(gt_accel.mean() * fps * fps),
            gt_max_acceleration=float(gt_accel.max() * fps * fps),
            boundary_mpjpe_mm=float(boundary_mpjpe * 1000),
            center_mpjpe_mm=float(center_mpjpe * 1000),
            outlier_count=len(outlier_frames),
            outlier_ratio=float(len(outlier_frames) / seq_len),
            num_clusters=len(clusters),
            hierarchy_errors=hierarchy_errors,
        )
        
        return analysis
    
    def _get_boundary_frames(self, seq_len: int) -> tuple:
        """Identify boundary and center frames based on sliding window."""
        window_starts = list(range(0, seq_len, self.stride))
        window_ends = [min(ws + self.window_size, seq_len) for ws in window_starts]
        
        boundary_frames = set()
        for ws, we in zip(window_starts, window_ends):
            for f in range(ws, min(ws + self.overlap_size, we)):
                boundary_frames.add(f)
            for f in range(max(we - self.overlap_size, ws), we):
                boundary_frames.add(f)
        
        center_frames = set(range(seq_len)) - boundary_frames
        return boundary_frames, center_frames
    
    def _cluster_outliers(self, outlier_frames: np.ndarray, gap_threshold: int = 5) -> List[List[int]]:
        """Cluster consecutive outlier frames."""
        if len(outlier_frames) == 0:
            return []
        
        clusters = []
        current_cluster = [int(outlier_frames[0])]
        
        for i in range(1, len(outlier_frames)):
            if outlier_frames[i] - outlier_frames[i-1] <= gap_threshold:
                current_cluster.append(int(outlier_frames[i]))
            else:
                clusters.append(current_cluster)
                current_cluster = [int(outlier_frames[i])]
        
        clusters.append(current_cluster)
        return clusters
    
    def save_analysis(
        self,
        seq_name: Any,
        analysis: SequenceAnalysis,
        jts_rec: Optional[np.ndarray] = None,
        jts_clean: Optional[np.ndarray] = None,
    ):
        """Save analysis results to files."""
        seq_name_safe = str(seq_name).replace('/', '_')
        
        # Save JSON analysis
        json_path = os.path.join(self.dump_dir, f'batch_{seq_name_safe}_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2)
        
        # Save numpy arrays if provided
        if jts_rec is not None and jts_clean is not None:
            npz_path = os.path.join(self.dump_dir, f'batch_{seq_name_safe}_arrays.npz')
            np.savez(npz_path, jts_rec=jts_rec, jts_clean=jts_clean)
        
        print(f"[Analysis] {seq_name}: MPJPE={analysis.mpjpe_mm:.1f}mm, "
              f"Vel={analysis.gt_mean_velocity_ms:.3f}m/s, "
              f"Outliers={analysis.outlier_ratio*100:.1f}%")
    
    def aggregate_analyses(self, analyses: List[SequenceAnalysis]) -> Dict[str, float]:
        """Aggregate multiple sequence analyses into summary statistics."""
        if not analyses:
            return {}
        
        return {
            'mean_mpjpe_mm': np.mean([a.mpjpe_mm for a in analyses]),
            'std_mpjpe_mm': np.std([a.mpjpe_mm for a in analyses]),
            'mean_boundary_mpjpe_mm': np.mean([a.boundary_mpjpe_mm for a in analyses]),
            'mean_center_mpjpe_mm': np.mean([a.center_mpjpe_mm for a in analyses]),
            'mean_outlier_ratio': np.mean([a.outlier_ratio for a in analyses]),
            'total_sequences': len(analyses),
        }
