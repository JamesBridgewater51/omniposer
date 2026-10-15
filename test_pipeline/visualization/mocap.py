"""
MoCap dataset visualization using matplotlib 3D plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .base import BaseVisualizer, VisualizationData, SMPL_SKELETON_CONNECTIONS
from .registry import VisualizerRegistry

# @VisualizerRegistry.register('aist')
# @VisualizerRegistry.register('totalcapture')
@VisualizerRegistry.register('3dpw')
class MoCaPVisualizer(BaseVisualizer):
    """Matplotlib-based visualizer for MoCap datasets."""
    
    def __init__(self, train_config, test_config, smplx_model):
        super().__init__(train_config, test_config, smplx_model)
        self.fig = None
        self.ax = None
    
    def setup(self) -> None:
        """Initialize matplotlib figure."""
        if self._is_initialized:
            return
        
        plt.ion()
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        
        def on_key_press(event):
            if event.key in ('q', 'Q'):
                self.cleanup()
        
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
        self._is_initialized = True
    
    def visualize(self, data: VisualizationData) -> None:
        """Render MoCap visualization."""
        if not self.should_visualize():
            return
        
        if self.fig is None or (self._is_initialized and not plt.fignum_exists(self.fig.number)):
            self._is_initialized = False
        if not self._is_initialized:
            self.setup()
        
        # Convert tensors to numpy
        jts_rec_tries = [jts.numpy(force=True) for jts in data.jts_rec_tries]
        verts_rec_tries = [verts.numpy(force=True) for verts in data.verts_rec_tries]
        
        if data.jts_clean is None:
            print("[Warning] MoCap visualization requires ground truth data")
            return
        
        jts_clean = data.jts_clean.numpy(force=True)
        
        bs, T, num_joints, _ = jts_clean.shape
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        
        for bs_idx in range(min(bs, 1)):
            if bs_idx % self.test_config.vis_interval != 0:
                continue
            
            for t in range(T):
                if not plt.get_fignums():
                    return
                
                self.ax.clear()
                
                # Plot ground truth
                joints_gt = jts_clean[bs_idx, t]
                self.ax.scatter(
                    joints_gt[:, 0], joints_gt[:, 1], joints_gt[:, 2],
                    c='red', s=50, alpha=0.8, label='GT Joints'
                )
                
                for conn in SMPL_SKELETON_CONNECTIONS:
                    if conn[0] < num_joints and conn[1] < num_joints:
                        p1, p2 = joints_gt[conn[0]], joints_gt[conn[1]]
                        self.ax.plot(
                            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                            'r-', linewidth=2, alpha=0.8, label='_nolegend_'
                        )
                
                # Plot reconstructions (contract guarantees aligned try lists)
                for try_idx, (jts_rec, verts_rec, mask_ids) in enumerate(
                    zip(jts_rec_tries, verts_rec_tries, data.occluded_joint_ids_tries)
                ):
                    joints_rec = jts_rec[bs_idx, t]
                    color = colors[try_idx % len(colors)]
                    
                    # Plot vertices
                    self.ax.scatter(
                        verts_rec[bs_idx, t, :, 0],
                        verts_rec[bs_idx, t, :, 1],
                        verts_rec[bs_idx, t, :, 2],
                        c=color, s=1, alpha=0.08, label=f'Rec {try_idx} Verts'
                    )
                    
                    # Plot joints
                    self.ax.scatter(
                        joints_rec[:, 0], joints_rec[:, 1], joints_rec[:, 2],
                        c=color, s=50, alpha=0.8, label=f'Rec {try_idx} Joints'
                    )
                    
                    # Plot skeleton
                    for conn in SMPL_SKELETON_CONNECTIONS:
                        if conn[0] < num_joints and conn[1] < num_joints:
                            p1, p2 = joints_rec[conn[0]], joints_rec[conn[1]]
                            self.ax.plot(
                                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                f'{color[0]}-', linewidth=2, alpha=0.6, label='_nolegend_'
                            )
                    
                    # Highlight masked joints
                    if mask_ids.size > 0:
                        masked_joints = joints_rec[mask_ids]
                        self.ax.scatter(
                            masked_joints[:, 0], masked_joints[:, 1], masked_joints[:, 2],
                            c='yellow', s=100, marker='x', alpha=1.0,
                            label='Masked' if try_idx == 0 else '_nolegend_'
                        )
                
                self.ax.set_xlabel('X')
                self.ax.set_ylabel('Y')
                self.ax.set_zlabel('Z')
                self.ax.set_aspect('equal')
                self.ax.legend()
                
                plt.tight_layout()
                plt.draw()
                plt.pause(0.001)
        
        plt.ioff()
        plt.show()
    
    def cleanup(self) -> None:
        """Close matplotlib figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        self._is_initialized = False
