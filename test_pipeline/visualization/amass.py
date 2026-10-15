"""
AMASS dataset visualization using Open3D.
"""
import os
import tempfile
import numpy as np
from typing import Optional
from PIL import Image, ImageDraw

from .base import BaseVisualizer, VisualizationData, COLOR_GT, COLOR_VIS, COLOR_OCC
from .registry import VisualizerRegistry
from utils.vis_util import vis_skeleton, vis_foot_contact, update_cam
from utils.other_utils import LIMBS_BODY_SMPL


@VisualizerRegistry.register('amass')
@VisualizerRegistry.register('nymeria')
class AMASSVisualizer(BaseVisualizer):
    """Open3D-based visualizer for AMASS dataset."""
    
    def __init__(self, train_config, test_config, smplx_model):
        super().__init__(train_config, test_config, smplx_model)
        self.vis = None
        self.should_exit = False
        self.fix_cam = True
        self.T_bc = np.array([
            [0, 0, -1, 10],
            [-1, 0, 0, 0],
            [0, -1, 0, 2],
            [0, 0, 0, 1]
        ])
    
    def setup(self) -> None:
        """Initialize Open3D visualizer."""
        if self._is_initialized:
            return
        
        import open3d as o3d
        
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        
        def key_callback(vis):
            self.should_exit = True
        
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.vis.register_key_callback(ord('Q'), key_callback)
        self.vis.register_key_callback(ord('q'), key_callback)
        
        opt = self.vis.get_render_option()
        opt.line_width = 5.0
        
        self.vis.add_geometry(mesh_frame)
        
        # Create ground plane
        ground = o3d.geometry.TriangleMesh.create_box(width=10.0, height=10.0, depth=0.1)
        ground_trans = np.array([
            [1, 0, 0, -5],
            [0, 1, 0, 0],
            [0, 0, 1, -0.1],
            [0, 0, 0, 1]
        ])
        ground.transform(ground_trans)
        ground.paint_uniform_color([0.8, 0.8, 0.8])
        self.vis.add_geometry(ground)
        
        self._is_initialized = True
        print('AMASS Visualization ready:')
        print('[Q/q] to EXIT visualization')
    
    def visualize(self, data: VisualizationData) -> None:
        """Render AMASS visualization."""
        if not self.should_visualize():
            return
        
        if not self._is_initialized:
            self.setup()
        
        import open3d as o3d
        
        temp_dir = tempfile.mkdtemp(dir=self.test_config.save_root)
        frames = []
        
        # Convert tensors to numpy
        jts_rec_tries = [jts.numpy(force=True) for jts in data.jts_rec_tries]
        verts_rec_tries = [verts.numpy(force=True) for verts in data.verts_rec_tries]
        
        if data.jts_clean is not None:
            jts_clean = data.jts_clean.numpy(force=True)
            verts_clean = data.verts_clean.numpy(force=True)
        
        bs, T, *_ = jts_rec_tries[0].shape
        
        for bs_idx in range(0, bs, 1):
            if bs_idx % self.test_config.vis_interval != 0:
                continue
            
            for t in range(T - 1):
                if self.should_exit:
                    self.should_exit = False
                    break
                
                geometries = []
                
                # Ground truth skeleton and mesh
                if data.jts_clean is not None:
                    skeleton_gt = vis_skeleton(
                        joints=jts_clean[bs_idx, t],
                        limbs=LIMBS_BODY_SMPL,
                        add_trans=np.array([0, 0.8, 2.5])
                    )
                    geometries.extend(skeleton_gt)
                    
                    body_mesh_clean = o3d.geometry.TriangleMesh()
                    body_mesh_clean.vertices = o3d.utility.Vector3dVector(verts_clean[bs_idx, t])
                    body_mesh_clean.triangles = o3d.utility.Vector3iVector(self.smplx_model.faces)
                    body_mesh_clean.compute_vertex_normals()
                    body_mesh_clean.paint_uniform_color(COLOR_GT)
                    transform = np.identity(4)
                    transform[1, 3] = 2.0
                    body_mesh_clean.transform(transform)
                    geometries.append(body_mesh_clean)
                
                # Reconstructed skeletons and meshes
                for idx, (jts_rec, verts_rec, color_vis, color_occ, mask_ids) in enumerate(zip(
                    jts_rec_tries, verts_rec_tries,
                    data.color_vis_tries, data.color_occ_tries,
                    data.occluded_joint_ids_tries
                )):
                    skeleton_rec = vis_skeleton(
                        joints=jts_rec[bs_idx, t],
                        limbs=LIMBS_BODY_SMPL,
                        add_trans=np.array([0, -idx * 0.5, 2.5]),
                        mask_scheme=self.test_config.test_mask_action,
                        cur_mask_joint_id=mask_ids.tolist(),
                        color_occ=np.array(color_occ),
                        color_vis=np.array(color_vis)
                    )
                    geometries.extend(skeleton_rec)
                    
                    body_mesh_rec = o3d.geometry.TriangleMesh()
                    body_mesh_rec.vertices = o3d.utility.Vector3dVector(verts_rec[bs_idx, t])
                    body_mesh_rec.triangles = o3d.utility.Vector3iVector(self.smplx_model.faces)
                    body_mesh_rec.compute_vertex_normals()
                    body_mesh_rec.paint_uniform_color(color_occ)
                    transform = np.identity(4)
                    transform[1, 3] = -idx * 0.5
                    body_mesh_rec.transform(transform)
                    geometries.append(body_mesh_rec)
                
                # Add all geometries
                for geom in geometries:
                    self.vis.add_geometry(geom)
                
                # Update camera
                if self.fix_cam:
                    ctr = self.vis.get_view_control()
                    cam_param = ctr.convert_to_pinhole_camera_parameters()
                    cam_param = update_cam(cam_param, self.T_bc)
                    ctr.convert_from_pinhole_camera_parameters(cam_param)
                
                self.vis.poll_events()
                self.vis.update_renderer()
                
                # Capture frame
                temp_path = os.path.join(temp_dir, f'frame_{bs_idx:04d}_{t:04d}.png')
                self.vis.capture_screen_image(temp_path)
                
                # Add legend
                self._add_legend(temp_path, data, len(jts_rec_tries))
                frames.append(temp_path)
                
                # Remove geometries
                for geom in geometries:
                    self.vis.remove_geometry(geom)
    
    def _add_legend(self, image_path: str, data: VisualizationData, num_tries: int):
        """Add legend to captured frame."""
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        box_size = 20
        padding = 10
        x, y = padding, padding
        
        for idx in range(num_tries):
            color_vis = tuple([int(c * 255) for c in data.color_vis_tries[idx]])
            color_occ = tuple([int(c * 255) for c in data.color_occ_tries[idx]])
            
            draw.rectangle([x, y, x + box_size, y + box_size], fill=color_vis)
            draw.rectangle([x, y + box_size + 5, x + box_size, y + 2 * box_size + 5], fill=color_occ)
            
            text_x = x + box_size + padding
            draw.multiline_text((text_x, y), f"Rank {idx} VIS", fill='black')
            draw.multiline_text((text_x, y + box_size + 5), f"Rank {idx} OCC", fill='black')
            
            y += 2 * box_size + padding
        
        img.save(image_path)
    
    def cleanup(self) -> None:
        """Release Open3D resources."""
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None
        self._is_initialized = False
