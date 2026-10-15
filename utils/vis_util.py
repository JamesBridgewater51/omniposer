import open3d as o3d
from utils.other_utils import *
import random
import pyrender
import trimesh
from PIL import Image
from einops import einsum
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt

import numpy as np

from typing import List
from typing import Optional
import open3d as o3d
import time
import torch
from cprint import cprint

COLOR_VIS_O3D = [random.random(), random.random(), random.random()]
COLOR_OCC_O3D = [random.random(), random.random(), random.random()]
COLOR_GT_O3D = [255 / 255, 102 / 255, 102 / 255]
COLOR_CONTACT_1 = [0 / 255, 128 / 255, 0 / 255]
COLOR_CONTACT_0 = [128 / 255, 0 / 255, 0 / 255]

def render_prox_overlay(
    output_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    transf_matrix: np.ndarray,
    cam2world: np.ndarray,
    focal_length: np.ndarray,
    camera_center: np.ndarray,
    bg_image_path: str
):
    """
    Renders SMPL vertices overlaid on a background image for the PROX dataset.

    Args:
        output_path (str): Path to save the rendered frame.
        vertices (np.ndarray): SMPL vertices in canonical coordinate system, shape [V, 3].
        faces (np.ndarray): SMPL faces, shape [F, 3].
        transf_matrix (np.ndarray): 4x4 matrix transforming from world to canonical space.
        cam2world (np.ndarray): 4x4 matrix transforming from camera to world space.
        focal_length (np.ndarray): Camera focal lengths [fx, fy].
        camera_center (np.ndarray): Camera principal point [cx, cy].
        bg_image_path (str): Path to the background RGB image.
    """
    # 1. Load background image and get dimensions
    bg_image = Image.open(bg_image_path).convert("RGB")
    W, H = bg_image.size

    # 2. Define material for the body mesh
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        alphaMode='BLEND',
        baseColorFactor=(0.3, 0.7, 0.3, 0.8)  # Semi-transparent green
    )

    # 3. Transform vertices from canonical space to camera space
    # Inverse of canonical transform: canonical -> world
    inv_transf_matrix = np.linalg.inv(transf_matrix)
    # Inverse of camera transform: world -> camera
    world2cam = np.linalg.inv(cam2world)

    # Apply transformations
    verts_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    verts_cam_homo = (world2cam @ inv_transf_matrix @ verts_homo.T).T
    verts_cam = verts_cam_homo[:, :3]

    # 4. Create the pyrender mesh and scene
    body_mesh = trimesh.Trimesh(vertices=verts_cam, faces=faces)
    render_mesh = pyrender.Mesh.from_trimesh(body_mesh, material=material)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(render_mesh)

    # 5. Set up the camera and light
    camera = pyrender.IntrinsicsCamera(
        fx=focal_length[0], fy=focal_length[1],
        cx=camera_center[0], cy=camera_center[1]
    )
    # The camera pose is identity because vertices are already in camera space
    scene.add(camera, pose=np.eye(4))

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))

    # 6. Render the scene
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    # 7. Composite the render over the background image
    render_image = Image.fromarray(color, 'RGBA')
    bg_image.paste(render_image, (0, 0), render_image)
    
    # 8. Save the final image
    bg_image.save(output_path)

def vis_skeleton(joints, limbs,
                 add_trans=None,
                 mask_scheme=None, cur_mask_joint_id=None,
                 start=0, end=0, t=0,
                 color_occ=COLOR_OCC_O3D, color_vis=COLOR_VIS_O3D):
    """
    Input:
        joints: numpy [n_joints, 3], joint positions 
        limbs: limb topology
        add_trans: numpy [3], additional translation for skeleton
        mask_scheme: occlusion mask scheme, 'lower'/'full'/'video'
        start/end: start/end frame for full-body occlusion mask if mask_scheme=='full'
        t: current timestep (for full-body occlusion visualization)
    Output:
        skeleton_list: open3d body skeleton (a list of open3d arrows)
    """
    skeleton_list = []
    for limb in limbs:
        p0 = joints[limb[0]]
        p1 = joints[limb[1]]
        
        # Variables to store masking status if cur_mask_joint_id is used
        is_p0_masked = False
        is_p1_masked = False
        
        # Flag to indicate if this limb was processed as two half-limbs
        processed_as_half_limbs = False

        # Check for half-limb visualization if mask_scheme requires it
        if mask_scheme is not None and mask_scheme != 'full' and cur_mask_joint_id is not None:
            is_p0_masked = limb[0] in cur_mask_joint_id
            is_p1_masked = limb[1] in cur_mask_joint_id
            
            # If exactly one end is masked, draw two half-limbs with different colors
            if is_p0_masked != is_p1_masked: # XOR condition
                mid_point = (p0 + p1) / 2.0
                
                # Create first half-limb (p0 to mid_point)
                segment_vec1 = mid_point - p0
                segment_len1 = np.linalg.norm(segment_vec1)
                if segment_len1 > 1e-6: # Avoid creating zero-length arrows
                    arrow_half1 = o3d.geometry.TriangleMesh.create_arrow(
                        cylinder_radius=0.03, cone_radius=0.001,
                        cylinder_height=segment_len1, cone_height=0.001
                    )
                    mat1 = rotation_matrix_from_vectors(np.array([0, 0, 1]), segment_vec1)
                    transformation1 = np.identity(4)
                    transformation1[0:3, 0:3] = mat1
                    transformation1[:3, 3] = p0
                    if add_trans is not None:
                        transformation1[0:3, 3] += add_trans
                    arrow_half1.transform(transformation1)
                    arrow_half1.paint_uniform_color(color_occ if is_p0_masked else color_vis)
                    arrow_half1.compute_vertex_normals()
                    skeleton_list.append(arrow_half1)

                # Create second half-limb (mid_point to p1)
                segment_vec2 = p1 - mid_point
                segment_len2 = np.linalg.norm(segment_vec2)
                if segment_len2 > 1e-6: # Avoid creating zero-length arrows
                    arrow_half2 = o3d.geometry.TriangleMesh.create_arrow(
                        cylinder_radius=0.03, cone_radius=0.001,
                        cylinder_height=segment_len2, cone_height=0.001
                    )
                    mat2 = rotation_matrix_from_vectors(np.array([0, 0, 1]), segment_vec2)
                    transformation2 = np.identity(4)
                    transformation2[0:3, 0:3] = mat2
                    transformation2[:3, 3] = mid_point # Start from midpoint
                    if add_trans is not None:
                        transformation2[0:3, 3] += add_trans
                    arrow_half2.transform(transformation2)
                    arrow_half2.paint_uniform_color(color_occ if is_p1_masked else color_vis)
                    arrow_half2.compute_vertex_normals()
                    skeleton_list.append(arrow_half2)
                
                processed_as_half_limbs = True
        
        if processed_as_half_limbs:
            continue # Move to the next limb

        # If not processed as half-limbs, create a single full arrow for the limb
        bone_length = np.linalg.norm(p1 - p0)
        if bone_length < 1e-6: # Skip zero-length bones
            continue

        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.03, cone_radius=0.001,
                                                       cylinder_height=bone_length, cone_height=0.001)
        
        # Ensure p1 - p0 is not a zero vector before calling rotation_matrix_from_vectors
        # The bone_length check above should handle this.
        direction_vector = p1 - p0
        mat = rotation_matrix_from_vectors(np.array([0, 0, 1]), direction_vector)
        
        transformation = np.identity(4)
        transformation[0:3, 0:3] = mat
        transformation[:3, 3] = p0
        if add_trans is not None:
            transformation[0:3, 3] += add_trans
        arrow.transform(transformation)

        # Original coloring logic for the full arrow
        if mask_scheme is None:
            arrow.paint_uniform_color(COLOR_GT_O3D)
        elif mask_scheme == 'full':
            if t >= start and t < end:
                arrow.paint_uniform_color(color_occ)
            else:
                arrow.paint_uniform_color(color_vis)
        else: # This is for schemes like 'lower'/'video' etc.
              # And we are here if:
              # 1. cur_mask_joint_id was None for this scheme.
              # 2. cur_mask_joint_id was not None, but both joints had the same mask status (both masked or neither masked).
            if cur_mask_joint_id is not None:
                # is_p0_masked and is_p1_masked were computed earlier.
                # Since (is_p0_masked != is_p1_masked) was false, they are equal.
                if is_p0_masked: # True if limb[0] was in cur_mask_joint_id (implies both are masked)
                    arrow.paint_uniform_color(color_occ)
                else: # False if limb[0] was not in cur_mask_joint_id (implies neither are masked)
                    arrow.paint_uniform_color(color_vis)
            else: # cur_mask_joint_id was None for this scheme.
                arrow.paint_uniform_color(color_vis) # Default to visible
        
        arrow.compute_vertex_normals()
        skeleton_list.append(arrow)
        
    return skeleton_list


def vis_foot_contact(joints, contact_lbl, add_trans=None):
    """
    Input:
        joints: numpy [n_joints, 3], joint positions
        contact_lbl: numpy, [4]
        add_trans: numpy [3], additional translation for skeleton
    Output:
        foot_sphere_list: sphere of foot joints with color denoting contact labels (a list of open3d spheres)
    """
    foot_sphere_list = []
    for idx, foot_joint_idx in enumerate([7, 10, 8, 11]): # Assuming these are the correct indices for feet
        transformation = np.identity(4)
        transformation[:3, 3] = joints[foot_joint_idx]
        if add_trans is not None: # <--- add_trans is used here
            transformation[0:3, 3] += add_trans
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        if contact_lbl[idx] == 1:
            sphere.paint_uniform_color(COLOR_CONTACT_1)  # green, in contact
        else:
            sphere.paint_uniform_color(COLOR_CONTACT_0)  # red, not in contact
        sphere.transform(transformation)
        sphere.compute_vertex_normals() # Good practice to add this for consistent rendering
        foot_sphere_list.append(sphere)
    return foot_sphere_list


def blend_with_background(image: np.ndarray, background_color: tuple) -> np.ndarray:
    """Blend RGBA image with solid background color."""
    if image.shape[2] != 4:
        return image

    alpha = image[:, :, 3:4] / 255.0
    background = np.ones_like(image[:, :, :3]) * np.array(background_color) * 255
    blended = image[:, :, :3] * alpha + background * (1 - alpha)
    return blended.astype(np.uint8)


def create_skeleton_point_cloud(
    joints_wrt_world: np.ndarray,
    visible_joints_mask: np.ndarray,
    input_smplh: bool = False,
    num_samples_per_bone: int = 100,
    return_colors: Optional[bool] = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | np.ndarray
):
    """Create a point cloud representing the skeleton by densely sampling along bones.

    Args:
        joints_wrt_world: Joint positions in world coordinates, shape [num_joints, 3]
        visible_joints_mask: Boolean mask indicating valid joints, shape [num_joints]
        input_smplh: If True, input uses SMPLH convention (52 joints), otherwise COCO (17 joints)
        num_samples_per_bone: Number of points to sample along each bone
        return_colors: If True, returns colors for points (green=valid, red=invalid)

    Returns:
        points: Sampled 3D points forming the skeleton, shape [num_points, 3]
        colors: Optional point colors, shape [num_points, 3]
    """
    # Verify input dimensions based on convention
    if input_smplh:
        assert joints_wrt_world.shape[0] == 22, (
            f"SMPLH joints should have 22 joints, got {joints_wrt_world.shape[0]}"
        )
        # SMPLH kinematic tree (parent indices for each joint)
        # Only includes body joints, not hand joints
        kintree = SMPLH_KINTREE
    else:
        assert joints_wrt_world.shape[0] == 17, (
            f"COCO joints should have 17 joints, got {joints_wrt_world.shape[0]}"
        )
        # COCO kinematic tree (parent indices for each joint)
        kintree = EGOEXO4D_BODYPOSE_KINTREE_PARENTS

    # Initialize list to store sampled points and colors
    vis_pts = []
    invis_pts = []
    visc = [] if return_colors else None
    invisc = [] if return_colors else None

    # Define colors
    green = np.array([0, 255, 0]).astype(np.uint8)
    red = np.array([0, 255, 0]).astype(np.uint8)

    # Iterate through kinematic tree
    for joint_idx, parent_idx in enumerate(kintree):
        if parent_idx == -1:
            continue

        # Get joint positions
        start = joints_wrt_world[parent_idx]
        end = joints_wrt_world[joint_idx]

        # Skip if either contains NaN
        if np.isnan(start).any() or np.isnan(end).any():
            continue

        # Get validity of both joints
        start_valid = visible_joints_mask[parent_idx]
        end_valid = visible_joints_mask[joint_idx]

        # Sample points along bone
        for t in np.linspace(0, 1, num_samples_per_bone):
            point = start + t * (end - start)
            if start_valid and end_valid:
                vis_pts.append(point)
                if return_colors:
                    visc.append(green)
            elif not start_valid and not end_valid:
                invis_pts.append(point)
                if return_colors:
                    invisc.append(red)
            else:
                if (start_valid and t > 0.5) or (end_valid and t <= 0.5):
                    invis_pts.append(point)
                    if return_colors:
                        invisc.append(red)
                else:
                    vis_pts.append(point)
                    if return_colors:
                        visc.append(green)

    if len(vis_pts) == 0:
        vis_ret = (
            (np.zeros((0, 3)), np.zeros((0, 3))) if return_colors else np.zeros((0, 3))
        )
    else:
        vis_ret = (
            (np.stack(vis_pts), np.stack(visc)) if return_colors else np.stack(vis_pts)
        )

    if len(invis_pts) == 0:
        invis_ret = (
            (np.zeros((0, 3)), np.zeros((0, 3))) if return_colors else np.zeros((0, 3))
        )
    else:
        invis_ret = (
            (np.stack(invis_pts), np.stack(invisc))
            if return_colors
            else np.stack(invis_pts)
        )

    return vis_ret, invis_ret


def plot_synchronous_3d_animations_multi_modal(
    num_of_sets: int,
    rows: int,
    cols: int,
    data: list,  # List of numpy arrays, one per modality
    modality_names: list = None,  # Optional list of names for modalities
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors

    """
    Visualizes multiple sets of synchronous 3D point cloud animations from multiple
    modalities using Plotly subplots.

    Args:
        num_of_sets (int): The total number of data sets to visualize.
        rows (int): The number of rows in the subplot grid.
        cols (int): The number of columns in the subplot grid.
        data (list): A list containing numpy arrays, one for each modality.
                     Each array must have the shape: (num_sets, timesteps, num_kpts, 3).
        modality_names (list, optional): A list of strings corresponding to the name
                                         of each modality for the legend. If None,
                                         modalities will be named "Mod 1", "Mod 2", etc.
                                         Defaults to None.
    """

    if rows * cols < num_of_sets:
        raise ValueError(
            f"Grid size ({rows}x{cols}) is too small for {num_of_sets} sets.",
        )

    num_modalities = len(data)
    if num_modalities == 0:
        raise ValueError("The 'data' list cannot be empty.")

    # --- Validate Data Shapes and Modality Names ---
    if modality_names and len(modality_names) != num_modalities:
        raise ValueError(
            f"Number of modality names ({len(modality_names)}) must match number of data arrays ({num_modalities}).",
        )

    # Use the shape of the first modality's data as reference
    ref_shape = data[0].shape
    if ref_shape[0] != num_of_sets:
        raise ValueError(
            f"Mismatch between num_of_sets ({num_of_sets}) and data dimension 0 ({ref_shape[0]}) for modality 0.",
        )
    if len(ref_shape) != 4 or ref_shape[3] != 3:
        raise ValueError(
            "Data arrays must be 4-dimensional (sets, timesteps, kpts, 3).",
        )

    timesteps = ref_shape[1]
    _num_kpts = ref_shape[2]

    # Check consistency across all modalities
    for mod_idx in range(1, num_modalities):
        mod_shape = data[mod_idx].shape
        if mod_shape[0] != num_of_sets:
            raise ValueError(
                f"Mismatch between num_of_sets ({num_of_sets}) and data dimension 0 ({mod_shape[0]}) for modality {mod_idx}.",
            )
        if mod_shape[1] != ref_shape[1]:
            raise ValueError(
                f"Shape mismatch (timesteps) between modality 0 {ref_shape[1:]} and modality {mod_idx} {mod_shape[1:]}.",
            )
        if mod_shape[-1] != ref_shape[-1]:
            raise ValueError(
                f"Shape mismatch (dims) between modality 0 {ref_shape[2:]} and modality {mod_idx} {mod_shape[2:]}.",
            )

    # --- Setup Visuals ---
    # Use a qualitative color scheme from Plotly
    colors = plotly.colors.qualitative.Vivid
    # Define a list of marker symbols
    markers = [
        "circle",
        "circle-open",
        "cross",
        "diamond",
        "diamond-open",
        "square",
        "square-open",
    ]

    # Generate default modality names if not provided
    if modality_names is None:
        modality_names = [f"Mod {i + 1}" for i in range(num_modalities)]

    # --- Create Figure ---
    specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=[f"Set {i + 1}" for i in range(num_of_sets)],
    )

    # --- Add Initial Traces (Timestep 0) ---
    # The order traces are added here *must* be maintained when creating frames
    for i in range(num_of_sets):  # Iterate through sets first
        subplot_row = (i // cols) + 1
        subplot_col = (i % cols) + 1
        for mod_idx in range(num_modalities):  # Iterate through modalities second
            mod_data_array = data[mod_idx]
            data_t0 = mod_data_array[
                i,
                0,
                :,
                :,
            ]  # Data for set i, modality mod_idx, timestep 0

            color = colors[mod_idx % len(colors)]  # Cycle through colors
            marker_symbol = markers[mod_idx % len(markers)]  # Cycle through markers
            name = f"Set {i + 1} {modality_names[mod_idx]}"

            fig.add_trace(
                go.Scatter3d(
                    x=data_t0[:, 0],
                    y=data_t0[:, 1],
                    z=data_t0[:, 2],
                    mode="markers",
                    marker=dict(color=color, size=1.5, symbol=marker_symbol),
                    name=name,
                ),
                row=subplot_row,
                col=subplot_col,
            )

    # --- Create Animation Frames ---
    frames = []
    num_traces_total = num_of_sets * num_modalities  # Total traces added initially

    for t in range(timesteps):  # Loop through each timestep for the frame
        frame_data = []  # Holds the data for *all* traces for this specific frame/timestep

        # Iterate in the *exact same order* as traces were added initially
        for i in range(num_of_sets):  # Sets first
            for mod_idx in range(num_modalities):  # Modalities second
                mod_data_array = data[mod_idx]
                data_t = mod_data_array[
                    i,
                    t,
                    :,
                    :,
                ]  # Data for set i, modality mod_idx, timestep t

                # Append only the coordinate data for this trace for this frame
                frame_data.append(
                    go.Scatter3d(x=data_t[:, 0], y=data_t[:, 1], z=data_t[:, 2]),
                )

        frames.append(
            go.Frame(
                data=frame_data,  # Contains data for all num_traces_total traces
                name=f"t={t}",
                # Specify that this frame updates all the initially added traces
                traces=list(range(num_traces_total)),
            ),
        )

    fig.frames = frames

    # --- Configure Layout and Animation Controls --- (Identical to previous version)

    # Buttons
    fig.update_layout(
        title_text="Synchronous Multi-Modality 3D Point Cloud Animations",
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            },
        ],
    )

    # Slider
    sliders = [{"pad": {"t": 30, "b": 10}, "len": 0.9, "x": 0.1, "y": 0, "steps": []}]
    for t in range(timesteps):
        slider_step = {
            "args": [
                [f"t={t}"],
                {
                    "frame": {"duration": 100, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": f"Time {t}",
            "method": "animate",
        }
        sliders[0]["steps"].append(slider_step)

    fig.update_layout(sliders=sliders)

    # --- Customize Scene Layouts (Identical to previous version) ---
    scene_config = dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectratio=dict(x=1, y=1, z=1),
        aspectmode="data",
        # Consider setting axis ranges based on global min/max across all data if needed
    )
    layout_update_dict = {}
    for i in range(num_of_sets):
        scene_id = f"scene{i + 1}" if i > 0 else "scene"
        layout_update_dict[scene_id] = scene_config

    fig.update_layout(**layout_update_dict)
    fig.update_layout(
        height=max(400, rows * 350),
        width=max(600, cols * 350),
    )  # Slightly increase size per plot
    fig.update_layout(hovermode="closest", legend_title_text="Modalities")

    # Show legend
    fig.update_layout(showlegend=True)

    # --- Show Figure ---
    fig.show()


def test_plot_synchronous_3d_animations_mutli_modal():
    import math

    # --- Generate Example Data ---
    NUM_SETS = 6  # Reduced for clarity, use 8 if preferred
    TIMESTEPS = 60
    NUM_KPTS = 15

    # Create data for 3 modalities
    num_modalities_example = 3
    all_data = []  # List to hold data arrays for each modality

    # Modality 1: Expanding/Contracting Sphere
    data_mod1 = np.zeros((NUM_SETS, TIMESTEPS, NUM_KPTS, 3))
    # Modality 2: Figure-eight Motion
    data_mod2 = np.zeros((NUM_SETS, TIMESTEPS, NUM_KPTS, 3))
    # Modality 3: Random Walk within a Box
    data_mod3 = np.zeros((NUM_SETS, TIMESTEPS, NUM_KPTS, 3))

    # Generate points on a sphere for modality 1 base
    phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians
    indices = np.arange(0, NUM_KPTS)
    z_sphere = 1 - (indices / float(NUM_KPTS - 1)) * 2  # z goes from 1 to -1
    radius_sphere = np.sqrt(1 - z_sphere * z_sphere)  # radius at z
    theta_sphere = phi * indices  # golden angle increment
    x_sphere_base = radius_sphere * np.cos(theta_sphere)
    y_sphere_base = radius_sphere * np.sin(theta_sphere)

    for s in range(NUM_SETS):
        # Set-specific parameters
        set_offset = np.array(
            [s % 3 * 10, s // 3 * 10, 0],
        )  # Basic spatial offset for sets
        set_speed_factor = 1.0 + s * 0.1
        set_scale_factor = 5.0 + s  # Size factor

        # Initialize positions for random walk (Modality 3)
        current_pos_mod3 = (
            np.random.rand(NUM_KPTS, 3) * set_scale_factor * 0.5
            - set_scale_factor * 0.25
            + set_offset
        )

        for t in range(TIMESTEPS):
            time_angle = t * 0.1 * set_speed_factor

            # Modality 1: Pulsating Sphere
            scale = set_scale_factor * (
                1 + 0.3 * np.sin(time_angle * 2)
            )  # Pulsating radius
            data_mod1[s, t, :, 0] = x_sphere_base * scale + set_offset[0]
            data_mod1[s, t, :, 1] = y_sphere_base * scale + set_offset[1]
            data_mod1[s, t, :, 2] = z_sphere * scale + set_offset[2]

            # Modality 2: Figure Eight
            x_fig8 = set_scale_factor * 0.8 * np.sin(time_angle)
            y_fig8 = (
                set_scale_factor * 0.5 * np.sin(time_angle * 2)
            )  # Double frequency for figure eight
            z_fig8 = np.linspace(
                -set_scale_factor * 0.2,
                set_scale_factor * 0.2,
                NUM_KPTS,
            )  # Spread points in Z
            data_mod2[s, t, :, 0] = x_fig8 + set_offset[0]
            data_mod2[s, t, :, 1] = y_fig8 + set_offset[1]
            data_mod2[s, t, :, 2] = (
                z_fig8 + set_offset[2] + t * 0.05
            )  # Slow drift upwards

            # Modality 3: Random Walk step
            step = (
                (np.random.rand(NUM_KPTS, 3) - 0.5) * 0.1 * set_scale_factor
            )  # Small random step
            current_pos_mod3 += step
            # Simple boundary reflection
            box_size = set_scale_factor * 0.6
            min_bound = set_offset - box_size / 2
            max_bound = set_offset + box_size / 2
            current_pos_mod3 = np.clip(
                current_pos_mod3,
                min_bound,
                max_bound,
            )  # Keep within bounds (simple clip)
            data_mod3[s, t, :, :] = current_pos_mod3

    # Add generated data arrays to the list
    all_data.append(data_mod1)
    all_data.append(data_mod2)
    all_data.append(data_mod3)

    # Optional: Define names for the modalities
    mod_names = ["SpherePulse", "FigureEight", "RandWalk"]

    # --- Define grid layout ---
    total_plots = NUM_SETS
    cols_layout = math.ceil(math.sqrt(total_plots))
    rows_layout = math.ceil(total_plots / cols_layout)
    print(
        f"Using layout: {rows_layout} rows x {cols_layout} cols for {num_modalities_example} modalities.",
    )

    # --- Call the plotting function ---
    plot_synchronous_3d_animations_multi_modal(
        num_of_sets=NUM_SETS,
        rows=rows_layout,
        cols=cols_layout,
        data=all_data,  # Pass the list of numpy arrays
        modality_names=mod_names,  # Pass the list of names
    )

    print("Plot generation complete. Check the displayed Plotly figure.")


def OpenGLlookAt(eye, target, up):
    """
    Compute a view matrix looking from `eye` towards `target` with the specified `up` vector.

    Parameters
    ----------
    eye : ndarray of dtype float64
        The position of the camera in world space. Must be a 1D array of shape (3,).
    target : ndarray of dtype float64
        The point in world space where the camera is looking. Must be a 1D array of shape (3,).
    up : ndarray
        The up direction for the camera in world space. Must be a 1D array of shape (3,).

    Returns
    -------
    view_matrix (T_cam_world) : ndarray
        A 4x4 view matrix as a NumPy array.

    Notes
    -----
    This function creates a view matrix for a right-handed coordinate system with a direct
    analogy to the traditional OpenGL `lookAt` function.

    Examples
    --------
    >>> eye = np.array([1.0, 1.0, 1.0])
    >>> target = np.array([0.0, 0.0, 0.0])
    >>> up = np.array([0.0, 1.0, 0.0])
    >>> view_matrix = lookAt(eye, target, up)
    >>> np.linalg.inv(view_matrix).shape
    (4, 4)

    """
    direction = eye - target
    if direction.any():
        direction /= np.linalg.norm(direction)

    right = np.cross(up, direction)
    if right.any():
        right /= np.linalg.norm(right)

    up = np.cross(direction, right)
    if up.any():
        up /= np.linalg.norm(up)

    R = np.array([right, up, direction])

    T = np.eye(4)
    T[0:3, 3] = -eye

    M = np.eye(4)
    M[:3, :3] = R
    view_matrix = M @ T

    return view_matrix

def visualize_temporal_sequence(points_sequence, alpha=0.2, point_size=5):
    """
    Visualize a 3D point sequence with temporal color gradients.
    
    Args:
        points_sequence: np.ndarray of shape [seq_len, num_points, 3]
        alpha: Transparency for each point (0-1)
        point_size: Size of each point in the scatter plot
    """
    seq_len, num_points, _ = points_sequence.shape
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color gradient (from blue to red)
    colors = plt.cm.jet(np.linspace(0, 1, seq_len))
    
    # Plot all points with temporal coloring
    for t in range(seq_len):
        ax.scatter(
            points_sequence[t, :, 0],  # x
            points_sequence[t, :, 1],  # y
            points_sequence[t, :, 2],  # z
            c=np.tile(colors[t][None, :], (num_points, 1)),  # color for all points at this timestep
            s=point_size,
            alpha=alpha,
            edgecolors='none'
        )
    
    # Add colorbar to show temporal progression
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=seq_len-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Timestep')
    cbar.set_ticks(np.linspace(0, seq_len-1, 5))
    cbar.set_ticklabels(np.linspace(0, seq_len-1, 5, dtype=int))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.set_title(f'Temporal Point Cloud Sequence ({seq_len} timesteps)')
    plt.tight_layout()
    plt.show()
