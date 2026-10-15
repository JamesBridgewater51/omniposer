import torch
import numpy as np
import smplx # Make sure smplx is installed and importable
from einops import einsum
from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, transform_mat
from smplx.body_models import SMPLX # For type hinting and accessing constants
from utils import dist_util
import torch.nn.functional as F
from utils.konia_transform import rotation_matrix_to_angle_axis
from typing import Optional, Tuple
from torch import Tensor

def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32
) -> Tensor:
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, transforms, rel_transforms



# --- Main Functions ---
def local_to_global_pose_rotations(
    smplx_model: SMPLX,
    global_orient_aa: torch.Tensor,
    body_pose_aa: torch.Tensor,
    return_joint_positions: bool = False # Added for visualization
) -> Tuple[torch.Tensor,torch.Tensor,  Optional[torch.Tensor]]:
    """
    Converts local body pose (root axis-angle, local body joint axis-angles)
    to global rotation matrices and optionally joint positions for all SMPL-X joints.
    Assumes hand and facial poses are zero (or will be taken from smplx_model.pose_mean).

    Args:
        smplx_model (SMPLX): An instance of the SMPLX model.
        global_orient_aa (torch.Tensor): Root orientation as axis-angles (batch_size, 3).
        body_pose_aa (torch.Tensor): Local axis-angle rotations for SMPL-X body joints
                                     (batch_size, smplx_model.NUM_BODY_JOINTS * 3).
        return_joint_positions (bool): If True, also returns global joint positions.

    Returns:
        torch.Tensor | None: Global joint positions (batch_size, 55, 3) if return_joint_positions is True.
        torch.Tensor: Global rotation matrices for all 55 SMPL-X joints
                      (batch_size, 55, 3, 3).
        torch.Tensor: Rotation matrices relative to root for all 55 SMPL-X joints
                      (batch_size, 55, 3, 3).
    """
    batch_size = global_orient_aa.shape[0]
    device = global_orient_aa.device
    dtype = global_orient_aa.dtype

    global_orient_aa = global_orient_aa.reshape(batch_size, 3)
    body_pose_aa = body_pose_aa.reshape(batch_size, smplx_model.NUM_BODY_JOINTS * 3)

    jaw_pose_zeros = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    leye_pose_zeros = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    reye_pose_zeros = torch.zeros((batch_size, 3), device=device, dtype=dtype)

    if smplx_model.use_pca:
        left_hand_pose_input = torch.zeros((batch_size, smplx_model.num_pca_comps), device=device, dtype=dtype)
        right_hand_pose_input = torch.zeros((batch_size, smplx_model.num_pca_comps), device=device, dtype=dtype)
    else:
        left_hand_pose_input = torch.zeros((batch_size, smplx_model.NUM_HAND_JOINTS * 3), device=device, dtype=dtype)
        right_hand_pose_input = torch.zeros((batch_size, smplx_model.NUM_HAND_JOINTS * 3), device=device, dtype=dtype)

    if smplx_model.use_pca:
        _left_hand_pose_aa = torch.einsum(
            'bi,ij->bj', [left_hand_pose_input, smplx_model.left_hand_components]
        )
        _right_hand_pose_aa = torch.einsum(
            'bi,ij->bj', [right_hand_pose_input, smplx_model.right_hand_components]
        )
    else:
        _left_hand_pose_aa = left_hand_pose_input
        _right_hand_pose_aa = right_hand_pose_input

    full_pose_flat_aa = torch.cat([
        global_orient_aa, body_pose_aa, jaw_pose_zeros,
        leye_pose_zeros, reye_pose_zeros, _left_hand_pose_aa, _right_hand_pose_aa
    ], dim=1)

    posed_full_pose_aa = full_pose_flat_aa + smplx_model.pose_mean.unsqueeze(0).to(device)

    num_total_joints = smplx_model.J_regressor.shape[0]
    rot_mats_local = batch_rodrigues(posed_full_pose_aa.view(-1, 3)).view(
        batch_size, num_total_joints, 3, 3
    )

    betas_zeros = torch.zeros((batch_size, smplx_model.num_betas), device=device, dtype=dtype)
    expression_zeros = torch.zeros((batch_size, smplx_model.num_expression_coeffs), device=device, dtype=dtype)
    shape_components = torch.cat([betas_zeros, expression_zeros], dim=-1)
    
    combined_blend_shapes = torch.cat([smplx_model.shapedirs, smplx_model.expr_dirs], dim=-1).to(device)
    v_template_batch = smplx_model.v_template.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    v_shaped = v_template_batch + blend_shapes(shape_components, combined_blend_shapes)
    
    t_pose_joints = vertices2joints(smplx_model.J_regressor.to(device), v_shaped)
    parents = smplx_model.parents.to(device)

    J_transformed, T_world_joints, T_rest_pose_joints = batch_rigid_transform(rot_mats_local, t_pose_joints, parents, dtype=dtype)
    T_root_joints = einsum(T_world_joints[:, 0:1, :, :].inverse(), T_world_joints, 'B M J K, B P K L -> B P J L')

    if return_joint_positions:
        return J_transformed, T_world_joints[:, :, :3, :3], T_root_joints[:, :, :3, :3]
    else:
        return T_world_joints[:, :, :3, :3], T_root_joints[:, :, :3, :3]

def global_to_local_pose_rotations(
    smplx_model: SMPLX,
    global_joint_rotations_matrix: torch.Tensor # Shape (batch_size, N, 3, 3) where N is 55 or 22
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts global joint rotation matrices to local root orientation (axis-angle)
    and local body pose (axis-angle).
    Handles cases for full 55 SMPL-X joints or 22 main body joints.

    Args:
        smplx_model (SMPLX): An instance of the SMPLX model.
        global_joint_rotations_matrix (torch.Tensor): Global rotation matrices
            (batch_size, num_input_joints, 3, 3). num_input_joints can be 55 or 22.

    Returns:
        torch.Tensor: Root orientation as axis-angles (batch_size, 3).
        torch.Tensor: Local axis-angle rotations for SMPL-X body joints
                      (batch_size, smplx_model.NUM_BODY_JOINTS * 3), i.e., 21*3=63.
    """
    batch_size = global_joint_rotations_matrix.shape[0]
    num_input_joints = global_joint_rotations_matrix.shape[1]
    device = global_joint_rotations_matrix.device
    dtype = global_joint_rotations_matrix.dtype

    # ==== new code -=====
    if num_input_joints == 22:
        # Use parent definitions for the first 22 joints
        # Assumes the 22 joints are the standard SMPL/SMPL-H body joints
        # (Pelvis, LHip, RHip, Spine1, ..., RWrist)
        # SMPLH_KINTREE is often used for this. smplx_model.parents is for full 55.
        # We need the kinematic tree for these 22 joints.
        # smplx_model.parents[:22] would give the parents for the first 22 joints in the SMPL-X ordering.
        parents_for_kinematics = smplx_model.parents[:num_input_joints].to(device)
        
        # The pose_mean is for the full 55 joints. We need the part relevant to these 22.
        # SMPL-X pose_mean is (global_orient, body_pose (21), jaw, leye, reye, lhand, rhand)
        # The first 22 entries of pose_mean are:
        # pose_mean[0:3] -> global_orient
        # pose_mean[3:3+21*3] -> body_pose (for joints 1-21 of SMPL-X, which are LHip to RWrist)
        # So, for 22 joints (0 to 21), the relevant pose_mean parts are:
        # joint 0 (root): pose_mean[0:3]
        # joints 1-21 (body): pose_mean[3 : 3 + 21*3]
        # We need to construct a pose_mean_subset for these 22 joints.
        pose_mean_subset = torch.cat([
            smplx_model.pose_mean[:3], # global_orient for joint 0
            smplx_model.pose_mean[3 : 3 + (num_input_joints - 1) * 3] # body_pose for joints 1 to 21
        ], dim=0).unsqueeze(0).to(device) # (1, 22*3)

    elif num_input_joints == smplx_model.J_regressor.shape[0]: # Should be 55 for full SMPL-X
        parents_for_kinematics = smplx_model.parents.to(device)
        pose_mean_subset = smplx_model.pose_mean.unsqueeze(0).to(device) # (1, 55*3)
    else:
        raise ValueError(f"Unsupported number of input joints: {num_input_joints}. Expected 22 or {smplx_model.J_regressor.shape[0]}.")
    # ========

    all_local_rot_matrices = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_input_joints, 1, 1)
    all_local_rot_matrices[:, 0] = global_joint_rotations_matrix[:, 0]

    for j_idx in range(1, num_input_joints):
        parent_idx = parents_for_kinematics[j_idx]
        # ==== new code -=====
        # Ensure parent_idx is valid for the current global_joint_rotations_matrix
        if parent_idx == -1 or parent_idx >= num_input_joints: # parent_idx should not be >= num_input_joints if parents_for_kinematics is correct
            # This case should ideally not be hit if parents_for_kinematics is correctly sliced for num_input_joints
            all_local_rot_matrices[:, j_idx] = global_joint_rotations_matrix[:, j_idx]
            continue
        # ========

        parent_global_rot_inv = torch.inverse(global_joint_rotations_matrix[:, parent_idx])
        child_global_rot = global_joint_rotations_matrix[:, j_idx]
        all_local_rot_matrices[:, j_idx] = torch.matmul(parent_global_rot_inv, child_global_rot)

    # Convert all local rotation matrices (for num_input_joints) to axis-angle
    all_local_pose_aa_flat_subset = rotation_matrix_to_angle_axis(all_local_rot_matrices.reshape(-1,3,3)).reshape(batch_size, num_input_joints * 3)
    
    # Subtract the corresponding subset of the model's pose_mean
    local_pose_aa_unposed_subset = all_local_pose_aa_flat_subset - pose_mean_subset

    # Extract global_orient_aa and body_pose_aa
    # global_orient is always the first 3 values (local rotation of joint 0)
    global_orient_aa_rec = local_pose_aa_unposed_subset[:, :3]
    
    # ==== new code -=====
    # body_pose_aa_rec is always 21*3 dimensions.
    # It corresponds to the local unposed rotations of SMPL-X body joints (typically joints 1-21 in SMPL-X order).
    if num_input_joints == 22:
        # The input 22 joints are assumed to be [Root, LHip, RHip, ..., RWrist].
        # Local unposed rotations for joints 1 to 21 from local_pose_aa_unposed_subset
        # correspond to the 21 body joints.
        body_pose_aa_rec = local_pose_aa_unposed_subset[:, 3 : 3 + smplx_model.NUM_BODY_JOINTS * 3]
    elif num_input_joints == smplx_model.J_regressor.shape[0]: # 55 joints
        # Original extraction logic for 55 joints:
        # local_pose_aa_unposed_subset is (B, 55*3)
        # body_pose starts after global_orient (3 dims) in the full pose vector.
        body_pose_start_idx_in_full_pose = 3 
        body_pose_end_idx_in_full_pose = body_pose_start_idx_in_full_pose + smplx_model.NUM_BODY_JOINTS * 3
        body_pose_aa_rec = local_pose_aa_unposed_subset[:, body_pose_start_idx_in_full_pose:body_pose_end_idx_in_full_pose]
    else:
        # This case is already handled by the ValueError above, but as a fallback:
        body_pose_aa_rec = torch.zeros((batch_size, smplx_model.NUM_BODY_JOINTS * 3), device=device, dtype=dtype)
    # ========

    return global_orient_aa_rec, body_pose_aa_rec


# --- Visualization Function ---
def visualize_global_rotations(
    joint_positions: torch.Tensor,      # (B, 55, 3)
    joint_orientations: torch.Tensor,   # (B, 55, 3, 3)
    parents: torch.Tensor,              # (55)
    batch_idx: int = 0,
    axis_scale: float = 0.05
):
    """
    Visualizes the SMPL-X skeleton with global joint rotations as RGB axes.
    Requires `trimesh` to be installed.
    """
    try:
        import trimesh
        from trimesh import creation
    except ImportError:
        print("Trimesh library is required for visualization. Please install it: pip install trimesh")
        return

    if joint_positions.ndim == 2: # If only one sample, unsqueeze batch dim
        joint_positions = joint_positions.unsqueeze(0)
        joint_orientations = joint_orientations.unsqueeze(0)

    pos = joint_positions[batch_idx].detach().cpu().numpy()
    orient = joint_orientations[batch_idx].detach().cpu().numpy()
    parents_np = parents.cpu().numpy()

    scene = trimesh.Scene()

    # Add skeleton lines
    for i, p_idx in enumerate(parents_np):
        if p_idx != -1: # If not root
            pt1 = pos[i]
            pt2 = pos[p_idx]
            line = trimesh.load_path(np.array([pt1, pt2]))
            scene.add_geometry(line)
    
    # Add coordinate axes at each joint
    for i in range(pos.shape[0]):
        # Create a 4x4 transformation matrix for the axis
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = orient[i]
        transform_matrix[:3, 3] = pos[i]
        
        axis_marker = creation.axis(origin_size=0.005, axis_radius=0.002, axis_length=axis_scale)
        axis_marker.apply_transform(transform_matrix)
        scene.add_geometry(axis_marker)

    scene.show()


# --- Test Function ---
def test_inverse_conversion(smplx_model_instance: SMPLX):
    print("\n--- Testing Inverse Conversion ---")
    batch_size = 2
    device = smplx_model_instance.betas.device # Use device from model

    # Generate random local poses
    rand_global_orient_aa = torch.rand(batch_size, 3, device=device) * 0.5 # Small rotations
    rand_body_pose_aa = torch.rand(batch_size, smplx_model_instance.NUM_BODY_JOINTS * 3, device=device) * 0.5

    # 1. Local to Global
    global_rots_pred, _ = local_to_global_pose_rotations(
        smplx_model_instance,
        rand_global_orient_aa,
        rand_body_pose_aa
    )

    # 2. Global to Local
    orient_aa_rec, body_pose_aa_rec = global_to_local_pose_rotations(
        smplx_model_instance,
        global_rots_pred
    )

    # Compare original and reconstructed (axis-angle can be tricky due to 2pi periodicity and v vs -v for pi)
    # For a more robust check, convert reconstructed axis-angles back to rotation matrices
    # and compare with the source local rotation matrices derived from initial AA.
    
    # Check global_orient
    diff_orient = torch.abs(rand_global_orient_aa - orient_aa_rec)
    # Normalize angles to [-pi, pi] before comparison for robustness, or compare matrices.
    # For simplicity, direct comparison with tolerance.
    print(f"Max absolute difference in global_orient_aa: {torch.max(diff_orient).item()}")
    if torch.allclose(rand_global_orient_aa, orient_aa_rec, atol=1e-4): # Tolerance for float + m2a errors
        print("Global orient reconstruction: SUCCESSFUL")
    else:
        print("Global orient reconstruction: FAILED")
        # print("Original global_orient_aa:\n", rand_global_orient_aa)
        # print("Reconstructed global_orient_aa:\n", orient_aa_rec)

    # Check body_pose
    diff_body_pose = torch.abs(rand_body_pose_aa - body_pose_aa_rec)
    print(f"Max absolute difference in body_pose_aa: {torch.max(diff_body_pose).item()}")
    if torch.allclose(rand_body_pose_aa, body_pose_aa_rec, atol=1e-4):
        print("Body pose reconstruction: SUCCESSFUL")
    else:
        print("Body pose reconstruction: FAILED")
        # print("Original body_pose_aa (first 5 components):\n", rand_body_pose_aa[:, :5])
        # print("Reconstructed body_pose_aa (first 5 components):\n", body_pose_aa_rec[:, :5])
    
    print("Note: Small discrepancies are expected due to floating point precision and axis-angle conversion ambiguities, especially near 0 or PI rotations.")


if __name__ == '__main__':
    model_folder = 'data/body_models/smplx_model' # !!! UPDATE THIS PATH !!!
    
    try:
        # It's good practice to set use_pca=False and flat_hand_mean=True
        # if you are primarily working with body_pose and global_orient
        # and want hand/face poses to be zero or determined by pose_mean.
        smplx_model_instance = smplx.create(
            model_path=model_folder, # smplx.create expects folder containing model_type subfolder
            model_type="smplx",
            gender='neutral', 
            flat_hand_mean=True, # Hands start flat, not in mean pose
            use_pca=False,       # Use full axis-angle for hands (though we zero them)
            batch_size=1         # Default batch_size for nn.Parameters
        ).to(dist_util.dev()).eval()
        print(f"SMPLX model loaded. Device: {smplx_model_instance.betas.device}")
    except Exception as e:
        print(f"Could not load SMPLX model. Error: {e}")
        print(f"Please ensure '{model_folder}/smplx/SMPLX_NEUTRAL.npz' (or .pkl) exists and is accessible.")
        smplx_model_instance = None

    if smplx_model_instance:
        print(f"SMPLX model has {smplx_model_instance.J_regressor.shape[0]} joints.")

        # --- Example: Local to Global and Visualize ---
        example_batch_size = 1
        example_global_orient = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32, device=dist_util.dev())
        example_body_pose = torch.zeros((example_batch_size, smplx_model_instance.NUM_BODY_JOINTS * 3), dtype=torch.float32, device=dist_util.dev())
        
        # Bend left elbow (joint 18 in SMPLX hierarchy)
        # Body pose parameters are for joints 1 through 21 (NUM_BODY_JOINTS).
        # Joint 18 (LElbow) is (18-1) = 17th in this 0-indexed body_pose parameter list.
        idx_l_elbow_param = 17 
        example_body_pose[0, idx_l_elbow_param*3 : idx_l_elbow_param*3+3] = torch.tensor([0.0, -1.2, 0.0], device=dist_util.dev()) # Bend more for visibility
        # Bend right knee (joint 5 in SMPLX hierarchy)
        # Joint 5 (RKnee) is (5-1) = 4th in body_pose parameters
        idx_r_knee_param = 4
        example_body_pose[0, idx_r_knee_param*3 : idx_r_knee_param*3+3] = torch.tensor([0.8, 0.0, 0.0], device=dist_util.dev())


        global_rotations, global_positions = local_to_global_pose_rotations(
            smplx_model_instance,
            example_global_orient,
            example_body_pose,
            return_joint_positions=True
        )

        print("\nGlobal joint rotations shape:", global_rotations.shape)
        print("Global joint positions shape:", global_positions.shape)
        
        print("\nGlobal rotation of root (joint 0):\n", global_rotations[0, 0].detach().cpu().numpy())
        smplx_joint_idx_l_elbow = 18 # LElbow is joint 18 in the full hierarchy
        print(f"\nGlobal rotation of LElbow (joint {smplx_joint_idx_l_elbow}):\n", global_rotations[0, smplx_joint_idx_l_elbow].detach().cpu().numpy())

        # Visualize (call this if you have trimesh and a display environment)
        print("\nAttempting visualization... Close the trimesh window to continue.")
        visualize_global_rotations(
            global_positions,
            global_rotations,
            smplx_model_instance.parents,
            batch_idx=0,
            axis_scale=0.08 # Adjust scale as needed
        )

        # --- Test Inverse ---
        # Adjust batch size for test if needed
        smplx_model_instance_for_test = smplx.create(
            model_path=model_folder, model_type="smplx", gender='neutral',
            flat_hand_mean=True, use_pca=False, batch_size=2 # Test with batch_size > 1
        ).to(dist_util.dev()).eval()
        
        test_inverse_conversion(smplx_model_instance_for_test)