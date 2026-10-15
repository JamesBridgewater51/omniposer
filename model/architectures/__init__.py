"""
Task-agnostic neural network architectures.

Available architectures:
- UNetTCN: Task-agnostic UNet TCN for trajectory/sequence modeling
- GenericDiT: Task-agnostic Diffusion Transformer (black-box approach)
- SpecificPoseDiT: Task-specific DiT for pose estimation (handles traj/pose split)
"""

from model.architectures.unet_tcn import UNetTCN, UNetTCNWithControlNet
from model.architectures.generic_dit import GenericDiT
from model.architectures.pose_dit import SpecificPoseDiT

__all__ = [
    'UNetTCN',
    'UNetTCNWithControlNet',
    'GenericDiT',
    'SpecificPoseDiT',
]
