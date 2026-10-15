# fm/respace.py
import numpy as np
import torch
from .fm_posenet import FlowMatchingPoseNet
from .fm_trajnet import FlowMatchingTrajNet

class SpacedFlowMatchingTrajNet(FlowMatchingTrajNet):
    """
    A simple pass-through wrapper for API compatibility with the model factory.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SpacedFlowMatchingPoseNet(FlowMatchingPoseNet):
    """
    A simple pass-through wrapper for API compatibility with the model factory.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)