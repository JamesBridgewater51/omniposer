import torch as th
from safetensors.torch import save_file, load_file
from collections import defaultdict

def can_view_flatten(tensor: th.Tensor) -> bool:
    """
    Check if a tensor can be flattened with `.view(-1)` without errors.
    
    Args:
        tensor (torch.Tensor): Input tensor to check.
    
    Returns:
        bool: True if `.view(-1)` is safe, False otherwise.
    """
    # A tensor can be flattened with `.view(-1)` if it is contiguous in memory
    # or if its strides are compatible with a contiguous flattening.
    # Essentially, the stride of the last dimension should be 1,
    # and the product of sizes should match the storage size.
    
    if tensor.is_contiguous():
        return True
    
    # If not contiguous, check if strides allow flattening
    stride = tensor.stride()
    size = tensor.size()
    
    # Calculate expected stride for a contiguous tensor of the same shape
    expected_stride = [1]  # Start from the last dimension
    for i in range(len(size) - 1, 0, -1):
        expected_stride.insert(0, expected_stride[0] * size[i])
    
    # Check if the actual strides match the expected strides
    if stride == tuple(expected_stride):
        return True
    
    # Special case: If the tensor is empty, `.view(-1)` is always allowed
    if tensor.numel() == 0:
        return True
    
    return False

