# Copyright (c) Meta Platforms, Inc. and affiliates.
# Utility functions for config-based class instantiation.
# Adapted from NVIDIA's dnnlib pattern.

"""
Utility functions for dynamic class instantiation from config.

This module provides functions to construct Python classes by name from config,
following the pattern established by NVIDIA's dnnlib in StyleGAN/EDM projects.

Example usage:
    from utils.construct import construct_class_by_name
    
    # In config:
    network:
        class_name: model.architectures.UNetTCN
        input_dim: 12
        cond_dim: 66
        output_dim: 12
    
    # In code:
    net = construct_class_by_name(**config.network)
"""

import importlib
import re
import sys
import types
from typing import Any, Tuple


def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed).
    
    Args:
        obj_name: Fully qualified name like 'model.architectures.UNetTCN'
        
    Returns:
        Tuple of (module, local_object_name)
    """
    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [
        (".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)
    ]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)
            get_obj_from_module(module, local_obj_name)
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name)
        except ImportError:
            if not str(sys.exc_info()[1]).startswith(
                "No module named '" + module_name + "'"
            ):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)
            get_obj_from_module(module, local_obj_name)
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object.
    
    Args:
        module: Python module object
        obj_name: Object name within the module (can be nested like 'Class.method')
        
    Returns:
        The Python object
    """
    if obj_name == "":
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given fully qualified name.
    
    Args:
        name: Fully qualified name like 'model.architectures.UNetTCN'
        
    Returns:
        The Python object (class, function, etc.)
    """
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function.
    
    Args:
        *args: Positional arguments to pass to the function
        func_name: Fully qualified function name
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of calling the function
    """
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments.
    
    This is the main function for config-based class instantiation.
    
    Args:
        *args: Positional arguments to pass to the constructor
        class_name: Fully qualified class name like 'model.architectures.UNetTCN'
        **kwargs: Keyword arguments to pass to the constructor
        
    Returns:
        Instance of the class
        
    Example:
        config = {
            'class_name': 'model.architectures.UNetTCN',
            'input_dim': 12,
            'cond_dim': 66,
            'output_dim': 12,
        }
        net = construct_class_by_name(**config)
    """
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def construct_from_config(config: dict) -> Any:
    """Construct a class from a config dictionary.
    
    Expects the config to have a 'class_name' key with the fully qualified class name.
    All other keys are passed as keyword arguments to the constructor.
    
    Args:
        config: Dictionary with 'class_name' and constructor kwargs
        
    Returns:
        Instance of the class
        
    Example:
        config = {
            'class_name': 'model.architectures.UNetTCN',
            'input_dim': 12,
            'cond_dim': 66,
            'output_dim': 12,
        }
        net = construct_from_config(config)
    """
    config = dict(config)  # Make a copy to avoid modifying original
    class_name = config.pop('class_name')
    return construct_class_by_name(class_name=class_name, **config)


def is_valid_class_name(name: str) -> bool:
    """Check if a class name is valid and can be imported.
    
    Args:
        name: Fully qualified class name
        
    Returns:
        True if the class can be found, False otherwise
    """
    try:
        get_obj_by_name(name)
        return True
    except (ImportError, AttributeError):
        return False

