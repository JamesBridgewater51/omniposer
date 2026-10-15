"""
Visualizer registry for dataset-specific implementations.
"""
from typing import Dict, Type, Optional
from omegaconf import DictConfig

from .base import BaseVisualizer


class VisualizerRegistry:
    """Registry for dataset-specific visualizers."""
    
    _visualizers: Dict[str, Type[BaseVisualizer]] = {}
    
    @classmethod
    def register(cls, dataset_name: str):
        """Decorator to register a visualizer for a dataset."""
        def decorator(visualizer_cls: Type[BaseVisualizer]):
            cls._visualizers[dataset_name.lower()] = visualizer_cls
            return visualizer_cls
        return decorator
    
    @classmethod
    def get(cls, dataset_name: str) -> Optional[Type[BaseVisualizer]]:
        """Get visualizer class for a dataset."""
        return cls._visualizers.get(dataset_name.lower())
    
    @classmethod
    def list_registered(cls) -> list:
        """List all registered dataset names."""
        return list(cls._visualizers.keys())


def get_visualizer(
    dataset_name: str,
    train_config: DictConfig,
    test_config: DictConfig,
    fk_model,
) -> Optional[BaseVisualizer]:
    """
    Factory function to create appropriate visualizer.
    
    Args:
        dataset_name: Name of the dataset
        train_config: Training configuration
        test_config: Test configuration
        fk_model: FK model instance (smplx or custom)
        
    Returns:
        Visualizer instance or None if not available
    """
    visualizer_cls = VisualizerRegistry.get(dataset_name)
    if visualizer_cls is None:
        print(f"[Warning] No visualizer registered for dataset: {dataset_name}")
        return None
    
    return visualizer_cls(train_config, test_config, fk_model)


# Import visualizers to trigger registration
# These imports are at the bottom to avoid circular imports
def _register_visualizers():
    """Lazy import and register all visualizers."""
    try:
        from . import amass
    except ImportError:
        pass
    
    try:
        from . import mocap
    except ImportError:
        pass
    
    try:
        from . import egoexo
    except ImportError:
        pass
    
    try:
        from . import render_totalcapture
    except ImportError:
        pass

    try:
        from . import render_aist
    except ImportError:
        pass

    try:
        from . import render_3dpw
    except ImportError:
        pass

    try:
        from . import nymeria
    except ImportError:
        pass


_register_visualizers()
