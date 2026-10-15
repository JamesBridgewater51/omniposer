"""
Baseline registry for managing baseline method specifications.

Provides a singleton registry pattern for registering and retrieving baseline specs.
"""
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .spec import BaselineSpec


class BaselineRegistry:
    """
    Singleton registry for baseline method specifications.
    
    Usage:
        # Register a baseline using decorator
        @BaselineRegistry.register('my_baseline')
        def create_my_baseline_spec() -> BaselineSpec:
            return BaselineSpec(...)
        
        # Or register directly
        BaselineRegistry.register_spec('my_baseline', my_spec)
        
        # Retrieve a baseline spec
        spec = BaselineRegistry.get('my_baseline')
        
        # List all registered baselines
        names = BaselineRegistry.list_all()
    """
    
    _specs: Dict[str, 'BaselineSpec'] = {}
    _factories: Dict[str, Callable[[], 'BaselineSpec']] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a baseline spec factory function.
        
        The factory function should return a BaselineSpec instance.
        This allows lazy instantiation of specs.
        
        Args:
            name: Unique identifier for the baseline (lowercase)
            
        Returns:
            Decorator function
            
        Example:
            @BaselineRegistry.register('egoego')
            def create_egoego_spec() -> BaselineSpec:
                return BaselineSpec(
                    name='egoego',
                    compatible_mask_schemes=['egocentric_head_xyz_rot'],
                    ...
                )
        """
        def decorator(factory_fn: Callable[[], 'BaselineSpec']) -> Callable[[], 'BaselineSpec']:
            cls._factories[name.lower()] = factory_fn
            return factory_fn
        return decorator
    
    @classmethod
    def register_spec(cls, name: str, spec: 'BaselineSpec') -> None:
        """
        Directly register a BaselineSpec instance.
        
        Args:
            name: Unique identifier for the baseline (lowercase)
            spec: BaselineSpec instance to register
        """
        cls._specs[name.lower()] = spec
    
    @classmethod
    def get(cls, name: str) -> 'BaselineSpec':
        """
        Retrieve a baseline spec by name.
        
        If the spec was registered via factory, instantiates and caches it.
        
        Args:
            name: Baseline identifier (case-insensitive)
            
        Returns:
            BaselineSpec instance
            
        Raises:
            KeyError: If baseline not found
        """
        name_lower = name.lower()
        
        # Check cached specs first
        if name_lower in cls._specs:
            return cls._specs[name_lower]
        
        # Check factories and instantiate
        if name_lower in cls._factories:
            spec = cls._factories[name_lower]()
            cls._specs[name_lower] = spec
            return spec
        
        raise KeyError(
            f"Baseline '{name}' not found. "
            f"Available baselines: {cls.list_all()}"
        )
    
    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a baseline is registered."""
        name_lower = name.lower()
        return name_lower in cls._specs or name_lower in cls._factories
    
    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered baseline names."""
        all_names = set(cls._specs.keys()) | set(cls._factories.keys())
        return sorted(all_names)
    
    @classmethod
    def get_compatible_baselines(cls, mask_scheme: str) -> List[str]:
        """
        Get all baselines compatible with a given mask scheme.
        
        Args:
            mask_scheme: Name of the masking action
            
        Returns:
            List of baseline names that support this mask scheme
        """
        compatible = []
        for name in cls.list_all():
            spec = cls.get(name)
            if mask_scheme in spec.compatible_mask_schemes:
                compatible.append(name)
        return compatible
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered baselines. Mainly for testing."""
        cls._specs.clear()
        cls._factories.clear()
