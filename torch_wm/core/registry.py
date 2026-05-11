"""Module Registry for WMAgent-CARLA.

Registry Pattern implementation to dynamically register and retrieve
pluggable components (Losses, Callbacks, etc.).
"""

from typing import Dict, Type

from .base import BaseModule

class ModuleRegistry:
    """Global registry for framework modules."""

    _registry: Dict[str, Type[BaseModule]] = {}

    @classmethod
    def register(cls, arg=None):
        """Decorator to register a module class, can be used as @register or @register('name')."""
        def decorator(module_cls):
            name = arg if isinstance(arg, str) else module_cls().name()
            if name in cls._registry:
                return module_cls
            cls._registry[name] = module_cls
            return module_cls

        if callable(arg):
            return decorator(arg)
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseModule]:
        """Get module class by name."""
        if name not in cls._registry:
            raise KeyError(f"Module '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_modules(cls) -> list:
        """List all registered module names."""
        return list(cls._registry.keys())
