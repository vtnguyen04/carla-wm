import pytest
from torch_wm.core.registry import ModuleRegistry
from torch_wm.core.base import BaseModule

class MockComponent(BaseModule):
    def __init__(self):
        super().__init__()
        self._name = "mock_component"

def test_registry_decorator():
    @ModuleRegistry.register("test_mock")
    class TestMock(BaseModule):
        pass
    
    # Verify registration
    assert "test_mock" in ModuleRegistry._registry
    assert ModuleRegistry._registry["test_mock"] == TestMock

def test_registry_create():
    @ModuleRegistry.register("create_mock")
    class CreateMock(BaseModule):
        def __init__(self, value=10):
            self.value = value
        def name(self):
            return "create_mock"
        def is_enabled(self, config):
            return True
            
    cls = ModuleRegistry.get("create_mock")
    instance = cls(value=42)
    assert isinstance(instance, CreateMock)
    assert instance.value == 42
    
def test_registry_create_unregistered():
    with pytest.raises(KeyError):
        ModuleRegistry.get("non_existent")

def test_registry_keys():
    # Backup registry
    backup = ModuleRegistry._registry.copy()
    ModuleRegistry._registry.clear()
    
    @ModuleRegistry.register("key1")
    class K1(BaseModule): pass
    @ModuleRegistry.register("key2")
    class K2(BaseModule): pass
    
    assert set(ModuleRegistry.list_modules()) == {"key1", "key2"}
    
    # Restore registry
    ModuleRegistry._registry.clear()
    ModuleRegistry._registry.update(backup)
