import pytest
import os
from torch_wm.utils.yaml_config import load_yaml_config, merge_configs
from torch_wm.structs import AttrDict

def test_load_yaml_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\nb: {c: 2}")
    
    config = load_yaml_config(str(path))
    assert config.a == 1
    assert config.b.c == 2
    assert isinstance(config, AttrDict)

def test_merge_configs():
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"d": 3}, "e": 4}
    
    merged = merge_configs(base, override)
    assert merged["a"] == 1
    assert merged["b"]["c"] == 2
    assert merged["b"]["d"] == 3
    assert merged["e"] == 4
    
    # Test non-dict override
    assert merge_configs(base, None) == base
