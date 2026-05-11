import torch
import torch.nn as nn
from .yaml_config import load_yaml_config, merge_configs
from .collate_fn import CollateFn

def get_module_and_params(module, module_dict=None):
    """
    Utility to get module class and its parameters from configuration.
    Supports recursive processing for lists.
    """
    if module is None:
        return None, {}
    
    # NEW: Recursive support for lists
    if isinstance(module, list):
        module_classes = []
        module_params_list = []
        for m in module:
            m_class, m_params = get_module_and_params(m, module_dict)
            module_classes.append(m_class)
            module_params_list.append(m_params)
        return module_classes, module_params_list

    if isinstance(module, str):
        if module_dict and module in module_dict:
            return module_dict[module], {}
        if hasattr(nn, module):
            return getattr(nn, module), {}
        return None, {}

    if isinstance(module, dict):
        module_class_name = module.get("class")
        module_params = module.get("params", {})
        
        if module_class_name is None:
            return None, {}
            
        if module_dict and module_class_name in module_dict:
            return module_dict[module_class_name], module_params
        
        if hasattr(nn, module_class_name):
            return getattr(nn, module_class_name), module_params
            
        return None, module_params

    # If it's already a class or callable
    return module, {}

from .logger import get_logger, get_console
