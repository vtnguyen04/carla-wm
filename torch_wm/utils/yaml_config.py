import yaml
from torch_wm.structs import AttrDict

def load_yaml_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return AttrDict(config)

def merge_configs(base_config, override_config):
    if not isinstance(override_config, dict):
        return base_config
    
    for k, v in override_config.items():
        if k in base_config and isinstance(base_config[k], dict) and isinstance(v, dict):
            merge_configs(base_config[k], v)
        else:
            base_config[k] = v
    return base_config
