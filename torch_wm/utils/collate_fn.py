import torch
import numpy as np

class CollateFn:
    def __init__(self, inputs_params=None, targets_params=None):
        self.inputs_params = inputs_params
        self.targets_params = targets_params

    def __call__(self, batch):
        if not isinstance(batch, list):
            return batch
        
        def stack_recursive(items):
            if isinstance(items[0], dict):
                return {k: stack_recursive([it[k] for it in items]) for k in items[0].keys()}
            elif isinstance(items[0], np.ndarray):
                return torch.from_numpy(np.stack(items))
            elif isinstance(items[0], torch.Tensor):
                return torch.stack(items)
            else:
                return torch.tensor(items)

        collated_inputs = stack_recursive(batch)
        return collated_inputs, {}
