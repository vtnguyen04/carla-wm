import torch
import torch.utils.checkpoint as checkpoint

def checkpoint_forward(module, *inputs, use_checkpointing=False, training=False, residual=False, **kwargs):
    """
    Utility function to apply gradient checkpointing while adhering to DRY principles.
    Supports kwargs passing and optional residual connections.
    """
    # Check conditions for checkpointing
    # At least one input tensor must require grad to use checkpointing
    requires_grad = any(isinstance(x, torch.Tensor) and x.requires_grad for x in inputs)
    
    if use_checkpointing and training and requires_grad:
        def custom_forward(*args):
            out = module(*args, **kwargs)
            if residual:
                # Assuming first argument is the main input 'x'
                return out + args[0]
            return out
            
        return checkpoint.checkpoint(
            custom_forward,
            *inputs,
            use_reentrant=False
        )
    else:
        out = module(*inputs, **kwargs)
        if residual:
            return out + inputs[0]
        return out
