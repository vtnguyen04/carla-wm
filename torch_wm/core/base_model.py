from typing import List, Dict, Tuple, Optional, Any, Union
# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn
# TensorBoard - try multiple import paths
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        try:
            from tensorboard.summary import SummaryWriter
        except ImportError:
            class SummaryWriterStub:
                def __init__(self, *args, **kwargs): pass
                def add_scalar(self, *args, **kwargs): pass
                def add_image(self, *args, **kwargs): pass
                def add_figure(self, *args, **kwargs): pass
                def close(self, *args, **kwargs): pass
            SummaryWriter = SummaryWriterStub
            print("⚠️ Warning: TensorBoard not found. Logging will be disabled.")

# Other
from tqdm import tqdm
import os
import time
import glob
try:
    import wandb
except ImportError:
    wandb = None

# Neural Nets
from torch_wm import modules
from torch_wm.core.training_loop import TrainingLoopMixin

class Model(TrainingLoopMixin, modules.Module):

    def __init__(self, name="model"):
        super(Model, self).__init__()

        # Model Attributes
        self.compiled = False
        self.built = False
        self.name = name
        self.grad_scaler_state_dict = None

    def compile(self, losses, loss_weights=None, optimizer="Adam", metrics=None, decoders=None):
        from torch_wm.optimizers import optim_dict
        from torch_wm import schedulers

        # Optimizer
        if isinstance(optimizer, str):
            self.optimizer = optim_dict[optimizer](params=self.parameters())
        else:
            self.optimizer = optimizer

        # Model Step
        self.model_step = self.optimizer.param_groups[0]["lr_scheduler"].model_step

        # Losses
        if losses == None:
            self.compiled_losses = []
        elif isinstance(losses, list):
            self.compiled_losses = nn.ModuleList(losses)
        elif isinstance(losses, dict):
            self.compiled_losses = nn.ModuleDict(losses)
        else:
            self.compiled_losses = losses

        # Loss Weights
        if loss_weights == None:
            self.compiled_loss_weights = schedulers.ConstantScheduler(1.0)
        elif isinstance(loss_weights, list):
            self.compiled_loss_weights = nn.ModuleList([w if isinstance(w, nn.Module) else nn.Module() for w in loss_weights]) # Dummy modules if they aren't modules to allow list assignment
            # Wait, ConstantScheduler is NOT a Module.
            # If I want to store non-Modules in a list on a Module, I should hide them from __setattr__.
            object.__setattr__(self, 'compiled_loss_weights', loss_weights)
        elif isinstance(loss_weights, dict):
            object.__setattr__(self, 'compiled_loss_weights', loss_weights)
        else:
            self.compiled_loss_weights = loss_weights

        # Metrics
        if metrics == None:
            self.compiled_metrics = []
        elif isinstance(metrics, (list, dict)):
            object.__setattr__(self, 'compiled_metrics', metrics)
        else:
            self.compiled_metrics = metrics
            
        # Decoders
        if decoders == None:
            self.compiled_decoders = []
        elif isinstance(decoders, (list, dict)):
            object.__setattr__(self, 'compiled_decoders', decoders)
        else:
            self.compiled_decoders = decoders

        # Set Compiled to True
        self.compiled = True

        # Set Modules Name
        for name, module in self.named_modules():
            if not hasattr(module, "name"):
                module.name = name

    def build(self, outputs):

        # Map to Outputs
        self.losses = self.map_to_outputs(outputs, self.compiled_losses)
        self.loss_weights = self.map_to_outputs(outputs, self.compiled_loss_weights)
        self.decoders = self.map_to_outputs(outputs, self.compiled_decoders)
        self.metrics = self.map_to_outputs(outputs, self.compiled_metrics)

        # Transfer to Device
        self.losses = self.transfer_to_device(self.losses)
        self.decoders = self.transfer_to_device(self.decoders)
        self.metrics = self.transfer_to_device(self.metrics)

        # Set Built to true
        self.built = True

    def map_to_outputs(self, outputs, struct):

        """Convenience method to conform `struct` to `outputs` structure.

        Mappings performed:
            (1) Map a struct to a dict of outputs, using the output names.
            (2) Fill missing struct elements with None.
            (3) Map a single item to all outputs.

        Args:
            outputs: Model outputs predictions dict.
            struct: Arbitrary nested structure (dict, list, item).

        Returns:
            Dict mapping `struct` to `outputs` structure.

        """

        # None
        if struct == None:

            return struct

        # Dictionary
        elif isinstance(struct, dict):

            # Assert struct key in outputs
            for key in struct:
                if not key in outputs:
                    raise Exception("Found unexpected dict key: {}. Valid output names are: {}".format(key, outputs.keys()))

            # Fill missing key with None
            for key in outputs:
                if not key in struct:
                    struct[key] = None

        # List
        elif isinstance(struct, list):

            # Map list items to outputs, Fill missing items with None, Ignore extra items
            struct = {key: struct[i] if i < len(struct) else None for i, key in enumerate(outputs)}

        # Module / Tensor / tuple
        else:

            # Map item to all outputs
            struct = {key: struct for key in outputs}

        return struct

    def forward_model(self, inputs, targets, compute_metrics=True, verbose=0):

        """ forward_model method

        - forward
        - compute losses
        - compute metrics
        
        """

        # Init Batch Dict
        batch_losses = {}
        batch_metrics = {}
        batch_truths = {}
        batch_preds = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Additional Targets
        self.additional_targets = {}

        # Forward
        outputs = self.forward(inputs)

        # Format Outputs to dict
        if isinstance(outputs, dict):
            pass
        elif isinstance(outputs, list):
            outputs = {"output_" + str(key): value for key, value in enumerate(outputs)}
        else:
            outputs = {"output": outputs}

        # Map Targets to Outputs
        targets = self.map_to_outputs(outputs, targets)

        # Append Additional Targets
        for key in self.additional_targets:
            targets[key] = self.additional_targets[key]

        # Build Model
        if not self.built:
            self.build(outputs)

        # Outputs loop
        for key in outputs:

            # Loss Function
            if self.losses[key] != None:

                # Loss key
                key_loss = "loss_" + key

                # Loss
                batch_losses[key_loss] = self.losses[key](targets[key], outputs[key])

                # Weight Loss
                total_loss += batch_losses[key_loss] * self.loss_weights[key].get_val_step(self.model_step + 1)

            # Metric Functions
            if self.metrics[key] != None and compute_metrics:

                # To list
                if not isinstance(self.metrics[key], list):
                    metrics = [self.metrics[key]]
                else:
                    metrics = self.metrics[key]
                if not isinstance(self.decoders[key], list):
                    decoders = [self.decoders[key] for _ in metrics]
                else:
                    decoders = self.decoders[key]

                for metric, decoder in zip(metrics, decoders):

                    # Metric Key
                    key_metric = metric.name
                    if key_metric in batch_metrics:
                        key_metric += "_" + key

                    # Decoding
                    if decoder != None:
                        batch_truths[key_metric] = decoder(targets[key], from_logits=False) if targets[key] != None else None
                        batch_preds[key_metric] = decoder(outputs[key])
                    else:
                        batch_truths[key_metric] = targets[key]
                        batch_preds[key_metric] = outputs[key]

                    # Prediction Verbose
                    if verbose:
                        print("Groundtruths:\n", batch_truths[key_metric])
                        print("Predictions:\n", batch_preds[key_metric])

                    # Metric
                    batch_metrics[key_metric] = metric(batch_truths[key_metric], batch_preds[key_metric])

        # Module Infos / Losses
        for module in self.modules():

            # Module added losses during forward
            if hasattr(module, "added_losses"):
                for key, value in module.added_losses.items():
                    key_loss = "loss_" + key
                    batch_losses[key_loss] = value["loss"]
                    total_loss += batch_losses[key_loss] * value["weight"]
                module.reset_losses()

            # Module added infos during forward
            if hasattr(module, "infos") and module is not self: # Do not include self to avoid reset infos
                self.infos.update(module.infos)
                module.reset_infos()

            # Module added metrics during forward
            if hasattr(module, "added_metrics"):
                for key, value in module.added_metrics.items():
                    key_metric = key
                    batch_metrics[key_metric] = value
                module.reset_metrics()

        # Append Total loss
        if len(batch_losses) >= 1:
            batch_losses = dict({"loss": total_loss}, **batch_losses)
        else:
            batch_losses = {"loss": total_loss}

        return batch_losses, batch_metrics, batch_truths, batch_preds

    def train_step(self, inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training):

        """ train_step method

        - forward_model (forward + compute losses/metrics)
        - backward
        
        """

        # Automatic Mixed Precision Casting (model forward + loss computing)
        if "cuda" in str(self.device):
            # Precision Mapping (Symmetric Fix)
            prec_dtype = torch.float16 if precision == "float16" else torch.float32
            with torch.amp.autocast('cuda', enabled=precision!=torch.float32, dtype=prec_dtype):
                batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, compute_metrics=eval_training)
        else:
            batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, compute_metrics=eval_training)

        # Accumulated Steps
        loss = batch_losses["loss"] / accumulated_steps
        acc_step += 1

        # Backward: Accumulate gradients
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        # Continue Accumulating
        if acc_step < accumulated_steps:
            return batch_losses, batch_metrics, acc_step

        # Grad Scaler Info
        if grad_scaler is not None and grad_scaler.is_enabled():
            self.add_info("grad_scale", grad_scaler.get_scale())

        # Unscale Gradients
        if grad_scaler is not None:
            grad_scaler.unscale_(self.optimizer)

        # Optimizer Step and Update Scale
        if grad_scaler is not None:
            grad_scaler.step(self.optimizer)
            grad_scaler.update()
        else:
            self.optimizer.step()

        # Zero Gradients
        self.optimizer.zero_grad()
        acc_step = 0

        # Update Model Infos
        if len(self.optimizer.param_groups) > 1:
            for i, param_group in enumerate(self.optimizer.param_groups):

                # learning rate
                self.add_info("lr_{}".format(i), float(param_group['lr']))

                # grad norm
                if "grad_norm" in param_group:
                    self.add_info("grad_norm_{}".format(i), round(float(param_group['grad_norm']), 4))

                # grad infos
                if "grad_min" in param_group:
                    self.add_info("grad_min_{}".format(i), param_group['grad_min'])
                if "grad_max" in param_group:
                    self.add_info("grad_max_{}".format(i), param_group['grad_max'])
                if "grad_mean" in param_group:
                    self.add_info("grad_mean_{}".format(i), param_group['grad_mean'])
                if "grad_std" in param_group:
                    self.add_info("grad_std_{}".format(i), param_group['grad_std'])
        else:

            # learning rate
            self.add_info("lr", float(self.optimizer.param_groups[0]['lr']))

            # grad norm
            if "grad_norm" in self.optimizer.param_groups[0]:
                self.add_info("grad_norm", round(float(self.optimizer.param_groups[0]['grad_norm']), 4))

            # grad infos
            if "grad_min" in self.optimizer.param_groups[0]:
                self.add_info("grad_min", self.optimizer.param_groups[0]['grad_min'])
            if "grad_max" in self.optimizer.param_groups[0]:
                self.add_info("grad_max", self.optimizer.param_groups[0]['grad_max'])
            if "grad_mean" in self.optimizer.param_groups[0]:
                self.add_info("grad_mean", self.optimizer.param_groups[0]['grad_mean'])
            if "grad_std" in self.optimizer.param_groups[0]:
                self.add_info("grad_std", self.optimizer.param_groups[0]['grad_std'])

        # Add Info Model Step
        self.add_info("step", self.model_step.item())

        return batch_losses, batch_metrics, acc_step  

    def eval_step(self, inputs, targets, verbose=0):

        with torch.no_grad():
            batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, verbose=verbose)

        return batch_losses, batch_metrics, batch_truths, batch_preds

    def num_params(self, module=None):

        if module != None:
            if isinstance(module, list):
                return sum([self.num_params(m) for m in module])
            else:
                return sum([p.numel() for p in module.parameters()])
        else:
            return sum([p.numel() for p in self.parameters()])

    def summary(self, show_dict=False, show_modules=False):

        # Model Name
        print("Model name: {}".format(self.name))

        # Number Params
        print("Number Parameters: {:,}".format(self.num_params()))

        # Show Modules Params
        for key, value in self.named_children():
            print("{}: {:,} Parameters".format(key, self.num_params(value)))

        # Options
        if show_dict:
            self.show_dict()
        if show_modules:
            self.show_modules()

        # Modules Buffer
        for key, value in self.modules_buffer.items():
            print("{} Parameters: {:,}".format(key, self.num_params(value)))

    def show_dict(self, module=None):

        # Print
        print("State Dict:")

        # Default Dict
        if module != None:
            state_dict = module.state_dict(keep_vars=True)
        else:
            state_dict = self.state_dict(keep_vars=True)

        # Empty Dict
        if state_dict == {}:
            return

        # Show Dict
        max_len_id = len(str(len(state_dict)))
        max_len_key = max([len(key) for key in state_dict.keys()]) + 5
        for id, (key, value) in enumerate(state_dict.items()):
            print("{} {} type: {:<12} numel: {:<12} shape: {:<20} mean: {:<12.4f} std: {:<12.4f} min: {:<12.4f} max: {:<12.4f} dtype: {:<12} device: {}".format(str(id) + " " * (max_len_id - len(str(id))), key + " " * (max_len_key - len(key)), "param" if isinstance(value, nn.Parameter) else "buffer", value.numel(), str(tuple(value.size())), value.float().mean(), value.float().std(), value.float().min(), value.float().max(), str(value.dtype).replace("torch.", ""), str(value.device)))

    def show_modules(self, module=None):

        # Print
        print("Named Modules:")

        # Named Modules
        if module != None:
            named_modules = dict(module.named_modules())
        else:
            named_modules = dict(self.named_modules())

        # Show Modules
        max_len_id = len(str(len(named_modules)))
        max_len_key = max([len(key) for key in named_modules.keys()]) + 5
        max_len_class = max([len(type(value).__name__) for value in named_modules.values()]) + 5
        for id, (key, value) in enumerate(named_modules.items()):
            print("{} {} class: {} device: {}".format(str(id) + " " * (max_len_id - len(str(id))), key + " " * (max_len_key - len(key)), type(value).__name__ + " " * (max_len_class - len(type(value).__name__)), value.device if hasattr(value, "device") else ""))

    def save(self, path, save_optimizer=True, keep_last_k=None):
        
        # Save Model Checkpoint
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": None if not save_optimizer else {key: value.state_dict() for key, value in self.optimizer.items()} if isinstance(self.optimizer, dict) else self.optimizer.state_dict(),
            "model_step": self.model_step,
            "grad_scaler_state_dict": self.grad_scaler.state_dict() if hasattr(self, "grad_scaler") else None
            }, path)

        # Print Model state
        print("Model saved at step {}: {}".format(self.model_step, path))

        # Keep last k checkpoints
        if keep_last_k != None:

            # List checkpoints
            save_dir = os.path.dirname(path)
            checkpoints_list = glob.glob(os.path.join(save_dir, "*.ckpt"))
            checkpoints_list = sorted(checkpoints_list, key=lambda s: int(os.path.splitext(s)[0].split("/")[-1].split("_")[-1]))

            # Remove older_checkpoint
            while len(checkpoints_list) > keep_last_k:

                # Pop older_checkpoint
                older_checkpoint = checkpoints_list.pop(0)

                # Remove older_checkpoint
                os.remove(older_checkpoint)

                # Print
                print("Removed old checkpoint: {}".format(older_checkpoint))

    def load(self, path, load_optimizer=True, verbose=True, strict=True):

        # Print Load state
        if verbose:
            print("Load Model from {}".format(path))

        # Load Model Checkpoint
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load Model State Dict
        self.load_state_dict({key:value for key, value in checkpoint["model_state_dict"].items()}, strict=strict)

        # Load Optimizer State Dict
        if load_optimizer and checkpoint["optimizer_state_dict"] is not None:

            if isinstance(self.optimizer, dict):
                for key, value in self.optimizer.items():
                    value.load_state_dict(checkpoint["optimizer_state_dict"][key])
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Model Step, already loaded from optm
            self.model_step.fill_(checkpoint["model_step"])

        # Load Grad Scaler
        if "grad_scaler_state_dict" in checkpoint:
            self.grad_scaler_state_dict = checkpoint["grad_scaler_state_dict"]

        # Print Model state
        if verbose:
            print("Model loaded at step {}".format(self.model_step))


    # ── Training Loop Methods ──────────────────────────────────────────
    # fit(), evaluate(), on_epoch_begin/end, log_step, display_step, etc.
    # are provided by TrainingLoopMixin (core/training_loop.py)
    # This follows Single Responsibility Principle:
    #   - Model: forward, backward, compile, save/load
    #   - TrainingLoopMixin: epoch orchestration, logging, dashboard