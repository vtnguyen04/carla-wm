# Copyright 2025, Maxime Burchi.
# Modifications copyright 2026, Vo Thanh Nguyen.
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

"""Training Loop — Extracted from base_model.py for Single Responsibility.

Contains the full `fit()` training loop, evaluation loop, epoch callbacks,
logging helpers, and display utilities. These are NOT used by the offline
training pipeline (RL/train_offline.py) but are kept for the original
WMAgent online training workflow.

Usage:
    model = WMAgent(...)
    model.compile(...)
    model.fit(dataset_train, epochs=100, ...)
"""

import torch
import os
import time

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


class TrainingLoopMixin:
    """Mixin class providing fit(), evaluate(), and related training loop methods.
    
    Designed to be mixed into Model (base_model.py) via multiple inheritance.
    All methods assume self has: train_step, eval_step, forward_model,
    optimizer, model_step, device, infos, etc.
    """

    def on_train_begin(self):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, evaluate, save, log_figure, callback_path, epoch, inputs, targets, dataset_eval, eval_steps, verbose_eval, writer, recompute_metrics, keep_last_k):
        self.on_step_end(evaluate, save, log_figure, callback_path, epoch, epoch, inputs, targets, dataset_eval, eval_steps, verbose_eval, writer, recompute_metrics, keep_last_k=keep_last_k, tag="epoch")
        print()

    def on_step_end(self, evaluate, save, log_figure, callback_path, epoch, step, inputs, targets, dataset_eval, eval_steps, verbose_eval, writer, recompute_metrics, keep_last_k, tag="step"):
        # Evaluate Model
        if evaluate:
            self._evaluate(dataset_eval, writer, eval_steps, verbose_eval, recompute_metrics, tag="Evaluation-" + tag)
            self.train()

        # Save Checkpoint
        if save and callback_path:
            self.save(os.path.join(callback_path, "checkpoints_epoch_{}_step_{}.ckpt".format(epoch, self.model_step)), keep_last_k=keep_last_k)

        # Log Figure
        if log_figure and callback_path:
            self.eval()
            self.log_figure(step, inputs, targets, writer, tag)
            self.train()

    def log_figure(self, step, inputs, targets, writer, tag): 
        pass

    def display_step(self, losses, metrics, infos, epoch_iterator, step):
        description = ""

        for key, value in losses.items():
            description += "{}: {:.4f} - ".format(key, value / step)

        for key, value in metrics.items():
            description += "{}: {:.4f} - ".format(key, value / step)

        for key, value in infos.items():
            if key.startswith("lr"):
                description += "{}: {:.2e} - ".format(key, value)
            elif isinstance(value, float):
                description += "{}: {:.4f} - ".format(key, value)
            else:
                description += "{}: {} - ".format(key, value)

        epoch_iterator.set_description(description)

    def log_step(self, losses, metrics, infos, writer, step, tag, inputs=None, targets=None, outputs=None):
        for key, value in losses.items():
            writer.add_scalar(os.path.join(tag, key), value, step)
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({f"{tag}/{key}": value.item() if hasattr(value, 'item') else value}, step=step)
            except:
                pass

        for key, value in metrics.items():
            writer.add_scalar(os.path.join(tag, key), value, step)

        for key, value in infos.items():
            if isinstance(value, float) or isinstance(value, int):
                writer.add_scalar(os.path.join(tag, key), float(value), step)
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    writer.add_scalar(os.path.join(tag, key), float(value), step)

        if hasattr(self, '_wandb_viz') and self._wandb_viz is not None and inputs is not None:
            try:
                batch = {'inputs': inputs, 'targets': targets}
                self._wandb_viz.log_training_visualization(batch, outputs or {}, step)
                self._wandb_viz.log_summary(step)
            except Exception as viz_err:
                if step % 100 == 0:
                    print(f"  [WandB Viz] Error: {viz_err}")

    def print_step(self, losses, metrics, tag):
        for key, value in losses.items():
            print("{} {}: {:.4f}".format(tag, key, value))
        for key, value in metrics.items():
            print("{} {}: {:.4f}".format(tag, key, value))

    def fit(
        self, 
        dataset_train, 
        epochs, 
        dataset_eval=None, 
        eval_steps=None, 
        verbose_eval=0, 
        initial_epoch=0, 
        callback_path=None, 
        steps_per_epoch=None, 
        precision=torch.float32, 
        accumulated_steps=1, 
        eval_period_step=None, 
        eval_period_epoch=1,
        saving_period_epoch=1, 
        log_figure_period_step=None, 
        log_figure_period_epoch=1, 
        step_log_period=10, 
        eval_training=True,
        grad_init_scale=65536.0, 
        detect_anomaly=False, 
        recompute_metrics=False,
        wandb_logging=False,
        verbose_progress_bar=1,
        keep_last_k=None
    ):
        # Init wandb
        if callback_path is not None and wandb_logging:
            try:
                import wandb
                wandb.init(
                    project='world-model-carla',
                    config={
                        'epochs': epochs,
                        'batch_size': dataset_train.batch_size if hasattr(dataset_train, 'batch_size') else None,
                        'callback_path': callback_path
                    },
                    name=callback_path.split('/')[-1] if callback_path else 'world-model-run',
                    sync_tensorboard=True
                )
                print(f"✅ WandB initialized: {wandb.run.url}")
                
                try:
                    class WandBVisualizerStub:
                        def __init__(self, *args, **kwargs): pass
                    WandBVisualizer = WandBVisualizerStub
                    self._wandb_viz = WandBVisualizer(self, callback_path, log_period=step_log_period)
                    print("✅ WandB Visualizer initialized")
                except Exception as viz_err:
                    print(f"  WandB Visualizer init failed: {viz_err}")
                    self._wandb_viz = None
            except Exception as e:
                print(f"WandB init failed: {str(e)}")
                self._wandb_viz = None

        if not self.compiled:
            raise Exception("You must compile your model before training/testing.")

        # Mixed Precision Gradient Scaler
        self.grad_scaler = torch.cuda.amp.GradScaler(init_scale=grad_init_scale, enabled=(grad_init_scale != None) and (precision==torch.float16))
        if self.grad_scaler_state_dict is not None:
            self.grad_scaler.load_state_dict(self.grad_scaler_state_dict)
            self.grad_scaler_state_dict = None
        assert not (precision==torch.float16 and not self.grad_scaler.is_enabled()), "gradient scaling not enabled for float16 precision training!"

        torch.set_anomaly_enabled(detect_anomaly)

        acc_step = 0
        self.zero_grad()

        # Callbacks
        if callback_path is not None:
            if not os.path.isdir(callback_path):
                os.makedirs(callback_path, exist_ok=True)
            writer = SummaryWriter(os.path.join(callback_path, "logs"))
        else:
            writer = None

        try:
            self.on_train_begin()

            # Extract config dictionary gracefully
            cfg = getattr(self, "config", {})
            cfg_dict = dict(cfg) if isinstance(cfg, dict) else {}
            if hasattr(cfg, "__dict__"):
                cfg_dict.update(cfg.__dict__)
            for k in ["model_size", "hidden_size", "stoch_size", "discrete", "batch_size", "L", "H", "epochs", "epoch_length", "env_step_period", "buffer_capacity", "model_lr", "actor_lr", "critic_lr"]:
                if k in cfg:
                    cfg_dict[k] = cfg[k]
            # Optional algorithmic-specific key check (kept generic)
            if "algo_params" in cfg_dict and isinstance(cfg_dict["algo_params"], dict):
                cfg_dict.update(cfg_dict["algo_params"])

            # Live Dashboard
            dashboard = None
            if verbose_progress_bar >= 1:
                try:
                    class LiveDashboardStub:
                        def __init__(self, *args, **kwargs): pass
                        def __enter__(self): return self
                        def __exit__(self, exc_type, exc_val, exc_tb): pass
                        def start_epoch(self, *args, **kwargs): pass
                        def update(self, *args, **kwargs): pass
                        def finalize_epoch(self, *args, **kwargs): pass
                    LiveDashboard = LiveDashboardStub
                    dashboard = LiveDashboard(
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch if steps_per_epoch else 100,
                    )
                    dashboard.start()
                except Exception as e:
                    print(f"[Dashboard] Failed to start: {e}")
                    dashboard = None
            self._dashboard = dashboard

            for epoch in range(initial_epoch, epochs):
                epoch_start_time = time.time()

                if dashboard:
                    dashboard.begin_epoch(epoch)

                epoch_losses = {}
                epoch_metrics = {}
                self.reset_infos()
                self.train()
                self.on_epoch_begin(epoch=epoch + 1)

                for step, batch in enumerate(dataset_train):
                    if steps_per_epoch is not None and step >= steps_per_epoch * accumulated_steps:
                        break

                    inputs, targets = batch["inputs"], batch["targets"]
                    inputs = self.transfer_to_device(inputs)
                    targets = self.transfer_to_device(targets)

                    batch_losses, batch_metrics, acc_step = self.train_step(inputs=inputs, targets=targets, precision=precision, grad_scaler=self.grad_scaler, accumulated_steps=accumulated_steps, acc_step=acc_step, eval_training=eval_training)

                    for key, value in batch_losses.items():
                        raw_val = value.item() if hasattr(value, 'item') else float(value)
                        epoch_losses[key] = epoch_losses.get(key, 0.0) + raw_val
                    for key, value in batch_metrics.items():
                        raw_val = value.item() if hasattr(value, 'item') else float(value)
                        epoch_metrics[key] = epoch_metrics.get(key, 0.0) + raw_val

                    if acc_step > 0:
                        continue

                    if dashboard:
                        dashboard.update_train_step(step, batch_losses)

                    # Prevent OS window manager from freezing the process
                    import sys
                    if "pygame" in sys.modules:
                        try:
                            sys.modules["pygame"].event.pump()
                        except:
                            pass

                    if writer is not None and self.model_step % step_log_period == 0:
                        self.log_step(losses=batch_losses, metrics=batch_metrics, infos=self.infos, writer=writer, step=self.model_step, tag="Training-step", inputs=inputs, targets=targets, outputs=batch_losses)

                    self.on_step_end(
                        evaluate=self.model_step % eval_period_step == 0 if eval_period_step != None else False,
                        save=False, 
                        log_figure=self.model_step % log_figure_period_step == 0 if log_figure_period_step != None else False, 
                        callback_path=callback_path, 
                        epoch=epoch + 1,
                        step=self.model_step, 
                        inputs=inputs, 
                        targets=targets, 
                        dataset_eval=dataset_eval, 
                        eval_steps=eval_steps, 
                        verbose_eval=verbose_eval,
                        writer=writer,
                        recompute_metrics=recompute_metrics,
                        keep_last_k=keep_last_k
                    )

                    if steps_per_epoch is not None:
                        if step + 1 >= steps_per_epoch * accumulated_steps:
                            break

                for key, value in epoch_losses.items():
                    epoch_losses[key] = value / (steps_per_epoch * accumulated_steps if steps_per_epoch is not None else len(dataset_train))
                for key, value in epoch_metrics.items():
                    epoch_metrics[key] = value / (steps_per_epoch * accumulated_steps if steps_per_epoch is not None else len(dataset_train))

                elapsed_time = time.time() - epoch_start_time
                if dashboard:
                    dashboard.end_epoch(epoch_losses, elapsed_time)

                if writer is not None:
                    self.log_step(losses=epoch_losses, metrics=epoch_metrics, infos={}, writer=writer, step=epoch + 1, tag="Training-epoch", inputs=inputs, targets=targets, outputs=epoch_losses)

                self.on_epoch_end(
                    evaluate=(epoch + 1) % eval_period_epoch == 0 if eval_period_epoch != None else False,
                    save=(epoch + 1) % saving_period_epoch == 0 if saving_period_epoch != None else False, 
                    log_figure=(epoch + 1) % log_figure_period_epoch == 0 if log_figure_period_epoch != None else False, 
                    callback_path=callback_path, 
                    epoch=epoch + 1, 
                    inputs=inputs, 
                    targets=targets, 
                    dataset_eval=dataset_eval, 
                    eval_steps=eval_steps, 
                    verbose_eval=verbose_eval,
                    writer=writer,
                    recompute_metrics=recompute_metrics,
                    keep_last_k=keep_last_k
                )

            if dashboard:
                dashboard.stop()

        except Exception as e:
            if writer is not None:
                writer.add_text("Exceptions", "Date: {} \n{}".format(time.ctime(), str(e)), self.model_step)
            raise e

    def _evaluate(self, dataset, writer, eval_steps=None, verbose=0, recompute_metrics=False, tag="Evaluation", verbose_progress_bar=1):
        if dataset is not None:
            if not isinstance(dataset, list):
                dataset_list = [dataset]
            else:
                dataset_list = dataset
                
            for dataset_i, ds in enumerate(dataset_list):
                val_losses, val_metrics = self.evaluate(ds, eval_steps, verbose, recompute_metrics, verbose_progress_bar)
                if not hasattr(self, '_dashboard') or self._dashboard is None:
                    self.print_step(val_losses, val_metrics, "eval")
                if writer is not None:
                    self.log_step(losses=val_losses, metrics=val_metrics, infos={}, writer=writer, step=self.model_step, tag=os.path.join(tag, str(dataset_i)))

    def evaluate(self, dataset_eval, eval_steps=None, verbose=0, recompute_metrics=False, verbose_progress_bar=1):
        self.eval()
        self.reset_infos()
        epoch_losses = {}
        epoch_metrics = {}
        if recompute_metrics:
            epoch_truths = {}
            epoch_preds = {}

        dashboard = getattr(self, '_dashboard', None)
        if dashboard:
            dashboard.begin_eval(eval_steps if eval_steps else len(dataset_eval))

        for step, batch in enumerate(dataset_eval):
            inputs, targets = batch["inputs"], batch["targets"]
            inputs = self.transfer_to_device(inputs)
            targets = self.transfer_to_device(targets)

            batch_losses, batch_metrics, batch_truths, batch_preds = self.eval_step(inputs, targets, verbose)

            for key, value in batch_losses.items():
                epoch_losses[key] = epoch_losses[key] + value if key in epoch_losses else value.type(torch.float64)
            for key, value in batch_metrics.items():
                epoch_metrics[key] = epoch_metrics[key] + value if key in epoch_metrics else value.type(torch.float64)
            if recompute_metrics:
                for key, value in batch_truths.items():
                    epoch_truths[key] = epoch_truths[key] + value if key in epoch_truths else value
                for key, value in batch_preds.items():
                    epoch_preds[key] = epoch_preds[key] + value if key in epoch_preds else value

            if dashboard:
                dashboard.update_eval_step(step)

            if eval_steps:
                if step + 1 >= eval_steps:
                    break

        for key, value in epoch_losses.items():
            epoch_losses[key] = value / (eval_steps if eval_steps is not None else len(dataset_eval))

        if recompute_metrics:
            for key in epoch_metrics.keys():
                epoch_metrics[key] = self.metrics["outputs"](epoch_truths[key], epoch_preds[key])
        else:
            for key, value in epoch_metrics.items():
                epoch_metrics[key] = value / (eval_steps if eval_steps is not None else len(dataset_eval))

        if dashboard:
            eval_losses_flat = {k: v.item() if hasattr(v, 'item') else float(v) for k, v in epoch_losses.items()}
            dashboard.end_eval(eval_losses_flat)

        return epoch_losses, epoch_metrics
