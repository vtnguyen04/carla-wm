import collections
import concurrent.futures
import datetime
import json
import os
import re
import time

import numpy as np

from . import path
from carla_env.toolkit.utils import get_logger
import imageio # Add this import

log = get_logger(log_dir=".", job_name="logger")


class Logger:
    def __init__(self, step, outputs, multiplier=1):
        assert outputs, "Provide a list of logger outputs."
        self.step = step
        self.outputs = outputs
        self.multiplier = multiplier
        self._last_step = None
        self._last_time = None
        self._metrics = []

    def add(self, mapping, prefix=None):
        step = int(self.step) * self.multiplier
        for name, value in dict(mapping).items():
            name = f"{prefix}/{name}" if prefix else name
            value = np.asarray(value)
            if len(value.shape) not in (0, 1, 2, 3, 4):
                raise ValueError(f"Shape {value.shape} for name '{name}' cannot be " "interpreted as scalar, histogram, image, or video.")
            self._metrics.append((step, name, value))

    def scalar(self, name, value):
        self.add({name: value})

    def image(self, name, value):
        self.add({name: value})

    def video(self, name, value):
        self.add({name: value})

    def write(self, fps=False):
        if fps:
            value = self._compute_fps()
            if value is not None:
                self.scalar("fps", value)
        if not self._metrics:
            return
        for output in self.outputs:
            output(tuple(self._metrics))
        self._metrics.clear()

    def _compute_fps(self):
        step = int(self.step) * self.multiplier
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return None
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration


class AsyncOutput:
    def __init__(self, callback, parallel=True):
        self._callback = callback
        self._parallel = parallel
        if parallel:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._future = None

    def __call__(self, summaries):
        if self._parallel:
            self._future and self._future.result()
            self._future = self._executor.submit(self._callback, summaries)
        else:
            self._callback(summaries)


class TerminalOutput:
    def __init__(self, pattern=r".*", name=None):
        self._pattern = re.compile(pattern)
        self._name = name
        try:
            import rich.console

            self._console = rich.console.Console()
        except ImportError:
            self._console = None

    def __call__(self, summaries):
        step = max(s for s, _, _, in summaries)
        scalars = {k: float(v) for _, k, v in summaries if len(v.shape) == 0}
        scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
        formatted = {k: self._format_value(v) for k, v in scalars.items()}
        if self._console:
            if self._name:
                self._console.rule(f"[green bold]{self._name} (Step {step})")
            else:
                self._console.rule(f"[green bold]Step {step}")
            self._console.print(" [blue]/[/blue] ".join(f"{k} {v}" for k, v in formatted.items()))
            log.info("")
        else:
            message = " / ".join(f"{k} {v}" for k, v in formatted.items())
            message = f"[{step}] {message}"
            if self._name:
                message = f"[{self._name}] {message}"
            log.info(message)

    def _format_value(self, value):
        value = float(value)
        if value == 0:
            return "0"
        elif 0.01 < abs(value) < 10000:
            value = f"{value:.2f}"
            value = value.rstrip("0")
            value = value.rstrip("0")
            value = value.rstrip(".")
            return value
        else:
            value = f"{value:.1e}"
            value = value.replace(".0e", "e")
            value = value.replace("+0", "")
            value = value.replace("+", "")
            value = value.replace("-0", "-")
        return value


class JSONLOutput(AsyncOutput):
    def __init__(self, logdir, filename="metrics.jsonl", pattern=r".*", parallel=True):
        super().__init__(self._write, parallel)
        self._filename = filename
        self._pattern = re.compile(pattern)
        self._logdir = path.Path(logdir)
        self._logdir.mkdirs()

    def _write(self, summaries):
        bystep = collections.defaultdict(dict)
        for step, name, value in summaries:
            if len(value.shape) == 0 and self._pattern.search(name):
                bystep[step][name] = float(value)
        lines = "".join([json.dumps({"step": step, **scalars}) + "\n" for step, scalars in bystep.items()])
        with (self._logdir / self._filename).open("a") as f:
            f.write(lines)


class TensorBoardOutput(AsyncOutput):
    def __init__(self, logdir, fps=20, maxsize=1e9, parallel=True):
        super().__init__(self._write, parallel)
        self._logdir = str(logdir)
        if self._logdir.startswith("/gcs/"):
            self._logdir = self._logdir.replace("/gcs/", "gs://")
        self._fps = fps
        self._writer = None
        self._maxsize = self._logdir.startswith("gs://") and maxsize
        if self._maxsize:
            self._checker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._promise = None

    def _write(self, summaries):
        from torch.utils.tensorboard import SummaryWriter

        if not self._writer:
            log.info("Creating new TensorBoard event file writer using PyTorch.")
            self._writer = SummaryWriter(self._logdir)
            
        for step, name, value in summaries:
            try:
                if len(value.shape) == 0:
                    self._writer.add_scalar(name, value, step)
                elif len(value.shape) == 1:
                    if len(value) > 1024:
                        value = value.copy()
                        np.random.shuffle(value)
                        value = value[:1024]
                    self._writer.add_histogram(name, value, step)
                elif len(value.shape) == 2:
                    self._writer.add_image(name, value[None], step)
                elif len(value.shape) == 3:
                    if value.shape[-1] == 3: # HWC -> CHW
                        value = value.transpose(2, 0, 1)
                    self._writer.add_image(name, value, step)
                elif len(value.shape) == 4:
                    # PyTorch add_video expects (N, T, C, H, W)
                    # Current value is (T, H, W, C)
                    t, h, w, c = value.shape
                    if value.shape[-1] == 3: # THWC -> TCHW
                        v = value.transpose(0, 3, 1, 2)
                    else:
                        v = value
                    self._writer.add_video(name, v[None], step, fps=self._fps)
            except Exception as e:
                log.error(f"Error writing summary {name}: {e}")
        self._writer.flush()

    def _check(self):
        return 0

    def _video_summary(self, name, video, step):
        pass


class WandBOutput:
    def __init__(self, run_name, config, project=None, entity=None, pattern=r".*", resume=False):
        self._pattern = re.compile(pattern)
        import wandb

        wandb.init(
            name=run_name,
            entity=entity,
            project=project or "twister-carla",
            config=dict(config),
            resume=resume,
        )
        self._wandb = wandb

    def __call__(self, summaries):
        bystep = collections.defaultdict(dict)
        wandb = self._wandb
        for step, name, value in summaries:
            if len(value.shape) == 0 and self._pattern.search(name):
                bystep[step][name] = float(value)
            elif len(value.shape) == 1:
                bystep[step][name] = wandb.Histogram(value)
            elif len(value.shape) == 2:
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
                bystep[step][name] = wandb.Image(value)
            elif len(value.shape) == 3:
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                bystep[step][name] = wandb.Image(value)
            elif len(value.shape) == 4:
                assert value.shape[3] in [1, 3, 4], f"Invalid shape: {value.shape}"
                # WandB Video expects (T, C, H, W) or (N, T, C, H, W)
                value = np.transpose(value, [0, 3, 1, 2])
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                bystep[step][name] = wandb.Video(value[None], fps=self._fps if hasattr(self, '_fps') else 20)

        for step, metrics in bystep.items():
            self._wandb.log(metrics, step=step)


class MLFlowOutput:
    def __init__(self, run_name=None, resume_id=None, config=None, prefix=None):
        import mlflow

        self._mlflow = mlflow
        self._prefix = prefix
        self._setup(run_name, resume_id, config)

    def __call__(self, summaries):
        bystep = collections.defaultdict(dict)
        for step, name, value in summaries:
            if len(value.shape) == 0 and self._pattern.search(name):
                name = f"{self._prefix}/{name}" if self._prefix else name
                bystep[step][name] = float(value)
        for step, metrics in bystep.items():
            self._mlflow.log_metrics(metrics, step=step)

    def _setup(self, run_name, resume_id, config):
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "local")
        run_name = run_name or os.environ.get("MLFLOW_RUN_NAME")
        resume_id = resume_id or os.environ.get("MLFLOW_RESUME_ID")
        log.info(f"MLFlow Tracking URI: {tracking_uri}")
        log.info(f"MLFlow Run Name:     {run_name}")
        log.info(f"MLFlow Resume ID:    {resume_id}")
        if resume_id:
            runs = self._mlflow.search_runs(None, f'tags.resume_id="{resume_id}"')
            assert len(runs), ("No runs to resume found.", resume_id)
            self._mlflow.start_run(run_name=run_name, run_id=runs["run_id"].iloc[0])
            for key, value in config.items():
                self._mlflow.log_param(key, value)
        else:
            tags = {"resume_id": resume_id or ""}
            self._mlflow.start_run(run_name=run_name, tags=tags)


def _encode_gif(frames, fps):
    try:
        from io import BytesIO
        buffer = BytesIO()
        # imageio.mimsave expects frames to be (T, H, W, C)
        # It handles conversion to uint8 if input is floating
        imageio.mimsave(buffer, frames, format='gif', fps=fps)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        log.error(f"Error encoding GIF with imageio: {e}")
        raise OSError(f"Failed to encode GIF with imageio: {e}")