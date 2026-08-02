import warnings
from collections.abc import Mapping, Sequence
from itertools import count
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr
import torch.utils.data as utils
from torch import Tensor
from torch.types import Device
from torchmetrics import Metric
from torchmetrics.aggregation import MeanMetric

from .configtools import get_default_device, set_seed

type MetricCollection = Metric | Mapping[str, Metric] | Sequence[Metric] | None

__all__ = ['Trainer']


class Trainer:
    """A lightweight trainer for common PyTorch training loops."""

    history: list[dict[str, int | float]]

    def __init__(
        self,
        device: Device = None,
        amp: bool = False,
        precision: Literal['fp32', 'fp16', 'bf16'] = 'fp32',
        seed: int | None = None,
        deterministic: bool = False,
        benchmark: bool = False,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: Literal['norm', 'value'] = 'norm',
        max_epochs: int | None = None,
        max_steps: int | None = None,
        checkpoint_path: str | PathLike[str] | None = None,
        checkpoint_every_n_epochs: int = 1,
        verbose: bool = True,
    ):
        """Initialize a trainer.

        Args:
            device (Device, default: None): Device used for model, batches, and metrics.
                If `None` or `'auto'`, `get_default_device()` function is used to select
                the device.
            amp (bool, default: False): Whether to enable automatic mixed precision.
            precision ({'fp32', 'fp16', 'bf16'}, default: 'fp32'): Floating-point precision
                used by autocast when `amp=True`.
            seed (int | None, default: None): Random seed passed to `set_seed`.
            deterministic (bool, default: False): Whether to request deterministic PyTorch
                algorithms. We use a `warn_only` policy to avoid exceptions when deterministic
                algorithms are not available.
            benchmark (bool, default: False): Whether to enable cuDNN benchmarking.
            gradient_clip_val (float | None, default: None): Gradient clipping value. `None`
                disables gradient clipping.
            gradient_clip_algorithm ({'norm', 'value'}, default: 'norm'): Clipping algorithm
                used when `gradient_clip_val` is set.
            max_epochs (int | None, default: None): Number of training epochs to run. `None`
                disables this limit.
            max_steps (int | None, default: None): Maximum number of optimizer steps. `None`
                disables this limit.
            checkpoint_path (str | PathLike[str] | None, default: None): Checkpoint file or
                directory path. `None` disables automatic checkpointing.
            checkpoint_every_n_epochs (int, default: 1): Frequency for automatic checkpoint saves.
            verbose (bool, default: True): Whether to print training progress.
        """
        self.amp = amp
        self.precision = precision
        self.seed = seed
        self.deterministic = deterministic
        self.benchmark = benchmark
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.max_epochs = max_epochs
        self.max_steps = max_steps

        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            self.checkpoint_path = None

        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs

        self.verbose = verbose
        self.history = []
        self.global_step = 0

        if self.max_epochs is not None and self.max_epochs < 1:
            raise AssertionError('`max_epochs` must be at least 1.')
        if self.max_steps is not None and self.max_steps < 1:
            raise AssertionError('`max_steps` must be at least 1.')
        if self.gradient_clip_val is not None and self.gradient_clip_val < 0:
            raise AssertionError('`gradient_clip_val` must be non-negative.')
        if self.gradient_clip_algorithm not in {'norm', 'value'}:
            raise AssertionError('`gradient_clip_algorithm` must be `norm` or `value`.')
        if self.precision not in {'fp32', 'fp16', 'bf16'}:
            raise AssertionError('`precision` must be `fp32`, `fp16`, or `bf16`.')
        if self.checkpoint_every_n_epochs < 1:
            raise AssertionError('`checkpoint_every_n_epochs` must be at least 1.')

        set_seed(
            self.seed,
            deterministic=self.deterministic,
            benchmark=self.benchmark,
            warn_only=True,
        )

        self.device = (
            get_default_device()
            if device is None or device == 'auto'
            else torch.device(device)
        )
        self._amp_dtype = self._get_amp_dtype()
        self._scaler = torch.GradScaler(
            self.device.type,
            enabled=self.amp
            and self.precision == 'fp16'
            and self.device.type in {'cuda', 'xpu', 'cpu'},
        )

    def fit(
        self,
        model: nn.Module,
        train_dataloader: utils.DataLoader[Any],
        val_dataloader: utils.DataLoader[Any] | None = None,
        *,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: lr.LRScheduler | None = None,
        lr_scheduler_interval: Literal['epoch', 'step'] = 'epoch',
        lr_scheduler_monitor: str = 'val_loss',
        train_metrics: MetricCollection = None,
        val_metrics: MetricCollection = None,
        resume_from_checkpoint: str | PathLike[str] | None = None,
    ) -> None:
        """Train a model and return one dictionary of logs per epoch.

        Args:
            model (nn.Module):
                Model to train.
            train_dataloader (DataLoader[Any]):
                Dataloader for training batches.
            val_dataloader (DataLoader[Any] | None, default: None):
                Optional dataloader for validation batches.
            loss_fn (nn.Module):
                Loss module used for training and validation.
            optimizer (optim.Optimizer):
                Optimizer used to update model parameters.
            lr_scheduler (lr.LRScheduler | None, default: None):
                Optional learning-rate scheduler.
            lr_scheduler_interval ({'epoch', 'step'}, default: 'epoch'):
                Whether to step the scheduler after each epoch or optimizer step.
            lr_scheduler_monitor (str, default: 'val_loss'):
                Log key used when stepping `ReduceLROnPlateau`.
            train_metrics (MetricCollection, default: None):
                Optional TorchMetrics metric, mapping, or sequence for training logs.
            val_metrics (MetricCollection, default: None):
                Optional TorchMetrics metric, mapping, or sequence for validation logs.
            resume_from_checkpoint (str | PathLike[str] | None, default: None):
                Optional checkpoint path to resume from.

        Returns:
            list[dict[str, float]]: Per-epoch training history.
        """
        if lr_scheduler_interval not in {'epoch', 'step'}:
            raise AssertionError('`lr_scheduler_interval` must be `epoch` or `step`.')

        if lr_scheduler_interval == 'step':
            if isinstance(lr_scheduler, lr.ReduceLROnPlateau):
                raise AssertionError(
                    '`ReduceLROnPlateau` can only be stepped by epoch.'
                )

        model.to(self.device)
        train_metrics = self._prepare_metrics(train_metrics)
        val_metrics = self._prepare_metrics(val_metrics)
        start_epoch = 1

        if resume_from_checkpoint is not None:
            checkpoint = self.load_checkpoint(
                resume_from_checkpoint,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
            start_epoch = int(checkpoint.get('epoch', 0)) + 1
        else:
            self.history.clear()
            self.global_step = 0

        if self.verbose:
            print(f'Training on {self.device}...')

        if self.max_epochs is None:
            epochs = count(start_epoch)
        else:
            epochs = range(start_epoch, self.max_epochs + 1)

        for epoch in epochs:
            if self._training_limit_reached():
                break

            logs: dict[str, int | float] = {
                'epoch': epoch,
                'global_step': self.global_step,
            }

            train_logs, limit_reached = self._train_epoch(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                lr_scheduler_interval=lr_scheduler_interval,
                metrics=train_metrics,
            )
            logs.update({f'{name}': value for name, value in train_logs.items()})

            if val_dataloader is not None:
                val_logs = self._validate_epoch(
                    model=model,
                    dataloader=val_dataloader,
                    loss_fn=loss_fn,
                    metrics=val_metrics,
                )
                logs.update({f'val_{name}': value for name, value in val_logs.items()})

            self.history.append(logs)

            if lr_scheduler_interval == 'epoch':
                self._step_lr_scheduler(lr_scheduler, logs, lr_scheduler_monitor)

            if self.verbose:
                self._print_epoch(logs)

            self._save_checkpoint_if_needed(epoch, model, optimizer, lr_scheduler)

            if limit_reached:
                break

    def validate(
        self,
        model: nn.Module,
        dataloader: utils.DataLoader[Any],
        *,
        loss_fn: nn.Module,
        metrics: MetricCollection = None,
    ) -> dict[str, float]:
        """Evaluate a model once and return validation logs.

        Args:
            model (nn.Module):
                Model to evaluate.
            dataloader (DataLoader[Any]):
                Dataloader for evaluation batches.
            loss_fn (nn.Module):
                Loss module used to measure prediction error.
            metrics (MetricCollection, default: None):
                Optional TorchMetrics metric, mapping, or sequence for evaluation logs.

        Returns:
            dict[str, float]: Validation loss and metric logs.
        """
        model.to(self.device)
        prepared_metrics = self._prepare_metrics(metrics)
        return self._validate_epoch(model, dataloader, loss_fn, prepared_metrics)

    def save_checkpoint(
        self,
        path: str | PathLike[str],
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        lr_scheduler: lr.LRScheduler | None = None,
        *,
        epoch: int | None = None,
    ) -> None:
        """Save model, optimizer, scheduler, scaler, trainer config, and history.

        Args:
            path (str | PathLike[str]):
                Destination checkpoint path.
            model (nn.Module):
                Model whose state will be saved.
            optimizer (optim.Optimizer | None, default: None):
                Optional optimizer whose state will be saved.
            lr_scheduler (lr.LRScheduler | None, default: None):
                Optional scheduler whose state will be saved.
            epoch (int | None, default: None):
                Epoch number to store. `None` stores `len(history)`.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'epoch': len(self.history) if epoch is None else epoch,
            'global_step': self.global_step,
            'model': model.state_dict(),
            'history': self.history,
            'trainer': self._state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        if lr_scheduler is not None:
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
        if self._scaler.is_enabled():
            checkpoint['scaler'] = self._scaler.state_dict()
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(
        self,
        path: str | PathLike[str],
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        lr_scheduler: lr.LRScheduler | None = None,
        *,
        map_location: Device = None,
    ) -> dict[str, Any]:
        """Load a checkpoint into the model and optional optimizer/scheduler.

        Args:
            path (str | PathLike[str]):
                Source checkpoint path.
            model (nn.Module):
                Model whose state will be loaded.
            optimizer (optim.Optimizer | None, default: None):
                Optional optimizer whose state will be loaded when present in the checkpoint.
            lr_scheduler (lr.LRScheduler | None, default: None):
                Optional scheduler whose state will be loaded when present in the checkpoint.
            map_location (Device, default: None):
                Device mapping passed to `torch.load`. `None` uses the trainer device.

        Returns:
            dict[str, Any]: Loaded checkpoint dictionary.
        """
        if map_location is not None:
            map_location = torch.device(map_location)
        else:
            map_location = self.device

        checkpoint = torch.load(Path(path), map_location=map_location)
        model.load_state_dict(checkpoint['model'])

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        if self._scaler.is_enabled() and 'scaler' in checkpoint:
            self._scaler.load_state_dict(checkpoint['scaler'])

        self.history = list(checkpoint.get('history', []))
        self.global_step = int(checkpoint.get('global_step', 0))

        return checkpoint

    def _train_epoch(
        self,
        *,
        model: nn.Module,
        dataloader: utils.DataLoader[Any],
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: lr.LRScheduler | None,
        lr_scheduler_interval: Literal['epoch', 'step'],
        metrics: dict[str, Metric],
    ) -> tuple[dict[str, float], bool]:
        """Run one training epoch and report whether a training limit was reached."""
        model.train()
        self._reset_metrics(metrics)
        loss_metric = MeanMetric().to(self.device)
        limit_reached = False

        for batch in dataloader:
            batch = self._move_batch(batch)
            optimizer.zero_grad()

            with self._autocast_context():
                loss, preds, targets = self._default_step(model, batch, loss_fn)
            self._backward_and_step(loss, model, optimizer)
            self.global_step += 1

            if lr_scheduler_interval == 'step':
                self._step_lr_scheduler(lr_scheduler)

            loss_metric.update(loss.detach())
            self._update_metrics(metrics, preds, targets)

            if self._training_limit_reached():
                limit_reached = True
                break

        logs = self._build_logs(loss_metric, metrics)
        return logs, limit_reached

    def _validate_epoch(
        self,
        model: nn.Module,
        dataloader: utils.DataLoader[Any],
        loss_fn: nn.Module,
        metrics: dict[str, Metric],
    ) -> dict[str, float]:
        """Run one validation epoch and return aggregate logs."""
        model.eval()
        self._reset_metrics(metrics)
        loss_metric = MeanMetric().to(self.device)

        with torch.inference_mode():
            for batch in dataloader:
                batch = self._move_batch(batch)

                with self._autocast_context():
                    loss, preds, targets = self._default_step(model, batch, loss_fn)
                loss_metric.update(loss.detach())

                self._update_metrics(metrics, preds, targets)

        logs = self._build_logs(loss_metric, metrics)
        return logs

    def _default_step(
        self,
        model: nn.Module,
        batch: Any,
        loss_fn: nn.Module,
    ) -> tuple[Tensor, Any, Any]:
        """Compute loss, predictions, and targets for a single batch."""
        inputs, targets = self._split_batch(batch)
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        return loss, preds, targets

    def _backward_and_step(
        self,
        loss: Tensor,
        model: nn.Module,
        optimizer: optim.Optimizer,
    ) -> None:
        """Backpropagate loss and update optimizer parameters."""
        if self._scaler.is_enabled():
            self._scaler.scale(loss).backward()
            self._clip_gradients(model, optimizer, unscale=True)
            self._scaler.step(optimizer)
            self._scaler.update()
            return

        loss.backward()
        self._clip_gradients(model, optimizer, unscale=False)
        optimizer.step()

    def _clip_gradients(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        *,
        unscale: bool,
    ) -> None:
        """Apply configured gradient clipping to model parameters."""
        if self.gradient_clip_val is None:
            return
        if unscale and self._scaler.is_enabled():
            self._scaler.unscale_(optimizer)

        parameters = [p for p in model.parameters() if p.grad is not None]
        if self.gradient_clip_algorithm == 'norm':
            nn.utils.clip_grad_norm_(parameters, self.gradient_clip_val)
        else:
            nn.utils.clip_grad_value_(parameters, self.gradient_clip_val)

    def _prepare_metrics(self, metrics: MetricCollection) -> dict[str, Metric]:
        """Convert different types of metrics into a dictionary of metrics."""
        if metrics is None:
            return {}
        if isinstance(metrics, Metric):
            prepared = {'metric': metrics}
        elif isinstance(metrics, Mapping):
            prepared = dict(metrics)
        else:
            prepared = {}
            for metric in metrics:
                name = metric.__class__.__name__
                if name in prepared:
                    raise RuntimeError(f'Duplicate metric name: {name}.')
                prepared[name] = metric

        for metric in prepared.values():
            metric.to(self.device)
        return prepared

    @staticmethod
    def _reset_metrics(metrics: dict[str, Metric]) -> None:
        """Reset all metrics before an epoch."""
        for metric in metrics.values():
            metric.reset()

    @staticmethod
    def _update_metrics(metrics: dict[str, Metric], preds: Any, targets: Any) -> None:
        """Update metrics from predictions and targets."""
        if not metrics or preds is None or targets is None:
            return
        for metric in metrics.values():
            metric.update(preds.detach(), targets)

    def _build_logs(self, loss: Metric, metrics: dict[str, Metric]) -> dict[str, float]:
        """Build scalar loss and metric logs for an epoch."""
        logs = {'loss': self._to_float(loss.compute())}
        for name, metric in metrics.items():
            logs[name] = self._to_float(metric.compute())
        return logs

    def _save_checkpoint_if_needed(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: lr.LRScheduler | None,
    ) -> None:
        """Save an automatic checkpoint when the epoch matches the schedule."""
        if self.checkpoint_path is None:
            return
        if epoch % self.checkpoint_every_n_epochs != 0:
            return
        path = self._checkpoint_path_for_epoch(epoch)
        self.save_checkpoint(path, model, optimizer, lr_scheduler, epoch=epoch)

    def _step_lr_scheduler(
        self,
        lr_scheduler: lr.LRScheduler | None,
        logs: dict[str, float] | None = None,
        monitor: str = 'val_loss',
    ) -> None:
        """Step a learning-rate scheduler with optional monitored logs."""
        if lr_scheduler is None:
            return

        if isinstance(lr_scheduler, lr.ReduceLROnPlateau):
            if logs is None or monitor not in logs:
                raise ValueError(
                    f'LR scheduler monitor `{monitor}` is not available in logs.'
                )
            lr_scheduler.step(logs[monitor])
            return

        lr_scheduler.step()

    def _checkpoint_path_for_epoch(self, epoch: int) -> Path:
        """Return the automatic checkpoint path for an epoch."""
        if self.checkpoint_path is None:
            raise AssertionError('`checkpoint_path` is not configured.')
        if self.checkpoint_path.suffix:
            return self.checkpoint_path
        return self.checkpoint_path / f'epoch={epoch}.pth'

    def _state_dict(self) -> dict[str, Any]:
        """Return serializable trainer configuration state."""
        if self.checkpoint_path is not None:
            checkpoint_path = str(self.checkpoint_path)
        else:
            checkpoint_path = None

        return {
            'device': self.device.type,
            'amp': self.amp,
            'precision': self.precision,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'benchmark': self.benchmark,
            'gradient_clip_val': self.gradient_clip_val,
            'gradient_clip_algorithm': self.gradient_clip_algorithm,
            'max_epochs': self.max_epochs,
            'max_steps': self.max_steps,
            'checkpoint_path': checkpoint_path,
            'checkpoint_every_n_epochs': self.checkpoint_every_n_epochs,
        }

    def _split_batch(self, batch: Any) -> tuple[Any, Any]:
        """Split a batch into inputs and targets."""
        if isinstance(batch, Mapping):
            inputs = batch.get('inputs', batch.get('x'))
            targets = batch.get('targets', batch.get('y'))

            if inputs is None or targets is None:
                msg = 'Mapping batches must contain inputs / x and targets / y keys.'
                raise RuntimeError(msg)
            return inputs, targets

        if isinstance(batch, Sequence):
            if len(batch) == 2:
                return batch[0], batch[1]
            else:
                msg = 'More than two items found in sequence batch. '
                msg += 'Using only the first two items as inputs and targets.'
                warnings.warn(msg, RuntimeWarning, stacklevel=4)
                return batch[0], batch[1]

        msg = 'Batches must be mappings or sequences of at least two items.'
        raise RuntimeError(msg)

    def _move_batch(self, batch: Any) -> Any:
        """Recursively move tensors in a batch to the trainer device."""
        if isinstance(batch, Tensor):
            return batch.to(self.device)
        if isinstance(batch, Mapping):
            return {k: self._move_batch(v) for k, v in batch.items()}
        if isinstance(batch, tuple):
            return tuple(self._move_batch(v) for v in batch)
        if isinstance(batch, list):
            return [self._move_batch(v) for v in batch]
        return batch

    def _autocast_context(self) -> torch.autocast:
        """Create the autocast context for the current precision settings."""
        enabled = self.amp and self.precision in {'fp16', 'bf16'}
        return torch.autocast(self.device.type, dtype=self._amp_dtype, enabled=enabled)

    def _get_amp_dtype(self) -> torch.dtype:
        """Return the torch dtype used by AMP autocast."""
        if self.precision == 'bf16':
            return torch.bfloat16
        if self.precision == 'fp16':
            return torch.float16
        return torch.float32

    @staticmethod
    def _to_float(value: Any) -> float:
        """Convert a scalar metric value to a Python float."""
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise RuntimeError('`metric.compute()` must return a scalar tensor.')
            return value.item()
        return float(value)

    def _training_limit_reached(self) -> bool:
        """Return whether the optimizer-step limit was reached."""
        return self.max_steps is not None and self.global_step >= self.max_steps

    def _print_epoch(self, logs: dict[str, float]) -> None:
        """Print one formatted epoch log line."""
        epoch = int(logs.get('epoch', 0))
        metrics = ' | '.join(
            f'{name}: {value:.4f}'
            for name, value in logs.items()
            if name not in {'epoch', 'global_step'}
        )
        if self.max_epochs is None:
            print(f'Epoch [{epoch}] | {metrics}')
            return

        width = len(str(self.max_epochs))
        print(f'Epoch [{epoch:{width}d}/{self.max_epochs:{width}d}] | {metrics}')

    def __repr__(self) -> str:
        """Return a concise string representation of trainer configuration."""
        return (
            f'{self.__class__.__name__}('
            f'device={self.device!r}, '
            f'amp={self.amp!r}, '
            f'precision={self.precision!r}, '
            f'seed={self.seed!r}, '
            f'deterministic={self.deterministic!r}, '
            f'benchmark={self.benchmark!r}, '
            f'gradient_clip_val={self.gradient_clip_val!r}, '
            f'gradient_clip_algorithm={self.gradient_clip_algorithm!r}, '
            f'max_epochs={self.max_epochs!r}, '
            f'max_steps={self.max_steps!r}, '
            f'checkpoint_path={self.checkpoint_path!r}, '
            f'checkpoint_every_n_epochs={self.checkpoint_every_n_epochs!r}, '
            f'verbose={self.verbose!r}'
            ')'
        )
