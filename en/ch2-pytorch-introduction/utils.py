import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from torch import Tensor
from torch.types import Device
from torchmetrics import Metric

import dnnlpy

__all__ = [
    'evaluate',
    'load_checkpoint',
    'save_checkpoint',
    'train_and_evaluate',
    'train_one_epoch',
]


def train_one_epoch(
    model: nn.Module,
    dataloader: utils.DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    metric: Metric,
    device: torch.device,
) -> tuple[float, float]:
    """Train a model for one epoch and return average loss and metric values.

    Args:
        model (Module): Model to train.
        dataloader (DataLoader): Batches of input tensors and target tensors.
        loss_fn (Module): Loss module used to optimize the model.
        optimizer (Optimizer): Optimizer that updates model parameters.
        metric (Metric): TorchMetrics metric updated from model predictions and targets.
        device (torch.device): Device where batches are moved before the forward pass.

    Returns:
        tuple[float, float]: A tuple containing the average loss and computed metric value.
    """
    model.train()
    metric.reset()

    total_loss = 0.0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        metric.update(logits.detach(), y)

    avg_loss = total_loss / len(dataloader)
    avg_metric = metric.compute().item()
    return avg_loss, avg_metric


def evaluate(
    model: nn.Module,
    dataloader: utils.DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    metric: Metric,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate a model for one epoch and return average loss and metric values.

    Args:
        model (Module): Model to evaluate.
        dataloader (DataLoader): Batches of input tensors and target tensors.
        loss_fn (Module): Loss module used to measure prediction error.
        metric (Optimizer): TorchMetrics metric updated from model predictions and targets.
        device (torch.device): Device where batches are moved before the forward pass.

    Returns:
        tuple[float, float]: A tuple containing the average loss and computed metric value.
    """
    model.eval()
    metric.reset()

    total_loss = 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            metric.update(logits, y)

    avg_loss = total_loss / len(dataloader)
    avg_metric = metric.compute().item()
    return avg_loss, avg_metric


def train_and_evaluate(
    model: nn.Module,
    train_dl: utils.DataLoader[tuple[Tensor, Tensor]],
    val_dl: utils.DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    train_metric: Metric,
    val_metric: Metric,
    metric_name: str,
    num_epochs: int,
    device: Device = None,
) -> None:
    """Train and validate a model for multiple epochs.

    Args:
        model (Module): Model to train and evaluate.
        train_dl (DataLoader): Training dataloader.
        val_dl (DataLoader): Validation dataloader.
        loss_fn (Module): Loss module used for both training and validation.
        optimizer (Optimizer): Optimizer that updates model parameters.
        train_metric (Metric): Metric used on the training split.
        val_metric (Metric): Metric used on the validation split.
        metric_name (str): Name of the metric used for logging.
        num_epochs (int): Number of epochs to run.
        device (Device, default: None): Device to use. Defaults to `get_default_device()`
            when omitted.
    """
    if device is None:
        device = dnnlpy.get_default_device()
    else:
        device = torch.device(device)

    model.to(device)
    train_metric.to(device)
    val_metric.to(device)

    for epoch in range(1, num_epochs + 1):
        loss, score = train_one_epoch(
            model=model,
            dataloader=train_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metric=train_metric,
            device=device,
        )
        val_loss, val_score = evaluate(
            model=model,
            dataloader=val_dl,
            loss_fn=loss_fn,
            metric=val_metric,
            device=device,
        )

        w = len(str(num_epochs))
        print(
            f'Epoch [{epoch:{w}d}/{num_epochs:{w}d}] '
            f'| loss: {loss:.4f} '
            f'| {metric_name}: {score:.4f} '
            f'| val_loss: {val_loss:.4f} '
            f'| val_{metric_name}: {val_score:.4f}'
        )


def load_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: Device = None,
) -> tuple[int, list[dict[str, float]]]:
    """Load a checkpoint and restore the model and optimizer states.

    Args:
        path (str | os.PathLike[str]): Path to the checkpoint file.
        model (nn.Module): Model to restore.
        optimizer (optim.Optimizer): Optimizer to restore.
        device (Device, optional): Device to map the checkpoint to. Defaults to None,
            which uses `dnnlpy.get_default_device()`.

    Returns:
        tuple[int, list[dict[str, float]]]: A tuple containing the epoch number and
            training history loaded from the checkpoint.
    """
    if device is None:
        device = dnnlpy.get_default_device()
    else:
        device = torch.device(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    epoch = checkpoint['epoch']
    history = checkpoint['history']
    return epoch, history


def save_checkpoint(
    path: str | os.PathLike[str],
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    history: list[dict[str, float]],
) -> None:
    """Save a checkpoint containing the model and optimizer states.

    Args:
        path (str | os.PathLike[str]): Path to save the checkpoint file.
        epoch (int): Current epoch number.
        model (nn.Module): Model to save.
        optimizer (optim.Optimizer): Optimizer to save.
        history (list[dict[str, float]]): Training history to save.
    """
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history': history,
    }
    torch.save(checkpoint, path)
