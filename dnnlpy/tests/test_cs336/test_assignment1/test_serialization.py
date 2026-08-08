from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.optim as dopt

from .adapters import get_adamw_cls, run_load_checkpoint, run_save_checkpoint

__all__ = ['test_checkpointing']


class _TestNet(nn.Module):
    """A simple feedforward neural network for testing purposes."""

    def __init__(self, in_features: int = 100, out_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x


def are_optimizers_equal(
    opt1_state_dict: dict[str, Any],
    opt2_state_dict: dict[str, Any],
    atol: float = 1e-8,
    rtol: float = 1e-5,
):
    """Test if two optimizer state dictionaries are equal, including their parameter
    groups and states.
    """
    # Check if the keys of the main dictionaries are equal (e.g., 'state', 'param_groups')
    if opt1_state_dict.keys() != opt2_state_dict.keys():
        return False

    # Check parameter groups are identical
    if opt1_state_dict['param_groups'] != opt2_state_dict['param_groups']:
        return False

    # Check states
    state1 = opt1_state_dict['state']
    state2 = opt2_state_dict['state']

    if state1.keys() != state2.keys():
        return False

    for key in state1:
        # Assuming state contents are also dictionaries
        if state1[key].keys() != state2[key].keys():
            return False

        for sub_key in state1[key]:
            item1 = state1[key][sub_key]
            item2 = state2[key][sub_key]

            # If both items are tensors, use torch.allclose
            if torch.is_tensor(item1) and torch.is_tensor(item2):
                if not torch.allclose(item1, item2, atol=atol, rtol=rtol):
                    return False
            # For non-tensor items, check for direct equality
            elif item1 != item2:
                return False

    return True


def test_checkpointing(tmp_path: Path):
    """Test the checkpointing functionality by saving and loading a model and optimizer
    state, and verifying that the loaded states match the original states.
    """
    torch.manual_seed(42)
    d_input = 100
    d_output = 10
    num_iters = 10

    model = _TestNet(in_features=d_input, out_features=d_output)
    optimizer_cls = cast(type[dopt.AdamW], get_adamw_cls())
    optimizer = optimizer_cls(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    x = torch.randn(d_input)
    y = torch.randn(d_output)

    # Use 1000 optimization steps for testing
    for it in range(num_iters):
        optimizer.zero_grad()
        y_hat = model(x)
        loss = F.mse_loss(y_hat, y, reduction='sum')
        loss.backward()
        optimizer.step()

    ckpt_path = tmp_path / 'checkpoint.pt'

    # Save the model
    run_save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        iteration=it,
    )

    # Load the model back again
    new_model = _TestNet(in_features=d_input, out_features=d_output)
    new_optimizer = optimizer_cls(
        new_model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    loaded_iterations = run_load_checkpoint(ckpt_path, new_model, new_optimizer)
    assert it == loaded_iterations

    # Compare the loaded model state with the original model state
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    new_model_state = new_model.state_dict()
    new_optimizer_state = new_optimizer.state_dict()

    # Check that state dict keys match
    assert model_state.keys() == new_model_state.keys()
    assert optimizer_state.keys() == new_optimizer_state.keys()

    # compare the model state dicts
    for key in model_state:
        assert_close(model_state[key], new_model_state[key])

    # compare the optimizer state dicts
    assert are_optimizers_equal(optimizer_state, new_optimizer_state)
