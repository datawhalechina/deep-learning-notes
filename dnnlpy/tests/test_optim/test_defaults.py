import inspect
from typing import Any

import torch.optim as optim

import dnnlpy.optim as dopt


def _sig(function: Any, name: str):
    sig = inspect.signature(function.__init__)
    return sig.parameters[name].default


def test_optimizer_sigs_match_torch_optim_sigs():
    assert _sig(dopt.SGD, 'lr') == _sig(optim.SGD, 'lr')
    assert _sig(dopt.SGD, 'momentum') == _sig(optim.SGD, 'momentum')
    assert _sig(dopt.SGD, 'weight_decay') == _sig(optim.SGD, 'weight_decay')
    assert _sig(dopt.SGD, 'nesterov') == _sig(optim.SGD, 'nesterov')

    assert _sig(dopt.Adagrad, 'lr') == _sig(optim.Adagrad, 'lr')
    assert _sig(dopt.Adagrad, 'lr_decay') == _sig(optim.Adagrad, 'lr_decay')
    assert _sig(dopt.Adagrad, 'eps') == _sig(optim.Adagrad, 'eps')
    assert _sig(dopt.Adagrad, 'weight_decay') == _sig(optim.Adagrad, 'weight_decay')
    assert _sig(dopt.Adagrad, 'initial_accumulator_value') == _sig(
        optim.Adagrad, 'initial_accumulator_value'
    )

    assert _sig(dopt.RMSprop, 'lr') == _sig(optim.RMSprop, 'lr')
    assert _sig(dopt.RMSprop, 'rho') == _sig(optim.RMSprop, 'alpha')
    assert _sig(dopt.RMSprop, 'eps') == _sig(optim.RMSprop, 'eps')
    assert _sig(dopt.RMSprop, 'weight_decay') == _sig(optim.RMSprop, 'weight_decay')
    assert _sig(dopt.RMSprop, 'momentum') == _sig(optim.RMSprop, 'momentum')

    assert _sig(dopt.Adadelta, 'lr') == _sig(optim.Adadelta, 'lr')
    assert _sig(dopt.Adadelta, 'rho') == _sig(optim.Adadelta, 'rho')
    assert _sig(dopt.Adadelta, 'eps') == _sig(optim.Adadelta, 'eps')
    assert _sig(dopt.Adadelta, 'weight_decay') == _sig(optim.Adadelta, 'weight_decay')

    assert _sig(dopt.Adam, 'lr') == _sig(optim.Adam, 'lr')
    assert _sig(dopt.Adam, 'betas') == _sig(optim.Adam, 'betas')
    assert _sig(dopt.Adam, 'eps') == _sig(optim.Adam, 'eps')
    assert _sig(dopt.Adam, 'weight_decay') == _sig(optim.Adam, 'weight_decay')

    assert _sig(dopt.AdamW, 'lr') == _sig(optim.AdamW, 'lr')
    assert _sig(dopt.AdamW, 'betas') == _sig(optim.AdamW, 'betas')
    assert _sig(dopt.AdamW, 'eps') == _sig(optim.AdamW, 'eps')
    assert _sig(dopt.AdamW, 'weight_decay') == _sig(optim.AdamW, 'weight_decay')

    assert _sig(dopt.Muon, 'lr') == _sig(optim.Muon, 'lr')
    assert _sig(dopt.Muon, 'weight_decay') == _sig(optim.Muon, 'weight_decay')
    assert _sig(dopt.Muon, 'momentum') == _sig(optim.Muon, 'momentum')
    assert _sig(dopt.Muon, 'nesterov') == _sig(optim.Muon, 'nesterov')
    assert _sig(dopt.Muon, 'ns_coefficients') == _sig(optim.Muon, 'ns_coefficients')
    assert _sig(dopt.Muon, 'eps') == _sig(optim.Muon, 'eps')
    assert _sig(dopt.Muon, 'ns_steps') == _sig(optim.Muon, 'ns_steps')
