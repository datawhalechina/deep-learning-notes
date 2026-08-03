from typing import Literal

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from dnnlpy.models.vae import VAE, AutoEncoder

type LossFn = Literal['mse', 'bce']


def test_autoencoder_encode_decode_and_forward_form_same_reconstruction():
    model = AutoEncoder((1, 4, 4), hidden_dim=8, latent_dim=3)
    x = torch.randn(2, 1, 4, 4, requires_grad=True)

    latent = model.encode(x)
    decoded = model.decode(latent)
    actual = model(x)
    actual.mean().backward()

    assert latent.shape == (2, 3)
    assert decoded.shape == x.shape
    assert_close(actual, decoded)
    assert torch.all((0.0 <= actual) & (actual <= 1.0))
    assert x.grad is not None


def test_vae_reparameterize_matches_seeded_location_scale_sample():
    model = VAE((1, 2, 2), hidden_dim=4, latent_dim=2)
    mu = torch.tensor([[1.0, -1.0]])
    logvar = torch.log(torch.tensor([[4.0, 0.25]]))

    torch.manual_seed(0)
    actual = model.reparameterize(mu, logvar)

    torch.manual_seed(0)
    expected = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    assert_close(actual, expected)


def test_vae_forward_and_loss_support_backward():
    model = VAE((1, 4, 4), hidden_dim=8, latent_dim=3)
    x = torch.rand(2, 1, 4, 4)

    x_hat, mu, logvar = model(x)
    loss, recon_loss, kl_loss = model.loss(x_hat, x, mu, logvar)
    loss.backward()

    assert x_hat.shape == x.shape
    assert mu.shape == logvar.shape == (2, 3)
    assert loss.ndim == recon_loss.ndim == kl_loss.ndim == 0
    assert_close(loss, recon_loss + kl_loss)
    assert all(value.isfinite() for value in (loss, recon_loss, kl_loss))
    assert model.fc_mu.weight.grad is not None
    assert model.fc_logvar.weight.grad is not None


@pytest.mark.parametrize('loss_fn', ['mse', 'bce'])
def test_vae_loss_matches_torch_reference(loss_fn: LossFn):
    x = torch.tensor([[[[0.0, 1.0]]], [[[1.0, 0.0]]]])
    x_hat = torch.tensor([[[[0.25, 0.75]]], [[[0.8, 0.2]]]])
    mu = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    logvar = torch.log(torch.tensor([[1.0, 4.0], [0.25, 1.0]]))
    beta = 0.5

    actual, actual_recon, actual_kl = VAE.loss(
        x_hat, x, mu, logvar, loss_fn=loss_fn, beta=beta
    )

    B = x.size(0)
    if loss_fn == 'mse':
        expected_recon = F.mse_loss(x_hat, x, reduction='sum') / B
    else:
        expected_recon = F.binary_cross_entropy(x_hat, x, reduction='sum') / B

    expected_kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()) / B
    expected = expected_recon + beta * expected_kl

    assert_close(actual, expected)
    assert_close(actual_recon, expected_recon)
    assert_close(actual_kl, expected_kl)


def test_vae_loss_can_skip_normalization_and_rejects_unknown_loss():
    x = torch.zeros(2, 1, 2, 2)
    x_hat = torch.full_like(x, 0.5)
    mu = torch.zeros(2, 3)
    logvar = torch.zeros(2, 3)

    normalized = VAE.loss(x_hat, x, mu, logvar, loss_fn='mse')
    unnormalized = VAE.loss(x_hat, x, mu, logvar, loss_fn='mse', normalize=False)

    for actual, expected in zip(unnormalized, normalized, strict=True):
        assert_close(actual, expected * x.size(0))

    with pytest.raises(NotImplementedError, match='Unsupported loss function'):
        VAE.loss(x_hat, x, mu, logvar, loss_fn='mae')  # type: ignore[arg-type]
