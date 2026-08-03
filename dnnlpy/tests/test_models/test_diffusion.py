import pytest
import torch
from torch.testing import assert_close

from dnnlpy.models import ddpm


def test_add_noise_matches_closed_form_sample():
    x0 = torch.tensor([[[[1.0, -1.0], [0.5, 2.0]]]])
    betas = torch.tensor([0.1, 0.2, 0.3])

    torch.manual_seed(0)
    actual = ddpm.add_noise(x0, betas, timestep=1)

    torch.manual_seed(0)
    noise = torch.randn_like(x0)
    alpha_bar = torch.tensor(0.9 * 0.8)
    expected = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise

    assert_close(actual, expected)


def test_denoise_at_timestep_zero_returns_clean_sample_without_using_rng():
    x0 = torch.tensor([[[[1.0, -1.0], [0.5, 2.0]]]])
    xt = torch.randn_like(x0)
    betas = torch.tensor([0.1, 0.2, 0.3])
    rng_state = torch.random.get_rng_state()

    actual = ddpm.denoise(x0, xt, timestep=0, betas=betas)

    assert_close(actual, x0)
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_denoise_matches_posterior_sample_at_nonzero_timestep():
    x0 = torch.full((1, 1, 2, 2), 2.0)
    xt = torch.full_like(x0, 3.0)
    betas = torch.tensor([0.1, 0.2])

    torch.manual_seed(0)
    actual = ddpm.denoise(x0, xt, timestep=1, betas=betas)

    alpha_t = torch.tensor(0.8)
    alpha_bar_t = torch.tensor(0.9 * 0.8)
    alpha_bar_prev = torch.tensor(0.9)

    mean = (
        alpha_bar_prev.sqrt() * 0.2 / (1 - alpha_bar_t) * x0
        + alpha_t.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * xt
    )
    variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * 0.2

    torch.manual_seed(0)
    expected = mean + variance.sqrt() * torch.randn_like(x0)

    assert_close(actual, expected)


def test_ddpm_scheduler_initializes_linear_noise_schedule():
    scheduler = ddpm.DDPMScheduler(3, beta_start=0.1, beta_end=0.2)

    assert scheduler.num_train_timesteps == 3
    assert scheduler.num_inference_steps == 3
    assert_close(scheduler.betas, torch.tensor([0.1, 0.15, 0.2]))
    assert_close(scheduler.alphas, torch.tensor([0.9, 0.85, 0.8]))
    assert_close(scheduler.alphas_cumprod, torch.tensor([0.9, 0.765, 0.612]))
    assert torch.equal(scheduler.timesteps, torch.tensor([2, 1, 0]))


def test_ddpm_scheduler_add_noise_uses_each_batch_timestep():
    scheduler = ddpm.DDPMScheduler(3, beta_start=0.1, beta_end=0.2)
    original_samples = torch.ones(2, 1, 2, 2)
    noise = torch.full_like(original_samples, 2.0)
    timesteps = torch.tensor([0, 2])

    actual = scheduler.add_noise(original_samples, noise, timesteps)

    alpha_bars = torch.tensor([0.9, 0.612]).view(-1, 1, 1, 1)
    expected = alpha_bars.sqrt() * original_samples + (1 - alpha_bars).sqrt() * noise
    assert_close(actual, expected)


def test_ddpm_scheduler_add_noise_validates_inputs():
    scheduler = ddpm.DDPMScheduler(3)
    samples = torch.zeros(2, 1, 2, 2)

    with pytest.raises(AssertionError, match='must have the same shape'):
        scheduler.add_noise(samples, torch.zeros(1, 1, 2, 2), torch.tensor([0, 1]))

    with pytest.raises(AssertionError, match='must be a 1D tensor'):
        scheduler.add_noise(samples, torch.zeros_like(samples), torch.tensor([[0, 1]]))


def test_ddpm_scheduler_sets_inference_schedule_and_previous_timesteps():
    scheduler = ddpm.DDPMScheduler(10)

    scheduler.set_timesteps(5)

    assert scheduler.num_inference_steps == 5
    assert scheduler.timesteps.dtype == torch.long
    assert scheduler.timesteps.tolist() == [9, 6, 4, 2, 0]
    assert scheduler.previous_timestep(9) == 6
    assert scheduler.previous_timestep(4) == 2
    assert scheduler.previous_timestep(0) == -1


def test_ddpm_scheduler_rejects_too_many_inference_steps():
    scheduler = ddpm.DDPMScheduler(3)

    with pytest.raises(AssertionError, match='num_inference_steps.*range'):
        scheduler.set_timesteps(4)


def test_ddpm_scheduler_step_at_zero_matches_predicted_clean_sample():
    scheduler = ddpm.DDPMScheduler(3, beta_start=0.1, beta_end=0.2)
    sample = torch.ones(1, 1, 2, 2)
    model_output = torch.full_like(sample, 0.25)
    rng_state = torch.random.get_rng_state()

    actual = scheduler.step(model_output, timestep=0, sample=sample)

    alpha_bar = torch.tensor(0.9)
    expected = (sample - (1 - alpha_bar).sqrt() * model_output) / alpha_bar.sqrt()

    assert_close(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_ddpm_scheduler_step_matches_posterior_sample():
    scheduler = ddpm.DDPMScheduler(3, beta_start=0.1, beta_end=0.2)
    sample = torch.ones(1, 1, 2, 2)
    model_output = torch.full_like(sample, 0.25)

    torch.manual_seed(0)
    actual = scheduler.step(model_output, timestep=2, sample=sample)

    alpha_t = scheduler.alphas[2]
    alpha_bar_t = scheduler.alphas_cumprod[2]
    alpha_bar_prev = scheduler.alphas_cumprod[1]
    beta_t = scheduler.betas[2]

    pred_original = (
        sample - (1 - alpha_bar_t).sqrt() * model_output
    ) / alpha_bar_t.sqrt()

    mean = (
        alpha_bar_prev.sqrt() * beta_t / (1 - alpha_bar_t) * pred_original
        + alpha_t.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * sample
    )
    variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t

    torch.manual_seed(0)
    expected = mean + variance.sqrt() * torch.randn_like(sample)

    assert_close(actual, expected)


def test_sinusoidal_timestep_embedding_handles_odd_and_unit_dimensions():
    timesteps = torch.tensor([0, 1])

    actual = ddpm.SinusoidalTimestepEmbedding(5)(timesteps)
    unit = ddpm.SinusoidalTimestepEmbedding(1)(timesteps)

    assert actual.shape == (2, 5)
    assert_close(actual[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]))
    assert_close(actual[:, -1], torch.zeros(2))
    assert_close(unit, torch.zeros(2, 1))


def test_unet_forward_and_backward_support_configured_channels():
    x = torch.randn(1, 1, 4, 4, requires_grad=True)
    timesteps = torch.tensor([2])

    model = ddpm.UNet2DModel(
        in_channels=1,
        out_channels=2,
        block_out_channels=(8, 16),
        time_emb_dim=16,
    )

    output = model(x, timesteps)
    output.mean().backward()

    assert output.shape == (1, 2, 4, 4)
    assert x.grad is not None
    assert x.grad.isfinite().all()
    assert any(p.grad is not None for p in model.parameters())


def test_unet_requires_one_timestep_per_batch_item():
    model = ddpm.UNet2DModel(
        in_channels=1,
        out_channels=1,
        block_out_channels=(8,),
        time_emb_dim=8,
    )

    with pytest.raises(
        AssertionError, match='Batch size of x and timesteps must match'
    ):
        model(torch.randn(2, 1, 4, 4), torch.tensor([1]))
