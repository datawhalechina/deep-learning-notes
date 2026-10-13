from typing import Any, Literal

import dnnlpy.cs224n.assignment3 as a3
import pydantic as dt
import torch

import dnnlpy

__all__ = ['GPT2TrainingConfig']


DEVICE = dnnlpy.get_default_device()


class GPT2TrainingConfig(dt.BaseModel):
    model_config = dt.ConfigDict(arbitrary_types_allowed=True)

    # Dataset parameters
    path: str = 'roneneldan/TinyStories'
    num_stories: dt.PositiveInt | str | None = None

    # Model parameters
    model_size: Literal['tiny', 'small', 'medium', 'large', 'xl'] = 'small'
    model_cfg: a3.GPT2Config | None = None

    # Training parameters
    batch_size: dt.PositiveInt = 32
    chunk_size: dt.PositiveInt | None = None
    max_steps: dt.PositiveInt | None = None

    # Optimizer parameters
    lr: dt.PositiveFloat = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: dt.NonNegativeFloat = 0.01
    max_norm: dt.PositiveFloat = 1.0

    # Other parameters
    seed: int | None = None
    device: str | torch.device = DEVICE

    def model_post_init(self, context: Any) -> None:
        configs = {
            'tiny': a3.GPT2_TINY,
            'small': a3.GPT2_SMALL,
            'medium': a3.GPT2_MEDIUM,
            'large': a3.GPT2_LARGE,
            'xl': a3.GPT2_XL,
        }

        if self.model_cfg is None:
            self.model_cfg = configs[self.model_size]

        if self.chunk_size is None:
            self.chunk_size = self.model_cfg.context_length

        if not 2 <= self.chunk_size <= self.model_cfg.context_length + 1:
            raise ValueError('Invalid chunk_size.')

        if self.seed is not None:
            dnnlpy.set_seed(self.seed)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
