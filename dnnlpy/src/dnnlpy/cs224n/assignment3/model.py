import math
from typing import Literal, Self

import torch
import torch.nn as nn
from torch import Tensor
from transformers import GPT2LMHeadModel

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF

from .config import *
from .utils import state_dict_converter

type ModelSize = Literal['small', 'medium', 'large', 'xl']

__all__ = [
    'GPT2MLP',
    'GPT2Block',
    'GPT2CasualAttention',
    'GPT2Model',
]


class GPT2MLP(nn.Module):
    """Creates a feedforward neural network (MLP) for the GPT-2 model."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.fc1 = dnn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = dnn.Linear(4 * config.d_model, config.d_model)
        self.gelu = dnn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class GPT2CasualAttention(nn.Module):
    """Creates a self-attention layer for the GPT-2 model."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.attn = dnn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            fast=config.fast,
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_output, _ = self.attn(x, x, x, is_causal=True)
        return attn_output


class GPT2Block(nn.Module):
    """Creates a single block of the GPT-2 model, consisting of self-attention
    and feedforward layers.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.norm1 = dnn.LayerNorm(config.d_model)
        self.attn = GPT2CasualAttention(config)
        self.norm2 = dnn.LayerNorm(config.d_model)
        self.mlp = GPT2MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT2Model(nn.Module):
    """Creates the GPT-2 model for language modeling."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.tok_embed = dnn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = dnn.Embedding(config.context_length, config.d_model)

        self.backbone = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.n_layers)]
        )

        self.final_norm = dnn.LayerNorm(config.d_model)
        self.lm_head = dnn.Linear(config.d_model, config.vocab_size, bias=False)

        self.reset_parameters()

        if config.weight_tying:
            self.lm_head.weight = self.tok_embed.weight
            assert self.lm_head.weight is self.tok_embed.weight

    def reset_parameters(self):
        """Reset the parameters of the model using the initialization scheme from
        the original GPT-2 paper.

        Note that this initialization scheme is different from the default PyTorch
        initialization scheme, which uses Kaiming uniform initialization for linear
        layers and uniform initialization for embedding layers. The GPT-2 paper
        uses a normal distribution with mean 0 and standard deviation 0.02 for all
        linear and embedding layers, and initializes the bias terms to zero. The
        LayerNorm layers are initialized with weight 1 and bias 0.
        """
        for module in self.modules():
            if isinstance(module, dnn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, dnn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, dnn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for name, param in self.named_parameters():
            if name.endswith(('mlp.fc2.weight', 'attn.out_proj.weight')):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.config.context_length:
            raise AssertionError(
                f'Input sequence length {x.size(1)} exceeds context length '
                f'{self.config.context_length}.'
            )

        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        x = self.tok_embed(x) + self.pos_embed(pos)

        for block in self.backbone:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        x: Tensor,
        max_new_token: int,
        greedy: bool = False,
    ) -> Tensor:
        """Generate new tokens given a prompt `x`.

        Args:
            x (Tensor): Prompt token IDs with shape `(batch_size, sequence_length)`.
            max_new_token (int): Number of tokens to generate.
            greedy (bool, default: False): Whether to select the most likely token
                instead of sampling from the predicted distribution.

        Returns:
            Tensor: The prompt followed by the generated token IDs.
        """
        for _ in range(max_new_token):
            logits = self(x)
            logits = logits[:, -1, :]

            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = dF.softmax(logits, dim=-1)
                next_token = probs.multinomial(num_samples=1)

            x = torch.concat([x, next_token], dim=1)

        return x

    def loss(self, input_ids: Tensor, targets: Tensor) -> Tensor:
        """Compute the cross-entropy loss for language modeling."""
        logits = self(input_ids)
        logits = logits.reshape(-1, logits.size(-1))
        loss = dF.cross_entropy_loss(logits, targets.reshape(-1))
        return loss

    @classmethod
    def from_pretrained(cls, size: ModelSize) -> Self:
        """Load a pretrained GPT-2 model from HuggingFace."""
        match size:
            case 'small':
                config = GPT2_SMALL
            case 'medium':
                config = GPT2_MEDIUM
            case 'large':
                config = GPT2_LARGE
            case 'xl':
                config = GPT2_XL
            case _:
                raise RuntimeError(f'Invalid model size: {size}.')

        model = cls(config)

        # Load weights from HuggingFace
        if size == 'small':
            hf_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
        else:
            hf_model = GPT2LMHeadModel.from_pretrained(f'openai-community/gpt2-{size}')

        converted_state_dict = state_dict_converter(hf_model.state_dict())
        model.load_state_dict(converted_state_dict)

        return model
