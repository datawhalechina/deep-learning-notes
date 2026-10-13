import os
from functools import partial
from itertools import chain
from typing import Any, cast

import datasets as ds
import dnnlpy.cs224n.assignment3 as a3
import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF
import dnnlpy.optim as dopt
import rich_click as click
import torch
import wandb
from torchmetrics.classification import MulticlassAccuracy
from transformers import BatchEncoding, GPT2Tokenizer

from .config import GPT2TrainingConfig

type Batch = dict[str, list[Any]]

DEFAULT_CONFIG = GPT2TrainingConfig()


def load_tinystories(cfg: GPT2TrainingConfig) -> ds.Dataset:
    """Load the TinyStories dataset from Hugging Face."""
    if cfg.num_stories is None:
        train_ds = ds.load_dataset(cfg.path, split='train')
    else:
        train_ds = ds.load_dataset(cfg.path, split=f'train[:{cfg.num_stories}]')

    return train_ds


def tokenize_batch(tokenizer: GPT2Tokenizer, batch: Batch) -> BatchEncoding:
    result = tokenizer(
        batch['text'],
        add_special_tokens=False,
    )

    eos = cast(int, tokenizer.eos_token_id)
    result['input_ids'] = [ids + [eos] for ids in result['input_ids']]

    return result


def chunk_batch(chunk_size: int, batch: BatchEncoding) -> Batch:
    tokens = list(chain.from_iterable(batch['input_ids']))
    length = len(tokens) // chunk_size * chunk_size

    return {
        'input_ids': [tokens[i : i + chunk_size] for i in range(0, length, chunk_size)]
    }


def tokenize_tinystories(dataset: ds.Dataset, cfg: GPT2TrainingConfig) -> ds.Dataset:
    if cfg.chunk_size is None:
        raise AssertionError('`chunk_size` must be configured before tokenization.')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer = cast(GPT2Tokenizer, tokenizer)

    cpu_count = os.process_cpu_count()
    if cpu_count is None:
        raise RuntimeError('Unable to determine the number of CPU cores.')

    dataset = dataset.map(
        partial(tokenize_batch, tokenizer),
        batched=True,
        batch_size=cfg.batch_size,
        remove_columns=dataset.column_names,
        num_proc=cpu_count,
    )

    dataset = dataset.map(
        partial(chunk_batch, cfg.chunk_size),
        batched=True,
        batch_size=cfg.batch_size,
        remove_columns=dataset.column_names,
        num_proc=cpu_count // 4,
    )

    dataset.set_format(
        type='torch',
        columns=['input_ids'],
    )

    return dataset


def train_gpt2(cfg: GPT2TrainingConfig) -> None:
    run = wandb.init(project='cs224n-assignment3', config=cfg.to_dict())

    train_ds = load_tinystories(cfg)
    train_ds = tokenize_tinystories(train_ds, cfg)

    model = a3.GPT2Model(cfg.model_cfg).to(cfg.device)
    optimizer = dopt.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    train_metric = MulticlassAccuracy(cfg.model_cfg.vocab_size).to(cfg.device)

    for global_step, i in enumerate(range(0, len(train_ds), cfg.batch_size)):
        if cfg.max_steps is not None and global_step >= cfg.max_steps:
            break

        batch = train_ds[i : i + cfg.batch_size]

        token_ids = batch['input_ids'].to(cfg.device)
        inputs = token_ids[:, :-1]
        targets = token_ids[:, 1:]

        logits = model(inputs)
        loss = dF.cross_entropy_loss(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        loss.backward()

        acc = train_metric(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        with torch.no_grad():
            dnn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)

        optimizer.step()
        optimizer.zero_grad()

        run.log({'loss': loss.item(), 'acc': acc.item()}, step=global_step)

    run.finish()


@click.command(context_settings={'show_default': True})
@click.option(
    '--path',
    type=str,
    default=DEFAULT_CONFIG.path,
    help='Hugging Face dataset path.',
)
@click.option(
    '--num-stories',
    type=int | str,
    default=DEFAULT_CONFIG.num_stories,
    help='Number or percentage of training stories to load (for example, 1000 or 10%).',
)
@click.option(
    '--model-size',
    type=click.Choice(['small', 'medium', 'large', 'xl']),
    default=DEFAULT_CONFIG.model_size,
    help='Size of the GPT-2 model to train.',
)
@click.option(
    '--batch-size',
    type=click.IntRange(min=1),
    default=DEFAULT_CONFIG.batch_size,
    help='Number of sequences per batch.',
)
@click.option(
    '--chunk-size',
    type=click.IntRange(min=2),
    default=DEFAULT_CONFIG.chunk_size,
    help='Number of tokens per sequence.',
)
@click.option(
    '--max-steps',
    type=click.IntRange(min=1),
    default=DEFAULT_CONFIG.max_steps,
    help='Maximum training steps. If not specified, train on the entire dataset.',
)
@click.option(
    '--lr',
    type=click.FloatRange(min=0.0, min_open=True),
    default=DEFAULT_CONFIG.lr,
    help='Learning rate for the AdamW optimizer.',
)
@click.option(
    '--betas',
    type=click.Tuple([click.FloatRange(min=0.0, max=1.0, min_open=True)] * 2),
    default=DEFAULT_CONFIG.betas,
    help='Beta parameters for the AdamW optimizer (e.g., (beta1, beta2)).',
)
@click.option(
    '--weight-decay',
    type=click.FloatRange(min=0.0),
    default=DEFAULT_CONFIG.weight_decay,
    help='Weight decay for the AdamW optimizer.',
)
@click.option(
    '--max-norm',
    type=click.FloatRange(min=0.0, min_open=True),
    default=DEFAULT_CONFIG.max_norm,
    help='Maximum gradient norm for gradient clipping.',
)
@click.option(
    '--seed',
    type=int,
    default=DEFAULT_CONFIG.seed,
    help='Random seed for reproducibility. If not specified, a random seed will be used.',
)
@click.option(
    '--device',
    default=str(DEFAULT_CONFIG.device),
    help='Device to use for training (e.g., "cpu", "cuda:0").',
)
def main(**overrides: Any) -> None:
    """Train a GPT-2 model on TinyStories."""
    cfg = GPT2TrainingConfig(**overrides)
    train_gpt2(cfg)


if __name__ == '__main__':
    main()
