# dnnlpy

**dnnlpy** is the companion Python package for **Deep Learning Notes**. It provides code examples, helper functions, and small utilities used throughout the tutorial, similar in spirit to the `d2l` package for _Dive into Deep Learning_.

The package structure is similar to PyTorch, but keeps a clear boundary between reusable neural network building blocks and complete model implementations:

- `dnnlpy.nn` contains general neural network modules, such as attention layers, positional encodings, and other reusable components.
- `dnnlpy.nn.functional` contains stateless helper functions, such as functional attention implementations.
- `dnnlpy.optim` contains small optimizer implementations for teaching purposes, such as SGD and Adam.
- `dnnlpy.models` contains higher-level model architectures or model-specific components, such as ViT, DDPM, or other models introduced in the notes.
- `dnnlpy.cs336` contains implementations and utilities for the CS336 assignments.
- `dnnlpy.tokenizers` contains small tokenizer implementations for teaching purposes, such as a simple BPE tokenizer.

The APIs are designed to feel close to their PyTorch counterparts where practical, while still keeping the code lightweight and easy to read for tutorial purposes.

This package is intended as a lightweight code supplement rather than a general-purpose deep learning framework. Its goal is to make the examples in the notes easier to run, reuse, and extend.

## What is this package for?

The `dnnlpy` package is designed to support the code in the **Deep Learning Notes** tutorial.

It can be used to:

- Organize example code from the notes
- Provide reusable utility functions
- Reduce repeated boilerplate in notebooks and scripts
- Make tutorial examples easier to reproduce

In short, this package serves as the code companion to the tutorial.

## Requirements

- Python 3.12 or newer
- PyTorch 2.13 or newer

## Installation

Install the published package from PyPI with:

```bash
uv pip install dnnlpy
```

To install the latest version directly from this repository, use:

```bash
uv pip install "git+https://github.com/jshn9515/deep-learning-notes.git#subdirectory=dnnlpy"
```

This project uses [uv](https://docs.astral.sh/uv/) for local package development.

```bash
git clone https://github.com/jshn9515/deep-learning-notes.git
cd dnnlpy
uv pip install .
```

If you want to modify the package while working through the notes, editable installation is recommended:

```bash
uv pip install -e .
```

This way, changes to the source code take effect immediately without reinstalling the package each time.

## Examples

After installation, you can import reusable neural network modules from `dnnlpy.nn`:

```python
import dnnlpy.nn as dnn
import torch

attn = dnn.MultiheadAttention(embed_dim=16, num_heads=4)

query = torch.randn(2, 8, 16)
key = torch.randn(2, 8, 16)
value = torch.randn(2, 8, 16)

output = attn(query, key, value)
```

You can also import stateless functions from `dnnlpy.nn.functional`:

```python
import dnnlpy.nn.functional as dF
import torch

query = torch.randn(2, 4, 8, 16)
key = torch.randn(2, 4, 8, 16)
value = torch.randn(2, 4, 8, 16)

output, weights = dF.scaled_dot_product_attention(
    query,
    key,
    value,
    need_weights=True,
)
```

Higher-level model architectures live under `dnnlpy.models`:

```python
import torch
import dnnlpy.models.vit as vit

model = vit.ViTForImageClassification(
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
)

images = torch.randn(2, 3, 224, 224)
logits = model(images)
```

The `dnnlpy.models.mlp` package contains small NumPy modules for teaching manual
forward and backward passes:

```python
import dnnlpy.models.mlp as mlp
import numpy as np

model = mlp.MLP(input_dim=4, hidden_dim=8, num_classes=3)
loss_fn = mlp.CrossEntropyLoss()
optimizer = mlp.SGD(model.parameters(), lr=0.1)

x = np.random.randn(2, 4)
targets = np.array([0, 2])

logits = model(x)
loss = loss_fn(logits, targets)
model.backward(loss_fn.backward())

optimizer.step()
optimizer.zero_grad()
```

A simple rule of thumb is:

- Use `dnnlpy.nn` when a component is reusable across many models.
- Use `dnnlpy.models` when the code represents a complete architecture or is tightly coupled to one model family.

## License

This project is licensed under the **MIT License**.
