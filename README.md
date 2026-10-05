# Deep Learning Notes

[![publish](https://github.com/datawhalechina/deep-learning-notes/actions/workflows/quarto-ci.yml/badge.svg)](https://github.com/datawhalechina/deep-learning-notes/actions/workflows/quarto-ci.yml)
[![build](https://github.com/datawhalechina/deep-learning-notes/actions/workflows/dnnlpy-ci.yml/badge.svg)](https://github.com/datawhalechina/deep-learning-notes/actions/workflows/dnnlpy-ci.yml)
[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-5.18.0-ffcc00?logo=huggingface)](https://huggingface.co/docs/transformers/index)

**English** | [简体中文](README-zh.md)

![dnnl-title](assets/dnnl-title.png)

For a long time, I struggled with how to learn deep learning effectively.

_Dive into Deep Learning_ is a great introductory book, but deep learning is evolving incredibly quickly. After the Transformer, large models gradually became a major direction, and a growing ecosystem has emerged around them, including data processing, training optimization, model evaluation, inference systems, and post-training. There is no shortage of material online, but it is often scattered across papers, blog posts, courses, and code repositories. As I kept learning, I found it easy to end up with a collection of isolated concepts that were difficult to connect into a coherent picture.

So I decided to gradually organize what I encounter during my own learning process. These notes start with the basics, including an introduction to neural networks, PyTorch, optimization algorithms, CNNs, and RNNs. From there, they move on to attention and Transformers, and then continue into LLM-related topics, including implementing GPT-2 from scratch, large-scale model training engineering, data processing, scaling laws, model evaluation, LLM inference, post-training, and inference frameworks such as vLLM and SGLang.

Of course, I am still learning as well, so this repository is an ongoing record of that process. I will try to clearly explain the things I have learned and genuinely understood, while connecting formulas, code, and practical projects whenever possible. If you are also learning deep learning on your own, I hope these notes can be helpful to you.

> [!NOTE]
> **AI-assisted writing:** LLMs were used during the writing process of this tutorial to assist with drafting. After each generated draft, I review it myself and revise the content, logic, and wording based on my own understanding. Before publication, I also further check the relevant code and technical details. Despite this, the tutorial may still contain omissions or errors, and corrections and suggestions are always welcome.

## 📌 About These Notes

This project is primarily maintained and published in **Quarto Markdown**, and built as a static website. Quarto Markdown is a plain-text format based on Markdown, which makes it well suited for version control and continuous updates.

The content mainly includes:

- Neural network fundamentals, PyTorch, MLPs, and optimization algorithms
- CNNs, regularization, and normalization
- RNNs, LSTMs, GRUs, and Seq2Seq
- Attention, Transformers and FlashAttention
- Implementing and training MiniGPT, tokenization, and GPT-2
- LLM training engineering
- vLLM and SGLang inference and optimization

The corresponding Jupyter Notebook version of this project is available at [jshn9515/dnnl-notebooks](https://github.com/jshn9515/dnnl-notebooks). This repository is kept in sync with the main repository, and the notebooks can be opened directly in Google Colab. GitHub Actions Artifacts can also serve as a backup source for accessing the latest build outputs when repository synchronization fails or is temporarily unavailable.

If you prefer generating notebook files from the source yourself, you can also install Quarto locally and use the `quarto convert` command to convert `.qmd` files into Jupyter Notebooks. For example:

```bash
quarto convert path/to/file.qmd
```

## 🔧 Environment

All code in this repository has been tested in the following environment:

- Python 3.14
- PyTorch 2.14

See `pyproject.toml` for the full list of dependencies.

Before running the related content, please install the `dnnlpy` library. This library contains some custom implementations and utility functions used throughout the notes, and many examples will not run properly without it.

```bash
uv pip install dnnlpy
```

To install the latest version directly from this repository, use:

```bash
uv pip install "git+https://github.com/datawhalechina/deep-learning-notes.git#subdirectory=dnnlpy"
```

> [!NOTE]
> This project uses **Transformers v5**. If you are following other repositories or tutorials based on v4, there may be significant API differences (such as tokenizers and quantization configurations). Please refer to the [official migration guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) for adjustments.

## 🤝 Contributions

If you find an explanation unclear, notice a problem in the code, or have topics you would like me to add, feel free to contribute through Issues or Pull Requests.

Possible contributions include, but are not limited to:

- Pointing out errors or inaccuracies in the notes
- Adding clearer explanations, derivations, or code comments
- Suggesting improvements to structure, wording, or formatting
- Recommending topics or practical cases for future coverage

Since this is a project I am building and refining while learning, there will inevitably be places where my understanding is incomplete or my explanations are not precise enough. I read all helpful feedback carefully and try to improve the notes whenever possible.

If you would like to make a larger change, it is recommended to open an Issue first with a brief description so that we can discuss it in advance.

## 🙏 Acknowledgements

While organizing these notes, I have benefited from many excellent resources. In particular, _Dive into Deep Learning_ by Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola, as well as Professor Hung-yi Lee’s deep learning lecture series, have helped me greatly in understanding many core concepts in deep learning.

This website is built with [Quarto](https://quarto.org/).

The book cover design is inspired by [_Understanding Deep Learning_](https://udlbook.github.io/udlbook/).

## 📄 License

- The notes in this repository are licensed under **CC BY-NC 4.0**.
- The `dnnlpy` library is licensed under **MIT**.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/chart?repos=datawhalechina/deep-learning-notes&type=date&legend=top-left)](https://www.star-history.com/?repos=jshn9515%2Fdeep-learning-notes&type=date&legend=top-left)
