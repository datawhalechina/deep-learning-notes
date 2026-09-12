# Deep Learning Notes

[![publish](https://github.com/datawhalechina/deep-learning-notes/actions/workflows/quarto-ci.yml/badge.svg)](https://github.com/datawhalechina/deep-learning-notes/actions/workflows/quarto-ci.yml)
[![build](https://github.com/datawhalechina/deep-learning-notes/actions/workflows/dnnlpy-ci.yml/badge.svg)](https://github.com/datawhalechina/deep-learning-notes/actions/workflows/dnnlpy-ci.yml)
[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-5.16.0-ffcc00?logo=huggingface)](https://huggingface.co/docs/transformers/index)

**English** | [简体中文](README-zh.md)

![dnnl-title](assets/dnnl-title.png)

For a long time, I struggled with how to learn deep learning effectively.

_Dive into Deep Learning_ is a very good introductory book, but its update speed has already fallen somewhat behind the development of this field. After Transformer, more and more content such as ViT, DiT, LLM, Agent, as well as data processing, training optimization, inference, and post-training that gradually formed around large models, has continued to appear. Although there are many materials online, they are often scattered across papers, blogs, courses, and code repositories: today you learn Attention, tomorrow you study LoRA, and the day after tomorrow you start reading about vLLM, SGLang, and FlashAttention. In the end, what is often left is only fragments, and it is difficult to truly connect them into a system.

So I thought, I might as well systematically organize the content I have learned. These notes start from basic content such as neural networks, PyTorch, optimization algorithms, and CNNs, gradually move into Attention, Transformer, ViT, VAE, and DDPM, and further extend to modern LLMs: implementing GPT from scratch, training engineering, data processing, Scaling Laws, model evaluation, LLM Inference, and post-training. I will try to clearly explain the core ideas, formula derivations, code implementations, and common problems of each topic. This repository is the public version of these notes. If you are also self-studying deep learning, I hope it can give you some help.

> [!NOTE]
> **AI-assisted writing:** LLMs were used during the writing process of this tutorial to assist with drafting. After each generated draft, I review it myself and revise the content, logic, and wording based on my own understanding. Before publication, I also further check the relevant code and technical details. Despite this, the tutorial may still contain omissions or errors, and corrections and suggestions are always welcome.

## 📌 About These Notes

This project is primarily maintained and published in **Quarto Markdown**, and built as a static website. Quarto Markdown is a plain-text format based on Markdown, which makes it well suited for version control and continuous updates.

The content mainly includes:

- PyTorch fundamentals and Deep Learning training practice
- Introduction to Attention and Transformer
- Vision Transformer models such as ViT and Swin
- Generative models such as GAN, VAE, and DDPM
- Vision and multimodal models such as CLIP and BLIP
- Implementing GPT-2 from scratch and modern language models
- LLM data processing, training engineering, and Scaling Laws
- LLM Evaluation, Inference, and Serving
- Post-training methods such as Instruction Tuning, LoRA, DPO, and RLHF

The corresponding Jupyter Notebook version of this project is available at [jshn9515/dnnl-notebooks](https://github.com/jshn9515/dnnl-notebooks). This repository is kept in sync with the main repository, and the notebooks can be opened directly in Google Colab. GitHub Actions Artifacts can also serve as a backup source for accessing the latest build outputs when repository synchronization fails or is temporarily unavailable.

If you prefer generating notebook files from the source yourself, you can also install Quarto locally and use the `quarto convert` command to convert `.qmd` files into Jupyter Notebooks. For example:

```bash
quarto convert path/to/file.qmd
```

## 🔧 Environment

All code in this repository has been tested in the following environment:

- Python 3.14
- PyTorch 2.13

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
