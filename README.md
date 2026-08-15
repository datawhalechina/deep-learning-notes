# Deep Learning Notes

[![publish](https://github.com/jshn9515/deep-learning-notes/actions/workflows/quarto-ci.yml/badge.svg)](https://github.com/jshn9515/deep-learning-notes/actions/workflows/quarto-ci.yml)
[![build](https://github.com/jshn9515/deep-learning-notes/actions/workflows/dnnlpy-ci.yml/badge.svg)](https://github.com/jshn9515/deep-learning-notes/actions/workflows/dnnlpy-ci.yml)
[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-5.15.0-ffcc00?logo=huggingface)](https://huggingface.co/docs/transformers/index)

**English** | [简体中文](README-zh.md)

![dnnl-title](assets/dnnl-title.png)

For a long time, I struggled with how to learn deep learning effectively.

_Dive into Deep Learning_ is an excellent introductory book, but its update pace has gradually fallen behind the speed of progress in this field. Since the rise of Transformers, topics like CLIP, Diffusion, and vLLM have become increasingly important. Although there is no shortage of online material, most of it is scattered. One day you study Attention, the next day LoRA, and the day after that diffusion models. In the end, what often remains are only fragments, and it is hard to build a truly coherent understanding.

So I decided to systematically organize what I have learned. From the fundamentals of PyTorch, to Attention and Transformers, and then to GANs, CLIP, Stable Diffusion, and SAM3, I try to explain the core ideas, mathematical derivations, code implementations, and common pitfalls of each topic as clearly as possible. This repository is the public version of those notes. If you are also learning deep learning on your own, I hope it can be helpful to you.

## 📌 About These Notes

This project is primarily maintained and published in **Quarto Markdown**, and built as a static website. Quarto Markdown is a plain-text format based on Markdown, which makes it well suited for version control and continuous updates.

The content mainly includes:

- PyTorch fundamentals and engineering practice
- Attention mechanisms and Transformer-based models
- Generative models, such as GANs, VAEs, and diffusion models
- Multimodal models, such as CLIP
- The Hugging Face ecosystem and its practical use
- Practical notes covering the full workflow from data processing to training, inference, and deployment

The corresponding Jupyter Notebook version of this project is available at [jshn9515/dnnl-notebooks](https://github.com/jshn9515/dnnl-notebooks). This repository is kept in sync with the main repository, and the notebooks can be opened directly in Google Colab. GitHub Actions Artifacts can also serve as a backup source for accessing the latest build outputs when repository synchronization fails or is temporarily unavailable.

If you prefer generating Notebook files from the source yourself, you can also install Quarto locally and use the `quarto convert` command to convert `.qmd` files into Jupyter Notebooks. For example:

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
uv pip install "git+https://github.com/jshn9515/deep-learning-notes.git#subdirectory=dnnlpy"
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

This project website is built with [Quarto](https://quarto.org/).

## 📄 License

- The notes in this repository are licensed under **CC BY-NC 4.0**.
- The `dnnlpy` library is licensed under **MIT**.

## ⭐ Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=datawhalechina/deep-learning-notes&type=date&legend=top-left)](https://star-history.dera.page/#datawhalechina/deep-learning-notes&type=date&legend=top-left)
