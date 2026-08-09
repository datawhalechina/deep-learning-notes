# Contributing

Thank you for your interest in contributing to this project.

To keep the repository consistent and easy to maintain, please follow the steps below before submitting a Pull Request.

## 1. Install Quarto

This project is maintained primarily in **Quarto Markdown (`.qmd`)** format.

Please install Quarto first from the official website:

- Quarto Docs: [Quarto Get Started](https://quarto.org/docs/get-started/)

After installation, make sure the `quarto` command is available in your terminal.

## 2. Install project dependencies

This project uses `uv` to manage Python dependencies.

After cloning the repository, install the required dependencies with:

```bash
uv sync
```

This will create or update the project environment according to the dependency lock file.

## 3. Install `dnnlpy`

Before running or modifying the notes, please install the `dnnlpy` library:

```bash
pip install dnnlpy
```

If you need the latest package code directly from this repository, use:

```bash
uv pip install "git+https://github.com/jshn9515/deep-learning-notes.git#subdirectory=dnnlpy"
```

Some notebooks and examples rely on utilities and custom implementations provided by `dnnlpy`, so skipping this step may cause rendering or execution issues.

## 4. Edit the `.qmd` files

Please make your changes by editing the corresponding **Quarto Markdown (`.qmd`)** files.

When contributing:

- Follow existing `.qmd` writing style and project structure
- Keep explanations clear and concise
- If you add code examples, make sure they are readable and properly formatted
- If you modify formulas, derivations, or technical explanations, please check them carefully for correctness

## 5. Understand the Quarto build cache

This project uses a layered cache strategy for Quarto rendering:

- `_freeze` is the global Quarto freeze cache. It stores rendered results so pages do not need to be re-executed unnecessarily during future builds.
- `.jupyter_cache` is the code-cell-level execution cache. It is used for caching individual notebook/code-block execution results.

In short, `_freeze` helps cache rendered documents at the project/page level, while `.jupyter_cache` helps cache executed code blocks at a finer-grained level.

When modifying executable examples, please make sure the rendered output and relevant cache files are kept consistent.

For a broader overview of the repository structure and GitHub Actions workflows, see [architecture.md](architecture.md).

## 6. Re-render locally before submitting

Before opening a Pull Request, you **must** re-render the modified content locally and confirm that everything works correctly.

At minimum, please make sure:

- The page renders successfully with no build errors
- Code blocks and math formulas display correctly
- Links, formatting, and section structure look normal
- If needed, the `.qmd` file can still be converted into a notebook with Quarto

For example, you may use:

```bash
quarto render
```

If you need to convert a `.qmd` file into a notebook for checking, you can also use:

```bash
quarto convert path/to/file.qmd
```

Please do **not** submit a PR without verifying the local rendering result first.

## 7. Use a verified commit

This repository requires commits to be verified.

Before submitting a Pull Request, please make sure your local Git commits are signed with a valid GPG key so GitHub can mark them as **Verified**.

If you have not configured a GPG key yet, please follow GitHub's official guide:

- GitHub Docs: [Signing commits with GPG](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)

After configuration, confirm that your commits show the **Verified** badge on GitHub.

## 8. Submit a Pull Request

Once everything looks good locally, you can submit a Pull Request.

If your change is relatively large, it is recommended to open an Issue first to briefly describe your idea before starting implementation.

Typical contributions include:

- Fixing errors or unclear explanations
- Improving code examples or comments
- Correcting formatting or structure issues
- Adding better derivations or clearer technical explanations
- Suggesting or contributing new topics

Thank you for helping improve these notes.
