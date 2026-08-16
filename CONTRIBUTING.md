# Contributing

Thank you for your interest in contributing to this project!

To keep the repository consistent and easy to maintain, please follow the steps below before submitting a Pull Request.

## 1. Install Quarto

This project is maintained primarily in **Quarto Markdown (`.qmd`)** format.

Please install Quarto first from the official website:

- Quarto: [Quarto Get Started](https://quarto.org/docs/get-started/)

After installation, make sure the `quarto` command is available in your terminal.

## 2. Install project dependencies

This project uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies.

First, follow the instructions to install `uv` from the official repository:

- GitHub: [uv](https://github.com/astral-sh/uv#installation)

Then, clone this repository and navigate to the project root directory:

```bash
git clone jshn9515/deep-learning-notes.git
cd deep-learning-notes
```

After cloning the repository, install the required dependencies with:

```bash
uv sync --all-packages
```

This will create or update the project environment according to the `pyproject.toml`.

## 3. Install `dnnlpy`

Before running or modifying the notes, please install the `dnnlpy` library:

```bash
uv pip install dnnlpy
```

If you need the latest package code directly from this repository, use:

```bash
uv pip install ./dnnlpy
```

Some notebooks and examples rely on utilities and custom implementations provided by `dnnlpy`, so skipping this step may cause rendering or execution issues.

## 4. Edit the `.qmd` files

Please make your changes by editing the corresponding **Quarto Markdown (`.qmd`)** files.

When contributing:

- Follow existing `.qmd` writing style and project structure
- Keep explanations clear and concise
- If you add code examples, make sure they are readable and properly formatted
- If you modify formulas, derivations, or technical explanations, please check them carefully for correctness

## 5. Format your code with `ruff`

Before submitting a Pull Request, please format and check your Python code with [ruff](https://docs.astral.sh/ruff/). This helps keep the codebase consistent and avoids unnecessary formatting changes during review.

To install `ruff`, run:

```bash
uv tool install ruff
```

To format the code, run:

```bash
ruff format .
```

Then check for linting issues with:

```bash
ruff check .
```

You can automatically fix issues that Ruff considers safe to fix with:

```bash
ruff check . --fix
```

Please make sure both formatting and lint checks pass before submitting your Pull Request.

## 6. Use a verified commit

This repository requires commits to be verified.

Before submitting a Pull Request, please make sure your local Git commits are signed with a valid GPG key so GitHub can mark them as **Verified**.

If you have not configured a GPG key yet, please follow GitHub's official guide:

- GitHub Docs: [Signing commits with GPG](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)

After configuration, confirm that your commits show the **Verified** badge on GitHub.

## 7. Re-render locally before submitting

Before opening a Pull Request, you **must** re-render the modified content locally and confirm that everything works correctly.

At minimum, please make sure:

- The page renders successfully with no build errors
- Code blocks and math formulas display correctly
- Links, formatting, and section structure look normal
- If needed, the `.qmd` file can still be converted into a notebook with Quarto

For example, you may use:

```bash
quarto render --profile html
```

If you need to convert a `.qmd` file into a notebook for checking, you can also use:

```bash
quarto convert path/to/file.qmd
```

Please **do not** submit a PR without verifying the local rendering result first.

## 8. Submit a Pull Request

Once everything looks good locally, you can submit a Pull Request.

If your change is relatively large, it is recommended to open an Issue first to briefly describe your idea before starting implementation.

Typical contributions include:

- Fixing errors or unclear explanations
- Improving code examples or comments
- Correcting formatting or structure issues
- Adding better derivations or clearer technical explanations
- Suggesting or contributing new topics

Thank you for helping improve these notes!
