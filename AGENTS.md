# Repository Guidelines

## Project Structure & Module Organization

Quarto sources live in `zh/` and `en/`, grouped by chapter (for example, `zh/ch4-optimization-algorithms/`). Keep paired translations and their local figures aligned. Shared images belong in `assets/`. The reusable Python package is under `dnnlpy/src/`, with tests in `dnnlpy/tests/`. Course material lives in `cs224n/`, `cs234/` and `cs336/`; build and maintenance helpers live in `utils/`. Treat `_site/`, `_typst/`, `_freeze/`, and `_jupyter/` as generated output.

## Build, Test, and Development Commands

- `uv sync --all-packages`: install the workspace and `dnnlpy` dependencies.
- `quarto render --profile html`: build the complete website locally.
- `quarto render --profile typst-zh`: build the Chinese PDF; use `typst-en` for English.
- `pytest dnnlpy/tests`: run the full Python test suite. Append a test path or `-k expression` for focused checks.
- `ruff format .` and `ruff check .`: format and lint supported source files.
- `pre-commit run --all-files`: run Ruff and secret scanning before submission.

## Coding Style & Naming Conventions

Python uses four-space indentation, an 88-character line limit, single-quoted strings, and Google-style docstrings; Ruff is authoritative. Format every changed Python file before finishing. Name tests `test_*.py` and mirror package subdirectories. Quarto files follow `chN.M-descriptive-name.qmd`; preserve front-matter numbering, sidebar order, code fences, equations, citations, and relative asset paths.

## Documentation and Testing Guidelines

When the user says **"sync zh to en"**, load and follow `.agents/skills/sync-zh-to-en/SKILL.md` before inspecting or editing the requested documentation. Limit the synchronization to the scope the user identifies and preserve unrelated worktree changes. Render every changed page and verify math, links, assets, and navigation. Python changes require focused pytest coverage plus the relevant broader suite; do not claim a full render or hosted CI result unless it actually ran.

## Commit & Pull Request Guidelines

Use the established imperative prefixes, such as `DOC:`, `FIX:`, `FEA:`, `STY:`, or `MNT:`. Sign commits so GitHub marks them Verified. Keep pull requests focused, describe the motivation and affected paths, link related issues, list exact validation commands, and include screenshots for visible layout changes. Before submission, run `git diff --check` and avoid committing generated output or unrelated worktree changes.
