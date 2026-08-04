# syntax=docker/dockerfile:1

FROM ghcr.io/quarto-dev/quarto:1.11.0 AS builder

USER root

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

ENV UV_LINK_MODE=copy

COPY . .

RUN uv sync --python 3.14 --all-packages --group cpu
RUN quarto install chrome-headless-shell

ENV QUARTO_PYTHON=/workspace/.venv/bin/python

RUN quarto render --profile html --no-execute

FROM nginx:latest AS runtime

COPY --from=builder /workspace/_site/ /usr/share/nginx/html/

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD nginx -t || exit 1
