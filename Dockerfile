# syntax=docker/dockerfile:1

FROM ghcr.io/quarto-dev/quarto:1.11.0 AS builder

USER root

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . .

RUN quarto install chrome-headless-shell
RUN quarto render --profile html --no-execute

FROM nginx:latest AS runtime

COPY --from=builder /workspace/_site/ /usr/share/nginx/html/

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD nginx -t || exit 1
