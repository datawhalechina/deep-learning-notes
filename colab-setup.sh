#!/usr/bin/env bash
set -e

# Install Quarto CLI

VER=$(curl -s https://api.github.com/repos/quarto-dev/quarto-cli/releases/latest \
  | grep '"tag_name":' \
  | sed -E 's/.*"v([^"]+)".*/\1/')

echo "Installing Quarto ${VER}..."

wget -O quarto-linux-latest-amd64.deb \
  "https://github.com/quarto-dev/quarto-cli/releases/download/v${VER}/quarto-${VER}-linux-amd64.deb"

sudo dpkg -i quarto-linux-latest-amd64.deb

rm quarto-linux-latest-amd64.deb

echo "Quarto version: $(quarto --version)"

echo "Quarto ${VER} installed successfully."

# Install missing dependencies for Python

uv pip install --upgrade pip

uv pip install "git+https://github.com/jshn9515/deep-learning-notes.git#subdirectory=dnnlpy" --no-deps

uv pip install jupyter-cache torchmetrics evaluate

# Git user configuration

git config --global user.name "jshn9515"
git config --global user.email "jshn9510@gmail.com"
