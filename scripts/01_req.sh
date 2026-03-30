#!/bin/bash

# =============================================================
# 01_req.sh — Install OS requirements, pyenv, and Python
# Run standalone: source scripts/01_req.sh
# =============================================================

set -e

PYTHON_VERSION="3.10.11"
OS="$(uname -s)"

echo ""
echo "--- [1/3] Installing OS requirements ---"
echo "Detected OS: $OS"

if [[ "$OS" == "Linux" ]]; then
    SUDO=""
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
    $SUDO apt-get update
    $SUDO apt-get install -y make build-essential libssl-dev zlib1g-dev libffi-dev wget curl \
        libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev tk-dev

elif [[ "$OS" == "Darwin" ]]; then
    if ! command -v brew >/dev/null 2>&1; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        eval "$(/opt/homebrew/bin/brew shellenv)"   # Apple Silicon
        eval "$(/usr/local/bin/brew shellenv)"      # Intel
    fi

    brew update
    brew install openssl zlib readline sqlite wget curl pyenv bzip2 xz

else
    echo "Unsupported OS: $OS"
    exit 1
fi

echo ""
echo "--- Installing pyenv and Python $PYTHON_VERSION ---"

if [ ! -d "$HOME/.pyenv" ]; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
else
    echo "pyenv directory already exists, skipping installation."
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pyenv install "$PYTHON_VERSION" --force
pyenv local "$PYTHON_VERSION"

echo "Python $PYTHON_VERSION installed and set as local version."