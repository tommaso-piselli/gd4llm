#!/bin/bash

# =============================================================
# 02_venv.sh — Create virtualenv and install Python packages
# Run standalone: source scripts/02_venv.sh
# =============================================================

set -e

VENV_NAME="myenv"

echo ""
echo "--- [2/3] Setting up virtual environment ---"

rm -rf "$VENV_NAME"
python -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

echo ""
echo "--- Installing Python packages ---"

pip install --upgrade pip

pip install \
    langchain-core \
    langchain-openai \
    langchain-anthropic \
    langchain-google-genai \
    langchain-google-vertexai \
    networkx \
    pandas

echo "All packages installed in '$VENV_NAME'."
