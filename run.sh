#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==============================="
echo " GD4LLM"
echo "==============================="

source "$SCRIPT_DIR/setup.sh"
source "$HOME/.bashrc"

python "$SCRIPT_DIR/src/main.py"