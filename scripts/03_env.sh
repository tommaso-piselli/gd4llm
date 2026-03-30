#!/bin/bash

# =============================================================
# 03_env.sh — Load API key from config.txt and export it
# Run standalone: source scripts/03_env.sh
# =============================================================

set -e

CONFIG_FILE="config.txt"
OS="$(uname -s)"

echo ""
echo "--- [3/3] Loading environment variables ---"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found."
    exit 1
fi

OPENAI_API_KEY=$(grep "^OPENAI_API_KEY=" "$CONFIG_FILE" | sed 's/^OPENAI_API_KEY=//')

if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "Error: OPENAI_API_KEY not found in $CONFIG_FILE."
    exit 1
fi

export OPENAI_API_KEY
echo "OPENAI_API_KEY exported for the current session."

if [[ "$OS" == "Darwin" ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ "$OS" == "Linux" ]]; then
    SHELL_RC="$HOME/.bashrc"
fi

if grep -q "^export OPENAI_API_KEY=" "$SHELL_RC" 2>/dev/null; then
    sed -i'' "s|^export OPENAI_API_KEY=.*|export OPENAI_API_KEY=\"$OPENAI_API_KEY\"|" "$SHELL_RC"
    echo "OPENAI_API_KEY updated in $SHELL_RC."
else
    echo "export OPENAI_API_KEY=$OPENAI_API_KEY" >> "$SHELL_RC"
    echo "OPENAI_API_KEY added to $SHELL_RC."
fi
