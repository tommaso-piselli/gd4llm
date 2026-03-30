#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==============================="
echo " Starting setup..."
echo "==============================="

source "$SCRIPT_DIR/scripts/01_req.sh"
source "$SCRIPT_DIR/scripts/02_venv.sh"
source "$SCRIPT_DIR/scripts/03_env.sh"

echo ""
echo "==============================="
echo " Setup complete!"
echo "==============================="