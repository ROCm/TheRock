#!/bin/bash

# TODO: This script will be converted to python script
set -euxo pipefail

# Set VENV_DIR from argument or default
VENV_DIR="${1:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")/venv}"

# Validate requirements-test.txt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements-test.txt"

# Create virtual environment
python -m venv "$VENV_DIR"

# Install dependencies
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"
# Install torch, overwriting any previously installed versions.
"$VENV_DIR/bin/pip" install --index-url "$INDEX_URL" torch --no-cache-dir --force-reinstall
