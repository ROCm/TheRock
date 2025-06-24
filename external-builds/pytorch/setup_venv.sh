#!/bin/bash

set -euxo pipefail

# Set VENV_DIR from argument or default
VENV_DIR="${1:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")/venv}"

# Validate INDEX_URL
if [ -z "$INDEX_URL" ]; then
  echo "Error: INDEX_URL is not set" >&2
  exit 1
fi

# Validate requirements-test.txt
REQUIREMENTS_FILE="${VENV_DIR%/*}/requirements-test.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: $REQUIREMENTS_FILE does not exist" >&2
  exit 1
fi

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
  rm -rf "$VENV_DIR"
fi
python -m venv "$VENV_DIR"

# Install dependencies
"$VENV_DIR/bin/pip" install --upgrade pip --no-cache-dir || exit 1
"$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE" --no-cache-dir || exit 1
"$VENV_DIR/bin/pip" install --index-url "$INDEX_URL" torch || exit 1
