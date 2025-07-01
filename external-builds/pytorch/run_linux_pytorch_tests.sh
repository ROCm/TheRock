#!/bin/bash

set -euxo pipefail

# Determine directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"
PYTORCH_DIR="$ROOT_DIR/external-builds/pytorch/pytorch"
VENV_DIR="$ROOT_DIR/venv"
SKIP_FILE="$SCRIPT_DIR/skipped_tests.py"


# Set environment variables
export PYTORCH_PRINT_REPRO_ON_FAILURE=0
export PYTORCH_TEST_WITH_ROCM=1
export MIOPEN_CUSTOM_CACHE_DIR=$(mktemp -d)
export PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"
export PYTHONPATH="$PYTORCH_DIR/test:${PYTHONPATH:-}"

# Generate -k skip expression
K_EXPR=$(python "$SKIP_FILE")
echo "Excluding tests via -k: $K_EXPR"

# TODO: Add back "test/test_ops.py" and test/inductor/test_torchinductor.py
# when the AttributeError solved
# Run pytest
cd "$ROOT_DIR"
"$VENV_DIR/bin/python" -m pytest \
  "$PYTORCH_DIR/test/test_nn.py" \
  "$PYTORCH_DIR/test/test_torch.py" \
  "$PYTORCH_DIR/test/test_cuda.py" \
  "$PYTORCH_DIR/test/test_unary_ufuncs.py" \
  "$PYTORCH_DIR/test/test_binary_ufuncs.py" \
  "$PYTORCH_DIR/test/test_autograd.py" \
  -v \
  --continue-on-collection-errors \
  --import-mode=importlib \
  --maxfail=0 \
  $K_EXPR \
  -n 0
