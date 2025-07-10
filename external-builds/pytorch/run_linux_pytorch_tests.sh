#!/bin/bash

set -euxo pipefail

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"
PYTORCH_DIR="$ROOT_DIR/external-builds/pytorch/pytorch"
K_EXPR_SCRIPT="$SCRIPT_DIR/skipped_tests.py"

# Set up test environment
export PYTORCH_PRINT_REPRO_ON_FAILURE=0
export PYTORCH_TEST_WITH_ROCM=1
export MIOPEN_CUSTOM_CACHE_DIR=$(mktemp -d)
export PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"
export PYTHONPATH="$PYTORCH_DIR/test:${PYTHONPATH:-}"

# Generate -k skip expression
K_EXPR=$(python "$K_EXPR_SCRIPT")
echo "Excluding tests via -k: $K_EXPR"
# TODO: Add test/test_ops.py and test/inductor/test_torchinductor.py
# when AttributeError error is solved for Triton
# Run selected test files
pytest \
  "$PYTORCH_DIR/test/test_nn.py" \
  "$PYTORCH_DIR/test/test_torch.py" \
  "$PYTORCH_DIR/test/test_cuda.py" \
  "$PYTORCH_DIR/test/test_unary_ufuncs.py" \
  "$PYTORCH_DIR/test/test_binary_ufuncs.py" \
  "$PYTORCH_DIR/test/test_autograd.py" \
  --continue-on-collection-errors \
  --import-mode=importlib \
  -v \
  -k "$K_EXPR" \
  -n 0
