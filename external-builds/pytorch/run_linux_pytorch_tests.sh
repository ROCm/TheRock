#!/bin/bash

set -euxo pipefail

# Determine directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"
PYTORCH_DIR="$ROOT_DIR/external-builds/pytorch/pytorch"
VENV_DIR="$ROOT_DIR/venv"


# Set environment variables
export PYTORCH_PRINT_REPRO_ON_FAILURE=0
export PYTORCH_TEST_WITH_ROCM=1
export MIOPEN_CUSTOM_CACHE_DIR=$(mktemp -d)
export PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"
export PYTHONPATH="$PYTORCH_DIR/test:${PYTHONPATH:-}"

# Run pytest and log output
cd "$ROOT_DIR"
"$VENV_DIR/bin/python" -m pytest \
  "$PYTORCH_DIR/test/test_nn.py" \
  "$PYTORCH_DIR/test/test_torch.py" \
  "$PYTORCH_DIR/test/test_cuda.py" \
  "$PYTORCH_DIR/test/test_ops.py" \
  "$PYTORCH_DIR/test/test_unary_ufuncs.py" \
  "$PYTORCH_DIR/test/test_binary_ufuncs.py" \
  "$PYTORCH_DIR/test/test_autograd.py" \
  "$PYTORCH_DIR/test/inductor/test_torchinductor.py" \
  -v \
  --continue-on-collection-errors \
  --import-mode=importlib \
  -k "not test_unused_output_device_cuda and not test_pinned_memory_empty_cache" \
  --maxfail=0 \
  -n 0
