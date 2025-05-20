#!/bin/bash
set -xeuo pipefail

echo 'Running inside the container'

echo 'rocminfo' 
rocminfo

echo 'rocm-smi'
rocm-smi
#  Check for ROCm GPU availability
echo 'Checking for ROCm-compatible GPU...'
if ! rocminfo | grep -q 'Name: .*AMD'; then
  echo "ERROR: No ROCm-compatible GPU detected."
  exit 1
fi

echo 'Set path to .local/bin'
export PATH="$HOME/.local/bin:$PATH"

echo 'Check Python version'
python3 --version

echo 'List the pip packages'
pip list

echo 'Install pytest'
PIP_BREAK_SYSTEM_PACKAGES=1 pip install --no-index --find-links=/wheels pytest;
echo 'Run smoke tests'
pytest -v external-builds/pytorch/smoke-tests/

echo 'Task completed!'
