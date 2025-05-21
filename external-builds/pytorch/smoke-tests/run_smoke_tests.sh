#!/bin/bash
set -xeuo pipefail

echo 'Running inside the container'

echo 'Set path to .local/bin'
export PATH="$HOME/.local/bin:$PATH"

echo 'Check Python version'
python3 --version

echo 'List the pip packages'
pip list

echo 'Install pytest'
PIP_BREAK_SYSTEM_PACKAGES=1 pip install --no-index --find-links=/wheels pytest;

echo 'Run smoke tests'
pytest -v -p faulthandler -o faulthandler_timeout=5 -s external-builds/pytorch/smoke-tests/

echo 'Task completed!'
