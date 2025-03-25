#!/bin/bash
set -e

echo 'Running inside the container'

echo 'Set path to .local/bin'
export PATH="$HOME/.local/bin:$PATH"

echo 'Check Python version'
python3 --version

echo 'List the pip packages'
pip list

echo 'Install pytest'
PIP_BREAK_SYSTEM_PACKAGES=1 pip install -U pytest --timeout 60 -i https://pypi.org/simple

echo 'Run smoke tests'
pytest -v external-builds/pytorch/smoke-tests/pytorch_smoke_tests.py

echo 'Task completed!'
