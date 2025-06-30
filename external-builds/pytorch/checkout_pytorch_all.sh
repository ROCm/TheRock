#!/usr/bin/bash

# Causes bash process to die immediately after child process returns error
# to make sure that script does not continue logic if error has happened.
set -e
set -o pipefail

SCRIPT_DIR="$(cd $(dirname $0) && pwd)"
if ! source $SCRIPT_DIR/env_init.sh; then
    echo "Failed to init ROCK build environment"
    echo "Make sure that TheRock has been build first"
    exit 1
fi

if ! $SCRIPT_DIR/pytorch_torch_repo.py checkout; then
    echo "Failed to checkout pytorch"
    exit 1
fi

if ! pip install -r pytorch/requirements.txt; then
    echo "Failed to install python requirements with pip install -r pytorch/requirements.txt"
    exit 1
fi

pip install mkl-static mkl-include

if ! $SCRIPT_DIR/pytorch_vision_repo.py checkout; then
    echo "Failed to checkout pytorch vision"
    exit 1
fi

if ! $SCRIPT_DIR/pytorch_audio_repo.py checkout; then
    echo "Failed to checkout pytorch audio"
    exit 1
fi
