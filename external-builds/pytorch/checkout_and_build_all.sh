#!/usr/bin/bash

# Causes bash process to die immediately after child process returns error
# to make sure that script does not continue logic if error has happened.
set -e
set -o pipefail

SCRIPT_DIR="$(cd $(dirname $0) && pwd)"
if ! source $SCRIPT_DIR/env_init.sh; then
    echo "Failed to find python virtual-env"
    echo "Make sure that TheRock has been build first"
    exit 1
fi

if ! $SCRIPT_DIR/checkout_pytorch_all.sh; then
    echo "Failed to checkout pytorch, pytorch vision and pytorch audio"
    exit 1
fi

if ! $SCRIPT_DIR/build_pytorch_all.sh; then
    echo "Failed to build pytorch, pytorch vision and pytorch audio"
    exit 1
fi
