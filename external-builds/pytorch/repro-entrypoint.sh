#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Entrypoint for the PyTorch test reproducer image.
# Sources ROCm environment, activates the venv, then execs the user command.

set -euo pipefail

# Source ROCm SDK environment (paths captured at image build time)
if [ -f /etc/profile.d/rocm-env.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/rocm-env.sh
fi

# Activate virtual environment
if [ -f /workspace/.venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source /workspace/.venv/bin/activate
fi

exec "$@"
