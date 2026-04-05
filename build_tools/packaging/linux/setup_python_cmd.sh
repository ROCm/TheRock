#!/bin/bash

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

# Resolve PYTHON_CMD from --os-profile and optionally install that runtime.
#
# Mapping (install packages + PYTHON_CMD):
#
# - ubuntu* / debian* -> apt: python3.12, python3.12-venv, python3-pip -> python3.12
# - sles* -> zypper: python313, python313-pip -> python3.13
# - else (e.g. rhel10) -> dnf: python3.12, python3.12-pip -> python3.12
#
# Use --install-runtime in CI so Python install lives in this script (not the workflow
# prerequisites). The workflow may run a tiny bootstrap if python3 is missing before
# calling this script.
#
# --output-format matches get_s3_config.py: env, json, github.
#
# Sample usage
# ------------
#
# CI (install + append PYTHON_CMD to GITHUB_ENV):
#
#     bash build_tools/packaging/linux/setup_python_cmd.sh \
#         --os-profile ubuntu2404 --install-runtime >> "$GITHUB_ENV"
#
# Resolve only (no package manager):
#
#     bash build_tools/packaging/linux/setup_python_cmd.sh --os-profile rhel10 --output-format json

set -euo pipefail

# Defaults
OS_PROFILE=""
INSTALL_RUNTIME=false
OUTPUT_FORMAT="github"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --os-profile)
            OS_PROFILE="$2"
            shift 2
            ;;
        --install-runtime)
            INSTALL_RUNTIME=true
            shift
            ;;
        --output-format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$OS_PROFILE" ]]; then
    echo "Error: --os-profile is required" >&2
    exit 1
fi

# Map os_profile to the interpreter command name
resolve_python_cmd() {
    local os_profile="$1"

    if [[ "$os_profile" == ubuntu* ]] || [[ "$os_profile" == debian* ]]; then
        echo "python3.12"
    elif [[ "$os_profile" == sles* ]]; then
        echo "python3.13"
    else
        echo "python3.12"
    fi
}

# Install Python and pip with apt/zypper/dnf
install_python_runtime() {
    local os_profile="$1"

    if [[ "$os_profile" == ubuntu* ]] || [[ "$os_profile" == debian* ]]; then
        # Non-interactive apt (CI containers); avoids debconf prompts
        export DEBIAN_FRONTEND=noninteractive
        # Refresh package index so install sees current repos; -qq reduces log noise
        apt-get update -qq
        apt-get install -y --no-install-recommends \
            python3.12 \
            python3.12-venv \
            python3-pip
    elif [[ "$os_profile" == sles* ]]; then
        # zypper refresh ~= apt update (metadata before install)
        zypper --non-interactive refresh
        zypper --non-interactive install -y \
            python313 \
            python313-pip
    else
        # RHEL UBI: --allowerasing resolves curl vs curl-minimal style conflicts when pulling deps
        dnf install -y --allowerasing \
            python3.12 \
            python3.12-pip
    fi
}

# Install first so emitted PYTHON_CMD exists on PATH for later workflow steps
if [[ "$INSTALL_RUNTIME" == true ]]; then
    install_python_runtime "$OS_PROFILE"
fi

PYTHON_CMD=$(resolve_python_cmd "$OS_PROFILE")

# Emit output in requested format
case "$OUTPUT_FORMAT" in
    json)
        echo "{\"python_cmd\": \"$PYTHON_CMD\"}"
        ;;
    github)
        # One NAME=value line; often appended to GITHUB_ENV
        echo "PYTHON_CMD=$PYTHON_CMD"
        ;;
    env)
        # Shell: eval "$(...)" or copy-paste
        echo "export PYTHON_CMD=$PYTHON_CMD"
        ;;
    *)
        echo "Error: Unknown output format: $OUTPUT_FORMAT" >&2
        exit 1
        ;;
esac
