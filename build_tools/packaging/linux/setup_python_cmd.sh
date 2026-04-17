#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install Python runtime by --os-profile (optional) and emit PYTHON_CMD for CI.
#
# Package mapping by os_profile (default: distro python3 on PATH):
#
# - ubuntu* / debian* -> apt: python3, python3-venv, python3-pip -> PYTHON_CMD=python3
# - sles* -> zypper: python313, python313-pip (SLE/BCI; unversioned python3 / python3-pip are not valid package names) -> PYTHON_CMD=python3.13
# - else (e.g. UBI 10 / RHEL-like) -> dnf: python3, python3-pip -> PYTHON_CMD=python3
#
# Optional --python-version X.Y (e.g. 3.12): install that stream where supported
# (apt + dnf). Use on UBI 9 / RHEL 9 when default python3 is older than you need.
# On SLES, --python-version is ignored (zypper names vary); distro python3 is used.
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
# UBI 9 / older RHEL-like: pin 3.12 from AppStream / repos:
#
#     bash build_tools/packaging/linux/setup_python_cmd.sh \
#         --os-profile rhel9 --python-version 3.12 --install-runtime >> "$GITHUB_ENV"
#
# Emit PYTHON_CMD only (no package install):
#
#     bash build_tools/packaging/linux/setup_python_cmd.sh --os-profile rhel10 --output-format json

set -euo pipefail

# Defaults
OS_PROFILE=""
INSTALL_RUNTIME=false
OUTPUT_FORMAT="github"
PY_MM="" # e.g. 3.12 when --python-version set; empty = distro default python3

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
        --python-version)
            PY_MM="$2"
            shift 2
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

# Trim whitespace; lowercase for glob matching (e.g. SLES16 must match sles*)
OS_PROFILE="${OS_PROFILE#"${OS_PROFILE%%[![:space:]]*}"}"
OS_PROFILE="${OS_PROFILE%"${OS_PROFILE##*[![:space:]]}"}"
OS_PLC="${OS_PROFILE,,}"

if [[ -n "$PY_MM" ]] && ! [[ "$PY_MM" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "Error: --python-version must be MAJOR.MINOR (e.g. 3.12)" >&2
    exit 1
fi

if [[ -n "$PY_MM" ]]; then
    PYTHON_CMD="python${PY_MM}"
elif [[ "$OS_PLC" == sles* ]]; then
    # SLE/BCI: no installable "python3" metapackage; use python313 -> python3.13
    PYTHON_CMD="python3.13"
else
    PYTHON_CMD="python3"
fi

# Install Python and pip with apt/zypper/dnf
install_python_runtime() {
    local os_profile="$1"

    if [[ "$os_profile" == ubuntu* ]] || [[ "$os_profile" == debian* ]]; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq >&2
        if [[ -n "$PY_MM" ]]; then
            apt-get install -y --no-install-recommends \
                "python${PY_MM}" \
                "python${PY_MM}-venv" \
                "python${PY_MM}-pip" >&2
        else
            apt-get install -y --no-install-recommends \
                python3 \
                python3-venv \
                python3-pip >&2
        fi
    elif [[ "$os_profile" == sles* ]]; then
        if [[ -n "$PY_MM" ]]; then
            echo "Warning: --python-version is not applied on SLES; using python313 stack" >&2
        fi
        zypper --non-interactive refresh >&2
        zypper --non-interactive install -y \
            python313 \
            python313-pip >&2
    else
        # dnf: UBI 9 / RHEL 9 default python3 may be < 3.12; use --python-version 3.12 when needed
        if [[ -n "$PY_MM" ]]; then
            dnf install -y --allowerasing \
                "python${PY_MM}" \
                "python${PY_MM}-pip" >&2
        else
            dnf install -y --allowerasing \
                python3 \
                python3-pip >&2
        fi
    fi
}

# Install first so emitted PYTHON_CMD exists on PATH for later workflow steps
if [[ "$INSTALL_RUNTIME" == true ]]; then
    install_python_runtime "$OS_PLC"
fi

# Emit output in requested format
case "$OUTPUT_FORMAT" in
    json)
        echo "{\"python_cmd\": \"$PYTHON_CMD\"}"
        ;;
    github)
        echo "PYTHON_CMD=$PYTHON_CMD"
        ;;
    env)
        echo "export PYTHON_CMD=$PYTHON_CMD"
        ;;
    *)
        echo "Error: Unknown output format: $OUTPUT_FORMAT" >&2
        exit 1
        ;;
esac
