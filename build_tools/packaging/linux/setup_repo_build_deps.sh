#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install the system tools needed to build the amdrocm-repo package.
#
# Package mapping by --os-profile:
#
# - ubuntu* / debian* -> apt: debhelper-compat, dpkg-dev, build-essential
# - sles*             -> zypper: rpm-build
# - else (RHEL-like)  -> dnf: rpm-build
#
# gpg is installed for every profile: signed release lines need it to dearmor
# the deb keyring and to check the signing key fingerprint.
#
# Python is installed separately by setup_python_cmd.sh.
#
# Sample usage
# ------------
#
#     bash build_tools/packaging/linux/setup_repo_build_deps.sh --os-profile rhel10

set -euo pipefail

OS_PROFILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --os-profile)
            OS_PROFILE="$2"
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

if [[ "$OS_PLC" == ubuntu* ]] || [[ "$OS_PLC" == debian* ]]; then
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update
    sudo apt-get install -y \
        ca-certificates \
        gnupg \
        debhelper-compat \
        dpkg-dev \
        build-essential
elif [[ "$OS_PLC" == sles* ]]; then
    zypper --non-interactive refresh
    zypper --non-interactive install -y \
        ca-certificates \
        gpg2 \
        rpm-build
else
    dnf install -y --allowerasing \
        ca-certificates \
        gnupg2 \
        rpm-build
fi
