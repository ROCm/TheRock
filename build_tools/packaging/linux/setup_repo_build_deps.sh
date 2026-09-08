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
# Commands are prefixed with sudo only when this runs as a non-root user on a
# system that has it. The rpm build images run as root and ship no sudo.
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
            # Check for the value before reading it: this script runs under
            # "set -u", so a trailing "--os-profile" would otherwise abort with
            # bash's own unbound-variable error instead of the message below.
            if [[ $# -lt 2 ]]; then
                echo "Error: --os-profile requires a value" >&2
                exit 1
            fi
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

# Reject an unrecognised profile rather than letting it fall through to a
# package manager that is not the one it needs. Matched by family so a new
# release of a supported distro works without editing this script.
case "$OS_PLC" in
    ubuntu*|debian*|rhel*|sles*) ;;
    *)
        echo "Error: unrecognised --os-profile '$OS_PROFILE'" >&2
        echo "Expected an ubuntu*, debian*, rhel* or sles* profile" \
            "(this project builds ubuntu2404, rhel8, rhel10, sles16)" >&2
        exit 1
        ;;
esac

# Elevate only when sudo exists and we are not already root. The rpm build
# images (UBI, BCI) run as root and ship no sudo at all, so calling it there
# would fail; a developer following the documentation on a real RHEL or SLES
# machine is not root and does have it. An empty array expands to nothing, so
# the same command line serves both.
SUDO=()
if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
fi

if [[ "$OS_PLC" == ubuntu* ]] || [[ "$OS_PLC" == debian* ]]; then
    export DEBIAN_FRONTEND=noninteractive
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y \
        ca-certificates \
        gnupg \
        debhelper-compat \
        dpkg-dev \
        build-essential
elif [[ "$OS_PLC" == sles* ]]; then
    "${SUDO[@]}" zypper --non-interactive refresh
    "${SUDO[@]}" zypper --non-interactive install -y \
        ca-certificates \
        gpg2 \
        rpm-build
else
    "${SUDO[@]}" dnf makecache
    "${SUDO[@]}" dnf install -y \
        ca-certificates \
        gnupg2 \
        rpm-build
fi
