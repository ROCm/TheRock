#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# install_rocm_deps.sh
#
# Installs runtime dependencies for ROCm on various Linux distributions.
# Automatically detects the distribution and uses the appropriate package manager.
#
# Supported distributions:
#   - Ubuntu 22.04, 24.04 (apt)
#   - AlmaLinux 8 (dnf)
#   - Azure Linux 3 (tdnf)
#   - RHEL 8.10, 9.7, 10.1 (dnf)
#   - SLES 15.7, 16.0 (zypper)

set -e

# Detect distribution type from /etc/os-release
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    else
        echo "unknown"
    fi
}

# Detect distribution version from /etc/os-release
detect_version() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "${VERSION_ID:-unknown}"
    else
        echo "unknown"
    fi
}

DISTRO=$(detect_distro)
VERSION_ID=$(detect_version)
echo "Detected distribution: $DISTRO"
echo "Detected version: $VERSION_ID"

case "$DISTRO" in
    ubuntu)
        echo "Installing dependencies using apt..."
        apt-get update
        apt-get install -y --no-install-recommends \
            ca-certificates \
            curl \
            gpg \
            build-essential \
            libelf1 \
            libnuma1 \
            libunwind8 \
            libncurses6 \
            perl \
            file \
            python3 \
            python3-dev \
            python3-pip \
            kmod
        # libdw: libdw1t64 for Ubuntu 24.04+, libdw1 for older versions
        apt-get install -y --no-install-recommends libdw1t64 2>/dev/null || \
            apt-get install -y --no-install-recommends libdw1 || true
        # libssl: libssl3 for Ubuntu 22.04+, libssl1.1 for older versions
        apt-get install -y --no-install-recommends libssl3 2>/dev/null || \
            apt-get install -y --no-install-recommends libssl1.1 || true
        rm -rf /var/lib/apt/lists/*
        ;;

    almalinux)
        echo "Installing dependencies using dnf..."
        # Fix AlmaLinux repo to use direct baseurl instead of mirrorlist
        if [ -f /etc/yum.repos.d/almalinux.repo ]; then
            sed -i 's/^mirrorlist=/#mirrorlist=/g' /etc/yum.repos.d/almalinux.repo
            sed -i 's/^# baseurl=/baseurl=/g' /etc/yum.repos.d/almalinux.repo
        fi
        dnf install -y --setopt=install_weak_deps=False \
            ca-certificates \
            curl \
            libatomic \
            elfutils-libelf \
            elfutils-libs \
            numactl-libs \
            ncurses-libs \
            openssl-libs \
            perl \
            file \
            python3 \
            python3-devel \
            python3-pip \
            kmod
        dnf clean all
        ;;

    azurelinux)
        echo "Installing dependencies using tdnf..."
        tdnf install -y \
            ca-certificates \
            curl \
            tar \
            libatomic \
            elfutils-libelf \
            elfutils-libs \
            numactl-libs \
            libunwind \
            ncurses-libs \
            openssl-libs \
            perl \
            file \
            python3 \
            python3-devel \
            python3-pip \
            kmod
        tdnf clean all
        ;;

    rhel)
        echo "Installing dependencies using dnf for RHEL ${VERSION_ID}..."
        # Use --allowerasing to handle curl-minimal vs curl conflict in UBI images
        dnf install -y --setopt=install_weak_deps=False --allowerasing \
            ca-certificates \
            curl \
            libatomic \
            elfutils-libelf \
            elfutils-libs \
            numactl-libs \
            ncurses-libs \
            openssl-libs \
            perl \
            file \
            python3 \
            python3-devel \
            python3-pip \
            kmod
        dnf clean all
        ;;

    sles)
        echo "Installing dependencies using zypper for SLES ${VERSION_ID}..."

        # Refresh repos
        zypper --non-interactive refresh || true

        # Install dependencies
        zypper --non-interactive install --no-recommends \
            ca-certificates \
            curl \
            tar \
            libatomic1 \
            libelf1 \
            libdw1 \
            libnuma1 \
            ncurses-utils \
            perl \
            file \
            kmod \
            python3 \
            python3-devel \
            python3-pip

        zypper clean --all
        ;;

    *)
        echo "Error: Unsupported distribution: $DISTRO"
        echo "Supported distributions: ubuntu, almalinux, azurelinux, rhel, sles"
        exit 1
        ;;
esac

echo "Dependencies installed successfully for $DISTRO"
