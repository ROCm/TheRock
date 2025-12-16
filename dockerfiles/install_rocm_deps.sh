#!/bin/bash
# install_rocm_deps.sh
#
# Installs runtime dependencies for ROCm on various Linux distributions.
# Automatically detects the distribution and uses the appropriate package manager.
#
# Supported distributions:
#   - Ubuntu 22.04, 24.04 (apt)
#   - AlmaLinux 8 (dnf)
#   - Azure Linux 3 (tdnf)

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

DISTRO=$(detect_distro)
echo "Detected distribution: $DISTRO"

case "$DISTRO" in
    ubuntu)
        echo "Installing dependencies using apt..."
        apt-get update
        apt-get install -y --no-install-recommends \
            ca-certificates \
            curl \
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

    *)
        echo "Error: Unsupported distribution: $DISTRO"
        echo "Supported distributions: ubuntu, almalinux, azurelinux"
        exit 1
        ;;
esac

echo "Dependencies installed successfully for $DISTRO"
