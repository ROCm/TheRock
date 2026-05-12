#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# install_rocm_packages.sh
#
# Installs ROCm from deb/rpm packages via the system package manager.
# Automatically detects the distribution and configures the appropriate repository.
#
# Usage:
#   ./install_rocm_packages.sh <VERSION> <AMDGPU_FAMILY> [RELEASE_TYPE]
#
# Arguments:
#   VERSION          - Full version string (e.g., 7.13.0a20260322, 7.11.0)
#   AMDGPU_FAMILY    - AMD GPU family (e.g., gfx110x, gfx94x, gfx110X-all)
#   RELEASE_TYPE     - Release type: nightlies (default), prereleases, stable
#
# Examples:
#   ./install_rocm_packages.sh 7.13.0a20260322 gfx110x
#   ./install_rocm_packages.sh 7.13.0a20260322 gfx110x nightlies
#   ./install_rocm_packages.sh 7.12.0 gfx94x prereleases
#   ./install_rocm_packages.sh 7.11.0 gfx110x stable

set -euo pipefail

# Parse arguments
VERSION="${1:?Error: VERSION is required}"
AMDGPU_FAMILY="${2:?Error: AMDGPU_FAMILY is required}"
RELEASE_TYPE="${3:-nightlies}"

# ---------------------------------------------------------------------------
# Helper: extract MAJOR.MINOR from VERSION (e.g., 7.13.0a20260322 → 7.13)
# ---------------------------------------------------------------------------
extract_major_minor() {
    echo "$1" | grep -oE '^[0-9]+\.[0-9]+'
}

# ---------------------------------------------------------------------------
# Helper: normalize AMDGPU_FAMILY to package target
#   gfx110X-all → gfx110x
#   gfx94X-dcgpu → gfx94x
#   gfx110x → gfx110x (already normalized)
#   gfx1150 → gfx1150 (no change)
# ---------------------------------------------------------------------------
normalize_gpu_target() {
    echo "$1" | sed 's/-[a-zA-Z]*$//' | tr '[:upper:]' '[:lower:]'
}

# ---------------------------------------------------------------------------
# Helper: extract date from VERSION for nightly builds
#   7.13.0a20260322 → 20260322
# ---------------------------------------------------------------------------
extract_date_from_version() {
    echo "$1" | grep -oE '[0-9]{8}' | head -1 || echo ""
}

# ---------------------------------------------------------------------------
# Helper: detect distro and version from /etc/os-release
# ---------------------------------------------------------------------------
detect_distro_info() {
    if [ -f /etc/os-release ]; then
        # Source in subshell to avoid clobbering script variables (e.g., VERSION)
        DISTRO_ID=$(. /etc/os-release && echo "$ID")
        DISTRO_VERSION=$(. /etc/os-release && echo "${VERSION_ID:-unknown}")
    else
        echo "Error: Cannot detect distribution (/etc/os-release not found)" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Helper: map distro to repo name and package type
# ---------------------------------------------------------------------------
map_distro_to_repo() {
    local id="$1"
    local ver="$2"
    local major_ver="${ver%%.*}"

    case "$id" in
        ubuntu)
            # 24.04 → ubuntu2404, 22.04 → ubuntu2204
            # Use cut to keep MAJOR.MINOR even if VERSION_ID has a patch (e.g., 24.04.1)
            local major_minor
            major_minor=$(echo "$ver" | cut -d. -f1,2)
            REPO_DISTRO="ubuntu$(echo "$major_minor" | tr -d '.')"
            PKG_TYPE="deb"
            PKG_MGR="apt"
            ;;
        almalinux)
            # AlmaLinux uses RHEL repos
            REPO_DISTRO="rhel${major_ver}"
            PKG_TYPE="rpm"
            PKG_MGR="dnf"
            ;;
        azurelinux)
            REPO_DISTRO="azl${major_ver}"
            PKG_TYPE="rpm"
            PKG_MGR="tdnf"
            ;;
        rhel)
            REPO_DISTRO="rhel${major_ver}"
            PKG_TYPE="rpm"
            PKG_MGR="dnf"
            ;;
        sles)
            REPO_DISTRO="sles${major_ver}"
            PKG_TYPE="rpm"
            PKG_MGR="zypper"
            ;;
        *)
            echo "Error: Unsupported distribution: $id"
            echo "Supported: ubuntu, almalinux, azurelinux, rhel, sles"
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Helper: resolve nightly build directory from date
# ---------------------------------------------------------------------------
resolve_nightly_build_dir() {
    local date_str="$1"
    local repo_type="$2"  # deb or rpm

    if [ -z "$date_str" ]; then
        echo "Error: Cannot extract date from VERSION '$VERSION' for nightly builds" >&2
        echo "Nightly VERSION should contain a date, e.g., 7.13.0a20260322" >&2
        exit 1
    fi

    local listing_url="https://rocm.nightlies.amd.com/${repo_type}/"
    echo "Searching for nightly build directory matching date ${date_str}..." >&2

    local build_dir
    build_dir=$(curl -fsSL --connect-timeout 30 --retry 3 --retry-delay 5 \
        "$listing_url" | grep -oE "${date_str}-[0-9]+" | head -1) || true

    if [ -z "$build_dir" ]; then
        echo "Error: No nightly build found for date ${date_str}" >&2
        echo "Check available builds at: ${listing_url}" >&2
        exit 1
    fi

    echo "Found nightly build directory: ${build_dir}" >&2
    # Only the build dir goes to stdout (for capture by caller)
    echo "$build_dir"
}

# ---------------------------------------------------------------------------
# Helper: build the repo base URL
# ---------------------------------------------------------------------------
build_repo_url() {
    local release_type="$1"
    local pkg_type="$2"
    local repo_distro="$3"
    local nightly_build_dir="$4"

    case "$release_type" in
        nightlies)
            if [ "$pkg_type" = "deb" ]; then
                echo "https://rocm.nightlies.amd.com/deb/${nightly_build_dir}"
            else
                echo "https://rocm.nightlies.amd.com/rpm/${nightly_build_dir}/x86_64"
            fi
            ;;
        prereleases)
            if [ "$pkg_type" = "deb" ]; then
                echo "https://rocm.prereleases.amd.com/packages/${repo_distro}"
            else
                echo "https://rocm.prereleases.amd.com/packages/${repo_distro}/x86_64"
            fi
            ;;
        stable)
            if [ "$pkg_type" = "deb" ]; then
                echo "https://repo.amd.com/rocm/packages/${repo_distro}"
            else
                echo "https://repo.amd.com/rocm/packages/${repo_distro}/x86_64"
            fi
            ;;
        *)
            echo "Error: Unsupported release type: $release_type"
            echo "Supported: nightlies, prereleases, stable"
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Helper: get GPG key URL for signed repos
# ---------------------------------------------------------------------------
get_gpg_key_url() {
    local release_type="$1"
    case "$release_type" in
        prereleases)
            echo "https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg"
            ;;
        stable)
            echo "https://repo.amd.com/rocm/packages/gpg/rocm.gpg"
            ;;
        *)
            echo ""
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Configure DEB repository and install
# ---------------------------------------------------------------------------
install_deb() {
    local repo_url="$1"
    local gpg_key_url="$2"
    local meta_package="$3"
    local release_type="$4"

    echo "Configuring APT repository..."

    if [ -n "$gpg_key_url" ]; then
        # Signed repo: import GPG key (ASCII-armored, needs dearmor for apt)
        mkdir -p /etc/apt/keyrings
        curl -fsSL --connect-timeout 30 --retry 3 --retry-delay 5 \
            "$gpg_key_url" | gpg --dearmor -o /etc/apt/keyrings/amdrocm.gpg
        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] ${repo_url} stable main" \
            > /etc/apt/sources.list.d/rocm.list
    else
        # Unsigned repo (nightlies): use trusted=yes
        echo "deb [arch=amd64 trusted=yes] ${repo_url} stable main" \
            > /etc/apt/sources.list.d/rocm.list
    fi

    echo "Repository configured: $(cat /etc/apt/sources.list.d/rocm.list)"
    apt-get update

    echo "Installing ${meta_package}..."
    apt-get install -y --no-install-recommends "$meta_package"
    rm -rf /var/lib/apt/lists/*
}

# ---------------------------------------------------------------------------
# Configure RPM repository and install (dnf or tdnf)
# ---------------------------------------------------------------------------
install_rpm_dnf() {
    local repo_url="$1"
    local gpg_key_url="$2"
    local meta_package="$3"
    local release_type="$4"
    local pkg_mgr="$5"

    echo "Configuring YUM/DNF repository..."

    local gpgcheck=1
    local gpgkey_line="gpgkey=${gpg_key_url}"
    if [ -z "$gpg_key_url" ]; then
        gpgcheck=0
        gpgkey_line=""
    fi

    cat > /etc/yum.repos.d/rocm.repo << REPOEOF
[rocm]
name=ROCm
baseurl=${repo_url}
enabled=1
gpgcheck=${gpgcheck}
${gpgkey_line}
priority=50
REPOEOF

    echo "Repository configured:"
    cat /etc/yum.repos.d/rocm.repo

    if [ "$pkg_mgr" = "tdnf" ]; then
        # tdnf requires explicit GPG key import before install
        if [ -n "$gpg_key_url" ]; then
            curl -fsSL --connect-timeout 30 --retry 3 --retry-delay 5 \
                "$gpg_key_url" -o /tmp/rocm.gpg
            rpm --import /tmp/rocm.gpg
            rm -f /tmp/rocm.gpg
        fi
        tdnf clean all
        echo "Installing ${meta_package}..."
        tdnf install -y "$meta_package"
    else
        # dnf: --allowerasing needed for RHEL UBI images where curl-minimal conflicts with curl
        dnf clean all
        echo "Installing ${meta_package}..."
        dnf install -y --allowerasing "$meta_package"
        dnf clean all
    fi
}

# ---------------------------------------------------------------------------
# Configure RPM repository and install (zypper)
# ---------------------------------------------------------------------------
install_rpm_zypper() {
    local repo_url="$1"
    local gpg_key_url="$2"
    local meta_package="$3"
    local release_type="$4"

    echo "Configuring Zypper repository..."

    local gpgcheck=1
    local gpgkey_line="gpgkey=${gpg_key_url}"
    if [ -z "$gpg_key_url" ]; then
        gpgcheck=0
        gpgkey_line=""
    fi

    cat > /etc/zypp/repos.d/rocm.repo << REPOEOF
[rocm]
name=ROCm
baseurl=${repo_url}
enabled=1
gpgcheck=${gpgcheck}
${gpgkey_line}
priority=50
REPOEOF

    echo "Repository configured:"
    cat /etc/zypp/repos.d/rocm.repo

    zypper --non-interactive --gpg-auto-import-keys refresh
    echo "Installing ${meta_package}..."
    zypper --non-interactive install --no-recommends "$meta_package"
    zypper clean --all
}

# ===========================================================================
# Main
# ===========================================================================

MAJOR_MINOR=$(extract_major_minor "$VERSION")
GPU_TARGET=$(normalize_gpu_target "$AMDGPU_FAMILY")
META_PACKAGE="amdrocm${MAJOR_MINOR}-${GPU_TARGET}"

echo "=============================================="
echo "ROCm Package Installation"
echo "=============================================="
echo "Version:         ${VERSION}"
echo "Major.Minor:     ${MAJOR_MINOR}"
echo "AMDGPU Family:   ${AMDGPU_FAMILY}"
echo "GPU Target:      ${GPU_TARGET}"
echo "Meta Package:    ${META_PACKAGE}"
echo "Release Type:    ${RELEASE_TYPE}"
echo "=============================================="

# Detect distribution
detect_distro_info
map_distro_to_repo "$DISTRO_ID" "$DISTRO_VERSION"

echo "Distribution:    ${DISTRO_ID} ${DISTRO_VERSION}"
echo "Repo Distro:     ${REPO_DISTRO}"
echo "Package Type:    ${PKG_TYPE}"
echo "Package Manager: ${PKG_MGR}"
echo "=============================================="

# Resolve nightly build directory if needed
NIGHTLY_BUILD_DIR=""
if [ "$RELEASE_TYPE" = "nightlies" ]; then
    DATE_STR=$(extract_date_from_version "$VERSION")
    NIGHTLY_BUILD_DIR=$(resolve_nightly_build_dir "$DATE_STR" "$PKG_TYPE")
fi

# Build repo URL
REPO_URL=$(build_repo_url "$RELEASE_TYPE" "$PKG_TYPE" "$REPO_DISTRO" "$NIGHTLY_BUILD_DIR")
GPG_KEY_URL=$(get_gpg_key_url "$RELEASE_TYPE")

echo "Repo URL:        ${REPO_URL}"
echo "GPG Key URL:     ${GPG_KEY_URL:-none (unsigned)}"
echo "=============================================="

# Install based on package type
case "$PKG_MGR" in
    apt)
        install_deb "$REPO_URL" "$GPG_KEY_URL" "$META_PACKAGE" "$RELEASE_TYPE"
        ;;
    dnf|tdnf)
        install_rpm_dnf "$REPO_URL" "$GPG_KEY_URL" "$META_PACKAGE" "$RELEASE_TYPE" "$PKG_MGR"
        ;;
    zypper)
        install_rpm_zypper "$REPO_URL" "$GPG_KEY_URL" "$META_PACKAGE" "$RELEASE_TYPE"
        ;;
esac

# Create compatibility symlinks (fallback)
# Recent deb/rpm packages create symlinks via update-alternatives automatically.
# This fallback handles older packages or missing symlinks (e.g., share/).
CORE_DIR="/opt/rocm/core-${MAJOR_MINOR}"
if [ -d "$CORE_DIR" ]; then
    echo "Creating compatibility symlinks from /opt/rocm/ to ${CORE_DIR}/..."
    for subdir in bin lib include libexec share; do
        if [ -d "${CORE_DIR}/${subdir}" ] && [ ! -e "/opt/rocm/${subdir}" ]; then
            ln -sfn "${CORE_DIR}/${subdir}" "/opt/rocm/${subdir}"
            echo "  /opt/rocm/${subdir} -> ${CORE_DIR}/${subdir}"
        fi
    done
fi

# Verify installation
echo "=============================================="
echo "Verifying installation..."
missing_required=0
for dir in bin lib include; do
    if [ -d "/opt/rocm/${dir}" ]; then
        echo "ROCm ${dir} found at /opt/rocm/${dir}"
    else
        echo "Error: /opt/rocm/${dir} not found" >&2
        missing_required=1
    fi
done

if [ -x /opt/rocm/bin/hipcc ]; then
    echo "hipcc found at /opt/rocm/bin/hipcc"
else
    echo "Error: hipcc not found at /opt/rocm/bin/hipcc" >&2
    missing_required=1
fi

echo "=============================================="
if [ "$missing_required" -ne 0 ]; then
    echo "ROCm package installation verification failed" >&2
    exit 1
fi
echo "ROCm package installation complete"
echo "=============================================="
