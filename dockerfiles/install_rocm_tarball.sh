#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# install_rocm_tarball.sh
#
# Downloads and installs ROCm from a tarball.
# Supports nightlies, prereleases, devreleases, and stable releases.
#
# Usage:
#   ./install_rocm_tarball.sh <VERSION> <AMDGPU_FAMILY> [RELEASE_TYPE]
#
# Arguments:
#   VERSION          - Full version string (e.g., 7.11.0a20251211, 7.10.0)
#   AMDGPU_FAMILY    - AMD GPU family (e.g., gfx110X-all, gfx94X-dcgpu).
#                      Special value: 'multi-arch' downloads AMD's bundled
#                      tarball that contains kpack files for all GPU families
#                      (from the tarball-multi-arch/ path, except on the nightly
#                      host where every tarball shares one tarball/ directory).
#   RELEASE_TYPE     - Release type: nightlies (default), prereleases, devreleases, stable
#
# Examples:
#   ./install_rocm_tarball.sh 7.11.0a20251211 gfx110X-all
#   ./install_rocm_tarball.sh 7.11.0a20251211 gfx94X-dcgpu nightlies
#   ./install_rocm_tarball.sh 7.10.0rc2 gfx110X-all prereleases
#   ./install_rocm_tarball.sh 7.10.0 gfx94X-dcgpu stable
#   ./install_rocm_tarball.sh 7.13.0a20260515 multi-arch nightlies   # multi-arch

set -eu

# Parse arguments
VERSION="${1:?Error: VERSION is required}"
AMDGPU_FAMILY="${2:?Error: AMDGPU_FAMILY is required}"
RELEASE_TYPE="${3:-nightlies}"

# URL-encode '+' as '%2B' in VERSION (required for devreleases)
VERSION_ENCODED="${VERSION//+/%2B}"

# AMDGPU_FAMILY=multi-arch selects AMD's bundled all-GPU tarball. The tarball
# filename slot uses "multiarch" (no hyphen) while the directory name that
# carries it uses "multi-arch" (with hyphen) — handle both conventions
# explicitly here.
if [ "$AMDGPU_FAMILY" = "multi-arch" ]; then
    TARBALL_DIR="tarball-multi-arch"
    FAMILY_SLOT="multiarch"
else
    TARBALL_DIR="tarball"
    FAMILY_SLOT="$AMDGPU_FAMILY"
fi

# Directory holding the tarballs, by release type:
# - nightlies use: https://nightly.repo.amd.com/rocm/core/tarball/
# - stable releases use: https://repo.amd.com/rocm/${TARBALL_DIR}/
# - other releases use: https://rocm.{RELEASE_TYPE}.amd.com/${TARBALL_DIR}/
case "$RELEASE_TYPE" in
    nightlies)
        LISTING_URL="https://nightly.repo.amd.com/rocm/core/tarball/"
        ;;
    stable)
        LISTING_URL="https://repo.amd.com/rocm/${TARBALL_DIR}/"
        ;;
    *)
        LISTING_URL="https://rocm.${RELEASE_TYPE}.amd.com/${TARBALL_DIR}/"
        ;;
esac
TARBALL_URL="${LISTING_URL}therock-dist-linux-${FAMILY_SLOT}-${VERSION_ENCODED}.tar.gz"

echo "=============================================="
echo "ROCm Tarball Installation"
echo "=============================================="
echo "Version:         ${VERSION}"
echo "AMDGPU Family:   ${AMDGPU_FAMILY}"
echo "Release Type:    ${RELEASE_TYPE}"
echo "Tarball URL:     ${TARBALL_URL}"
echo "=============================================="

# Download tarball
TARBALL_FILE="/tmp/rocm-tarball.tar.gz"

echo "Downloading tarball..."
# Use curl with -fsSL: fail on errors, silent, show errors, follow redirects
# If direct URL fails, try fuzzy match (supports simplified AMDGPU_FAMILY like gfx110x)
if ! curl -fsSL -o "$TARBALL_FILE" "$TARBALL_URL" 2>/dev/null; then
    echo "Direct URL not found, searching for matching tarball..."
    # Case-insensitive search for tarball matching FAMILY_SLOT and VERSION.
    # The HTML listing contains literal '+' (not URL-encoded), so use VERSION
    # with '+' escaped as '\+' for PCRE rather than VERSION_ENCODED.
    VERSION_REGEX="${VERSION//+/\\+}"
    MATCHED_FILE=$(curl -fsSL "$LISTING_URL" 2>/dev/null \
        | grep -ioP "therock-dist-linux-[^\"]*${FAMILY_SLOT}[^\"]*-${VERSION_REGEX}\.tar\.gz" \
        | head -1) || true
    if [ -z "$MATCHED_FILE" ]; then
        echo "Error: No tarball found matching '${AMDGPU_FAMILY}' and version '${VERSION}'"
        echo "Tried direct URL: ${TARBALL_URL}"
        echo "Tried fuzzy search at: ${LISTING_URL}"
        echo "Hint: specify the full AMDGPU_FAMILY (e.g., gfx110X-all) or check available tarballs"
        exit 1
    fi
    # URL-encode '+' in the matched filename for the download URL
    TARBALL_URL="${LISTING_URL}${MATCHED_FILE//+/%2B}"
    echo "Found matching tarball: ${MATCHED_FILE}"
    echo "Downloading from: ${TARBALL_URL}"
    curl -fsSL -o "$TARBALL_FILE" "$TARBALL_URL" || {
        echo "Error: Failed to download tarball from $TARBALL_URL"
        exit 1
    }
fi

# Verify download
if [ ! -f "$TARBALL_FILE" ] || [ ! -s "$TARBALL_FILE" ]; then
    echo "Error: Downloaded file is empty or does not exist"
    exit 1
fi

# Install directory is fixed to /opt/rocm-{VERSION}
ROCM_INSTALL_DIR="/opt/rocm-${VERSION}"

# Extract tarball to versioned directory
echo "Extracting tarball to ${ROCM_INSTALL_DIR}..."
mkdir -p "$ROCM_INSTALL_DIR"
tar -xzf "$TARBALL_FILE" -C "$ROCM_INSTALL_DIR"

# Clean up downloaded file
rm -f "$TARBALL_FILE"
echo "Tarball extracted and cleaned up"

# Create symlink /opt/rocm -> /opt/rocm-{VERSION} for compatibility
ln -sfn "$ROCM_INSTALL_DIR" /opt/rocm
echo "Created symlink: /opt/rocm -> $ROCM_INSTALL_DIR"

# Verify bin and lib folder exists after extraction
echo "Verifying installation..."
for dir in bin include lib libexec share; do
    if [ ! -d "$ROCM_INSTALL_DIR/$dir" ]; then
        echo "Error: ROCm $dir directory not found"
        exit 1
    fi
    echo "ROCm $dir found in $ROCM_INSTALL_DIR/$dir"
done

echo "=============================================="
echo "ROCm installed successfully to $ROCM_INSTALL_DIR"
echo "ROCM_PATH=$ROCM_INSTALL_DIR"
echo "PATH should include: $ROCM_INSTALL_DIR/bin"
echo "=============================================="
echo ""
echo "Note: If running this script standalone (not in Docker),"
echo "you need to set up environment variables manually."
echo ""
echo "Add these lines to your ~/.bashrc:"
echo ""
echo "  export ROCM_PATH=/opt/rocm"
echo '  export PATH="$ROCM_PATH/bin:$PATH"'
echo ""
echo "Then run: source ~/.bashrc"
echo ""
echo "(In Docker, this is handled by ENV in the Dockerfile)"
echo "=============================================="
