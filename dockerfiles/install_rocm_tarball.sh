#!/bin/bash
# install_rocm_tarball.sh
#
# Downloads and installs ROCm from a tarball.
# Supports nightlies, prereleases, and devreleases.
#
# Usage:
#   ./install_rocm_tarball.sh <VERSION> <AMDGPU_FAMILY> [RELEASE_TYPE]
#
# Arguments:
#   VERSION          - Full version string (e.g., 7.11.0a20251211)
#   AMDGPU_FAMILY    - AMD GPU family (e.g., gfx110X-all, gfx94X-dcgpu)
#   RELEASE_TYPE     - Release type: nightlies (default), prereleases, devreleases
#
# Examples:
#   ./install_rocm_tarball.sh 7.11.0a20251211 gfx110X-all
#   ./install_rocm_tarball.sh 7.11.0a20251211 gfx94X-dcgpu nightlies
#   ./install_rocm_tarball.sh 7.10.0rc2 gfx110X-all prereleases

set -e

# Parse arguments
VERSION="${1:?Error: VERSION is required}"
AMDGPU_FAMILY="${2:?Error: AMDGPU_FAMILY is required}"
RELEASE_TYPE="${3:-nightlies}"

# URL-encode '+' as '%2B' in VERSION (required for devreleases)
VERSION_ENCODED="${VERSION//+/%2B}"

# Build tarball URL: https://rocm.{RELEASE_TYPE}.amd.com/tarball/therock-dist-linux-{FAMILY}-{VERSION}.tar.gz
TARBALL_URL="https://rocm.${RELEASE_TYPE}.amd.com/tarball/therock-dist-linux-${AMDGPU_FAMILY}-${VERSION_ENCODED}.tar.gz"

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
curl -fsSL -o "$TARBALL_FILE" "$TARBALL_URL" || {
    echo "Error: Failed to download tarball from $TARBALL_URL"
    exit 1
}

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
for dir in bin clients include lib libexec share; do
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
