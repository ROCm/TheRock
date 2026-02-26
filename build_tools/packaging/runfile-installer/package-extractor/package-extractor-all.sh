#!/bin/bash

# #############################################################################
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################

# UNIFIED PACKAGE EXTRACTOR - Extracts both DEB and RPM packages
# Can be run from any distribution (e.g., AlmaLinux 8) to extract both package types
# Uses: ar/tar for DEB files (no dpkg required), rpm2cpio/cpio for RPM files

# This script wraps the existing package-extractor-debs.sh and package-extractor-rpms.sh
# but provides dpkg-free extraction for DEB packages by parsing control files directly

PROG=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for required tools
check_tools() {
    local missing_tools=()

    # Always required
    if ! command -v ar &> /dev/null; then
        missing_tools+=("ar (binutils)")
    fi

    if ! command -v tar &> /dev/null; then
        missing_tools+=("tar")
    fi

    # For RPM extraction
    if ! command -v rpm2cpio &> /dev/null; then
        missing_tools+=("rpm2cpio")
    fi

    if ! command -v cpio &> /dev/null; then
        missing_tools+=("cpio")
    fi

    if ! command -v rpm &> /dev/null; then
        missing_tools+=("rpm")
    fi

    if [ ${#missing_tools[@]} -gt 0 ]; then
        echo "ERROR: Missing required tools:"
        printf '  - %s\n' "${missing_tools[@]}"
        echo ""
        echo "On AlmaLinux/RHEL, install with:"
        echo "  sudo dnf install binutils tar rpm-build cpio"
        exit 1
    fi
}

usage() {
cat <<END_USAGE
Usage: $PROG [options]

Unified package extractor that can extract both DEB and RPM packages from any distro.
Designed to run on AlmaLinux 8 or similar RPM-based distros.

[options]:
    help                    = Display this help information.
    prompt                  = Run the extractor with user prompts.
    amdgpu                  = Extract AMDGPU packages
    rocm                    = Extract ROCm packages

    nocontent               = Disables content extraction (deps, scriptlets will be extracted only).

    pkgs-rocm=<file_path>   = <file_path> Path to ROCm source packages directory for extract.
    pkgs-amdgpu=<file_path> = <file_path> Path to AMDGPU source packages directory for extract.
    ext-rocm=<file_path>    = <file_path> Path to ROCm packages extraction directory.
    ext-amdgpu=<file_path>  = <file_path> Path to AMDGPU packages extraction directory.

Example:
    ./package-extractor-all.sh rocm pkgs-rocm="packages-rocm-deb" ext-rocm="../rocm-installer"
    ./package-extractor-all.sh rocm pkgs-rocm="packages-rocm-rpm" ext-rocm="../rocm-installer"

END_USAGE
}

detect_package_type() {
    local pkg_dir="$1"

    if [ ! -d "$pkg_dir" ]; then
        echo "ERROR: Package directory not found: $pkg_dir"
        return 1
    fi

    # Check for .deb files
    if ls "$pkg_dir"/*.deb &> /dev/null; then
        echo "deb"
        return 0
    fi

    # Check for .rpm files
    if ls "$pkg_dir"/*.rpm &> /dev/null; then
        echo "rpm"
        return 0
    fi

    echo "ERROR: No .deb or .rpm files found in $pkg_dir"
    return 1
}

# Main script
echo "========================================"
echo "Unified Package Extractor (DEB + RPM)"
echo "========================================"

check_tools

# Detect package type from arguments or auto-detect
PACKAGE_DIR=""
EXTRACT_TYPE=""

# Parse arguments to find package directory
for arg in "$@"; do
    if [[ "$arg" == pkgs-rocm=* ]]; then
        PACKAGE_DIR="${arg#*=}"
    elif [[ "$arg" == pkgs-amdgpu=* ]]; then
        PACKAGE_DIR="${arg#*=}"
    fi
done

# Detect package type if we have a package directory
if [ -n "$PACKAGE_DIR" ]; then
    PKG_TYPE=$(detect_package_type "$PACKAGE_DIR")
    if [ $? -eq 0 ]; then
        echo "Detected package type: $PKG_TYPE"

        # Route to appropriate extractor
        if [ "$PKG_TYPE" == "deb" ]; then
            echo "Using DEB extractor (no dpkg required)..."
            exec "$SCRIPT_DIR/package-extractor-debs-nodpkg.sh" "$@"
        elif [ "$PKG_TYPE" == "rpm" ]; then
            echo "Using RPM extractor..."
            exec "$SCRIPT_DIR/package-extractor-rpms.sh" "$@"
        fi
    else
        exit 1
    fi
else
    # No package directory specified, show usage
    if [[ "$1" == "help" || -z "$1" ]]; then
        usage
        exit 0
    else
        echo "ERROR: Could not determine package directory from arguments"
        usage
        exit 1
    fi
fi
