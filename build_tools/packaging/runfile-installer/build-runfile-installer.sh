#!/bin/bash

# #############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# This script orchestrates the full ROCm runfile installer build process by:
# 1. Calling setup-installer.sh to pull packages from AMD repositories
# 2. Calling build-installer.sh to extract packages and build the .run file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_INSTALLER_DIR="$SCRIPT_DIR/build-installer"

# Arguments to pass to each script
SETUP_ARGS=()
BUILD_ARGS=()
SHOW_HELP=0

# Phase control flags
SKIP_SETUP=0
SKIP_BUILD=0

###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $0 [options]

This script performs a complete ROCm runfile installer build:
  1. Pull packages from AMD repositories (setup-installer.sh)
  2. Extract packages and build .run file (build-installer.sh)

[Setup Options] - Passed to setup-installer.sh:
    rocm                  = Setup only ROCm packages (skip AMDGPU).
    amdgpu                = Setup only AMDGPU packages (skip ROCm).
    amdgpu-mode=all       = Setup AMDGPU packages for all supported distributions (default).
    amdgpu-mode=single    = Setup AMDGPU packages for current distro only.
    rocm-mode=native      = Pull DEB packages using native OS (default).
    rocm-mode=chroot      = Pull DEB packages using Ubuntu chroot (for RPM-based OS).
    rocm-archs=<archs>    = Set GPU architectures to pull (e.g., gfx94x,gfx950). Default: gfx94x,gfx950.
    pull=nightlies        = Pull ROCm packages from nightlies repository.
    pull=prereleases      = Pull ROCm packages from prereleases repository (default).
    pullbuild=<version>   = Set ROCm build number to pull (e.g., 7.11).
    pullrocmver=<version> = Set ROCm version for package names (e.g., 7.11).
    pullrocmpkgver=<ver>  = Set explicit package version (e.g., 7.11.0-2).

[Build Options] - Passed to build-installer.sh:
    noextract             = Disable package extraction.
    norocm                = Disable ROCm package extraction.
    noamdgpu              = Disable AMDGPU package extraction.
    noextractcontent      = Disable package extraction content (deps and scriptlets only).
    contentlist           = List all files extracted to content directories.
    norunfile             = Disable makeself build of installer runfile.
    nogui                 = Disable GUI building.

[Script Options]:
    help                  = Display this help information.
    skip-setup            = Skip package setup/pull phase (only run build).
    skip-build            = Skip build phase (only run setup/pull).

Examples:
    # Full build with defaults (prereleases, all AMDGPU distros)
    $0

    # Pull only ROCm packages and build
    $0 rocm

    # Pull only AMDGPU for current distro and build
    $0 amdgpu amdgpu-mode=single

    # Pull from nightlies with specific build number
    $0 pull=nightlies pullbuild=20260211-21893116598

    # Pull packages for specific GPU architectures
    $0 rocm-archs=gfx94x,gfx950,gfx103x

    # Build without GUI
    $0 nogui

    # Pull packages but don't create runfile (for testing extraction)
    $0 norunfile

    # Only pull packages (skip build phase)
    $0 skip-build

    # Only build (assumes packages already pulled)
    $0 skip-setup
END_USAGE
}

###### Main script #############################################################

# Record start time
SCRIPT_START_TIME=$(date +%s)

echo ==============================
echo ROCM RUNFILE INSTALLER BUILDER
echo ==============================

# Verify build-installer directory exists
if [ ! -d "$BUILD_INSTALLER_DIR" ]; then
    echo -e "\e[31mERROR: build-installer directory not found at: $BUILD_INSTALLER_DIR\e[0m"
    exit 1
fi

# Parse arguments and categorize them
while (($#)); do
    case "$1" in
    help)
        usage
        exit 0
        ;;
    skip-setup)
        SKIP_SETUP=1
        shift
        ;;
    skip-build)
        SKIP_BUILD=1
        shift
        ;;
    # Setup-specific arguments
    rocm|amdgpu)
        SETUP_ARGS+=("$1")
        shift
        ;;
    amdgpu-mode=*|rocm-mode=*|rocm-archs=*|pull=*|pullbuild=*|pullrocmver=*|pullrocmpkgver=*)
        SETUP_ARGS+=("$1")
        shift
        ;;
    # Build-specific arguments
    noextract|norocm|noamdgpu|noextractcontent|contentlist|norunfile|nogui)
        BUILD_ARGS+=("$1")
        shift
        ;;
    *)
        echo -e "\e[33mWARNING: Unknown option: $1\e[0m"
        shift
        ;;
    esac
done

# Auto-configure build arguments based on setup arguments
# If rocm specified without amdgpu, disable AMDGPU extraction in build
if [[ " ${SETUP_ARGS[@]} " =~ " rocm " ]] && [[ ! " ${SETUP_ARGS[@]} " =~ " amdgpu " ]]; then
    # User specified only rocm, so disable AMDGPU extraction in build unless explicitly enabled
    if [[ ! " ${BUILD_ARGS[@]} " =~ " noamdgpu " ]]; then
        BUILD_ARGS+=("noamdgpu")
        echo "Auto-added 'noamdgpu' to build args (rocm-only build)"
    fi
fi

# If amdgpu specified without rocm, disable ROCm extraction in build
if [[ " ${SETUP_ARGS[@]} " =~ " amdgpu " ]] && [[ ! " ${SETUP_ARGS[@]} " =~ " rocm " ]]; then
    # User specified only amdgpu, so disable ROCm extraction in build unless explicitly enabled
    if [[ ! " ${BUILD_ARGS[@]} " =~ " norocm " ]]; then
        BUILD_ARGS+=("norocm")
        echo "Auto-added 'norocm' to build args (amdgpu-only build)"
    fi
fi

# Verify at least one phase is enabled
if [ $SKIP_SETUP -eq 1 ] && [ $SKIP_BUILD -eq 1 ]; then
    echo -e "\e[31mERROR: Cannot skip both setup and build phases!\e[0m"
    echo "Use 'help' to see usage information."
    exit 1
fi

# Display build configuration
echo "Build Configuration:"
echo "  Script Directory:      $SCRIPT_DIR"
echo "  Build Scripts Dir:     $BUILD_INSTALLER_DIR"
echo "  Skip Setup Phase:      $([ $SKIP_SETUP -eq 1 ] && echo "Yes" || echo "No")"
echo "  Skip Build Phase:      $([ $SKIP_BUILD -eq 1 ] && echo "Yes" || echo "No")"
echo "  Setup Arguments:       ${SETUP_ARGS[@]:-<none>}"
echo "  Build Arguments:       ${BUILD_ARGS[@]:-<none>}"
echo ""

# Change to build-installer directory
cd "$BUILD_INSTALLER_DIR" || {
    echo -e "\e[31mERROR: Failed to change to build-installer directory\e[0m"
    exit 1
}

###### Phase 1: Setup and Pull Packages ########################################

if [ $SKIP_SETUP -eq 0 ]; then
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Phase 1: Setup and Pull Packages"
    echo "------------------------------------------------------------------------"
    echo ""

    echo "Running: ./setup-installer.sh ${SETUP_ARGS[@]}"
    echo ""

    ./setup-installer.sh "${SETUP_ARGS[@]}"
    SETUP_EXIT_CODE=$?

    if [ $SETUP_EXIT_CODE -ne 0 ]; then
        echo ""
        echo -e "\e[31m----------------------------------------\e[0m"
        echo -e "\e[31mERROR: Package setup/pull failed!\e[0m"
        echo -e "\e[31mExit code: $SETUP_EXIT_CODE\e[0m"
        echo -e "\e[31m----------------------------------------\e[0m"
        exit $SETUP_EXIT_CODE
    fi

    echo ""
    echo -e "\e[32mPhase 1 Complete: Packages pulled successfully\e[0m"
else
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Phase 1: Skipped (skip-setup specified)"
    echo "------------------------------------------------------------------------"
    echo ""
fi

###### Phase 2: Build Installer ################################################

if [ $SKIP_BUILD -eq 0 ]; then
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Phase 2: Extract Packages and Build Runfile"
    echo "------------------------------------------------------------------------"
    echo ""

    echo "Running: ./build-installer.sh ${BUILD_ARGS[@]}"
    echo ""

    ./build-installer.sh "${BUILD_ARGS[@]}"
    BUILD_EXIT_CODE=$?

    if [ $BUILD_EXIT_CODE -ne 0 ]; then
        echo ""
        echo -e "\e[31m----------------------------------------\e[0m"
        echo -e "\e[31mERROR: Installer build failed!\e[0m"
        echo -e "\e[31mExit code: $BUILD_EXIT_CODE\e[0m"
        echo -e "\e[31m----------------------------------------\e[0m"
        exit $BUILD_EXIT_CODE
    fi

    echo ""
    echo -e "\e[32mPhase 2 Complete: Installer built successfully\e[0m"
else
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Phase 2: Skipped (skip-build specified)"
    echo "------------------------------------------------------------------------"
    echo ""
fi

###### Complete ################################################################

# Calculate total build time
SCRIPT_END_TIME=$(date +%s)
SCRIPT_ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

# Convert seconds to hours, minutes, seconds
SCRIPT_HOURS=$((SCRIPT_ELAPSED / 3600))
SCRIPT_MINUTES=$(((SCRIPT_ELAPSED % 3600) / 60))
SCRIPT_SECONDS=$((SCRIPT_ELAPSED % 60))

echo ""
echo "------------------------------------------------------------------------"
echo "Build Pipeline Complete!"
echo "------------------------------------------------------------------------"
echo ""

echo -e "\e[32mAll phases completed successfully!\e[0m"
echo ""
echo "Total Time: ${SCRIPT_HOURS}h ${SCRIPT_MINUTES}m ${SCRIPT_SECONDS}s (${SCRIPT_ELAPSED} seconds)"

# Display output information if build was run
if [ $SKIP_BUILD -eq 0 ]; then
    echo ""
    echo "Output Location: $SCRIPT_DIR/build/"
    if [ -d "$SCRIPT_DIR/build" ]; then
        echo "Runfile(s):"
        ls -lh "$SCRIPT_DIR/build"/*.run 2>/dev/null || echo "  (No .run files found)"
    fi
fi

exit 0
