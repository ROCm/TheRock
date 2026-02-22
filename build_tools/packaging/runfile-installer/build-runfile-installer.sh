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

# Build tag and run ID (captured from pulltag and pullrunid)
# Note: These may be pre-set by config files, so only initialize if not already set
PULL_TAG="${PULL_TAG:-}"
PULL_RUNID="${PULL_RUNID:-}"
BUILD_TAG="${BUILD_TAG:-}"
BUILD_RUNID="${BUILD_RUNID:-}"
BUILD_PULL_TAG="${BUILD_PULL_TAG:-}"

###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $0 [options]

This script performs a complete ROCm runfile installer build:
  1. Pull packages from AMD repositories (setup-installer.sh)
  2. Extract packages and build .run file (build-installer.sh)

[General Options]:
    config=<file>         = Load configuration from file (command-line args override config).
                            Preset configs available in build-installer/config/ directory:
                            - config/nightly.config
                            - config/prerelease.config
                            - config/release.config
                            - config/dev.config
                            Config file sets both setup and build variables.

[Setup Options] - Passed to setup-installer.sh:
    rocm                  = Setup only ROCm packages (skip AMDGPU).
    amdgpu                = Setup only AMDGPU packages (skip ROCm).
    
    amdgpu-mode=all       = Setup AMDGPU packages for all supported distributions (default).
    amdgpu-mode=single    = Setup AMDGPU packages for current distro only.
    
    rocm-mode=native      = Pull DEB packages using native OS (default).
    rocm-mode=chroot      = Pull DEB packages using Ubuntu chroot (for RPM-based OS).
    rocm-archs=<archs>    = Set GPU architectures to pull (e.g., gfx94x,gfx950). Default: gfx94x,gfx950.
    
    pull=<release-type>   = Pull ROCm packages from specified repository (dev, nightly, prerelease, release).
    pulltag=<tag>         = Set ROCm build tag (e.g., 20260123 for nightly).
    pullrunid=<runid>     = Set ROCm component build run ID (e.g., 21274498502 for nightly).
    pullrocmver=<version> = Set ROCm version for package names (e.g., 7.12, 7.11).
    pullpkg=<package>     = Set base package name with optional type prefix (default: amdrocm-core-sdk).
                            Syntax: pullpkg=[type:]<package>
                            - arch:<package> = Architecture-specific (has -gfxXYZ suffix, default)
                              Example: pullpkg=arch:amdrocm-core-sdk or pullpkg=amdrocm-core-sdk
                            - base:<package> = Base package (no -gfxXYZ suffix)
                              Example: pullpkg=base:amdrocm-amdsmi
    pullpkgextra=<pkgs>   = Add extra packages (comma-separated) with optional type prefix.
                            Syntax: pullpkgextra=[type:]pkg1,[type:]pkg2
                            - arch:<package> = Architecture-specific (has -gfxXYZ suffix, default)
                            - base:<package> = Base package (no -gfxXYZ suffix)
                            Example: pullpkgextra=arch:amdrocm-opencl,base:amdrocm-llvm

[Build Options] - Passed to build-installer.sh:
    noextract             = Disable package extraction.
    norocm                = Disable ROCm package extraction.
    noamdgpu              = Disable AMDGPU package extraction.
    noextractcontent      = Disable package extraction content (deps and scriptlets only).
    contentlist           = List all files extracted to content directories.
    norunfile             = Disable makeself build of installer runfile.
    nogui                 = Disable GUI building.
    buildtag=<tag>        = Set the build tag (default: 1).
    buildrunid=<id>       = Set the Runfile build run ID (default: 99999).
    buildpulltag=<tag>    = Set a tag/name for the builds package pull information. (ie. pulltag-pullid)
    mscomp=<mode>         = Makeself compression (build speed vs file size):
                            prodsmall  = XZ (slowest, ~30% smaller, requires xz-utils)
                            prodmedium = Pbzip2 (slower, ~15-20% smaller, universal)
                            normal     = Gzip -9 (slow, baseline, universal)
                            prodfast   = Pigz -9 (3-4x faster, same size, universal)
                            dev        = Pigz -6 (5-6x faster, ~5% larger, universal)

[Script Options]:
    help                  = Display this help information.
    skip-setup            = Skip package setup/pull phase (only run build).
    skip-build            = Skip build phase (only run setup/pull).

Examples:
    # Using preset configs (recommended)
    $0 config=config/nightly.config                           # Nightly build (7.12.0, prodfast)
    $0 config=config/dev.config                               # Dev build (7.12.0, gfx110x only)
    $0 config=config/prerelease.config                        # Prerelease RC0 (7.11.0, prodmedium)
    $0 config=config/release.config                           # Release build (7.11.0, prodmedium)

    # Basic builds
    $0                                                        # Default build (both ROCm and AMDGPU)
    $0 rocm                                                   # ROCm only build
    $0 amdgpu amdgpu-mode=single                              # AMDGPU for current distro only

    # Pull from specific builds (with actual values from preset configs)
    $0 pull=nightly pulltag=20260212 pullrunid=21933875966 pullrocmver=7.12.0     # Nightly
    $0 pull=dev pulltag=20260219 pullrunid=22188089855 pullrocmver=7.12.0         # Dev
    $0 pull=prerelease pulltag=rc0 pullrocmver=7.11.0                             # Prerelease RC0
    $0 pull=release pullrocmver=7.11.0                                            # Release

    # GPU architectures
    $0 rocm-archs=gfx110x,gfx94x                              # Specific GPU architectures
    $0 rocm-archs=gfx110x                                     # Single GPU architecture

    # Custom packages
    $0 pullpkg=arch:amdrocm-core                              # Custom main package
    $0 pullpkgextra=arch:amdrocm-opencl,base:amdrocm-llvm     # Add extra packages

    # Build options
    $0 nogui                                                  # Build without GUI
    $0 norunfile                                              # Pull and extract only (no .run file)
    $0 mscomp=prodfast                                        # Use fast compression

    # Build phases
    $0 skip-build                                             # Only pull packages
    $0 skip-setup                                             # Only build (packages already pulled)
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
    config=*)
        # Forward to both setup and build scripts
        SETUP_ARGS+=("$1")
        BUILD_ARGS+=("$1")
        shift
        ;;
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
    pulltag=*)
        PULL_TAG="${1#*=}"
        SETUP_ARGS+=("$1")
        shift
        ;;
    pullrunid=*)
        PULL_RUNID="${1#*=}"
        SETUP_ARGS+=("$1")
        shift
        ;;
    buildtag=*)
        BUILD_TAG="${1#*=}"
        BUILD_ARGS+=("$1")
        shift
        ;;
    buildrunid=*)
        BUILD_RUNID="${1#*=}"
        BUILD_ARGS+=("$1")
        shift
        ;;
    buildpulltag=*)
        BUILD_PULL_TAG="${1#*=}"
        shift
        ;;
    amdgpu-mode=*|rocm-mode=*|rocm-archs=*|pull=*|pullrocmver=*|pullpkg=*|pullpkgextra=*|pullrocmpkgver=*)
        SETUP_ARGS+=("$1")
        shift
        ;;
    # Build-specific arguments
    noextract|norocm|noamdgpu|noextractcontent|contentlist|norunfile|nogui|mscomp=*)
        BUILD_ARGS+=("$1")
        shift
        ;;
    *)
        echo -e "\e[31mERROR: Unknown option: $1\e[0m"
        echo ""
        usage
        exit 1
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

# Display general configuration
echo "Configuration:"
echo "  Script Directory:   $SCRIPT_DIR"
echo "  Build Scripts Dir:  $BUILD_INSTALLER_DIR"
echo "  Skip Setup Phase:   $([ $SKIP_SETUP -eq 1 ] && echo "Yes" || echo "No")"
echo "  Skip Build Phase:   $([ $SKIP_BUILD -eq 1 ] && echo "Yes" || echo "No")"
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

    echo "Setup Configuration:"
    echo "  Setup Arguments: ${SETUP_ARGS[@]:-<none>}"
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

    # Set buildtag to value of pulltag if former isn't set.
    if [ -z "$BUILD_TAG" ] && [ -n "$PULL_TAG" ]; then
        BUILD_TAG="$PULL_TAG"
        BUILD_ARGS+=("buildtag=$BUILD_TAG")
    fi

    # Construct buildpulltag if not already provided
    if [ -z "$BUILD_PULL_TAG" ]; then
        if [ -n "$PULL_TAG" ] && [ -n "$PULL_RUNID" ]; then
            BUILD_PULL_TAG="$PULL_TAG-$PULL_RUNID"
        fi
    fi

    # Add buildpulltag if set (but not if it's empty or just a dash)
    if [ -n "$BUILD_PULL_TAG" ] && [ "$BUILD_PULL_TAG" != "-" ]; then
        BUILD_ARGS+=("buildpulltag=$BUILD_PULL_TAG")
    fi

    echo "Build Configuration:"
    echo "  Build Arguments: ${BUILD_ARGS[@]:-<none>}"
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
echo -e "\e[36mTotal Time: ${SCRIPT_HOURS}h ${SCRIPT_MINUTES}m ${SCRIPT_SECONDS}s (${SCRIPT_ELAPSED} seconds)\e[0m"

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
