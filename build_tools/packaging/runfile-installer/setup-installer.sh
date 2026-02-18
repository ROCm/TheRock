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

# Package Puller Input Config

# ROCm configuration type and version
PULL_CONFIG_ROCM="prereleases"         # nightlies / prereleases / release
PULL_CONFIG_ROCM_BUILDNUM="7.11"       # 20260211-21893116598 / 7.11 (repo url)
PULL_CONFIG_ROCM_VER_PKG="7.11"        # 7.11 (package name)

# ROCm package version control (optional - defaults to latest)
PULL_ROCM_PKG_VERSION=""               # Explicit package version (e.g., 7.11.0-2 for release, 7.11.0~0-21265702726 for prereleases)

# AMDGPU configuration type and version
PULL_CONFIG_AMDGPU="release"            # release / hidden
PULL_CONFIG_AMDGPU_BUILDNUM="31.10"     # 3x.xx.xx / .3x.xx.xx
PULL_CONFIG_AMDGPU_HASH=""              # Hash for hidden repos (only used when PULL_CONFIG_AMDGPU="hidden")

# Package configs (relative to package-puller directory)
# These will be dynamically generated from templates by setup_puller_config()
PULLER_CONFIG_DEB="../build-config/rocm-${PULL_CONFIG_ROCM}-${PULL_CONFIG_ROCM_BUILDNUM}-deb.config"
PULLER_CONFIG_RPM="../build-config/rocm-${PULL_CONFIG_ROCM}-${PULL_CONFIG_ROCM_BUILDNUM}-rpm.config"
PULLER_CONFIG_DIR_AMDGPU="../build-config"

# Package Puller Output directories - separate for DEB and RPM
PULLER_OUTPUT_DIR_DEB="../package-extractor/packages-rocm-deb"
PULLER_OUTPUT_DIR_RPM="../package-extractor/packages-rocm-rpm"
PULLER_OUTPUT_DIR_AMDGPU_BASE="../package-extractor/packages-amdgpu"

# GPU architectures to include in package pulls
# Modify this array to control which GPU architectures are downloaded
ROCM_GFX_ARCHS=(gfx94x gfx950)
# Full list of available architectures:
# ROCM_GFX_ARCHS=(gfx90x gfx94x gfx950 gfx103x gfx110x gfx1150 gfx1151 gfx1152 gfx1153 gfx120x)

# Packages list (will be generated dynamically by generate_package_lists function)
PULLER_PACKAGES_DEB=""
PULLER_PACKAGES_RPM=""
PULLER_PACKAGES_AMDGPU="amdgpu-dkms"

# Setup control flags (default: both rocm/amdgpu enabled)
SETUP_ROCM=0
SETUP_AMDGPU=0
SETUP_AMDGPU_MODE="all"  # Default: all distros
SETUP_ROCM_MODE="chroot" # Default: native (use current OS), Options: native, chroot


###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $PROG [options]

[options]:
    help                  = Display this help information.

    rocm                  = Setup only ROCm packages (skip AMDGPU).
    amdgpu                = Setup only AMDGPU packages (skip ROCm).

    amdgpu-mode=all       = Setup AMDGPU packages for all supported distributions (default).
    amdgpu-mode=single    = Setup AMDGPU packages for current distro only.

    rocm-mode=native      = Pull DEB packages using native OS (default).
    rocm-mode=chroot      = Pull DEB packages using Ubuntu chroot (for use on RPM-based OS).
    rocm-archs=<archs>    = Set GPU architectures to pull (comma-separated or single, e.g., gfx94x,gfx950 or gfx94x). Default: gfx94x,gfx950.

    pull=nightlies        = Pull ROCm packages from nightlies repository.
    pull=prereleases      = Pull ROCm packages from prereleases repository.
    pullbuild=<version>      = Set ROCm build number to pull (e.g., 7.11 for prereleases, 20260123-21274498502 for nightlies).
    pullrocmver=<version>    = Set ROCm version for package names (e.g., 7.12, 7.11). Default: 7.11.
    pullrocmpkgver=<version> = Set explicit package version for both RPM and DEB. Examples:
                               - Release: 7.11.0-2
                               - Prerelease: 7.11.0~0-21265702726 (use ~N- format, auto-converts to ~rcN/~preN)
                               If not specified, pulls latest from configured repo.

Examples:
    ./setup-installer.sh                                      # Setup both ROCm and AMDGPU for all distros (default)
    ./setup-installer.sh rocm                                 # Setup only ROCm packages
    ./setup-installer.sh amdgpu                               # Setup only AMDGPU packages for all distros
    ./setup-installer.sh amdgpu amdgpu-mode=single            # Setup AMDGPU for current distro only
    
    ./setup-installer.sh rocm rocm-mode=chroot                # Pull DEB packages using chroot on AlmaLinux
    ./setup-installer.sh rocm-archs=gfx94x,gfx950,gfx103x     # Pull for specific GPU architectures
    ./setup-installer.sh rocm-archs=gfx94x                    # Pull for single GPU architecture
    
    ./setup-installer.sh pull=nightlies pullbuild=20260123-21274498502  # Pull from nightlies with specific build
    ./setup-installer.sh pull=prereleases pullbuild=7.11      # Pull from prereleases build 7.11
    ./setup-installer.sh pullrocmver=7.11                     # Use ROCm 7.11 package names
    ./setup-installer.sh pullrocmpkgver=7.11.0-2              # Pull specific release version
    ./setup-installer.sh pull=prereleases pullrocmpkgver=7.11.0~0-21265702726  # Pull specific prerelease build

END_USAGE
}

generate_package_lists() {
    echo -------------------------------------------------------------
    echo "Generating ROCm package lists from GPU architectures..."

    # RPM and DEB have different package naming conventions
    # We need to build separate lists for each

    local rpm_packages=""
    local deb_packages=""

    # Determine version patterns for RPM and DEB
    local rpm_version_suffix=""
    local deb_version_suffix=""

    if [[ -n "$PULL_ROCM_PKG_VERSION" ]]; then
        # Explicit package version specified
        # Convert ~N- to ~rcN- for RPM, ~preN- for DEB (prerelease format auto-conversion)
        local rpm_ver="$PULL_ROCM_PKG_VERSION"
        local deb_ver="$PULL_ROCM_PKG_VERSION"

        # Check if version has ~N- pattern (where N is a digit)
        if [[ "$PULL_ROCM_PKG_VERSION" =~ ~([0-9]+)- ]]; then
            local rc_num="${BASH_REMATCH[1]}"
            rpm_ver="${PULL_ROCM_PKG_VERSION//\~${rc_num}-/\~rc${rc_num}-}"
            deb_ver="${PULL_ROCM_PKG_VERSION//\~${rc_num}-/\~pre${rc_num}-}"
        fi

        rpm_version_suffix="-${rpm_ver}"
        deb_version_suffix="=${deb_ver}"

        echo "Using explicit package version:"
        echo "  RPM: $rpm_ver"
        echo "  DEB: =$deb_ver (APT version pinning)"

    else
        # Default: use base version (pulls latest available)
        # Both RPM and DEB use same format: amdrocm-amdsmi{version}-{arch}
        # This matches the original behavior before version control was added
        rpm_version_suffix=""  # Will use embedded version format
        deb_version_suffix=""  # Will use embedded version format
        echo "Using default version pattern (latest): ${PULL_CONFIG_ROCM_VER_PKG}"
        echo "  Format: amdrocm-amdsmi${PULL_CONFIG_ROCM_VER_PKG}-{arch}"
    fi

    # Build RPM package list
    for gfx_arch in "${ROCM_GFX_ARCHS[@]}"; do
        if [[ -z "$rpm_version_suffix" ]]; then
            # Default: version embedded in package name (amdrocm-amdsmi7.11-gfx950)
            rpm_packages="$rpm_packages amdrocm-amdsmi${PULL_CONFIG_ROCM_VER_PKG}-${gfx_arch}"
        else
            # Explicit version: embedded version format (amdrocm-amdsmi7.11-gfx950-7.11.0-2)
            rpm_packages="$rpm_packages amdrocm-amdsmi${PULL_CONFIG_ROCM_VER_PKG}-${gfx_arch}${rpm_version_suffix}"
        fi
    done

    # Build DEB package list
    for gfx_arch in "${ROCM_GFX_ARCHS[@]}"; do
        if [[ -z "$deb_version_suffix" ]]; then
            # Default: version embedded in package name (amdrocm-amdsmi7.11-gfx950)
            deb_packages="$deb_packages amdrocm-amdsmi${PULL_CONFIG_ROCM_VER_PKG}-${gfx_arch}"
        else
            # Explicit version: embedded version + APT pinning (amdrocm-amdsmi7.11-gfx950=7.11.0-2)
            deb_packages="$deb_packages amdrocm-amdsmi${PULL_CONFIG_ROCM_VER_PKG}-${gfx_arch}${deb_version_suffix}"
        fi
    done

    # Trim leading spaces
    rpm_packages="${rpm_packages# }"
    deb_packages="${deb_packages# }"

    # Set package lists if not already set via environment variables
    PULLER_PACKAGES_RPM="${PULLER_PACKAGES_RPM:-$rpm_packages}"
    PULLER_PACKAGES_DEB="${PULLER_PACKAGES_DEB:-$deb_packages}"

    echo "GPU Architectures: ${ROCM_GFX_ARCHS[*]}"
    echo "RPM Packages: $PULLER_PACKAGES_RPM"
    echo "DEB Packages: $PULLER_PACKAGES_DEB"
    echo "Generating ROCm package lists...Complete"
}

setup_puller_config() {
    echo -------------------------------------------------------------
    echo "Setting up package puller configuration files..."

    # Ensure build-config directory exists
    BUILD_CONFIG_DIR="build-config"
    mkdir -p "$BUILD_CONFIG_DIR"

    # Template directory for ROCm configs
    TEMPLATE_DIR="package-puller/config/therock/rocm/${PULL_CONFIG_ROCM}"

    # Template files
    TEMPLATE_DEB="${TEMPLATE_DIR}/rocm-${PULL_CONFIG_ROCM}-deb.config"
    TEMPLATE_RPM="${TEMPLATE_DIR}/rocm-${PULL_CONFIG_ROCM}-rpm.config"

    # Output files (in build-config directory)
    OUTPUT_DEB="${BUILD_CONFIG_DIR}/rocm-${PULL_CONFIG_ROCM}-${PULL_CONFIG_ROCM_BUILDNUM}-deb.config"
    OUTPUT_RPM="${BUILD_CONFIG_DIR}/rocm-${PULL_CONFIG_ROCM}-${PULL_CONFIG_ROCM_BUILDNUM}-rpm.config"

    # Check if templates exist
    if [ ! -f "$TEMPLATE_DEB" ]; then
        echo -e "\e[31mERROR: Template file not found: $TEMPLATE_DEB\e[0m"
        exit 1
    fi
    if [ ! -f "$TEMPLATE_RPM" ]; then
        echo -e "\e[31mERROR: Template file not found: $TEMPLATE_RPM\e[0m"
        exit 1
    fi

    echo "Using ROCm config type: ${PULL_CONFIG_ROCM}"
    echo "Using ROCm version: ${PULL_CONFIG_ROCM_BUILDNUM}"

    # Generate DEB config from template
    echo "Generating DEB config: $OUTPUT_DEB"
    sed "s/{{ROCM_VERSION}}/${PULL_CONFIG_ROCM_BUILDNUM}/g" "$TEMPLATE_DEB" > "$OUTPUT_DEB"

    # Generate RPM config from template
    echo "Generating RPM config: $OUTPUT_RPM"
    sed "s/{{ROCM_VERSION}}/${PULL_CONFIG_ROCM_BUILDNUM}/g" "$TEMPLATE_RPM" > "$OUTPUT_RPM"

    echo -e "\e[32mROCm package puller configuration files generated successfully.\e[0m"
    echo "Setting up package puller configuration files...Complete"
}

setup_puller_config_amdgpu() {
    echo -------------------------------------------------------------
    echo "Setting up AMDGPU package puller configuration files..."

    # Ensure build-config directory exists
    BUILD_CONFIG_DIR="build-config"
    mkdir -p "$BUILD_CONFIG_DIR"

    # Template directory for AMDGPU configs
    TEMPLATE_DIR="package-puller/config/therock/amdgpu/${PULL_CONFIG_AMDGPU}"

    echo "Using AMDGPU config type: ${PULL_CONFIG_AMDGPU}"
    echo "Using AMDGPU version: ${PULL_CONFIG_AMDGPU_BUILDNUM}"

    # If using hidden config type, validate hash is provided
    if [[ "${PULL_CONFIG_AMDGPU}" == "hidden" ]]; then
        if [[ -z "${PULL_CONFIG_AMDGPU_HASH}" ]]; then
            echo -e "\e[31mERROR: PULL_CONFIG_AMDGPU_HASH must be set when using hidden config type\e[0m"
            exit 1
        fi
        echo "Using AMDGPU hash: ${PULL_CONFIG_AMDGPU_HASH}"
    fi

    # Determine EL9 version format based on AMDGPU major version
    # Extract major version (first number before first dot)
    AMDGPU_MAJOR_VER="${PULL_CONFIG_AMDGPU_BUILDNUM%%.*}"

    # Legacy format (30.x): el/9.6/
    # New format (31.x+): el/9/
    if [[ "$AMDGPU_MAJOR_VER" -le 30 ]]; then
        EL9_VERSION="9.6"
        echo "Using legacy EL9 path format: el/9.6/ (AMDGPU ${AMDGPU_MAJOR_VER}.x)"
    else
        EL9_VERSION="9"
        echo "Using new EL9 path format: el/9/ (AMDGPU ${AMDGPU_MAJOR_VER}.x)"
    fi

    # List of all supported distro tags
    DISTRO_TAGS=("ub24" "ub22" "el10" "el9" "el8" "sle16" "sle15" "amzn23")

    # Generate config file for each distro
    for distro_tag in "${DISTRO_TAGS[@]}"; do
        TEMPLATE_FILE="${TEMPLATE_DIR}/amdgpu-${PULL_CONFIG_AMDGPU}-${distro_tag}.config"
        OUTPUT_FILE="${BUILD_CONFIG_DIR}/amdgpu-${PULL_CONFIG_AMDGPU}-${PULL_CONFIG_AMDGPU_BUILDNUM}-${distro_tag}.config"

        # Check if template exists
        if [ ! -f "$TEMPLATE_FILE" ]; then
            echo -e "\e[31mERROR: Template file not found: $TEMPLATE_FILE\e[0m"
            exit 1
        fi

        # Generate config from template
        echo "Generating AMDGPU config for ${distro_tag}: $OUTPUT_FILE"
        sed -e "s/{{AMDGPU_VERSION}}/${PULL_CONFIG_AMDGPU_BUILDNUM}/g" \
            -e "s/{{EL9_VERSION}}/${EL9_VERSION}/g" \
            -e "s/{{AMDGPU_HASH}}/${PULL_CONFIG_AMDGPU_HASH}/g" \
            "$TEMPLATE_FILE" > "$OUTPUT_FILE"
    done

    echo -e "\e[32mAMDGPU package puller configuration files generated successfully.\e[0m"
    echo "Setting up AMDGPU package puller configuration files...Complete"
}

os_release() {
    if [[ -r  /etc/os-release ]]; then
        . /etc/os-release

        DISTRO_NAME=$ID
        DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
        DISTRO_MAJOR_VER=${DISTRO_VER%.*}

        case "$ID" in
        ubuntu)
            PULL_DISTRO_TYPE=deb
            PULL_DISTRO_PACKAGE_TYPE=deb
            if [[ $DISTRO_VER == 24.04 ]]; then
                DISTRO_TAG="ub24"
            elif [[ $DISTRO_VER == 22.04 ]]; then
                DISTRO_TAG="ub22"
            else
                echo "ERROR: Unsupported Ubuntu version: $DISTRO_VER"
                exit 1
            fi
            ;;
        debian)
            PULL_DISTRO_TYPE=deb
            PULL_DISTRO_PACKAGE_TYPE=deb
            if [[ $DISTRO_MAJOR_VER == 13 ]]; then
                DISTRO_TAG="ub24"  # Debian 13 uses same config as Ubuntu 24.04
            elif [[ $DISTRO_MAJOR_VER == 12 ]]; then
                DISTRO_TAG="ub22"  # Debian 12 uses same config as Ubuntu 22.04
            else
                echo "ERROR: Unsupported Debian version: $DISTRO_VER"
                exit 1
            fi
            ;;
        almalinux|rhel|ol|rocky)
            PULL_DISTRO_TYPE=el
            PULL_DISTRO_PACKAGE_TYPE=rpm
            if [[ "$DISTRO_MAJOR_VER" == "10" ]]; then
                DISTRO_TAG="el10"
            elif [[ "$DISTRO_MAJOR_VER" == "9" ]]; then
                DISTRO_TAG="el9"
            elif [[ "$DISTRO_MAJOR_VER" == "8" ]]; then
                DISTRO_TAG="el8"
                if [[ "$ID" == "almalinux" ]]; then
                    echo "Detected AlmaLinux $DISTRO_VER (ManyLinux)"
                fi
            else
                echo "ERROR: Unsupported EL version: $DISTRO_VER"
                exit 1
            fi
            ;;
        sles)
            PULL_DISTRO_TYPE=sle
            PULL_DISTRO_PACKAGE_TYPE=rpm
            DISTRO_TAG="sle15"
            ;;
        amzn)
            PULL_DISTRO_TYPE=el
            PULL_DISTRO_PACKAGE_TYPE=rpm
            DISTRO_TAG="amzn23"
            ;;
        *)
            echo "ERROR: $ID is not a supported OS"
            echo "Supported OSes: Ubuntu, Debian, AlmaLinux, RHEL, Oracle Linux, Rocky Linux, SLES, Amazon Linux"
            exit 1
            ;;
        esac
    else
        echo "ERROR: /etc/os-release not found. Unsupported OS."
        exit 1
    fi

    echo "Setup running on $DISTRO_NAME $DISTRO_VER."
}

install_tools() {
    echo -------------------------------------------------------------
    echo "Installing required tools for $DISTRO_NAME $DISTRO_VER..."

    if [ $PULL_DISTRO_TYPE == "el" ]; then
        echo "Installing tools for EL-based system..."

        # Install sudo package
        dnf install -y sudo wget
        if [[ $? -ne 0 ]]; then
            echo -e "\e[31mERROR: Failed to install sudo package.\e[0m"
            exit 1
        fi

        echo "sudo package installed successfully."
    else
        echo "Skipping tool installation (only EL-based systems supported)."
    fi

    echo "Installing required tools...Complete"
}

configure_setup_rocm() {
    echo ++++++++++++++++++++++++++++++++

    if [ $PULL_DISTRO_PACKAGE_TYPE == "deb" ]; then
        echo "Configuring for DEB packages."

        PULLER_CONFIG="${PULLER_CONFIG:-$PULLER_CONFIG_DEB}"
        if [[ -n $PULLER_CONFIG_DEB ]]; then
            PULLER_CONFIG=$PULLER_CONFIG_DEB
        fi

        echo "Using configuration: $PULLER_CONFIG"

    elif [ $PULL_DISTRO_PACKAGE_TYPE == "rpm" ]; then
        echo "Configuring for RPM packages."

        PULLER_CONFIG="${PULLER_CONFIG:-$PULLER_CONFIG_RPM}"
        if [[ -n $PULLER_CONFIG_RPM ]]; then
            PULLER_CONFIG=$PULLER_CONFIG_RPM
        fi

        echo "Using configuration: $PULLER_CONFIG"
        
    else
        echo "Invalid Distro Package Type: $PULL_DISTRO_PACKAGE_TYPE"
        exit 1
    fi
}

configure_setup_amdgpu() {
    echo ++++++++++++++++++++++++++++++++
    echo "Configuring AMDGPU for $DISTRO_NAME $DISTRO_VER (tag: $DISTRO_TAG)."

    # Build config file name: amdgpu-<type>-<version>-<distro>.config
    PULLER_CONFIG="${PULLER_CONFIG_DIR_AMDGPU}/amdgpu-${PULL_CONFIG_AMDGPU}-${PULL_CONFIG_AMDGPU_BUILDNUM}-${DISTRO_TAG}.config"

    echo "Using AMDGPU configuration: $PULLER_CONFIG"
}

setup_rocm_packages() {
    # Move all ROCm packages to single directory
    # Parameters:
    #   $1 - package type ("deb" or "rpm")
    #   $2 - base output directory (e.g., $PULLER_OUTPUT_DIR_DEB or $PULLER_OUTPUT_DIR_RPM)

    local pkg_type="$1"
    local output_base="$2"

    echo "Moving all ROCm packages to single directory..."

    if [ -d "$output_base" ]; then
        echo -e "\e[93mExtraction directory exists. Removing: $output_base\e[0m"
        $SUDO rm -rf "$output_base"
    fi

    mv packages/packages-amd "$output_base"
    echo -e "\e[32m${pkg_type^^} packages pulled to: $output_base\e[0m"
}

setup_rocm_deb() {
    # Pull ROCm DEB packages
    pushd package-puller
        echo -------------------------------------------------------------
        echo "Setting up for ROCm components..."
        echo "========================================="
        echo "Pulling DEB packages..."
        echo "========================================="

        ./package-puller-deb.sh amd config="$PULLER_CONFIG_DEB" pkg="$PULLER_PACKAGES_DEB"

        # check if package pull was successful
        if [[ $? -ne 0 ]]; then
            echo -e "\e[31mFailed pull of ROCm DEB packages.\e[0m"
            exit 1
        fi

        # Move packages to output directory
        setup_rocm_packages "deb" "$PULLER_OUTPUT_DIR_DEB"

        echo ""
        echo "Setting up for ROCm components...Complete."
    popd
}

setup_rocm_deb_chroot() {
    # Pull ROCm DEB packages using chroot method (for RPM-based host OS)
    pushd package-puller
        echo -------------------------------------------------------------
        echo "Setting up for ROCm components (chroot mode)..."
        echo "========================================="
        echo "Pulling DEB packages using Ubuntu chroot..."
        echo "========================================="

        ./package-puller-deb-chroot.sh amd config="$PULLER_CONFIG_DEB" pkg="$PULLER_PACKAGES_DEB"

        # check if package pull was successful
        if [[ $? -ne 0 ]]; then
            echo -e "\e[31mFailed pull of ROCm DEB packages (chroot).\e[0m"
            exit 1
        fi

        # Move packages to output directory
        setup_rocm_packages "deb" "$PULLER_OUTPUT_DIR_DEB"

        echo ""
        echo "Setting up for ROCm components (chroot)...Complete."
    popd
}

setup_rocm_rpm() {
    # Pull ROCm RPM packages
    pushd package-puller
        echo -------------------------------------------------------------
        echo "Setting up for ROCm components..."
        echo "========================================="
        echo "Pulling RPM packages..."
        echo "========================================="

        ./package-puller-el.sh amd config="$PULLER_CONFIG_RPM" pkg="$PULLER_PACKAGES_RPM"

        # check if package pull was successful
        if [[ $? -ne 0 ]]; then
            echo -e "\e[31mFailed pull of ROCm RPM packages.\e[0m"
            exit 1
        fi

        # Move packages to output directory
        setup_rocm_packages "rpm" "$PULLER_OUTPUT_DIR_RPM"

        echo ""
        echo "Setting up for ROCm components...Complete."
    popd
}

setup_rocm() {
    configure_setup_rocm

    if [ "$SETUP_ROCM_MODE" == "chroot" ]; then
        # Chroot mode: pull DEB packages via chroot
        if [ $PULL_DISTRO_PACKAGE_TYPE == "rpm" ]; then
            # On RPM-based system with chroot mode: pull both RPM and DEB
            echo "Chroot mode enabled on RPM-based system: Pulling both RPM and DEB packages"
            echo "  - Pulling RPM packages for current system"
            setup_rocm_rpm
            echo "  - Pulling DEB packages via chroot"
            setup_rocm_deb_chroot
        elif [ $PULL_DISTRO_PACKAGE_TYPE == "deb" ]; then
            # On DEB-based system with chroot mode: pull DEB via chroot
            echo "Chroot mode enabled on DEB-based system: Pulling DEB packages via chroot"
            setup_rocm_deb_chroot
        else
            echo "Invalid Distro Package Type: $PULL_DISTRO_PACKAGE_TYPE"
            exit 1
        fi
    else
        # Native mode: pull packages only for current distro type
        if [ $PULL_DISTRO_PACKAGE_TYPE == "deb" ]; then
            echo "Native mode: Pulling DEB packages"
            setup_rocm_deb
        elif [ $PULL_DISTRO_PACKAGE_TYPE == "rpm" ]; then
            echo "Native mode: Pulling RPM packages"
            setup_rocm_rpm
        else
            echo "Invalid Distro Package Type: $PULL_DISTRO_PACKAGE_TYPE"
            exit 1
        fi
    fi
}

setup_amdgpu() {
    configure_setup_amdgpu

    # Pull AMDGPU packages
    pushd package-puller
        echo -------------------------------------------------------------
        echo "Setting up for AMDGPU components..."
        echo "========================================="
        echo "Pulling AMDGPU packages for $DISTRO_NAME $DISTRO_VER..."
        echo "========================================="

        # Call the appropriate package puller based on distro type
        if [ $PULL_DISTRO_TYPE == "deb" ]; then
            ./package-puller-deb.sh amd config="$PULLER_CONFIG" pkg="$PULLER_PACKAGES_AMDGPU"
        elif [ $PULL_DISTRO_TYPE == "el" ]; then
            ./package-puller-el.sh amd config="$PULLER_CONFIG" pkg="$PULLER_PACKAGES_AMDGPU"
        elif [ $PULL_DISTRO_TYPE == "sle" ]; then
            ./package-puller-sle.sh amd config="$PULLER_CONFIG" pkg="$PULLER_PACKAGES_AMDGPU"
        else
            echo -e "\e[31mUnsupported distro type: $PULL_DISTRO_TYPE\e[0m"
            exit 1
        fi

        # check if package pull was successful
        if [[ $? -ne 0 ]]; then
            echo -e "\e[31mFailed pull of AMDGPU packages.\e[0m"
            exit 1
        fi

        # Build output directory with distro tag subdirectory
        PULLER_OUTPUT="${PULLER_OUTPUT_DIR_AMDGPU_BASE}/${DISTRO_TAG}"

        # Create base directory if it doesn't exist
        if [ ! -d "$PULLER_OUTPUT_DIR_AMDGPU_BASE" ]; then
            mkdir -p "$PULLER_OUTPUT_DIR_AMDGPU_BASE"
        fi

        if [ -d $PULLER_OUTPUT ]; then
            echo -e "\e[93mExtraction directory exists. Removing: $PULLER_OUTPUT\e[0m"
            $SUDO rm -rf $PULLER_OUTPUT
        fi
        mv packages/packages-amd $PULLER_OUTPUT
        echo -e "\e[32mAMDGPU packages pulled to: $PULLER_OUTPUT\e[0m"

        echo ""
        echo "Setting up for AMDGPU components...Complete."
    popd
}

setup_amdgpu_all() {
    # Pull AMDGPU packages for all distributions
    pushd package-puller
        echo -------------------------------------------------------------
        echo "Setting up for AMDGPU components (all distributions)..."
        echo "========================================="
        echo "Pulling AMDGPU packages for all supported distributions..."
        echo "========================================="

        # Call the multi-distro package puller with config variables
        AMDGPU_CONFIG_TYPE="$PULL_CONFIG_AMDGPU" AMDGPU_CONFIG_VER="$PULL_CONFIG_AMDGPU_BUILDNUM" ./package-puller-amdgpu-all.sh

        # Check if package pull had critical failures
        PULL_RESULT=$?
        if [[ $PULL_RESULT -ne 0 ]]; then
            echo -e "\e[93mWARNING: Some AMDGPU package pulls failed. Check the output above for details.\e[0m"
            echo -e "\e[93mContinuing with available packages...\e[0m"
        fi

        echo ""
        echo "Setting up for AMDGPU components (all distributions)...Complete."
    popd
}

####### Main script ###############################################################

echo ============================
echo ROCM RUNFILE INSTALLER SETUP
echo ============================

SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)
echo SUDO: $SUDO

os_release

# parse args
while (($#))
do
    case "$1" in
    help)
        usage
        exit 0
        ;;
    rocm)
        echo "Enabling ROCm setup only."
        SETUP_ROCM=1
        shift
        ;;
    amdgpu)
        echo "Enabling AMDGPU setup only."
        SETUP_AMDGPU=1
        shift
        ;;
    amdgpu-mode=*)
        SETUP_AMDGPU_MODE="${1#*=}"
        if [[ "$SETUP_AMDGPU_MODE" != "all" && "$SETUP_AMDGPU_MODE" != "single" ]]; then
            echo "ERROR: Invalid amdgpu-mode: $SETUP_AMDGPU_MODE"
            echo "Valid options: amdgpu-mode=all or amdgpu-mode=single"
            exit 1
        fi
        echo "AMDGPU mode set to: $SETUP_AMDGPU_MODE"
        shift
        ;;
    rocm-mode=*)
        SETUP_ROCM_MODE="${1#*=}"
        if [[ "$SETUP_ROCM_MODE" != "native" && "$SETUP_ROCM_MODE" != "chroot" ]]; then
            echo "ERROR: Invalid rocm-mode: $SETUP_ROCM_MODE"
            echo "Valid options: rocm-mode=native or rocm-mode=chroot"
            exit 1
        fi
        echo "ROCm mode set to: $SETUP_ROCM_MODE"
        shift
        ;;
    pull=*)
        PULL_CONFIG_ROCM="${1#*=}"
        if [[ "$PULL_CONFIG_ROCM" != "nightlies" && "$PULL_CONFIG_ROCM" != "prereleases" ]]; then
            echo "ERROR: Invalid pull type: $PULL_CONFIG_ROCM"
            echo "Valid options: pull=nightlies or pull=prereleases"
            exit 1
        fi
        echo "ROCm pull config type set to: $PULL_CONFIG_ROCM"
        shift
        ;;
    pullbuild=*)
        PULL_CONFIG_ROCM_BUILDNUM="${1#*=}"
        echo "ROCm pull config build number set to: $PULL_CONFIG_ROCM_BUILDNUM"
        shift
        ;;
    pullrocmver=*)
        PULL_CONFIG_ROCM_VER_PKG="${1#*=}"
        echo "ROCm version set to: $PULL_CONFIG_ROCM_VER_PKG"
        shift
        ;;
    rocm-archs=*)
        ARCHS_INPUT="${1#*=}"
        # Convert comma-separated string to array
        IFS=',' read -ra ROCM_GFX_ARCHS <<< "$ARCHS_INPUT"
        echo "GPU architectures set to: ${ROCM_GFX_ARCHS[*]}"
        shift
        ;;
    pullrocmpkgver=*)
        PULL_ROCM_PKG_VERSION="${1#*=}"
        echo "ROCm package version set to: $PULL_ROCM_PKG_VERSION"
        shift
        ;;
    *)
        echo "Unknown option: $1"
        shift
        ;;
    esac
done

# If neither rocm nor amdgpu specified, enable both (default behavior)
if [[ $SETUP_ROCM == 0 && $SETUP_AMDGPU == 0 ]]; then
    echo "No specific setup specified, enabling both ROCm and AMDGPU (default)."
    SETUP_ROCM=1
    SETUP_AMDGPU=1
fi

# Recreate build-config directory (clean slate for each run)
BUILD_CONFIG_DIR="build-config"
if [ -d "$BUILD_CONFIG_DIR" ]; then
    echo "Removing existing $BUILD_CONFIG_DIR directory"
    rm -rf "$BUILD_CONFIG_DIR"
fi
mkdir -p "$BUILD_CONFIG_DIR"
echo "Created $BUILD_CONFIG_DIR directory"

# Generate ROCm package lists from GPU architecture array
generate_package_lists

# Generate package puller configuration files from templates
setup_puller_config
setup_puller_config_amdgpu

# Install required tools
install_tools

echo Running Package Puller...

if [[ $SETUP_ROCM == 1 ]]; then
    setup_rocm
fi

if [[ $SETUP_AMDGPU == 1 ]]; then
    if [[ "$SETUP_AMDGPU_MODE" == "all" ]]; then
        setup_amdgpu_all
    elif [[ "$SETUP_AMDGPU_MODE" == "single" ]]; then
        setup_amdgpu
    else
        echo "ERROR: Invalid SETUP_AMDGPU_MODE: $SETUP_AMDGPU_MODE"
        exit 1
    fi
fi

echo Running Package Puller...Complete

