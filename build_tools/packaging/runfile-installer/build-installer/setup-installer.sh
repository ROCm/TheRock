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
PULL_CONFIG_RELEASE_TYPE=""            # dev / nightly / prerelease / release
PULL_CONFIG_TAG=""                     # Build tag (e.g., rc0, 20260211)
PULL_CONFIG_RUNID=""                   # Run ID (21893116598 for nightly)
PULL_CONFIG_ROCM_VER=""                # 7.11.0
PULL_CONFIG_PKG="amdrocm-core-sdk"     # Base package name (e.g., amdrocm-core-sdk, amdrocm-dev-tools)
PULL_CONFIG_PKG_TYPE="arch"            # Package type: "arch" (has -gfxXYZ suffix) or "base" (no suffix)
PULL_CONFIG_PKG_EXTRA=()               # Additional packages (array, comma-separated via pullpkgextra=)

# ROCm package version control (optional - defaults to latest)
PULL_ROCM_PKG_VERSION=""               # Explicit package version (e.g., 7.11.0-2 for release, 7.11.0~0-21265702726 for prerelease)

# AMDGPU configuration type and version
PULL_CONFIG_AMDGPU="release"            # release / hidden
PULL_CONFIG_AMDGPU_BUILDNUM="31.10"     # 3x.xx.xx / .3x.xx.xx
PULL_CONFIG_AMDGPU_HASH=""              # Hash for hidden repos (only used when PULL_CONFIG_AMDGPU="hidden")

# Package configs (relative to package-puller directory)
# These will be dynamically generated from templates by setup_puller_config_rocm()
PULLER_CONFIG_DEB=""
PULLER_CONFIG_RPM=""
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

# Configuration
ROCM_RELEASE_TYPES=(dev nightly prerelease release)


###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $PROG [options]

[options]:
    help                  = Display this help information.

    config=<file>         = Load configuration from file (command-line args override config).
                            Preset configs available in config/ directory:
                            - config/nightly.config
                            - config/prerelease.config
                            - config/release.config
                            - config/dev.config

    rocm                  = Setup only ROCm packages (skip AMDGPU).
    amdgpu                = Setup only AMDGPU packages (skip ROCm).

    amdgpu-mode=all       = Setup AMDGPU packages for all supported distributions (default).
    amdgpu-mode=single    = Setup AMDGPU packages for current distro only.

    rocm-mode=native      = Pull DEB packages using native OS.
    rocm-mode=chroot      = Pull DEB packages using Ubuntu chroot.
    rocm-archs=<archs>    = Set GPU architectures to pull (comma-separated or single, e.g., gfx94x,gfx950 or gfx94x). Default: gfx94x,gfx950.

    pull=<release-type>   = Pull ROCm packages from specified repository (dev, nightly, prerelease, release).
    pulltag=<tag>            = Set ROCm build tag (e.g., 20260123 for nightly).
    pullrunid=<runid>        = Set ROCm run ID (e.g., 21274498502 for nightly).
    pullrocmver=<version>    = Set ROCm version for package names (e.g., 7.12.0, 7.11.0).
    pullpkg=<package>        = Set base package name with optional type prefix (default: amdrocm-core-sdk).
                               Syntax: pullpkg=[type:]<package>
                               - arch:<package> = Architecture-specific (has -gfxXYZ suffix, default)
                                 Example: pullpkg=arch:amdrocm-core-sdk or pullpkg=amdrocm-core-sdk
                               - base:<package> = Base package (no -gfxXYZ suffix)
                                 Example: pullpkg=base:amdrocm-amdsmi
                               Other options: amdrocm-dev-tools, amdrocm-core, etc.
    pullpkgextra=<packages>  = Add extra packages to pull (comma-separated) with optional type prefix.
                               Syntax: pullpkgextra=[type:]pkg1,[type:]pkg2
                               - arch:<package> = Architecture-specific (has -gfxXYZ suffix, default)
                               - base:<package> = Base package (no -gfxXYZ suffix)
                               Example: pullpkgextra=arch:amdrocm-opencl,base:amdrocm-llvm
                               Or without prefix: pullpkgextra=rocm-llvm,rocm-device-libs (defaults to arch)
    pullrocmpkgver=<version> = DISABLED - Support for version package pull - disabled.

Examples:
    # Basic usage
    ./setup-installer.sh                                      # Setup both ROCm and AMDGPU for all distros (default)
    ./setup-installer.sh rocm                                 # Setup only ROCm packages
    ./setup-installer.sh amdgpu                               # Setup only AMDGPU packages for all distros
    ./setup-installer.sh amdgpu amdgpu-mode=single            # Setup AMDGPU for current distro only

    # GPU architectures
    ./setup-installer.sh rocm-archs=gfx94x,gfx950             # Pull for specific GPU architectures
    ./setup-installer.sh rocm-archs=gfx110x                   # Pull for single GPU architecture

    # Using preset configs
    ./setup-installer.sh config=config/nightly.config         # Use nightly preset
    ./setup-installer.sh config=config/dev.config             # Use dev preset
    ./setup-installer.sh config=config/prerelease.config      # Use prerelease preset
    ./setup-installer.sh config=config/release.config         # Use release preset

    # Pull from specific builds (with actual values from preset configs)
    ./setup-installer.sh pull=nightly pulltag=20260212 pullrunid=21933875966 pullrocmver=7.12.0    # Nightly build
    ./setup-installer.sh pull=dev pulltag=20260219 pullrunid=22188089855 pullrocmver=7.12.0        # Dev build
    ./setup-installer.sh pull=prerelease pulltag=rc0 pullrocmver=7.11.0                            # Prerelease RC0
    ./setup-installer.sh pull=release pullrocmver=7.11.0                                            # Release build

    # Custom packages
    ./setup-installer.sh pullpkg=arch:amdrocm-core                                                  # Arch-specific package
    ./setup-installer.sh pullpkg=base:amdrocm-amdsmi                                                # Base package
    ./setup-installer.sh pullpkgextra=arch:amdrocm-opencl,base:amdrocm-llvm                         # Extra packages

END_USAGE
}

os_release() {
    if [[ -r  /etc/os-release ]]; then
        . /etc/os-release

        DISTRO_NAME=$ID
        DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
        DISTRO_MAJOR_VER=${DISTRO_VER%.*}

        case "$ID" in
        almalinux)
            PULL_DISTRO_TYPE=el
            PULL_DISTRO_PACKAGE_TYPE=rpm
            if [[ "$DISTRO_MAJOR_VER" == "10" ]]; then
                DISTRO_TAG="el10"
            elif [[ "$DISTRO_MAJOR_VER" == "9" ]]; then
                DISTRO_TAG="el9"
            elif [[ "$DISTRO_MAJOR_VER" == "8" ]]; then
                DISTRO_TAG="el8"
                echo "Detected AlmaLinux $DISTRO_VER (ManyLinux)"
            else
                echo "ERROR: Unsupported AlmaLinux version: $DISTRO_VER"
                exit 1
            fi
            ;;
        *)
            echo "ERROR: $ID is not a supported OS"
            echo "Supported OS: AlmaLinux"
            exit 1
            ;;
        esac
    else
        echo "ERROR: /etc/os-release not found. Unsupported OS."
        exit 1
    fi

    echo "Setup running on $DISTRO_NAME $DISTRO_VER."
}

read_config() {
    # Check for config= argument and source it BEFORE parsing other args
    # This allows command-line args to override config file values
    local CONFIG_FILE=""

    for arg in "$@"; do
        case "$arg" in
            config=*)
                CONFIG_FILE="${arg#*=}"
                break
                ;;
        esac
    done

    if [[ -n "$CONFIG_FILE" ]]; then
        echo -------------------------------------------------------------
        echo "Loading configuration from: $CONFIG_FILE"

        # Check if file exists
        if [[ ! -f "$CONFIG_FILE" ]]; then
            echo -e "\e[31mERROR: Config file not found: $CONFIG_FILE\e[0m"
            exit 1
        fi

        # Check if file is readable
        if [[ ! -r "$CONFIG_FILE" ]]; then
            echo -e "\e[31mERROR: Config file not readable: $CONFIG_FILE\e[0m"
            exit 1
        fi

        # Source the config file
        source "$CONFIG_FILE"
        echo "Configuration loaded successfully."
        echo "Note: Command-line arguments will override config values."
        echo -------------------------------------------------------------
    fi
}

validate_args() {
    echo -------------------------------------------------------------
    echo "Validating configuration arguments..."

    local validation_failed=0

    # Validate PULL_CONFIG_RELEASE_TYPE is set and valid
    if [[ -z "$PULL_CONFIG_RELEASE_TYPE" ]]; then
        echo -e "\e[31mERROR: PULL_CONFIG_RELEASE_TYPE not set. Use pull= argument.\e[0m"
        echo "Valid values: ${ROCM_RELEASE_TYPES[*]}"
        validation_failed=1
    elif [[ ! " ${ROCM_RELEASE_TYPES[@]} " =~ " ${PULL_CONFIG_RELEASE_TYPE} " ]]; then
        echo -e "\e[31mERROR: Invalid pull= value: $PULL_CONFIG_RELEASE_TYPE\e[0m"
        echo "Valid values: ${ROCM_RELEASE_TYPES[*]}"
        validation_failed=1
    fi

    # Validate PULL_CONFIG_TAG based on release type
    case "$PULL_CONFIG_RELEASE_TYPE" in
        dev|nightly)
            if [[ -z "$PULL_CONFIG_TAG" ]]; then
                echo -e "\e[31mERROR: pulltag= required for $PULL_CONFIG_RELEASE_TYPE builds (8 characters)\e[0m"
                echo "Example: pulltag=20260123"
                validation_failed=1
            elif [[ ${#PULL_CONFIG_TAG} -gt 8 ]]; then
                echo "WARNING: Tag longer than 8 characters, truncating to: ${PULL_CONFIG_TAG:0:8}"
                PULL_CONFIG_TAG="${PULL_CONFIG_TAG:0:8}"
            elif [[ ${#PULL_CONFIG_TAG} -lt 8 ]]; then
                echo -e "\e[31mERROR: Tag must be 8 characters, got: $PULL_CONFIG_TAG (${#PULL_CONFIG_TAG} characters)\e[0m"
                validation_failed=1
            fi
            ;;
        prerelease|release)
            [[ -z "$PULL_CONFIG_TAG" ]] && echo "Note: pulltag not set (optional for $PULL_CONFIG_RELEASE_TYPE)"
            ;;
    esac

    # Validate PULL_CONFIG_RUNID based on release type
    # Only required for dev and nightly builds
    case "$PULL_CONFIG_RELEASE_TYPE" in
        dev|nightly)
            if [[ -z "$PULL_CONFIG_RUNID" ]]; then
                echo -e "\e[31mERROR: pullrunid= required for $PULL_CONFIG_RELEASE_TYPE builds\e[0m"
                echo "Example: pullrunid=21893116598"
                validation_failed=1
            fi
            ;;
    esac

    # Validate PULL_CONFIG_ROCM_VER is set
    if [[ -z "$PULL_CONFIG_ROCM_VER" ]]; then
        echo -e "\e[31mERROR: PULL_CONFIG_ROCM_VER not set. Use pullrocmver= argument.\e[0m"
        echo "Example: pullrocmver=7.11.0"
        validation_failed=1
    fi

    if [ $validation_failed -eq 1 ]; then
        echo -e "\e[31mValidation failed. Exiting.\e[0m"
        exit 1
    fi

    echo "Configuration arguments validated successfully."
}

write_version() {
    echo -------------------------------------------------------------
    echo Writing version...

    i=0
    VERSION_FILE="VERSION"

    while IFS= read -r line; do
        case $i in
            0) INSTALLER_VERSION="$line" ;;
        esac

        i=$((i+1))
    done < "$VERSION_FILE"

    if [[ -n $PULL_CONFIG_ROCM_VER ]]; then
        echo "INSTALLER_VERSION = $INSTALLER_VERSION"
        echo "ROCM_VER          = $PULL_CONFIG_ROCM_VER"

        # Update the version file
        echo "$INSTALLER_VERSION" > "$VERSION_FILE"
        echo "$PULL_CONFIG_ROCM_VER" >> "$VERSION_FILE"
    fi
}

install_tools() {
    echo -------------------------------------------------------------
    echo "Installing required tools for $DISTRO_NAME $DISTRO_VER..."

    if [ $PULL_DISTRO_TYPE == "el" ]; then
        echo "Installing tools for EL-based system..."

        # Check if all required tools are already installed
        if command -v sudo &> /dev/null && command -v wget &> /dev/null; then
            echo "All required tools are already installed (sudo, wget)"
        else
            # One or more tools missing, install all
            echo "Installing required tools: sudo wget"
            dnf install -y sudo wget
            if [[ $? -ne 0 ]]; then
                echo -e "\e[31mERROR: Failed to install tools.\e[0m"
                exit 1
            fi
            echo "Tools installed successfully"
        fi
    else
        echo "Skipping tool installation (only EL-based systems supported)."
    fi

    echo "Installing required tools...Complete"
}

normalize_package_name() {
    # Normalize package names for RPM vs DEB conventions
    # RPM uses "-devel" suffix, DEB uses "-dev" suffix
    # Args: $1 = package name
    # Returns: "rpm_name|deb_name" via echo

    local pkg_name="$1"
    local pull_pkg_rpm="$pkg_name"
    local pull_pkg_deb="$pkg_name"

    # Only match -devel or -dev as a suffix or followed by a digit (e.g., -dev-tools, -devel7.11)
    # This prevents false matches in words like "device"
    if [[ "$pkg_name" =~ -devel([^a-z]|$) ]]; then
        # Package contains "-devel" - keep for RPM, convert to "-dev" for DEB
        pull_pkg_deb="${pkg_name//-devel/-dev}"
    elif [[ "$pkg_name" =~ -dev([^a-z]|$) ]]; then
        # Package contains "-dev" (but not "-devel") - keep for DEB, convert to "-devel" for RPM
        pull_pkg_rpm="${pkg_name//-dev/-devel}"
    fi

    # Return both values separated by pipe
    echo "${pull_pkg_rpm}|${pull_pkg_deb}"
}

generate_package_lists_base() {
    # Generate package lists for base packages (no architecture suffix)
    # Args: $1=rpm_pkg, $2=deb_pkg, $3=rocm_ver, $4=rpm_version_suffix, $5=deb_version_suffix
    # Returns: Sets GENERATED_RPM_PACKAGES and GENERATED_DEB_PACKAGES

    local pull_pkg_rpm="$1"
    local pull_pkg_deb="$2"
    local rocm_ver="$3"
    local rpm_version_suffix="$4"
    local deb_version_suffix="$5"

    # Build single package WITHOUT architecture suffix
    if [[ -z "$rpm_version_suffix" ]]; then
        # Default: version embedded in package name (amdrocm-amdsmi7.11)
        GENERATED_RPM_PACKAGES="${pull_pkg_rpm}${rocm_ver}"
    else
        # Explicit version: embedded version format (amdrocm-amdsmi7.11-7.11.0-2)
        GENERATED_RPM_PACKAGES="${pull_pkg_rpm}${rocm_ver}${rpm_version_suffix}"
    fi

    if [[ -z "$deb_version_suffix" ]]; then
        # Default: version embedded in package name (amdrocm-amdsmi7.11)
        GENERATED_DEB_PACKAGES="${pull_pkg_deb}${rocm_ver}"
    else
        # Explicit version: embedded version + APT pinning (amdrocm-amdsmi7.11=7.11.0-2)
        GENERATED_DEB_PACKAGES="${pull_pkg_deb}${rocm_ver}${deb_version_suffix}"
    fi
}

generate_package_lists_arch() {
    # Generate package lists for architecture-specific packages (with -gfxXYZ suffix)
    # Args: $1=rpm_pkg, $2=deb_pkg, $3=rocm_ver, $4=rpm_version_suffix, $5=deb_version_suffix
    # Returns: Sets GENERATED_RPM_PACKAGES and GENERATED_DEB_PACKAGES

    local pull_pkg_rpm="$1"
    local pull_pkg_deb="$2"
    local rocm_ver="$3"
    local rpm_version_suffix="$4"
    local deb_version_suffix="$5"

    local rpm_packages=""
    local deb_packages=""

    # Build RPM package list with architecture suffixes
    for gfx_arch in "${ROCM_GFX_ARCHS[@]}"; do
        if [[ -z "$rpm_version_suffix" ]]; then
            # Default: version embedded in package name (amdrocm-core-sdk7.11-gfx950)
            rpm_packages="$rpm_packages ${pull_pkg_rpm}${rocm_ver}-${gfx_arch}"
        else
            # Explicit version: embedded version format (amdrocm-core-sdk7.11-gfx950-7.11.0-2)
            rpm_packages="$rpm_packages ${pull_pkg_rpm}${rocm_ver}-${gfx_arch}${rpm_version_suffix}"
        fi
    done

    # Build DEB package list with architecture suffixes
    for gfx_arch in "${ROCM_GFX_ARCHS[@]}"; do
        if [[ -z "$deb_version_suffix" ]]; then
            # Default: version embedded in package name (amdrocm-core-sdk7.11-gfx950)
            deb_packages="$deb_packages ${pull_pkg_deb}${rocm_ver}-${gfx_arch}"
        else
            # Explicit version: embedded version + APT pinning (amdrocm-core-sdk7.11-gfx950=7.11.0-2)
            deb_packages="$deb_packages ${pull_pkg_deb}${rocm_ver}-${gfx_arch}${deb_version_suffix}"
        fi
    done

    # Trim leading spaces
    GENERATED_RPM_PACKAGES="${rpm_packages# }"
    GENERATED_DEB_PACKAGES="${deb_packages# }"
}

generate_package_lists() {
    echo -------------------------------------------------------------
    echo "Generating ROCm package lists from GPU architectures..."

    # RPM and DEB have different package naming conventions
    # We need to build separate lists for each

    # Strip patch version (XX.YY.ZZ -> XX.YY) for package list creation
    # Package names use major.minor format only (e.g., amdrocm-core-sdk7.11-gfx950)
    local rocm_ver="${PULL_CONFIG_ROCM_VER%.*}"

    local rpm_packages=""
    local deb_packages=""

    # Handle RPM vs DEB package naming conventions (RPM uses "devel", DEB uses "dev")
    local normalized=$(normalize_package_name "$PULL_CONFIG_PKG")
    local pull_pkg_rpm="${normalized%|*}"
    local pull_pkg_deb="${normalized#*|}"

    # Only show message if names differ
    if [[ "$pull_pkg_rpm" != "$pull_pkg_deb" ]]; then
        echo "Package naming: RPM uses '$pull_pkg_rpm', DEB uses '$pull_pkg_deb'"
    fi

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
        # RPM and DEB may use different package names (e.g., "devel" vs "dev")
        # This matches the original behavior before version control was added
        rpm_version_suffix=""  # Will use embedded version format
        deb_version_suffix=""  # Will use embedded version format
        echo "Using default version pattern (latest): ${rocm_ver}"
    fi

    # Check package type and build appropriate package lists
    if [[ "$PULL_CONFIG_PKG_TYPE" == "base" ]]; then
        echo "Package type: base (no -gfxXYZ suffix)"
        generate_package_lists_base "$pull_pkg_rpm" "$pull_pkg_deb" "$rocm_ver" "$rpm_version_suffix" "$deb_version_suffix"
        rpm_packages="$GENERATED_RPM_PACKAGES"
        deb_packages="$GENERATED_DEB_PACKAGES"
    else
        # Default: arch type (architecture-specific packages with -gfxXYZ suffix)
        echo "Package type: arch (with -gfxXYZ suffix)"
        generate_package_lists_arch "$pull_pkg_rpm" "$pull_pkg_deb" "$rocm_ver" "$rpm_version_suffix" "$deb_version_suffix"
        rpm_packages="$GENERATED_RPM_PACKAGES"
        deb_packages="$GENERATED_DEB_PACKAGES"
    fi

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

generate_package_lists_extra() {
    echo -------------------------------------------------------------
    echo "Adding extra ROCm packages to package lists..."

    # Check if there are any extra packages
    if [ ${#PULL_CONFIG_PKG_EXTRA[@]} -eq 0 ]; then
        echo "No extra packages specified."
        echo "Adding extra packages...Complete"
        return
    fi

    # Strip patch version (XX.YY.ZZ -> XX.YY) for package list creation
    local rocm_ver="${PULL_CONFIG_ROCM_VER%.*}"

    local rpm_packages=""
    local deb_packages=""

    # Process each extra package
    echo "Processing extra packages: ${PULL_CONFIG_PKG_EXTRA[*]}"
    for pkg_with_type in "${PULL_CONFIG_PKG_EXTRA[@]}"; do
        # Parse optional type prefix (base: or arch:)
        local pkg_type="arch"  # Default type
        local pkg_name="$pkg_with_type"

        if [[ "$pkg_with_type" =~ ^(base|arch):(.+)$ ]]; then
            pkg_type="${BASH_REMATCH[1]}"
            pkg_name="${BASH_REMATCH[2]}"
        fi

        echo "  Package: $pkg_name (type: $pkg_type)"

        # Normalize package name for RPM/DEB
        local normalized=$(normalize_package_name "$pkg_name")
        local pull_pkg_rpm="${normalized%|*}"
        local pull_pkg_deb="${normalized#*|}"

        # Show normalization if names differ
        if [[ "$pull_pkg_rpm" != "$pull_pkg_deb" ]]; then
            echo "    RPM='$pull_pkg_rpm', DEB='$pull_pkg_deb'"
        fi

        # Generate package names based on type
        if [[ "$pkg_type" == "base" ]]; then
            # Base package: no architecture suffix
            rpm_packages="$rpm_packages ${pull_pkg_rpm}${rocm_ver}"
            deb_packages="$deb_packages ${pull_pkg_deb}${rocm_ver}"
        else
            # Arch-specific package: add architecture suffix for each GPU arch
            for gfx_arch in "${ROCM_GFX_ARCHS[@]}"; do
                rpm_packages="$rpm_packages ${pull_pkg_rpm}${rocm_ver}-${gfx_arch}"
                deb_packages="$deb_packages ${pull_pkg_deb}${rocm_ver}-${gfx_arch}"
            done
        fi
    done

    # Trim leading spaces
    rpm_packages="${rpm_packages# }"
    deb_packages="${deb_packages# }"

    # Append to existing package lists
    PULLER_PACKAGES_RPM="$PULLER_PACKAGES_RPM $rpm_packages"
    PULLER_PACKAGES_DEB="$PULLER_PACKAGES_DEB $deb_packages"

    # Trim any extra spaces
    PULLER_PACKAGES_RPM="${PULLER_PACKAGES_RPM# }"
    PULLER_PACKAGES_DEB="${PULLER_PACKAGES_DEB# }"

    echo "Extra packages added (${#PULL_CONFIG_PKG_EXTRA[@]} package(s))"
    echo "Updated RPM Packages: $PULLER_PACKAGES_RPM"
    echo "Updated DEB Packages: $PULLER_PACKAGES_DEB"
    echo "Adding extra packages...Complete"
}

setup_puller_config_rocm() {
    echo -------------------------------------------------------------
    echo "Setting up ROCm package puller configuration files..."

    # Ensure build-config directory exists
    BUILD_CONFIG_DIR="../build-config"
    mkdir -p "$BUILD_CONFIG_DIR"

    # Template directory for ROCm configs
    TEMPLATE_DIR="../package-puller/config/therock/rocm/${PULL_CONFIG_RELEASE_TYPE}"

    # Template files
    TEMPLATE_DEB="${TEMPLATE_DIR}/rocm-${PULL_CONFIG_RELEASE_TYPE}-deb.config"
    TEMPLATE_RPM="${TEMPLATE_DIR}/rocm-${PULL_CONFIG_RELEASE_TYPE}-rpm.config"

    # Build version string from tag and runid
    local version_string=""
    if [[ -n "${PULL_CONFIG_TAG}" ]] && [[ -n "${PULL_CONFIG_RUNID}" ]]; then
        version_string="${PULL_CONFIG_TAG}-${PULL_CONFIG_RUNID}"
    elif [[ -n "${PULL_CONFIG_TAG}" ]]; then
        version_string="${PULL_CONFIG_TAG}"
    elif [[ -n "${PULL_CONFIG_RUNID}" ]]; then
        version_string="${PULL_CONFIG_RUNID}"
    fi

    # Set output config file paths
    PULLER_CONFIG_DEB="${BUILD_CONFIG_DIR}/rocm-${PULL_CONFIG_RELEASE_TYPE}-${version_string}-deb.config"
    PULLER_CONFIG_RPM="${BUILD_CONFIG_DIR}/rocm-${PULL_CONFIG_RELEASE_TYPE}-${version_string}-rpm.config"

    # Check if templates exist
    if [ ! -f "$TEMPLATE_DEB" ]; then
        echo -e "\e[31mERROR: Template file not found: $TEMPLATE_DEB\e[0m"
        exit 1
    fi
    
    if [ ! -f "$TEMPLATE_RPM" ]; then
        echo -e "\e[31mERROR: Template file not found: $TEMPLATE_RPM\e[0m"
        exit 1
    fi

    echo "Using ROCm config type: ${PULL_CONFIG_RELEASE_TYPE}"
    echo "Using ROCm tag        : ${PULL_CONFIG_TAG}"
    echo "Using ROCm run ID     : ${PULL_CONFIG_RUNID}"

    # Generate DEB config from template
    echo "Generating DEB config: $PULLER_CONFIG_DEB"
    sed "s/{{VERSION_STRING}}/${version_string}/g" "$TEMPLATE_DEB" > "$PULLER_CONFIG_DEB"

    # Generate RPM config from template
    echo "Generating RPM config: $PULLER_CONFIG_RPM"
    sed "s/{{VERSION_STRING}}/${version_string}/g" "$TEMPLATE_RPM" > "$PULLER_CONFIG_RPM"

    echo -e "\e[32mROCm package puller configuration files generated successfully.\e[0m"
    echo "Setting up ROCm package puller configuration files...Complete"
}

setup_puller_config_amdgpu() {
    echo -------------------------------------------------------------
    echo "Setting up AMDGPU package puller configuration files..."

    # Ensure build-config directory exists
    BUILD_CONFIG_DIR="../build-config"
    mkdir -p "$BUILD_CONFIG_DIR"

    # Template directory for AMDGPU configs
    TEMPLATE_DIR="../package-puller/config/therock/amdgpu/${PULL_CONFIG_AMDGPU}"

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
    pushd ../package-puller
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
    pushd ../package-puller
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
    pushd ../package-puller
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
    pushd ../package-puller
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
    pushd ../package-puller
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

# Record start time
SETUP_START_TIME=$(date +%s)

echo ============================
echo SETUP INSTALLER
echo ============================

SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)
echo SUDO: $SUDO

os_release

# Load config file if specified (allows command-line args to override)
read_config "$@"

# parse args
while (($#))
do
    case "$1" in
    config=*)
        # Already processed before argument parsing loop
        # Skip to allow other args to override config values
        shift
        ;;
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
        PULL_CONFIG_RELEASE_TYPE="${1#*=}"
        echo "ROCm pull config type set to: $PULL_CONFIG_RELEASE_TYPE"
        shift
        ;;
    pulltag=*)
        PULL_CONFIG_TAG="${1#*=}"
        echo "ROCm pull config tag set to: $PULL_CONFIG_TAG"
        shift
        ;;
    pullrunid=*)
        PULL_CONFIG_RUNID="${1#*=}"
        echo "ROCm pull config run ID set to: $PULL_CONFIG_RUNID"
        shift
        ;;
    pullrocmver=*)
        PULL_CONFIG_ROCM_VER="${1#*=}"
        echo "ROCm version set to: $PULL_CONFIG_ROCM_VER"
        shift
        ;;
    pullpkg=*)
        # Parse package name with optional type prefix (e.g., "base:amdrocm-amdsmi" or "arch:amdrocm-core-sdk")
        # Default type is "arch" if no prefix specified
        pkg_arg="${1#*=}"

        if [[ "$pkg_arg" =~ ^(base|arch):(.+)$ ]]; then
            # Type prefix specified (e.g., "base:amdrocm-amdsmi")
            PULL_CONFIG_PKG_TYPE="${BASH_REMATCH[1]}"
            PULL_CONFIG_PKG="${BASH_REMATCH[2]}"
            echo "ROCm package set to: $PULL_CONFIG_PKG (type: $PULL_CONFIG_PKG_TYPE)"
        else
            # No prefix, default to "arch" type for backward compatibility
            PULL_CONFIG_PKG="$pkg_arg"
            PULL_CONFIG_PKG_TYPE="arch"
            echo "ROCm package set to: $PULL_CONFIG_PKG (type: arch - default)"
        fi
        shift
        ;;
    pullpkgextra=*)
        PKGS_INPUT="${1#*=}"
        # Convert comma-separated string to array
        # Each element can have optional type prefix (e.g., "base:pkg" or "arch:pkg")
        IFS=',' read -ra PULL_CONFIG_PKG_EXTRA <<< "$PKGS_INPUT"
        echo "Extra ROCm packages set to: ${PULL_CONFIG_PKG_EXTRA[*]}"
        echo "Note: Use [type:]package syntax - arch:package (default) or base:package"
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
        echo -e "\e[31mERROR: Support for version package pull - disabled.\e[0m"
        exit 1
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

# Validate required arguments if ROCm setup is enabled
if [[ $SETUP_ROCM == 1 ]]; then
    validate_args
fi

# Recreate build-config directory (clean slate for each run)
BUILD_CONFIG_DIR="../build-config"
if [ -d "$BUILD_CONFIG_DIR" ]; then
    echo "Removing existing $BUILD_CONFIG_DIR directory"
    rm -rf "$BUILD_CONFIG_DIR"
fi
mkdir -p "$BUILD_CONFIG_DIR"
echo "Created $BUILD_CONFIG_DIR directory"

# Install required tools
install_tools

# Generate ROCm package lists from GPU architecture array
generate_package_lists

# Add extra packages to the package lists
generate_package_lists_extra

echo Running Package Puller...

if [[ $SETUP_ROCM == 1 ]]; then
    # Generate ROCm package puller configuration files from templates
    setup_puller_config_rocm
    setup_rocm
fi

if [[ $SETUP_AMDGPU == 1 ]]; then
    # Generate AMDGPU package puller configuration files from templates
    setup_puller_config_amdgpu
    if [[ "$SETUP_AMDGPU_MODE" == "all" ]]; then
        setup_amdgpu_all
    elif [[ "$SETUP_AMDGPU_MODE" == "single" ]]; then
        setup_amdgpu
    else
        echo "ERROR: Invalid SETUP_AMDGPU_MODE: $SETUP_AMDGPU_MODE"
        exit 1
    fi
fi

# Write ROCM_VER to VERSION file if ROCm setup was performed
if [[ $SETUP_ROCM == 1 ]]; then
    write_version
fi

echo Running Package Puller...Complete

# Calculate and display setup time
SETUP_END_TIME=$(date +%s)
SETUP_ELAPSED=$((SETUP_END_TIME - SETUP_START_TIME))

# Convert seconds to hours, minutes, seconds
SETUP_HOURS=$((SETUP_ELAPSED / 3600))
SETUP_MINUTES=$(((SETUP_ELAPSED % 3600) / 60))
SETUP_SECONDS=$((SETUP_ELAPSED % 60))

echo ""
echo ==============================
echo "Setup completed successfully!"
echo "=============================="
echo -e "\e[36mTotal setup time: ${SETUP_HOURS}h ${SETUP_MINUTES}m ${SETUP_SECONDS}s (${SETUP_ELAPSED} seconds)\e[0m"
echo ==============================
echo ""

