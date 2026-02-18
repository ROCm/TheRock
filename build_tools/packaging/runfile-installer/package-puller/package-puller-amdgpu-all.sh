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

# Multi-distro AMDGPU package puller
# Downloads AMDGPU packages for all supported distributions from a single machine

PROG=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration directory - now uses build-config with dynamic versioning
# Can be overridden by environment variables from setup-installer
AMDGPU_CONFIG_TYPE="${AMDGPU_CONFIG_TYPE:-release}"
AMDGPU_CONFIG_VER="${AMDGPU_CONFIG_VER:-30.30}"
CONFIG_DIR="../build-config"

# Base output directory
OUTPUT_BASE_DIR="../package-extractor/packages-amdgpu"

# AMDGPU packages to download
AMDGPU_PACKAGES="amdgpu-dkms amdgpu-dkms-firmware"

# Distro configurations - now using dynamic naming
declare -A DISTRO_CONFIGS=(
    ["ub24"]="amdgpu-${AMDGPU_CONFIG_TYPE}-${AMDGPU_CONFIG_VER}-ub24.config"
    ["ub22"]="amdgpu-${AMDGPU_CONFIG_TYPE}-${AMDGPU_CONFIG_VER}-ub22.config"
    ["el10"]="amdgpu-${AMDGPU_CONFIG_TYPE}-${AMDGPU_CONFIG_VER}-el10.config"
    ["el9"]="amdgpu-${AMDGPU_CONFIG_TYPE}-${AMDGPU_CONFIG_VER}-el9.config"
    ["el8"]="amdgpu-${AMDGPU_CONFIG_TYPE}-${AMDGPU_CONFIG_VER}-el8.config"
    ["sle16"]="amdgpu-${AMDGPU_CONFIG_TYPE}-${AMDGPU_CONFIG_VER}-sle16.config"
    ["sle15"]="amdgpu-${AMDGPU_CONFIG_TYPE}-${AMDGPU_CONFIG_VER}-sle15.config"
    ["amzn23"]="amdgpu-${AMDGPU_CONFIG_TYPE}-${AMDGPU_CONFIG_VER}-amzn23.config"
)

declare -A DISTRO_TYPES=(
    ["ub24"]="deb"
    ["ub22"]="deb"
    ["el10"]="rpm"
    ["el9"]="rpm"
    ["el8"]="rpm"
    ["sle16"]="rpm"
    ["sle15"]="rpm"
    ["amzn23"]="rpm"
)

###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $PROG [options]

This script downloads AMDGPU packages for all supported distributions.
It can be run from any distribution (recommended: AlmaLinux 8).

[options]:
    help        = Display this help information.
    distro=TAG  = Download packages for a specific distro only (ub24, ub22, el10, el9, el8, sle16, sle15, amzn23).
    all         = Download packages for all distributions (default).

Examples:
    ./package-puller-amdgpu-all.sh                 # Download for all distros
    ./package-puller-amdgpu-all.sh distro=ub24     # Download only Ubuntu 24.04 packages
    ./package-puller-amdgpu-all.sh distro=el9      # Download only RHEL 9 packages

END_USAGE
}

print_status() {
    local msg=$1
    echo -e "\e[32m===== $msg =====\e[0m"
}

print_error() {
    local msg=$1
    echo -e "\e[31mERROR: $msg\e[0m"
}

print_warning() {
    local msg=$1
    echo -e "\e[93mWARNING: $msg\e[0m"
}

download_deb_packages() {
    local distro_tag=$1
    local config_file=$2
    local output_dir=$3

    print_status "Downloading DEB packages for $distro_tag"

    # Read config file to get repository URL
    if [ ! -f "$config_file" ]; then
        print_error "Config file not found: $config_file"
        return 1
    fi

    source "$config_file"

    if [ -z "$AMDGPU_REPO" ]; then
        print_error "AMDGPU_REPO not defined in $config_file"
        return 1
    fi

    # Create output directory
    mkdir -p "$output_dir"

    # Extract the repository URL from the AMDGPU_REPO format
    # Format: "deb https://repo.radeon.com/amdgpu/30.20.1/ubuntu jammy main"
    REPO_URL=$(echo "$AMDGPU_REPO" | awk '{print $2}')
    DISTRO_CODENAME=$(echo "$AMDGPU_REPO" | awk '{print $3}')

    print_status "Repository: $REPO_URL"
    print_status "Distro codename: $DISTRO_CODENAME"

    # Download Packages file to get package information
    PACKAGES_URL="${REPO_URL}/dists/${DISTRO_CODENAME}/main/binary-amd64/Packages.gz"

    print_status "Downloading package index from: $PACKAGES_URL"
    wget -q -O "$output_dir/Packages.gz" "$PACKAGES_URL"

    if [ $? -ne 0 ]; then
        print_warning "Repository not available for $distro_tag at $PACKAGES_URL"
        print_warning "Skipping $distro_tag (repository may not exist for this AMDGPU version)"
        rm -rf "$output_dir"
        return 2
    fi

    gunzip -f "$output_dir/Packages.gz"

    # Download each package
    for pkg in $AMDGPU_PACKAGES; do
        print_status "Searching for package: $pkg"

        # Extract package filename from Packages file using awk for better parsing
        # This handles packages with names like "amdgpu-dkms" or "amdgpu-dkms-firmware"
        PKG_FILE=$(awk -v pkg="$pkg" '
            /^Package:/ {
                pkgname=$2;
            }
            /^Filename:/ {
                filename=$2;
            }
            /^$/ {
                if (pkgname == pkg || index(pkgname, pkg) == 1) {
                    print filename;
                    exit;
                }
                pkgname="";
                filename="";
            }
        ' "$output_dir/Packages")

        if [ -z "$PKG_FILE" ]; then
            print_warning "Package $pkg not found in repository"
            continue
        fi

        PKG_URL="${REPO_URL}/${PKG_FILE}"
        PKG_NAME=$(basename "$PKG_FILE")

        print_status "Downloading $PKG_NAME from $PKG_URL"
        wget -q -O "$output_dir/$PKG_NAME" "$PKG_URL"

        if [ $? -eq 0 ]; then
            echo -e "\e[32mSuccessfully downloaded: $PKG_NAME\e[0m"
        else
            print_error "Failed to download $PKG_NAME"
        fi
    done

    # Cleanup
    rm -f "$output_dir/Packages"

    return 0
}

download_rpm_packages() {
    local distro_tag=$1
    local config_file=$2
    local output_dir=$3

    print_status "Downloading RPM packages for $distro_tag"

    # Read config file to get repository URL
    if [ ! -f "$config_file" ]; then
        print_error "Config file not found: $config_file"
        return 1
    fi

    source "$config_file"

    if [ -z "$AMDGPU_REPO" ]; then
        print_error "AMDGPU_REPO not defined in $config_file"
        return 1
    fi

    # Extract baseurl from AMDGPU_REPO
    # Format is a repo file content with baseurl=...
    REPO_BASEURL=$(echo "$AMDGPU_REPO" | grep -oP 'baseurl=\K[^\s]+')

    if [ -z "$REPO_BASEURL" ]; then
        print_error "Failed to extract baseurl from config"
        return 1
    fi

    # Remove trailing slash from REPO_BASEURL to avoid double slashes
    REPO_BASEURL="${REPO_BASEURL%/}"

    print_status "Repository: $REPO_BASEURL"

    # Create output directory
    mkdir -p "$output_dir"

    # Download repodata to get package information
    REPOMD_URL="${REPO_BASEURL}/repodata/repomd.xml"

    print_status "Downloading repomd.xml from: $REPOMD_URL"
    wget -q -O "$output_dir/repomd.xml" "$REPOMD_URL"

    if [ $? -ne 0 ]; then
        print_warning "Repository not available for $distro_tag at $REPOMD_URL"
        print_warning "Skipping $distro_tag (repository may not exist for this AMDGPU version)"
        rm -rf "$output_dir"
        return 2
    fi

    # Get primary.xml.gz location using awk to parse XML
    PRIMARY_HREF=$(awk -F'"' '/type="primary"/ {getline; if ($0 ~ /href=/) print $2}' "$output_dir/repomd.xml")

    if [ -z "$PRIMARY_HREF" ]; then
        # Try alternative parsing method
        PRIMARY_HREF=$(grep 'type="primary"' "$output_dir/repomd.xml" -A 5 | grep 'location href' | sed 's/.*href="\([^"]*\)".*/\1/')
    fi

    if [ -z "$PRIMARY_HREF" ]; then
        print_error "Failed to extract primary database location from repomd.xml"
        rm -f "$output_dir/repomd.xml"
        return 1
    fi

    PRIMARY_URL="${REPO_BASEURL}/${PRIMARY_HREF}"

    print_status "Downloading primary metadata from: $PRIMARY_URL"
    wget -q -O "$output_dir/primary.xml.gz" "$PRIMARY_URL"

    if [ $? -ne 0 ]; then
        print_error "Failed to download primary.xml.gz from $PRIMARY_URL"
        rm -f "$output_dir/repomd.xml"
        return 1
    fi

    gunzip -f "$output_dir/primary.xml.gz" 2>/dev/null

    if [ ! -f "$output_dir/primary.xml" ]; then
        print_error "Failed to decompress primary.xml.gz"
        rm -f "$output_dir/repomd.xml" "$output_dir/primary.xml.gz"
        return 1
    fi

    # Download each package
    for pkg in $AMDGPU_PACKAGES; do
        print_status "Searching for package: $pkg"

        # Extract package location from primary.xml
        # Look for package name and extract the location href (search for partial name match)
        PKG_LOCATION=$(awk -v pkg="$pkg" '
            /<package type="rpm">/ {inpkg=1; loc=""; name=""}
            inpkg && /<name>/ {
                # Extract name between tags
                gsub(/.*<name>/, "", $0);
                gsub(/<\/name>.*/, "", $0);
                name=$0;
            }
            inpkg && /location href=/ {
                # Extract href value
                gsub(/.*href="/, "", $0);
                gsub(/".*/, "", $0);
                loc=$0;
            }
            inpkg && /<\/package>/ {
                if (index(name, pkg) == 1) {
                    print loc;
                    exit;
                }
                inpkg=0;
            }
        ' "$output_dir/primary.xml")

        if [ -z "$PKG_LOCATION" ]; then
            print_warning "Package $pkg not found in repository"
            continue
        fi

        PKG_URL="${REPO_BASEURL}/${PKG_LOCATION}"
        PKG_NAME=$(basename "$PKG_LOCATION")

        print_status "Downloading $PKG_NAME from $PKG_URL"
        wget -q -O "$output_dir/$PKG_NAME" "$PKG_URL"

        if [ $? -eq 0 ]; then
            echo -e "\e[32mSuccessfully downloaded: $PKG_NAME\e[0m"
        else
            print_error "Failed to download $PKG_NAME"
        fi
    done

    # Cleanup
    rm -f "$output_dir/repomd.xml" "$output_dir/primary.xml"

    return 0
}

download_for_distro() {
    local distro_tag=$1

    if [ -z "${DISTRO_CONFIGS[$distro_tag]}" ]; then
        print_error "Unknown distro tag: $distro_tag"
        return 1
    fi

    local config_file="${CONFIG_DIR}/${DISTRO_CONFIGS[$distro_tag]}"
    local distro_type="${DISTRO_TYPES[$distro_tag]}"
    local output_dir="${OUTPUT_BASE_DIR}/${distro_tag}"

    # Create base directory if it doesn't exist
    if [ ! -d "$OUTPUT_BASE_DIR" ]; then
        mkdir -p "$OUTPUT_BASE_DIR"
    fi

    echo -e "\e[36mProcessing $distro_tag (type: $distro_type)...\e[0m"

    if [ "$distro_type" == "deb" ]; then
        download_deb_packages "$distro_tag" "$config_file" "$output_dir"
    elif [ "$distro_type" == "rpm" ]; then
        download_rpm_packages "$distro_tag" "$config_file" "$output_dir"
    else
        print_error "Unknown distro type: $distro_type"
        return 1
    fi

    return $?
}

####### Main script ###############################################################

echo "========================================"
echo "AMDGPU Multi-Distro Package Puller"
echo "========================================"

SPECIFIC_DISTRO=""

# Parse arguments
while (($#)); do
    case "$1" in
    help)
        usage
        exit 0
        ;;
    distro=*)
        SPECIFIC_DISTRO="${1#*=}"
        shift
        ;;
    all)
        SPECIFIC_DISTRO=""
        shift
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
done

# Change to script directory
pushd "$SCRIPT_DIR" > /dev/null

# Download packages
if [ -n "$SPECIFIC_DISTRO" ]; then
    # Download for specific distro only
    download_for_distro "$SPECIFIC_DISTRO"
    EXIT_CODE=$?
else
    # Download for all distros
    EXIT_CODE=0
    SUCCESS_COUNT=0
    SKIP_COUNT=0
    FAIL_COUNT=0
    SKIPPED_DISTROS=()
    FAILED_DISTROS=()

    for distro_tag in "${!DISTRO_CONFIGS[@]}"; do
        download_for_distro "$distro_tag"
        RESULT=$?

        if [ $RESULT -eq 0 ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        elif [ $RESULT -eq 2 ]; then
            # Return code 2 = repository not available (skip)
            SKIP_COUNT=$((SKIP_COUNT + 1))
            SKIPPED_DISTROS+=("$distro_tag")
        else
            # Return code 1 = actual error
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAILED_DISTROS+=("$distro_tag")
            EXIT_CODE=1
        fi
        echo ""
    done

    # Print summary
    echo ""
    echo "========================================"
    echo "AMDGPU Package Download Summary"
    echo "========================================"
    echo "Successful: $SUCCESS_COUNT"
    if [ $SKIP_COUNT -gt 0 ]; then
        echo "Skipped (repo not available): $SKIP_COUNT - ${SKIPPED_DISTROS[*]}"
    fi
    if [ $FAIL_COUNT -gt 0 ]; then
        echo "Failed: $FAIL_COUNT - ${FAILED_DISTROS[*]}"
    fi
    echo "========================================"

    # Print final status based on results
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        if [ $SKIP_COUNT -gt 0 ]; then
            print_status "AMDGPU package download complete! (Some distros skipped due to unavailable repositories)"
        else
            print_status "AMDGPU package download complete!"
        fi
    else
        print_error "Some package downloads failed. Check the output above for details."
    fi
fi

popd > /dev/null

exit $EXIT_CODE
