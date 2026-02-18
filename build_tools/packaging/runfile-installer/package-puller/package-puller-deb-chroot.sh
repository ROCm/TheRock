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

# This script downloads DEB packages using an Ubuntu chroot environment
# Designed to run on RPM-based systems (AlmaLinux, RHEL, etc.)

# Chroot configuration
CHROOT_DIR="$PWD/ubuntu-chroot"
UBUNTU_VERSION="focal"  # Ubuntu 20.04 (uses older compression compatible with EL8's dpkg)
UBUNTU_MIRROR="http://archive.ubuntu.com/ubuntu"

# Inputs
ROCM_REPO=
AMDGPU_REPO=
GRAPHICS_REPO=

# Packaging repos
PACKAGE_REPO=$PWD/packages
SETUP_PATH=$PWD/setup

# Logs
PULL_LOGS_DIR="$PWD/logs"
PULL_CURRENT_LOG="$PULL_LOGS_DIR/pull_chroot_$(date +%s).log"

# Config
PACKAGES="rocm"
VERBOSE=0

REPO_LIST=(rocm-build.list amdgpu-build.list graphics-build.list rocm.list amdgpu.list amdgpu-proprietary.list)
GPG_LIST=(rocm.gpg amdrocm.gpg)

PROMPT_USER=0
DUMP_AMD_PKGS=0
DUMP_NON_AMD_PKGS=0

###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $PROG [Options] [Pull_Config] [Packages] [Output]

[Options}:
    help    = Display this help information.

    prompt  = Run the package puller with user prompts.
    amd     = Copy amd-specific packages out.
    other   = Copy non-amd packages out.
    verbose = Run the package puller with verbose logging.

[Pull_Config]:
    config=<file_path> = <file_path> Path to a .config file with create settings in the format of create-default.config.

[Packages]:
    pkg=<package list> = <package-list> List of Package/Packages to pull

[Output]:
    out=<file_path>    = <file_path> Path to output directory for pulled packages

Example (pull by config):
-------------------------

    ./package-puller-deb-chroot.sh config="config/rocm-7.11-22.04.config" out="/packages" prompt amd

END_USAGE
}

check_host_os() {
    if [[ -r  /etc/os-release ]]; then
        . /etc/os-release

        DISTRO_NAME=$ID
        DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')

        echo "Running on $DISTRO_NAME $DISTRO_VER (chroot mode for DEB package pulling)"
    else
        echo "Cannot determine host OS"
        exit 1
    fi
}

prompt_user() {
    if [[ $PROMPT_USER == 1 ]]; then
        read -p "$1" option
    else
        option=y
    fi
}

install_chroot_tools() {
    echo "========================================="
    echo "Installing chroot tools on host..."
    echo "========================================="

    echo "Checking for required tools..."
    local tools_to_install=""

    # Check if debootstrap is installed
    if ! command -v debootstrap &> /dev/null; then
        echo "debootstrap not found. Will install."
        tools_to_install="$tools_to_install debootstrap"
    else
        echo "debootstrap is already installed."
        debootstrap --version
    fi

    # Check if dpkg is installed (needed for extracting .deb packages)
    if ! command -v dpkg &> /dev/null; then
        echo "dpkg not found. Will install."
        tools_to_install="$tools_to_install dpkg"
    else
        echo "dpkg is already installed."
    fi

    # Install missing tools
    if [[ -n "$tools_to_install" ]]; then
        echo "Installing:$tools_to_install"
        $SUDO dnf install -y $tools_to_install
        if [[ $? -ne 0 ]]; then
            echo -e "\e[31mERROR: Failed to install required tools\e[0m"
            echo "Make sure EPEL repository is enabled."
            exit 1
        fi
        echo "Tool installation complete."
    fi

    echo "Chroot tools ready."
}

create_ubuntu_chroot() {
    echo "========================================="
    echo "Creating Ubuntu chroot environment..."
    echo "========================================="

    if [ -d "$CHROOT_DIR" ]; then
        echo "Chroot directory exists: $CHROOT_DIR"
        prompt_user "Remove and recreate? (y/n): "
        if [[ $option == "y" || $option == "Y" ]]; then
            echo "Removing existing chroot..."
            $SUDO rm -rf "$CHROOT_DIR"
            echo "Existing chroot removed."
        else
            echo "Using existing chroot."
            return 0
        fi
    fi

    echo "Creating Ubuntu $UBUNTU_VERSION chroot at: $CHROOT_DIR"
    echo "This may take several minutes (downloading minimal base system)..."
    echo "Running: debootstrap --variant=minbase --arch=amd64 $UBUNTU_VERSION $CHROOT_DIR $UBUNTU_MIRROR"

    # Use --variant=minbase for a smaller, more reliable base system
    # This is especially important when running in containers
    $SUDO debootstrap --variant=minbase --arch=amd64 $UBUNTU_VERSION "$CHROOT_DIR" $UBUNTU_MIRROR

    if [[ $? -ne 0 ]]; then
        echo -e "\e[31mERROR: Failed to create Ubuntu chroot\e[0m"
        echo "Check log for details: $PULL_CURRENT_LOG"
        echo ""
        echo "Troubleshooting tips:"
        echo "  1. Make sure you have dpkg installed: dnf install -y dpkg"
        echo "  2. Check if debootstrap.log exists in $CHROOT_DIR for more details"
        if [ -f "$CHROOT_DIR/debootstrap/debootstrap.log" ]; then
            echo ""
            echo "Last 20 lines of debootstrap.log:"
            tail -20 "$CHROOT_DIR/debootstrap/debootstrap.log"
        fi
        exit 1
    fi

    echo "Ubuntu chroot created successfully."
}

setup_chroot_environment() {
    echo "========================================="
    echo "Configuring chroot environment..."
    echo "========================================="

    # Mount necessary filesystems
    echo "Mounting /proc, /sys, /dev in chroot..."
    echo "  - Mounting /proc..."
    $SUDO mount --bind /proc "$CHROOT_DIR/proc" 2>/dev/null || true
    echo "  - Mounting /sys..."
    $SUDO mount --bind /sys "$CHROOT_DIR/sys" 2>/dev/null || true
    echo "  - Mounting /dev..."
    $SUDO mount --bind /dev "$CHROOT_DIR/dev" 2>/dev/null || true
    echo "  - Mounting /dev/pts..."
    $SUDO mount --bind /dev/pts "$CHROOT_DIR/dev/pts" 2>/dev/null || true

    # Copy resolv.conf for network access
    echo "Setting up network configuration..."
    $SUDO cp /etc/resolv.conf "$CHROOT_DIR/etc/resolv.conf"
    echo "  - Copied /etc/resolv.conf"

    # Create package output directory inside chroot
    echo "Creating package directory in chroot..."
    $SUDO mkdir -p "$CHROOT_DIR/packages"
    echo "  - Created $CHROOT_DIR/packages"

    echo "Chroot environment configured."
}

install_ca_certificates_in_chroot() {
    echo "========================================="
    echo "Installing ca-certificates in chroot..."
    echo "========================================="

    echo "Installing ca-certificates, wget, and gnupg for HTTPS support..."
    echo "NOTE: Only using Ubuntu's default HTTP repositories at this stage."

    # Run apt-get commands inside chroot to install ca-certificates, wget, and gnupg
    # At this point, only Ubuntu's default repos (HTTP) are configured
    $SUDO chroot "$CHROOT_DIR" /bin/bash -c "apt-get update && apt-get install -y ca-certificates wget gnupg"

    if [[ $? -ne 0 ]]; then
        echo -e "\e[31mERROR: Failed to install ca-certificates, wget, and gnupg\e[0m"
        cleanup_chroot
        exit 1
    fi

    echo "ca-certificates, wget, and gnupg installed successfully."
}

setup_gpg_in_chroot() {
    echo "========================================="
    echo "Setting up GPG in chroot..."
    echo "========================================="

    if [[ $ROCM_REPO =~ "amdrocm.gpg" ]] || [[ $ROCM_REPO =~ "rocm.gpg" ]]; then
        # Determine GPG key URL based on repo type (release vs prereleases)
        if [[ $ROCM_REPO =~ "rocm.prereleases.amd.com" ]]; then
            GPG_KEY_URL="https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg"
            echo "Using prereleases GPG key: $GPG_KEY_URL"
        else
            GPG_KEY_URL="https://repo.amd.com/rocm/packages/gpg/rocm.gpg"
            echo "Using release GPG key: $GPG_KEY_URL"
        fi

        echo "Downloading ROCm GPG key inside chroot to /etc/apt/keyrings/amdrocm.gpg..."
        $SUDO chroot "$CHROOT_DIR" /bin/bash -c "set -e && set -o pipefail && mkdir --parents --mode=0755 /etc/apt/keyrings && wget $GPG_KEY_URL -O - | gpg --dearmor | tee /etc/apt/keyrings/amdrocm.gpg > /dev/null"

        if [[ $? -ne 0 ]]; then
            echo -e "\e[31mERROR: Failed to download GPG key\e[0m"
            cleanup_chroot
            exit 1
        fi

        echo "GPG key downloaded successfully to /etc/apt/keyrings/amdrocm.gpg"
    else
        echo "Skipping GPG setup (no GPG key reference found in ROCM_REPO)"
    fi

    echo "GPG setup complete."
}

configure_apt_sources() {
    echo "========================================="
    echo "Configuring APT sources in chroot..."
    echo "========================================="

    # Create repository list files from config variables
    echo "Creating ROCm repository configuration..."
    local repo_count=0

    # Create ROCm repo list if defined
    if [[ -n "$ROCM_REPO" ]]; then
        echo "  - Creating rocm.list from ROCM_REPO"
        echo "$ROCM_REPO" | $SUDO tee "$CHROOT_DIR/etc/apt/sources.list.d/rocm.list" > /dev/null
        repo_count=$((repo_count + 1))
        echo "    Repository: $ROCM_REPO"
    fi

    # Create AMDGPU repo list if defined
    if [[ -n "$AMDGPU_REPO" ]]; then
        echo "  - Creating amdgpu.list from AMDGPU_REPO"
        echo "$AMDGPU_REPO" | $SUDO tee "$CHROOT_DIR/etc/apt/sources.list.d/amdgpu.list" > /dev/null
        repo_count=$((repo_count + 1))
        echo "    Repository: $AMDGPU_REPO"
    fi

    # Create Graphics repo list if defined
    if [[ -n "$GRAPHICS_REPO" ]]; then
        echo "  - Creating graphics.list from GRAPHICS_REPO"
        echo "$GRAPHICS_REPO" | $SUDO tee "$CHROOT_DIR/etc/apt/sources.list.d/graphics.list" > /dev/null
        repo_count=$((repo_count + 1))
        echo "    Repository: $GRAPHICS_REPO"
    fi

    echo "Created $repo_count repository list files."

    echo "APT sources configured."
}

download_packages_in_chroot() {
    echo "========================================="
    echo "Downloading packages in chroot..."
    echo "========================================="

    echo "Packages to download: $PACKAGES"
    echo "Creating chroot download script..."

    # Create a script to run inside the chroot
    cat > /tmp/chroot-download.sh <<'CHROOT_SCRIPT'
#!/bin/bash
set -e

echo "========================================="
echo "Chroot download script started"
echo "========================================="
echo "Arguments: $@"
echo "Working directory: $(pwd)"
echo ""

echo "Step 1: Updating APT cache with ROCm repositories..."
apt-get update
echo "APT cache updated successfully."
echo ""

echo "Step 2: Installing apt-fast for faster downloads..."
apt-get install -y apt-fast || {
    echo "apt-fast not available, using apt-get"
    APT_CMD="apt-get"
}
echo ""

# Use apt-fast if available, otherwise apt-get
if command -v apt-fast &> /dev/null; then
    echo "Using apt-fast for downloads"
    APT_CMD="apt-fast"
else
    echo "Using apt-get for downloads"
    APT_CMD="apt-get"
fi
echo ""

echo "Step 3: Changing to /packages directory..."
cd /packages
echo "Current directory: $(pwd)"
echo ""

# Download packages specified in arguments
echo "Step 4: Downloading packages..."
pkg_num=1
for pkg in "$@"; do
    echo "----------------------------------------"
    echo "[$pkg_num] Downloading: $pkg"
    echo "Resolving dependencies..."
    $APT_CMD download $(apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts --no-breaks --no-replaces --no-enhances $pkg | grep "^\w" | sort -u)
    echo "[$pkg_num] Completed: $pkg"
    pkg_num=$((pkg_num + 1))
    echo ""
done

echo "========================================="
echo "Package download complete."
pkg_count=$(ls -1 /packages/*.deb 2>/dev/null | wc -l)
echo "Total packages downloaded: $pkg_count"
echo "========================================="
CHROOT_SCRIPT

    echo "Making chroot script executable..."
    chmod +x /tmp/chroot-download.sh

    echo "Copying script to chroot environment..."
    $SUDO cp /tmp/chroot-download.sh "$CHROOT_DIR/tmp/"

    echo "Script copied. Listing chroot /tmp directory:"
    $SUDO ls -la "$CHROOT_DIR/tmp/chroot-download.sh"

    # Execute the download script inside chroot
    echo "Executing download in chroot..."
    echo "Command: chroot $CHROOT_DIR /tmp/chroot-download.sh $PACKAGES"
    $SUDO chroot "$CHROOT_DIR" /tmp/chroot-download.sh $PACKAGES

    if [[ $? -ne 0 ]]; then
        echo -e "\e[31mERROR: Package download failed in chroot\e[0m"
        cleanup_chroot
        exit 1
    fi

    echo "Packages downloaded successfully."
}

copy_packages_out() {
    echo "========================================="
    echo "Copying packages out of chroot..."
    echo "========================================="

    # Create output directory
    if [ ! -d "$PACKAGE_REPO" ]; then
        mkdir -p "$PACKAGE_REPO"
    fi

    # Copy packages from chroot to host
    if [[ $DUMP_AMD_PKGS == 1 ]]; then
        echo "Filtering AMD packages..."
        $SUDO mkdir -p "$PACKAGE_REPO/packages-amd"
        for pkg in "$CHROOT_DIR/packages"/*.deb; do
            pkg_name=$(basename "$pkg")
            if [[ "$pkg_name" =~ (rocm|amdgpu|hip|hsa|miopen|rccl|rocblas|rocsparse|rocfft|rocrand|rocsolver|rocthrust|rocprim|hipcub|migraphx|rocalution|composablekernel) ]]; then
                echo "  Copying: $pkg_name"
                $SUDO cp "$pkg" "$PACKAGE_REPO/packages-amd/"
            fi
        done
    else
        # Copy all packages
        echo "Copying all packages..."
        $SUDO cp "$CHROOT_DIR/packages"/*.deb "$PACKAGE_REPO/" 2>/dev/null || true
    fi

    # Count packages
    pkg_count=$(ls -1 "$PACKAGE_REPO"/*.deb 2>/dev/null | wc -l)
    if [[ $DUMP_AMD_PKGS == 1 ]]; then
        pkg_count=$(ls -1 "$PACKAGE_REPO/packages-amd"/*.deb 2>/dev/null | wc -l)
    fi

    echo "Copied $pkg_count packages to: $PACKAGE_REPO"
}

cleanup() {
    echo "========================================="
    echo "Cleaning up repo files and GPG keys from chroot..."
    echo "========================================="

    # Remove any list files from chroot
    for index in ${REPO_LIST[@]}; do
        if [ -f "$CHROOT_DIR/etc/apt/sources.list.d/$index" ]; then
            echo -e "\e[93m=-=-=-= Removing $index =-=-=-=\e[0m"
            $SUDO rm "$CHROOT_DIR/etc/apt/sources.list.d/$index"
        fi
    done

    # Remove any gpg files from chroot
    for index in ${GPG_LIST[@]}; do
        if [ -f "$CHROOT_DIR/etc/apt/keyrings/$index" ]; then
            echo -e "\e[93m=-=-=-= Removing $index =-=-=-=\e[0m"
            $SUDO rm "$CHROOT_DIR/etc/apt/keyrings/$index"
        fi
    done

    echo "Cleanup complete."
}

cleanup_chroot() {
    echo "========================================="
    echo "Cleaning up chroot environment..."
    echo "========================================="

    # Unmount filesystems
    echo "Unmounting chroot filesystems..."
    $SUDO umount "$CHROOT_DIR/dev/pts" 2>/dev/null || true
    $SUDO umount "$CHROOT_DIR/dev" 2>/dev/null || true
    $SUDO umount "$CHROOT_DIR/sys" 2>/dev/null || true
    $SUDO umount "$CHROOT_DIR/proc" 2>/dev/null || true

    # Optionally remove chroot directory
    prompt_user "Remove chroot directory? (y/n): "
    if [[ $option == "y" || $option == "Y" ]]; then
        echo "Removing chroot directory..."
        $SUDO rm -rf "$CHROOT_DIR"
        echo "Chroot directory removed."
    else
        echo "Keeping chroot directory for reuse: $CHROOT_DIR"
    fi
}

read_config() {
    echo "Reading configuration from: $1"

    if [ ! -f "$1" ]; then
        echo -e "\e[31mERROR: Config file not found: $1\e[0m"
        exit 1
    fi

    # Source the config file
    . "$1"

    # Set variables from config
    if [[ -n $ROCM_REPO_URL ]]; then
        ROCM_REPO="$ROCM_REPO_URL"
    fi

    if [[ -n $AMDGPU_REPO_URL ]]; then
        AMDGPU_REPO="$AMDGPU_REPO_URL"
    fi

    if [[ -n $UBUNTU_CODENAME ]]; then
        UBUNTU_VERSION="$UBUNTU_CODENAME"
    fi

    echo "Configuration loaded."
}

####### Main script ###############################################################

# Create logs directory
if [ ! -d $PULL_LOGS_DIR ]; then
    mkdir -p $PULL_LOGS_DIR
fi

exec > >(tee -a "$PULL_CURRENT_LOG") 2>&1

echo "==============================="
echo "PACKAGE PULLER - DEB (CHROOT)"
echo "==============================="

PROG=${0##*/}
SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)

check_host_os

if [ "$#" -lt 1 ]; then
   echo "Missing argument"
   usage
   exit 1
fi

# Parse args
while (($#))
do
    case "$1" in
    help)
        usage
        exit 0
        ;;
    prompt)
        echo "Enabling user prompts."
        PROMPT_USER=1
        shift
        ;;
    amd)
        echo "Enabling AMD package filter."
        DUMP_AMD_PKGS=1
        shift
        ;;
    other)
        echo "Enabling non-AMD package dump."
        DUMP_NON_AMD_PKGS=1
        shift
        ;;
    verbose)
        echo "Enabling verbose logging."
        VERBOSE=1
        shift
        ;;
    config=*)
        CONFIG_FILE="${1#*=}"
        echo "Using configuration: $CONFIG_FILE"
        read_config "$CONFIG_FILE"
        shift
        ;;
    pkg=*)
        PACKAGES="${1#*=}"
        echo "Packages specified: $PACKAGES"
        shift
        ;;
    out=*)
        PACKAGE_REPO="${1#*=}"
        echo "Output directory: $PACKAGE_REPO"
        shift
        ;;
    *)
        echo "Unknown option: $1"
        shift
        ;;
    esac
done

# Main execution flow
echo ""
echo "Starting chroot-based DEB package pulling..."
echo "CHROOT_DIR: $CHROOT_DIR"
echo "PACKAGE_REPO: $PACKAGE_REPO"
echo "PACKAGES: $PACKAGES"
echo ""

echo "[Step 1/9] Installing chroot tools..."
install_chroot_tools

echo "[Step 2/9] Creating Ubuntu chroot..."
create_ubuntu_chroot

echo "[Step 3/9] Setting up chroot environment..."
setup_chroot_environment

echo "[Step 4/9] Installing ca-certificates in chroot..."
install_ca_certificates_in_chroot

echo "[Step 5/9] Setting up GPG in chroot..."
setup_gpg_in_chroot

echo "[Step 6/9] Configuring APT sources..."
configure_apt_sources

echo "[Step 7/9] Downloading packages in chroot..."
download_packages_in_chroot

echo "[Step 8/9] Copying packages out of chroot..."
copy_packages_out

echo "[Step 9/9] Cleaning up chroot..."
cleanup_chroot

echo ""
echo "========================================="
echo "Package pulling complete!"
echo "========================================="
echo "Log file: $PULL_CURRENT_LOG"
echo ""

exit 0
