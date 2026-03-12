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

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_EXTRACT="yes"
BUILD_INSTALLER="yes"
BUILD_UI="yes"

BUILD_DIR=../build
BUILD_DIR_UI=../build-UI

VERSION_FILE="$SCRIPT_DIR/VERSION"

INSTALLER_VERSION=
ROCM_VER=
BUILD_TAG="1"
BUILD_RUNID="99999"
BUILD_TAG_INFO=""
BUILD_INSTALLER_NAME=

AMDGPU_DKMS_FILE="../rocm-installer/component-amdgpu/amdgpu-dkms-ver.txt"
AMDGPU_DKMS_BUILD_NUM=

EXTRACT_DIR="../rocm-installer"
EXTRACT_TYPE=""
EXTRACT_ROCM="yes"
EXTRACT_AMDGPU="yes"
EXTRACT_AMDGPU_MODE="all"

# AlmaLinux 8.10 (EL8) requires specific makeself options
MAKESELF_OPT="--notemp --threads $(nproc)"
MAKESELF_OPT_CLEANUP=
MAKESELF_OPT_HEADER="--header ./rocm-makeself-header-pre.sh --help-header ../rocm-installer/VERSION"
MAKESELF_OPT_TAR=""        # EL8 does not support GNU tar format
MAKESELF_COMPRESS_MODE=""  # Compression mode (set by mscomp: dev1, dev2, etc.)
MAKESELF_OPT_COMPRESS=""   # Compression setting used by makeself
XZ_COMPRESS_LEVEL=9        # XZ compression level (1-9, lower=faster/larger, higher=slower/smaller)


###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $PROG [options]

[options}:
    help                 = Display this help information.

    config=<file>        = Load configuration from file (command-line args override config).
                           Preset configs available in config/ directory:
                           - config/nightly.config
                           - config/prerelease.config
                           - config/release.config
                           - config/dev.config

                           NOTE: When running via build-runfile-installer.sh, the same config
                           is sourced by both parent and child scripts. Each script sources
                           independently, then applies command-line overrides.

    noextract            = Disable package extraction.
    norocm               = Disable ROCm package extraction.
    noamdgpu             = Disable AMDGPU package extraction.
    noextractcontent     = Disable package extraction content. (Extract only deps and scriptlets)
    contentlist          = List all files extracted to content directories during package extraction.
    norunfile            = Disable makeself build of installer runfile.
    nogui                = Disable GUI building.
    buildtag=<tag>       = Set the build tag (default: 1).
    buildrunid=<id>      = Set the Runfile build run ID (default: 99999).
    buildtaginfo=<tag>   = Set a tag/name for the builds package pull information. (ie. pulltag-pullid)
    
    mscomp=<mode>        = Makeself compression mode (build speed vs file size):

                           Mode       Speed    Size      Compatibility    Use Case
                           ---------  -------  --------  ---------------  ------------------
                           hybrid     Slowest  Smallest  Universal        RECOMMENDED
                                      xz-9+xz-9 ~50-55%  (embedded xz)    ~7-8 GB installer
                                                                          Everything xz-9
                                                                          Max compression

                           hybridmix  Medium   ~73%      Universal        Best size + compat
                                      pigz+xz  ~8.5 GB   (embedded xz)    Tests xz-9
                                                                          No system deps

                           hybriddev  Faster   Medium    Universal        Fast compression
                                      xz-3+xz-3 ~70-75%  (embedded xz)    ~9-10 GB installer
                                                                          Everything xz-3
                                                                          4-6x faster build

                           devsmall   Slowest  Smallest  Standard (xz)    Max compression
                                      xz       ~70%      Most systems     Smallest file size
                                                                          Requires system xz

                           devmedium  Slower   ~80-85%   Universal        Balanced production
                                      pbzip2            (bzip2)           Near-xz compression

                           dev        5-6x     Larger    Universal        Development
                                      pigz -6  ~105%     (gzip)           Fast iteration

                           normal     Slow     Small     Universal        Standard default
                                      gzip -9  100%      (gzip)           Reliable baseline

                           Universal (gzip/bzip2/embedded xz): Works on all Linux including minimal installs
                           Standard (xz): Requires xz-utils on target (may not be in minimal installs)

END_USAGE
}

os_release() {
    if [[ -r  /etc/os-release ]]; then
        # shellcheck source=/dev/null
        . /etc/os-release

        DISTRO_NAME=$ID
        DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
        DISTRO_MAJOR_VER=${DISTRO_VER%.*}

        case "$ID" in
        almalinux)
            BUILD_DISTRO_PACKAGE_TYPE=rpm
            # Special handling for AlmaLinux 8.10 (ManyLinux)
            if [[ "$DISTRO_MAJOR_VER" == "10" ]]; then
                DISTRO_TAG="el10"
                BUILD_OS=el10
            elif [[ "$DISTRO_MAJOR_VER" == "9" ]]; then
                DISTRO_TAG="el9"
                BUILD_OS=el9
            elif [[ "$DISTRO_MAJOR_VER" == "8" ]]; then
                DISTRO_TAG="el8"
                BUILD_OS=el8
                echo "Detected AlmaLinux $DISTRO_VER (ManyLinux)"
                echo "Disable makeself tar options for EL8."
                MAKESELF_OPT_HEADER="--header ./rocm-makeself-header-pre.sh --help-header ../rocm-installer/VERSION"
                MAKESELF_OPT_TAR=""
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

    echo "Build running on $DISTRO_NAME $DISTRO_VER (tag: $DISTRO_TAG)."
}

format_size() {
    local bytes=$1
    local kb=$((bytes / 1024))
    local mb=$((kb / 1024))
    local gb=$((mb / 1024))

    if [[ $gb -gt 0 ]]; then
        local gb_dec=$(( (mb * 10 / 1024) % 10 ))
        echo "${gb}.${gb_dec} GB"
    elif [[ $mb -gt 0 ]]; then
        local mb_dec=$(( (kb * 10 / 1024) % 10 ))
        echo "${mb}.${mb_dec} MB"
    else
        echo "${kb} KB"
    fi
}

format_duration() {
    local duration=$1
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    echo -e "\e[36m${hours}h ${minutes}m ${seconds}s (${duration} seconds)\e[0m"
}

report_archive_time() {
    local label=$1
    local duration=$2
    echo -e "  \e[36m${label}: $(format_duration "$duration" | sed 's/\x1b\[36m//;s/\x1b\[0m//')\e[0m"
}

report_size() {
    local label=$1
    local size_info=$2
    echo -e "  \e[95m${label}: ${size_info}\e[0m"
}

show_compression_progress() {
    local archive_file=$1

    # Wait for archive file to be created
    while [[ ! -f "$archive_file" ]]; do
        sleep 0.5
    done

    # Show immediate feedback that compression has started
    printf "\r\033[K  Compressing..."

    local last_size=0
    local shown_initial=false

    # Loop until killed by parent process
    while true; do
        if [[ -f "$archive_file" ]]; then
            local current_size
            current_size=$(stat -c%s "$archive_file" 2>/dev/null || stat -f%z "$archive_file" 2>/dev/null || echo 0)

            local current_kb=$((current_size / 1024))

            # Show size as soon as we have any data
            if [[ $current_kb -gt 0 ]]; then
                if [[ $current_kb -ne $last_size ]]; then
                    printf "\r\033[K  Compressed: %s" "$(format_size "$current_size")"
                    last_size=$current_kb
                    shown_initial=true
                fi
            elif [[ "$shown_initial" == "false" ]]; then
                # Keep showing "Compressing..." while waiting for first data
                printf "\r\033[K  Compressing..."
            fi
        fi
        sleep 1
    done
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
        # shellcheck source=/dev/null
        source "$CONFIG_FILE"
        echo "Configuration loaded successfully."
        echo "Note: Command-line arguments will override config values."
        echo -------------------------------------------------------------
    fi
}

get_version() {
    i=0
    
    while IFS= read -r line; do
        case $i in
            0) INSTALLER_VERSION="$line" ;;
            1) ROCM_VER="$line" ;;
        esac
        
        i=$((i+1))
    done < "$VERSION_FILE"
}

write_version() {
    echo -------------------------------------------------------------
    echo Setting version and build info...

    get_version

    # Set the runfile installer name
    BUILD_INSTALLER_NAME="rocm-installer-$ROCM_VER-$BUILD_TAG-$BUILD_RUNID"

    # get the amdgpu-dkms build/version info
    if [ -f "$AMDGPU_DKMS_FILE" ]; then
        AMDGPU_DKMS_BUILD_NUM=$(cat "$AMDGPU_DKMS_FILE")
    fi

    echo "INSTALLER_VERSION        = $INSTALLER_VERSION"
    echo "ROCM_VER                 = $ROCM_VER"
    echo "BUILD_TAG                = $BUILD_TAG"
    echo "BUILD_RUNID              = $BUILD_RUNID"
    echo "BUILD_TAG_INFO           = $BUILD_TAG_INFO"
    echo "AMDGPU_DKMS_BUILD_NUM    = $AMDGPU_DKMS_BUILD_NUM"

    # Update the version file
    {
        echo "$INSTALLER_VERSION"
        echo "$ROCM_VER"
        echo "$BUILD_TAG"
        echo "$BUILD_RUNID"
        echo "$BUILD_TAG_INFO"
        echo "$AMDGPU_DKMS_BUILD_NUM"
    } > "$VERSION_FILE"

    echo "Installer name: $BUILD_INSTALLER_NAME"
}

print_directory_size() {
    local dir_path="$1"
    local dir_name="${2:-$(basename "$dir_path")}"
    local size_kb
    local size

    if [ -d "$dir_path" ]; then
        size_kb=$(du -sk "$dir_path" 2>/dev/null | awk '{print $1}')
        if [ -n "$size_kb" ]; then
            size=$(format_size $((size_kb * 1024)))
            echo "  Directory size: $size ($dir_name)"
        fi
    fi
}

generate_component_lists() {
    echo -------------------------------------------------------------
    echo Scanning components to build embedded lists...

    GFX_LIST=""
    COMPO_LIST=""

    local component_dir="../rocm-installer/component-rocm"

    if [ ! -d "$component_dir" ]; then
        echo "WARNING: component-rocm directory not found at: $component_dir"
        echo "GFX list will be empty."
    else
        # Extract GFX architectures (e.g., gfx94x, gfx942, gfx1030)
        # Look for patterns like gfx followed by numbers and optional letters
        local gfx_found=()
        for file in "$component_dir"/*; do
            if [[ -e "$file" ]]; then
                local filename
                filename=$(basename "$file")
                if [[ "$filename" =~ gfx[0-9]+[a-z]* ]]; then
                    gfx_found+=("${BASH_REMATCH[0]}")
                fi
            fi
        done

        # Remove duplicates and convert to space-separated list
        GFX_LIST=$(printf '%s\n' "${gfx_found[@]}" | sort -u | tr '\n' ' ' | sed 's/ *$//')
    fi

    # Component categories are fixed (defined in rocm-installer.sh)
    # These map to meta packages, not individual extracted packages
    COMPO_LIST="core core-dev dev-tools core-sdk opencl"

    echo "GFX architectures detected: ${GFX_LIST:-<none>}"
    echo "Component categories: $COMPO_LIST"
}

generate_headers() {
    echo -------------------------------------------------------------
    echo Generating makeself headers with embedded component lists...

    # Generate modern header (for non-EL8 distros)
    if [ -f "rocm-makeself-header.sh.template" ]; then
        sed -e "s|@@GFX_ARCHS_LIST@@|$GFX_LIST|g" \
            -e "s|@@COMPONENTS_LIST@@|$COMPO_LIST|g" \
            rocm-makeself-header.sh.template > rocm-makeself-header.sh
        echo "Generated: rocm-makeself-header.sh"
    else
        echo "ERROR: rocm-makeself-header.sh.template not found!"
        exit 1
    fi

    # Generate pre/EL8 header
    if [ -f "rocm-makeself-header-pre.sh.template" ]; then
        sed -e "s|@@GFX_ARCHS_LIST@@|$GFX_LIST|g" \
            -e "s|@@COMPONENTS_LIST@@|$COMPO_LIST|g" \
            rocm-makeself-header-pre.sh.template > rocm-makeself-header-pre.sh
        echo "Generated: rocm-makeself-header-pre.sh"
    else
        echo "ERROR: rocm-makeself-header-pre.sh.template not found!"
        exit 1
    fi
}

install_makeself() {
    echo ----------------------
    echo "Installing makeself..."

    # Check if makeself command is already available
    if command -v makeself &> /dev/null; then
        local makeself_version
        makeself_version=$(makeself --version)
        echo -e "\e[32mmakeself already installed\e[0m"
        echo -e "\e[32mVersion: $makeself_version\e[0m"
        return 0
    fi

    # Try to install from package manager first
    echo "Attempting to install makeself from package manager..."
    if [ "$BUILD_DISTRO_PACKAGE_TYPE" == "deb" ]; then
        $SUDO apt-get install -y makeself
    elif [ "$BUILD_DISTRO_PACKAGE_TYPE" == "rpm" ]; then
        $SUDO dnf install -y makeself
    fi

    # Check if package manager install succeeded
    if command -v makeself &> /dev/null; then
        local makeself_version
        makeself_version=$(makeself --version)
        echo -e "\e[32mmakeself installed successfully from package manager\e[0m"
        echo -e "\e[32mVersion: $makeself_version\e[0m"
        return 0
    fi

    # Package manager install failed, download and install from GitHub
    echo "Package manager install failed. Downloading makeself from GitHub..."

    local makeself_ver="2.4.5"
    local makeself_url="https://github.com/megastep/makeself/releases/download/release-$makeself_ver/makeself-$makeself_ver.run"

    # Download the makeself package
    echo "Downloading makeself package from github..."
    if ! wget -q "$makeself_url"; then
        echo -e "\e[31mmakeself package not found: $makeself_url.\e[0m"
        exit 1
    fi

    $SUDO chmod +x "makeself-$makeself_ver.run"

    # Install the makeself package
    echo "Installing makeself package..."
    bash "makeself-$makeself_ver.run"

    # Clean up
    echo "Cleaning up..."
    rm -f makeself-$makeself_ver.run

    # Add makeself to PATH
    echo "Adding makeself to PATH..."
    $SUDO ln -sf "$PWD/makeself-$makeself_ver/makeself.sh" /usr/local/bin/makeself

    echo Installing makeself...Complete
}

install_pigz() {
    echo ----------------------
    echo -e "\e[32mInstalling pigz (parallel gzip)...\e[0m"

    # Check if pigz is already installed
    if command -v pigz &> /dev/null; then
        echo "pigz is already installed: $(pigz --version 2>&1 | head -1)"
        return 0
    fi

    # Install pigz for AlmaLinux (build system)
    # Note: Creates gzip-compatible archives that decompress with standard gzip on ANY target system
    if [[ "$DISTRO_NAME" != "almalinux" ]]; then
        echo -e "\e[93mWARNING: Build system must be AlmaLinux.\e[0m"
        echo "Falling back to standard gzip compression"
        return 1
    fi

    echo "Installing pigz via dnf..."
    $SUDO dnf install -y pigz

    # Verify installation
    if command -v pigz &> /dev/null; then
        echo "pigz installed successfully: $(pigz --version 2>&1 | head -1)"
        echo "Target system requirement: gzip (universally available)"
        return 0
    else
        echo -e "\e[93mWARNING: pigz installation failed.\e[0m"
        return 1
    fi
}

install_xz() {
    echo ----------------------
    echo -e "\e[32mInstalling xz (best compression)...\e[0m"

    # Check if xz is already installed
    if command -v xz &> /dev/null; then
        echo "xz is already installed: $(xz --version 2>&1 | head -1)"
        return 0
    fi

    # Install xz for AlmaLinux (build system)
    # Note: Creates xz-compressed archives that decompress with standard xz on ANY target system
    if [[ "$DISTRO_NAME" != "almalinux" ]]; then
        echo -e "\e[93mWARNING: Build system must be AlmaLinux.\e[0m"
        echo "Falling back to standard gzip compression"
        return 1
    fi

    echo "Installing xz via dnf..."
    $SUDO dnf install -y xz

    # Verify installation
    if command -v xz &> /dev/null; then
        echo "xz installed successfully: $(xz --version 2>&1 | head -1)"
        echo "Target system requirement: xz (universally available)"
        return 0
    else
        echo -e "\e[93mWARNING: xz installation failed.\e[0m"
        return 1
    fi
}

install_pbzip2() {
    echo ----------------------
    echo -e "\e[32mInstalling pbzip2 (parallel bzip2)...\e[0m"

    # Check if pbzip2 is already installed
    if command -v pbzip2 &> /dev/null; then
        echo "pbzip2 is already installed: $(pbzip2 --version 2>&1 | head -1)"
        return 0
    fi

    # Install pbzip2 for AlmaLinux (build system)
    # Note: Creates bzip2-compatible archives that decompress with standard bzip2 on ANY target system
    if [[ "$DISTRO_NAME" != "almalinux" ]]; then
        echo -e "\e[93mWARNING: Build system must be AlmaLinux.\e[0m"
        echo "Falling back to standard gzip compression"
        return 1
    fi

    echo "Installing pbzip2 via dnf..."
    $SUDO dnf install -y pbzip2

    # Verify installation
    if command -v pbzip2 &> /dev/null; then
        echo "pbzip2 installed successfully: $(pbzip2 --version 2>&1 | head -1)"
        echo "Target system requirement: bzip2 (universally available)"
        return 0
    else
        echo -e "\e[93mWARNING: pbzip2 installation failed.\e[0m"
        return 1
    fi
}

install_ncurses_deb() {
    echo Installing ncurses libraries...

    # Check if ncurses development libraries are already installed
    if dpkg -l libncurses5-dev 2>/dev/null | grep -q "^ii" && \
       dpkg -l libncurses-dev 2>/dev/null | grep -q "^ii"; then
        echo "ncurses development libraries already installed"
    else
        echo "Installing ncurses development libraries"
        $SUDO apt-get install -y libncurses5-dev libncurses-dev
    fi

    # Verify static libraries exist
    if [ ! -f /usr/lib/x86_64-linux-gnu/libncurses.a ]; then
        echo "WARNING: Static ncurses library not found. Build will use dynamic linking."
        echo "Location checked: /usr/lib/x86_64-linux-gnu/libncurses.a"
    else
        echo "SUCCESS: Static ncurses library found: /usr/lib/x86_64-linux-gnu/libncurses.a"
    fi

    echo Installing ncurses libraries...Complete
}

install_ncurses_el() {
    echo Installing ncurses libraries...

    # Check if ncurses-devel is already installed
    if rpm -q ncurses-devel > /dev/null 2>&1; then
        echo "ncurses-devel already installed"
    else
        echo "Installing ncurses-devel"
        $SUDO dnf install -y ncurses-devel
    fi

    # For AlmaLinux 8, install ncurses-static from devel repo
    if [[ $DISTRO_NAME == "almalinux" ]] && [[ $DISTRO_VER == 8* ]]; then
        # Check if ncurses-static is already installed
        if rpm -q ncurses-static > /dev/null 2>&1; then
            echo "ncurses-static already installed"
        else
            echo "Installing ncurses-static for AlmaLinux 8..."

            # Create AlmaLinux Devel repository configuration
            echo "Creating AlmaLinux Devel repository configuration..."
            $SUDO tee /etc/yum.repos.d/almalinux-devel.repo > /dev/null <<'EOF'
[devel]
name=AlmaLinux $releasever - Devel
baseurl=https://repo.almalinux.org/almalinux/$releasever/devel/$basearch/os/
enabled=1
gpgcheck=1
countme=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-AlmaLinux
metadata_expire=86400
enabled_metadata=1
EOF

            echo "Devel repository configuration created."

            # Force metadata refresh
            echo "Refreshing repository metadata..."
            $SUDO dnf clean metadata
            $SUDO dnf makecache

            # Check if package is now available
            echo "Checking if ncurses-static is available..."
            dnf list ncurses-static || echo "WARNING: ncurses-static not found in package lists"

            # Install from devel repository
            echo "Installing ncurses-static from devel repository..."
            $SUDO dnf install -y ncurses-static || {
                echo "ERROR: Failed to install ncurses-static"
                return 1
            }

            # Verify installation
            if rpm -q ncurses-static >/dev/null 2>&1; then
                echo "SUCCESS: ncurses-static installed from devel repository"
            else
                echo "ERROR: ncurses-static package not found after installation"
                return 1
            fi
        fi
    else
        # For other EL distros, check if ncurses-static is already installed
        if rpm -q ncurses-static > /dev/null 2>&1; then
            echo "ncurses-static already installed"
        else
            echo "Installing ncurses-static..."
            $SUDO dnf install -y ncurses-static || echo "WARNING: ncurses-static not available"
        fi
    fi

    # Verify static libraries
    if [ ! -f /usr/lib64/libncurses.a ]; then
        echo "ERROR: Static ncurses library not found after installation."
        echo "Location checked: /usr/lib64/libncurses.a"
        echo "Build will fail. Please install ncurses-static manually."
        return 1
    else
        echo "SUCCESS: Static ncurses library found: /usr/lib64/libncurses.a"
    fi

    echo Installing ncurses libraries...Complete
}

install_tools_deb() {
    echo Installing DEB tools...

    # Define required tools to check (command names)
    local required_cmds=(wget cmake gcc g++ ar)

    # Define packages to install (package names)
    local required_pkgs=(wget cmake gcc g++ binutils)

    # Check if all required tools are already installed
    local all_installed=1
    for cmd in "${required_cmds[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            all_installed=0
            break
        fi
    done

    if [ $all_installed -eq 1 ]; then
        echo "All core build tools are already installed"
    else
        # One or more tools missing, install all
        echo "Installing core build tools: ${required_pkgs[*]}"
        $SUDO apt-get install -y "${required_pkgs[@]}"
    fi

    # Install ncurses libraries (only if UI build is enabled)
    if [ "$BUILD_UI" == "yes" ]; then
        install_ncurses_deb
    else
        echo "Skipping ncurses installation (GUI build disabled)"
    fi

    # Install makeself for .run creation (only if runfile build is enabled)
    if [ "$BUILD_INSTALLER" == "yes" ]; then
        install_makeself

        # Check the version of makeself and enable cleanup script support if >= 2.4.2
        makeself_version_min=2.4.2
        makeself_version=$(makeself --version)
        makeself_version=${makeself_version#Makeself version }

        if [[ "$(printf '%s\n' "$makeself_version_min" "$makeself_version" | sort -V | head -n1)" = "$makeself_version_min" ]]; then
            MAKESELF_OPT_CLEANUP+="--cleanup ../rocm-installer/cleanup-install.sh"
            echo Enabling cleanup script support.
        fi
    else
        echo "Skipping makeself installation (runfile build disabled)"
    fi

    echo Installing DEB tools...Complete
}

install_tools_el(){
    echo Installing EL tools...

    # Define required tools to check (command names)
    local required_cmds=(wget ar tar rpmbuild cpio dpkg cmake gcc g++)

    # Define packages to install (package names)
    local required_pkgs=(wget binutils tar rpm-build cpio dpkg cmake gcc gcc-c++)

    # Check if all required tools are already installed
    local all_installed=1
    for cmd in "${required_cmds[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            all_installed=0
            break
        fi
    done

    if [ $all_installed -eq 1 ]; then
        echo "All core build tools are already installed"
    else
        # One or more tools missing, install all
        echo "Installing core build tools: ${required_pkgs[*]}"
        $SUDO dnf install -y "${required_pkgs[@]}"
    fi

    # Install ncurses libraries (only if UI build is enabled)
    if [ "$BUILD_UI" == "yes" ]; then
        install_ncurses_el
    else
        echo "Skipping ncurses installation (GUI build disabled)"
    fi

    if [[ $DISTRO_NAME == "amzn" ]]; then
        # Amazon Linux may need additional packages
        if ! command -v bzip2 &> /dev/null; then
            $SUDO dnf install -y tar bzip2
        fi
    fi

    # Install makeself for .run creation (only if runfile build is enabled)
    if [ "$BUILD_INSTALLER" == "yes" ]; then
        install_makeself

        # Check the version of makeself and enable cleanup script support if >= 2.4.2
        makeself_version_min=2.4.2
        makeself_version=$(makeself --version)
        makeself_version=${makeself_version#Makeself version }

        if [[ "$(printf '%s\n' "$makeself_version_min" "$makeself_version" | sort -V | head -n1)" = "$makeself_version_min" ]]; then
            MAKESELF_OPT_CLEANUP+="--cleanup ../rocm-installer/cleanup-install.sh"
            echo Enabling cleanup script support.
        fi
    else
        echo "Skipping makeself installation (runfile build disabled)"
    fi

    echo Installing EL tools...Complete
}

install_tools() {
    echo -------------------------------------------------------------
    echo "Installing tools for $DISTRO_NAME $DISTRO_VER..."

    if [ $BUILD_DISTRO_PACKAGE_TYPE == "deb" ]; then
        install_tools_deb
    elif [ $BUILD_DISTRO_PACKAGE_TYPE == "rpm" ]; then
        install_tools_el
    else
        echo "ERROR: Invalid Distro Package Type: $BUILD_DISTRO_PACKAGE_TYPE"
        exit 1
    fi

    echo Installing tools...Complete
}

configure_compression() {
    echo -------------------------------------------------------------
    echo Configuring makeself compression...

    case "$MAKESELF_COMPRESS_MODE" in
        hybrid)
            # Hybrid: everything=xz-9, embedded xz-static, maximum compression
            HYBRID_COMPRESSION="yes"
            HYBRID_ALL_XZ="yes"
            XZ_COMPRESS_LEVEL=9
            MAKESELF_OPT_COMPRESS="--nocomp"
            echo "Compression: Hybrid (everything compressed with xz level 9)"
            echo "  - Main content: xz-9 compressed (6-8:1 ratio, best compression)"
            echo "  - Test packages: xz-9 compressed (12-15:1 ratio)"
            ;;
        hybridmix)
            # Hybrid Mix: tests=xz, main=pigz, embedded xz-static, no double-compression
            HYBRID_COMPRESSION="yes"
            MAKESELF_OPT_COMPRESS="--nocomp"
            echo "Compression: Hybrid Mix (tests=xz, main=pigz, no makeself compression)"
            echo "  - Main content: pigz-compressed (4:1 ratio, fast)"
            echo "  - Test packages: xz-compressed (12-15:1 ratio)"
            ;;
        hybriddev)
            # Hybrid Dev: everything=xz-3, embedded xz-static, fast compression
            HYBRID_COMPRESSION="yes"
            HYBRID_ALL_XZ="yes"
            XZ_COMPRESS_LEVEL=3
            MAKESELF_OPT_COMPRESS="--nocomp"
            echo "Compression: Hybrid Dev (everything compressed with xz level 3)"
            echo "  - Main content: xz-3 compressed (4-5:1 ratio, 4-6x faster than xz-9)"
            echo "  - Test packages: xz-3 compressed (8-10:1 ratio, 4-6x faster than xz-9)"
            ;;
        devsmall)
            # Install and use xz (best compression, slowest build)
            # SAFE: xz-compatible, works on most target systems
            if install_xz; then
                MAKESELF_OPT_COMPRESS="--xz"
                echo "Compression: XZ (best compression, standard xz-compatible)"
            else
                MAKESELF_OPT_COMPRESS=""
                echo "Compression: Standard Gzip (xz not available, universal)"
            fi
            ;;
        devmedium)
            # Install and use pbzip2 (parallel bzip2, better compression than gzip)
            # SAFE: bzip2-compatible, works on all target systems
            if install_pbzip2; then
                MAKESELF_OPT_COMPRESS="--pbzip2"
                echo "Compression: Pbzip2 (parallel bzip2, near-xz compression, universal)"
            else
                MAKESELF_OPT_COMPRESS="--bzip2"
                echo "Compression: Standard Bzip2 (pbzip2 not available, universal)"
            fi
            ;;
        dev)
            # Install and use pigz with compression level 6 (balanced)
            # SAFE: gzip-compatible, works on all target systems
            if install_pigz; then
                MAKESELF_OPT_COMPRESS="--pigz --complevel 6"
                echo "Compression: Pigz level 6 (fast, universal gzip-compatible)"
            else
                MAKESELF_OPT_COMPRESS="--complevel 6"
                echo "Compression: Gzip level 6 (pigz not available, universal)"
            fi
            ;;
        normal)
            # Explicit normal: standard gzip with level 9 (maximum compression)
            # SAFE: Universal compatibility
            MAKESELF_OPT_COMPRESS=""
            echo "Compression: Gzip level 9 (normal, universal)"
            ;;
        "")
            # No argument: standard gzip with level 9 (maximum compression)
            # SAFE: Universal compatibility
            MAKESELF_OPT_COMPRESS=""
            echo "Compression: Gzip level 9 (normal, universal)"
            ;;
        *)
            echo -e "\e[31mERROR: Invalid compression mode: $MAKESELF_COMPRESS_MODE\e[0m"
            exit 1
            ;;
    esac
}

extract_rocm_packages_deb() {
    echo "Extracting ROCm DEB packages for $BUILD_OS..."

    # Extract all ROCm DEB packages (common and gfx-specific)
    # The extractor script will auto-detect all packages-rocm*-deb directories
    echo "Extracting ROCm DEB packages (common and gfx-specific)..."

    if [ $BUILD_DISTRO_PACKAGE_TYPE == "rpm" ]; then
        # On RPM-based systems, use nodpkg extractor and extract to separate directory
        echo "Using nodpkg extractor for RPM-based system (nocontent mode)"
        PACKAGE_ROCM_DIR="$PWD/packages-rocm-deb" EXTRACT_FORMAT=deb ./package-extractor-debs-nodpkg.sh rocm ext-rocm="../rocm-installer/component-rocm-deb" nocontent
        extract_status=$?
    else
        # On DEB-based systems, use standard extractor
        PACKAGE_ROCM_DIR="$PWD/packages-rocm-deb" EXTRACT_FORMAT=deb ./package-extractor-debs.sh rocm ext-rocm="../rocm-installer/component-rocm" $EXTRACT_TYPE
        extract_status=$?
    fi

    if [[ $extract_status -ne 0 ]]; then
        echo -e "\e[31mFailed extraction of ROCm DEB packages.\e[0m"
        exit 1
    fi

    echo "ROCm DEB package extraction complete."
    if [ $BUILD_DISTRO_PACKAGE_TYPE == "rpm" ]; then
        print_directory_size "../rocm-installer/component-rocm-deb" "component-rocm-deb"
    else
        print_directory_size "../rocm-installer/component-rocm" "component-rocm"
    fi
}

extract_amdgpu_packages_deb() {
    echo "Extracting AMDGPU DEB packages for $BUILD_OS (tag: $DISTRO_TAG)..."

    # AMDGPU packages are stored in subdirectories: packages-amdgpu/<DISTRO_TAG>
    AMDGPU_PKG_DIR="packages-amdgpu/${DISTRO_TAG}"

    # Verify AMDGPU package directory exists
    if [ ! -d "$AMDGPU_PKG_DIR" ]; then
        echo -e "\e[31mERROR: $AMDGPU_PKG_DIR directory not found!\e[0m"
        echo "Please run setup-installer.sh first to download AMDGPU packages."
        exit 1
    fi

    # Extract the AMDGPU packages to component-amdgpu/<DISTRO_TAG>
    if ! PACKAGE_AMDGPU_DIR="$PWD/$AMDGPU_PKG_DIR" EXTRACT_FORMAT=deb ./package-extractor-debs.sh amdgpu ext-amdgpu="${EXTRACT_DIR}/component-amdgpu/${DISTRO_TAG}"; then
        echo -e "\e[31mFailed extraction of AMDGPU DEB packages.\e[0m"
        exit 1
    fi

    echo "AMDGPU DEB package extraction complete."
    print_directory_size "${EXTRACT_DIR}/component-amdgpu/${DISTRO_TAG}" "component-amdgpu/${DISTRO_TAG}"
}

extract_rocm_packages_rpm() {
    echo "Extracting ROCm RPM packages for $BUILD_OS..."

    # Extract all ROCm RPM packages (common and gfx-specific)
    # The extractor script will auto-detect all packages-rocm*-rpm directories
    echo "Extracting ROCm RPM packages (common and gfx-specific)..."
    if ! PACKAGE_ROCM_DIR="$PWD/packages-rocm-rpm" EXTRACT_FORMAT=rpm ./package-extractor-rpms.sh rocm ext-rocm="../rocm-installer" $EXTRACT_TYPE; then
        echo -e "\e[31mFailed extraction of ROCm RPM packages.\e[0m"
        exit 1
    fi

    echo "ROCm RPM package extraction complete."
    print_directory_size "../rocm-installer/component-rocm" "component-rocm"
}

extract_amdgpu_packages_rpm() {
    echo "Extracting AMDGPU RPM packages for $BUILD_OS (tag: $DISTRO_TAG)..."

    # AMDGPU packages are stored in subdirectories: packages-amdgpu/<DISTRO_TAG>
    AMDGPU_PKG_DIR="packages-amdgpu/${DISTRO_TAG}"

    # Verify AMDGPU package directory exists
    if [ ! -d "$AMDGPU_PKG_DIR" ]; then
        echo -e "\e[31mERROR: $AMDGPU_PKG_DIR directory not found!\e[0m"
        echo "Please run setup-installer.sh first to download AMDGPU packages."
        exit 1
    fi

    # Extract the AMDGPU packages to component-amdgpu/<DISTRO_TAG>
    if ! PACKAGE_AMDGPU_DIR="$PWD/$AMDGPU_PKG_DIR" EXTRACT_FORMAT=rpm ./package-extractor-rpms.sh amdgpu ext-amdgpu="${EXTRACT_DIR}/component-amdgpu/${DISTRO_TAG}"; then
        echo -e "\e[31mFailed extraction of AMDGPU RPM packages.\e[0m"
        exit 1
    fi

    echo "AMDGPU RPM package extraction complete."
    print_directory_size "${EXTRACT_DIR}/component-amdgpu/${DISTRO_TAG}" "component-amdgpu/${DISTRO_TAG}"
}

extract_amdgpu_packages_all() {
    echo "Extracting AMDGPU packages for all distros..."

    # AMDGPU packages are stored in subdirectories: packages-amdgpu/<distro_tag>
    # Find all subdirectories in packages-amdgpu/
    if [ ! -d "packages-amdgpu" ]; then
        echo -e "\e[31mERROR: packages-amdgpu directory not found!\e[0m"
        echo "Please run setup-installer.sh amdgpu-mode=all first to download AMDGPU packages for all distros."
        exit 1
    fi

    # Find all distro subdirectories
    local amdgpu_dirs=(packages-amdgpu/*/)

    if [ ${#amdgpu_dirs[@]} -eq 0 ] || [ ! -d "${amdgpu_dirs[0]}" ]; then
        echo -e "\e[31mERROR: No distro subdirectories found in packages-amdgpu/!\e[0m"
        echo "Please run setup-installer.sh amdgpu-mode=all first to download AMDGPU packages for all distros."
        exit 1
    fi

    echo "Found ${#amdgpu_dirs[@]} AMDGPU package directories to extract"

    # Extract packages from each distro-specific subdirectory
    for amdgpu_dir in "${amdgpu_dirs[@]}"; do
        # Remove trailing slash
        amdgpu_dir="${amdgpu_dir%/}"

        if [ -d "$amdgpu_dir" ]; then
            # Extract distro tag from directory name (e.g., packages-amdgpu/el8 -> el8)
            local distro_tag
            distro_tag="$(basename "$amdgpu_dir")"

            echo "Extracting AMDGPU packages from $amdgpu_dir (tag: $distro_tag)..."

            # Extract the AMDGPU packages to component-amdgpu/<distro_tag>
            if ! PACKAGE_AMDGPU_DIR="$PWD/$amdgpu_dir" ./package-extractor-all.sh amdgpu pkgs-amdgpu="$PWD/$amdgpu_dir" ext-amdgpu="${EXTRACT_DIR}/component-amdgpu/${distro_tag}"; then
                echo -e "\e[31mFailed extraction of AMDGPU packages from $amdgpu_dir.\e[0m"
                exit 1
            fi

            print_directory_size "${EXTRACT_DIR}/component-amdgpu/${distro_tag}" "component-amdgpu/${distro_tag}"
        fi
    done

    echo "AMDGPU-all package extraction complete."
    print_directory_size "${EXTRACT_DIR}/component-amdgpu" "component-amdgpu (total)"
}

extract_packages_rocm() {
    echo "Extracting ROCm packages..."

    if [ $EXTRACT_ROCM != "yes" ]; then
        echo "ROCm package extraction disabled."
        return
    fi

    # Check for RPM packages and extract if present
    if [ -d "packages-rocm-rpm" ]; then
        extract_rocm_packages_rpm
    fi

    # Check for DEB packages and extract if present
    if [ -d "packages-rocm-deb" ]; then
        extract_rocm_packages_deb
        echo disable DEB extraction.
    fi
}

extract_packages_amdgpu() {
    echo "Extracting AMDGPU packages..."

    # Check if AMDGPU extraction is enabled
    if [ $EXTRACT_AMDGPU == "yes" ]; then
        # Extract AMDGPU packages based on mode
        if [ $EXTRACT_AMDGPU_MODE == "all" ]; then
            # Extract AMDGPU-all packages (all distros)
            extract_amdgpu_packages_all
        else
            # Extract distro-specific AMDGPU packages
            if [ $BUILD_DISTRO_PACKAGE_TYPE == "deb" ]; then
                extract_amdgpu_packages_deb
            elif [ $BUILD_DISTRO_PACKAGE_TYPE == "rpm" ]; then
                extract_amdgpu_packages_rpm
            else
                echo -e "\e[31mERROR: Invalid Distro Package Type: $BUILD_DISTRO_PACKAGE_TYPE\e[0m"
                exit 1
            fi
        fi
    else
        echo "AMDGPU package extraction disabled."
    fi
}

extract_packages() {
    echo -------------------------------------------------------------
    echo Running Package Extractor...

    if [ $BUILD_EXTRACT == "yes" ]; then
        pushd ../package-extractor || exit

        # Extract ROCm packages
        extract_packages_rocm

        # Extract AMDGPU packages
        extract_packages_amdgpu

        popd || exit
    else
        echo Extract Packages disabled.
    fi

    echo Running Package Extractor...Complete
}

compress_tests() {
    echo -------------------------------------------------------------
    echo "Compressing tests..."
    echo -------------------------------------------------------------

    local INSTALLER_DIR="../rocm-installer"
    local TESTS_ARCHIVE="tests.tar.xz"
    local XZ_STATIC_SRC="$SCRIPT_DIR/tools/xz-static"
    local XZ_STATIC_DEST="$INSTALLER_DIR/bin/xz-static"

    # Create bin directory for embedded xz
    mkdir -p "$INSTALLER_DIR/bin"

    # Check for static xz binary and validate it's truly static
    local need_rebuild=0

    if [[ ! -f "$XZ_STATIC_SRC" ]]; then
        echo "xz-static not found, will build from source..."
        need_rebuild=1
    elif ldd "$XZ_STATIC_SRC" 2>&1 | grep -qv "not a dynamic executable"; then
        echo "xz-static exists but is dynamically linked, rebuilding for static..."
        need_rebuild=1
    fi

    if [[ $need_rebuild -eq 1 ]]; then
        echo ""
        if [[ -f "$SCRIPT_DIR/tools/build-xz-static.sh" ]]; then
            cd "$SCRIPT_DIR/tools" || exit 1

            # Remove old binary if it exists
            rm -f xz-static

            # Set environment variable to indicate non-interactive build
            export XZ_STATIC_NONINTERACTIVE=1

            if ./build-xz-static.sh; then
                echo ""
                echo -e "\e[32mxz-static built successfully.\e[0m"
                cd - >/dev/null || exit
            else
                echo -e "\e[31mERROR: Failed to build xz-static\e[0m"
                cd - >/dev/null || exit
                exit 1
            fi
        else
            echo -e "\e[31mERROR: build-xz-static.sh script not found\e[0m"
            exit 1
        fi
        echo ""
    fi

    # Copy static xz binary to installer
    echo "Embedding static xz binary..."
    cp "$XZ_STATIC_SRC" "$XZ_STATIC_DEST"
    chmod +x "$XZ_STATIC_DEST"

    local xz_bytes
    local xz_size

    xz_bytes=$(stat -c%s "$XZ_STATIC_DEST" 2>/dev/null || stat -f%z "$XZ_STATIC_DEST" 2>/dev/null)
    xz_size=$(format_size "$xz_bytes")
    echo "  xz-static embedded: $xz_size"

    # Verify it's actually static
    if ldd "$XZ_STATIC_DEST" 2>&1 | grep -q "not a dynamic executable"; then
        echo -e "  \e[32mBinary is statically linked (no dependencies).\e[0m"
    else
        echo -e "\e[93m  WARNING: xz binary may have dynamic dependencies.\e[0m"
    fi

    # Change to installer directory
    cd "$INSTALLER_DIR" || exit 1

    # Find all test component directories
    echo ""
    echo "Identifying test components..."

    local test_dirs=()
    for arch_dir in component-rocm/*/; do
        if [[ -d "$arch_dir" ]]; then
            for component in "$arch_dir"*test*/; do
                if [[ -d "$component" ]]; then
                    test_dirs+=("${component%/}")
                fi
            done
        fi
    done

    if [[ ${#test_dirs[@]} -eq 0 ]]; then
        echo "  No test packages found, skipping test compression."
        cd - >/dev/null || exit
        return 0
    fi

    echo "  Found ${#test_dirs[@]} test component(s):"
    printf '    - %s\n' "${test_dirs[@]}"

    # Calculate uncompressed size
    echo ""
    local test_size_kb
    local test_size_mb

    test_size_kb=$(du -sk "${test_dirs[@]}" 2>/dev/null | awk '{s+=$1} END {print s}')
    test_size_mb=$((test_size_kb / 1024))

    local test_size_gb_int=$((test_size_mb / 1024))
    local test_size_gb_dec=$(( (test_size_mb * 100 / 1024) % 100 ))

    # Remove any old test archives from previous builds
    echo ""
    echo "Removing old test archives..."
    rm -f tests.tar.xz 2>/dev/null
    echo -e "  \e[93mRemoved old archives.\e[0m"

    # Create xz-compressed tar archive of all test components
    echo ""
    echo "Creating xz-compressed test archive (level $XZ_COMPRESS_LEVEL)..."
    echo "  This may take several minutes..."

    local start_time
    start_time=$(date +%s)

    # Start progress monitor in background
    show_compression_progress "$TESTS_ARCHIVE" &
    local progress_pid=$!

    # Ensure progress monitor is killed on exit (Ctrl-C, error, or normal exit)
    trap 'kill $progress_pid 2>/dev/null; wait $progress_pid 2>/dev/null' EXIT INT TERM

    # Use relative paths for tar - compress with xz
    if tar -cf - "${test_dirs[@]}" 2>/dev/null | \
       xz "-$XZ_COMPRESS_LEVEL" -T"$(nproc)" --verbose 2>/tmp/xz-compression.log > "$TESTS_ARCHIVE"; then

        # Stop progress monitor
        kill $progress_pid 2>/dev/null
        wait $progress_pid 2>/dev/null
        trap - EXIT INT TERM
        echo ""
        echo -e "  \e[32mCompressed with xz (level $XZ_COMPRESS_LEVEL, $(nproc) threads)\e[0m"

        local end_time
        end_time=$(date +%s)

        local duration=$((end_time - start_time))

        if [[ ! -f "$TESTS_ARCHIVE" || ! -s "$TESTS_ARCHIVE" ]]; then
            echo -e "\e[31mERROR: Test archive creation failed or is empty\e[0m"
            cd - >/dev/null || exit
            exit 1
        fi

        report_archive_time "Archive created in" "$duration"
    else
        kill $progress_pid 2>/dev/null
        wait $progress_pid 2>/dev/null
        trap - EXIT INT TERM
        echo ""
        echo -e "\e[31mERROR: Failed to create test archive\e[0m"
        cd - >/dev/null || exit
        exit 1
    fi

    # Get archive size and calculate compression ratio
    local archive_size_kb
    local archive_size_mb

    archive_size_kb=$(du -k "$TESTS_ARCHIVE" | awk '{print $1}')
    archive_size_mb=$((archive_size_kb / 1024))

    local archive_size_gb_int=$((archive_size_mb / 1024))
    local archive_size_gb_dec=$(( (archive_size_mb * 100 / 1024) % 100 ))

    local ratio=$((test_size_kb / archive_size_kb))
    local reduction=$(( 100 - (archive_size_kb * 100 / test_size_kb) ))

    echo ""
    echo "Compression results:"
    echo "  Archive: $TESTS_ARCHIVE"
    report_size "Size" "${test_size_gb_int}.${test_size_gb_dec} GB -> ${archive_size_gb_int}.${archive_size_gb_dec} GB (ratio ${ratio}:1)"
    echo "  Size reduction: ${reduction}%"

    # Remove test directories from component tree
    echo ""
    echo "Removing uncompressed test components..."
    local removed_count=0
    for test_dir in "${test_dirs[@]}"; do
        if [[ -d "$test_dir" ]]; then
            rm -rf "$test_dir"
            ((removed_count++))
        fi
    done
    echo -e "  \e[93mRemoved $removed_count test component directories.\e[0m"

    cd - >/dev/null || exit
    echo ""
    echo "Test compression complete."
    echo "-------------------------------------------------------------"
}

compress_components() {
    echo "-------------------------------------------------------------"
    echo "Compressing components..."
    echo "-------------------------------------------------------------"

    local INSTALLER_DIR="../rocm-installer"
    cd "$INSTALLER_DIR" || exit 1

    # Find all component directories to compress (exclude test components already compressed)
    local component_dirs=()

    # Add component-rocm if it exists
    if [[ -d "component-rocm" ]]; then
        component_dirs+=("component-rocm")
    fi

    # Add component-amdgpu if it exists
    if [[ -d "component-amdgpu" ]]; then
        component_dirs+=("component-amdgpu")
    fi

    # Add component-rocm-deb if it exists (for cross-distro builds)
    if [[ -d "component-rocm-deb" ]]; then
        component_dirs+=("component-rocm-deb")
    fi

    if [[ ${#component_dirs[@]} -eq 0 ]]; then
        echo "  No component directories found, skipping main content compression."
        cd - >/dev/null || exit
        return 0
    fi

    # Calculate size of all component directories
    local main_size_kb
    local main_size_mb

    main_size_kb=$(du -sk "${component_dirs[@]}" 2>/dev/null | awk '{s+=$1} END {print s}')
    main_size_mb=$((main_size_kb / 1024))

    local main_size_gb_int=$((main_size_mb / 1024))
    local main_size_gb_dec=$(( (main_size_mb * 100 / 1024) % 100 ))
    echo "  Components to compress: ${component_dirs[*]}"

    # Remove any old component archives from previous builds
    echo ""
    echo "Removing old component archives..."
    rm -f components.tar.gz components.tar.xz 2>/dev/null
    echo -e "  \e[93mRemoved old archives.\e[0m"

    # Determine compression method and archive name based on mode
    local MAIN_ARCHIVE
    local COMPRESSION_TYPE

    if [[ "${HYBRID_ALL_XZ:-no}" == "yes" ]]; then
        MAIN_ARCHIVE="components.tar.xz"
        COMPRESSION_TYPE="xz"
    else
        MAIN_ARCHIVE="components.tar.gz"
        COMPRESSION_TYPE="pigz"
    fi

    # Compress all component directories
    echo ""
    local start_time
    start_time=$(date +%s)

    # Start progress monitor in background
    show_compression_progress "$MAIN_ARCHIVE" &
    local progress_pid=$!

    # Ensure progress monitor is killed on exit (Ctrl-C, error, or normal exit)
    trap 'kill $progress_pid 2>/dev/null; wait $progress_pid 2>/dev/null' EXIT INT TERM

    if [[ "$COMPRESSION_TYPE" == "xz" ]]; then
        echo "Creating xz-compressed components archive (level $XZ_COMPRESS_LEVEL)..."
        echo "  This may take several minutes..."

        if tar -cf - "${component_dirs[@]}" 2>/dev/null | \
           xz "-$XZ_COMPRESS_LEVEL" -T"$(nproc)" --verbose 2>/tmp/components-xz.log > "$MAIN_ARCHIVE"; then
            # Stop progress monitor
            kill $progress_pid 2>/dev/null
            wait $progress_pid 2>/dev/null
            trap - EXIT INT TERM
            echo ""
            echo -e "  \e[32mCompressed with xz (level $XZ_COMPRESS_LEVEL, $(nproc) threads)\e[0m"
        else
            kill $progress_pid 2>/dev/null
            wait $progress_pid 2>/dev/null
            trap - EXIT INT TERM
            echo ""
            echo -e "\e[31mERROR: Failed to create components archive with xz\e[0m"
            cd - >/dev/null || exit
            exit 1
        fi
    else
        echo "Creating pigz-compressed components archive..."
        echo "  This may take several minutes..."

        if command -v pigz &> /dev/null; then
            # Use pigz for parallel compression
            if tar -cf - "${component_dirs[@]}" 2>/dev/null | \
               pigz -6 -p "$(nproc)" > "$MAIN_ARCHIVE"; then
                # Stop progress monitor
                kill $progress_pid 2>/dev/null
                wait $progress_pid 2>/dev/null
                trap - EXIT INT TERM
                echo ""
                echo -e "  \e[32mCompressed with pigz (level 6, $(nproc) threads)\e[0m"
            else
                kill $progress_pid 2>/dev/null
                wait $progress_pid 2>/dev/null
                trap - EXIT INT TERM
                echo ""
                echo -e "\e[31mERROR: Failed to create components archive with pigz\e[0m"
                cd - >/dev/null || exit
                exit 1
            fi
        else
            # Fallback to gzip
            if tar -czf "$MAIN_ARCHIVE" "${component_dirs[@]}" 2>/dev/null; then
                # Stop progress monitor
                kill $progress_pid 2>/dev/null
                wait $progress_pid 2>/dev/null
                trap - EXIT INT TERM
                echo ""
                echo -e "  \e[32mCompressed with gzip\e[0m"
            else
                kill $progress_pid 2>/dev/null
                wait $progress_pid 2>/dev/null
                trap - EXIT INT TERM
                echo ""
                echo -e "\e[31mERROR: Failed to create components archive with gzip\e[0m"
                cd - >/dev/null || exit
                exit 1
            fi
        fi
    fi

    local end_time
    end_time=$(date +%s)

    local duration=$((end_time - start_time))

    if [[ ! -f "$MAIN_ARCHIVE" || ! -s "$MAIN_ARCHIVE" ]]; then
        echo -e "\e[31mERROR: Main archive creation failed or is empty\e[0m"
        cd - >/dev/null || exit
        exit 1
    fi

    report_archive_time "Archive created in" "$duration"

    # Get archive size and calculate compression ratio
    local archive_size_kb
    local archive_size_mb

    archive_size_kb=$(du -k "$MAIN_ARCHIVE" | awk '{print $1}')
    archive_size_mb=$((archive_size_kb / 1024))

    local archive_size_gb_int=$((archive_size_mb / 1024))
    local archive_size_gb_dec=$(( (archive_size_mb * 100 / 1024) % 100 ))

    local ratio=$((main_size_kb / archive_size_kb))
    local reduction=$(( 100 - (archive_size_kb * 100 / main_size_kb) ))

    echo ""
    echo "Compression results:"
    echo "  Archive: $MAIN_ARCHIVE"
    report_size "Size" "${main_size_gb_int}.${main_size_gb_dec} GB -> ${archive_size_gb_int}.${archive_size_gb_dec} GB (ratio ${ratio}:1)"
    echo "  Size reduction: ${reduction}%"

    # Remove uncompressed component directories
    echo ""
    echo "Removing uncompressed component directories..."
    local removed_count=0
    for comp_dir in "${component_dirs[@]}"; do
        if [[ -d "$comp_dir" ]]; then
            rm -rf "$comp_dir"
            ((removed_count++))
        fi
    done
    echo -e "  \e[93mRemoved $removed_count component directories.\e[0m"

    cd - >/dev/null || exit
    echo ""
    echo "Components compression complete."
    echo "-------------------------------------------------------------"
}

build_UI() {
    echo -------------------------------------------------------------
    echo Building Installer UI...

    if [ $BUILD_UI == "yes" ]; then
        if [ -d $BUILD_DIR_UI ]; then
            echo Removing UI Build directory.
            $SUDO rm -r $BUILD_DIR_UI
        fi

        echo Creating $BUILD_DIR_UI directory.
        mkdir $BUILD_DIR_UI

        pushd $BUILD_DIR_UI || exit
            # UI now reads VERSION file at runtime - no version parameters needed
            cmake ../build-installer
            if ! make; then
                echo -e "\e[31mFailed GUI build.\e[0m"
                exit 1
            fi

            # Verify static linking worked
            echo "Checking UI binary dependencies:"
            if ldd rocm_ui | grep -E "ncurses|menu|form|tinfo"; then
                echo "WARNING: UI binary has ncurses dynamic dependencies"
            else
                echo "SUCCESS: No ncurses dynamic dependencies found (fully static)"
            fi
        popd || exit
    else
        echo UI build disabled.
    fi

    echo Building Installer UI...Complete
}

build_installer() {
    echo -------------------------------------------------------------
    echo Building Installer Package...

    if [ ! -d $BUILD_DIR ]; then
        echo Creating $BUILD_DIR directory.
        mkdir $BUILD_DIR
    fi
    
    if [ $BUILD_INSTALLER == "yes" ]; then
        echo Building installer runfile...
        
        echo "MAKESELF_OPT_HEADER   = $MAKESELF_OPT_HEADER"
        echo "MAKESELF_OPT          = $MAKESELF_OPT"
        echo "MAKESELF_OPT_COMPRESS = $MAKESELF_OPT_COMPRESS"
        echo "MAKESELF_OPT_CLEANUP  = $MAKESELF_OPT_CLEANUP"
        echo "MAKESELF_OPT_TAR      = $MAKESELF_OPT_TAR"

        # shellcheck disable=SC2086
        if ! makeself $MAKESELF_OPT_HEADER $MAKESELF_OPT $MAKESELF_OPT_COMPRESS $MAKESELF_OPT_CLEANUP $MAKESELF_OPT_TAR ../rocm-installer "./$BUILD_DIR/$BUILD_INSTALLER_NAME.run" "ROCm Runfile Installer" ./install-init.sh; then
            echo -e "\e[31mFailed makeself build.\e[0m"
            exit 1
        fi

        echo Building installer runfile...Complete

        # Display the built runfile name and size
        RUNFILE_PATH="./$BUILD_DIR/$BUILD_INSTALLER_NAME.run"
        if [ -f "$RUNFILE_PATH" ]; then
            RUNFILE_SIZE_BYTES=$(stat -c%s "$RUNFILE_PATH" 2>/dev/null || stat -f%z "$RUNFILE_PATH" 2>/dev/null)
            RUNFILE_SIZE=$(format_size "$RUNFILE_SIZE_BYTES")
            echo ""
            echo -e "\e[32m========================================\e[0m"
            echo -e "\e[32mBuilt runfile: $BUILD_INSTALLER_NAME.run\e[0m"
            echo -e "\e[95mSize: $RUNFILE_SIZE ($RUNFILE_SIZE_BYTES bytes)\e[0m"
            echo -e "\e[32m========================================\e[0m"
        fi
    else
        echo Runfile build disabled.
    fi
    
    echo Building Installer Package...Complete
}


####### Main script ###############################################################

# Record start time
BUILD_START_TIME=$(date +%s)

echo ==============================
echo BUILD INSTALLER
echo ==============================

SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)
echo SUDO: "$SUDO"

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
    noextract)
        echo "Disabling package extraction."
        BUILD_EXTRACT="no"
        shift
        ;;
    norocm)
        echo "Disabling ROCm package extraction."
        EXTRACT_ROCM="no"
        shift
        ;;
    noamdgpu)
        echo "Disabling AMDGPU package extraction."
        EXTRACT_AMDGPU="no"
        shift
        ;;
    noextractcontent)
        echo "Disabling content extraction (deps and scriptlets only)."
        EXTRACT_TYPE="nocontent"
        shift
        ;;
    contentlist)
        echo "Enabling content file listing during extraction."
        EXTRACT_TYPE="contentlist"
        shift
        ;;
    norunfile)
        echo "Disabling runfile build."
        BUILD_INSTALLER="no"
        shift
        ;;
    nogui)
        echo "Disabling UI build."
        BUILD_UI="no"
        shift
        ;;
    buildtag=*)
        BUILD_TAG="${1#*=}"
        echo "Setting BUILD_TAG = $BUILD_TAG"
        shift
        ;;
    buildrunid=*)
        BUILD_RUNID="${1#*=}"
        echo "Setting BUILD_RUNID = $BUILD_RUNID"
        shift
        ;;
    buildtaginfo=*)
        BUILD_TAG_INFO="${1#*=}"
        echo "Setting BUILD_TAG_INFO = $BUILD_TAG_INFO"
        shift
        ;;
    mscomp=*)
        MAKESELF_COMPRESS_MODE="${1#*=}"
        case "$MAKESELF_COMPRESS_MODE" in
            hybrid)
                echo "Setting compression mode: hybrid (xz-9 for everything, maximum compression)"
                ;;
            hybridmix)
                echo "Setting compression mode: hybridmix (pigz + xz for tests, embedded xz-static)"
                ;;
            hybriddev)
                echo "Setting compression mode: hybriddev (xz-3 for everything, fast compression)"
                ;;
            devsmall)
                echo "Setting compression mode: devsmall (xz - best compression)"
                ;;
            devmedium)
                echo "Setting compression mode: devmedium (pbzip2 - balanced production)"
                ;;
            dev)
                echo "Setting compression mode: dev (pigz + complevel 6)"
                ;;
            normal)
                echo "Setting compression mode: normal (gzip -9)"
                ;;
            *)
                echo -e "\e[31mERROR: Invalid mscomp value: $MAKESELF_COMPRESS_MODE\e[0m"
                echo "Valid options: hybrid, hybridmix, hybriddev, devsmall, devmedium, dev, normal"
                exit 1
                ;;
        esac
        shift
        ;;
    *)
        echo "Unknown option: $1"
        shift
        ;;
    esac
done

# Install any required tools for the build
install_tools

# Configure compression (install pigz/lz4 if needed)
configure_compression

# Extract all ROCm/AMDGPU packages
extract_packages

# Setup version/build info (before compression to access amdgpu-dkms-ver.txt)
write_version

# Compress packages if hybrid mode is enabled
if [[ "$HYBRID_COMPRESSION" == "yes" ]]; then
    compress_tests
    compress_components
fi

# Generate component lists and headers
generate_component_lists
generate_headers

# Build the UI
build_UI

# Build the installer
build_installer

# Calculate and display build time
BUILD_END_TIME=$(date +%s)
BUILD_ELAPSED=$((BUILD_END_TIME - BUILD_START_TIME))

# Convert seconds to hours, minutes, seconds
BUILD_HOURS=$((BUILD_ELAPSED / 3600))
BUILD_MINUTES=$(((BUILD_ELAPSED % 3600) / 60))
BUILD_SECONDS=$((BUILD_ELAPSED % 60))

echo ""
echo ==============================
echo "Build completed successfully!"
echo "=============================="
echo -e "\e[36mTotal build time: ${BUILD_HOURS}h ${BUILD_MINUTES}m ${BUILD_SECONDS}s (${BUILD_ELAPSED} seconds)\e[0m"
echo ==============================
echo ""
