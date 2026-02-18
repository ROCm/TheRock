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

BUILD_EXTRACT="yes"
BUILD_INSTALLER="yes"
BUILD_UI="yes"

BUILD_DIR=build
BUILD_DIR_UI=build-UI

VERSION_FILE="./VERSION"

INSTALLER_VERSION=
ROCM_VER=
BUILD_NUMBER="${BUILD_NUMBER:-1}"
ROCK_RELEASE_TAG="${ROCK_RELEASE_TAG:-prerelease}"
BUILD_INSTALLER_NAME=

AMDGPU_DKMS_FILE="rocm-installer/component-amdgpu/amdgpu-dkms-ver.txt"
AMDGPU_DKMS_BUILD_NUM=

EXTRACT_DIR="../rocm-installer"
EXTRACT_TYPE=""
EXTRACT_ROCM="yes"
EXTRACT_AMDGPU="yes"
EXTRACT_AMDGPU_MODE="all"

# AlmaLinux 8.10 (EL8) requires specific makeself options
MAKESELF_OPT="--notemp --threads $(nproc)"
MAKESELF_OPT_CLEANUP=
MAKESELF_OPT_HEADER="--header ./rocm-makeself-header-pre.sh --help-header ./rocm-installer/VERSION"
MAKESELF_OPT_TAR=""  # EL8 does not support GNU tar format


###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $PROG [options]

[options}:
    help               = Display this help information.
    noextract          = Disable package extraction.
    norocm             = Disable ROCm package extraction.
    noamdgpu           = Disable AMDGPU package extraction.
    noextractcontent   = Disable package extraction content. (Extract only deps and scriptlets)
    contentlist        = List all files extracted to content directories during package extraction.
    norunfile          = Disable makeself build of installer runfile.
    nogui              = Disable GUI building.

Supported build systems:
    - Ubuntu (DEB packages)
    - AlmaLinux 8 (RPM packages - ManyLinux)
END_USAGE
}

os_release() {
    if [[ -r  /etc/os-release ]]; then
        . /etc/os-release

        DISTRO_NAME=$ID
        DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
        DISTRO_MAJOR_VER=${DISTRO_VER%.*}

        case "$ID" in
        ubuntu)
            BUILD_DISTRO_PACKAGE_TYPE=deb
            BUILD_OS=$DISTRO_VER
            if [[ $DISTRO_VER == 24.04 ]]; then
                DISTRO_TAG="ub24"
            elif [[ $DISTRO_VER == 22.04 ]]; then
                DISTRO_TAG="ub22"
            else
                echo "WARNING: Unsupported Ubuntu version: $DISTRO_VER, using ub24 as default"
                DISTRO_TAG="ub24"
            fi
            ;;
        debian)
            BUILD_DISTRO_PACKAGE_TYPE=deb
            BUILD_OS=$DISTRO_VER
            if [[ $DISTRO_MAJOR_VER == 13 ]]; then
                DISTRO_TAG="ub24"  # Debian 13 uses same config as Ubuntu 24.04
            elif [[ $DISTRO_MAJOR_VER == 12 ]]; then
                DISTRO_TAG="ub22"  # Debian 12 uses same config as Ubuntu 22.04
            else
                echo "WARNING: Unsupported Debian version: $DISTRO_VER, using ub24 as default"
                DISTRO_TAG="ub24"
            fi
            ;;
        almalinux|rhel|ol|rocky)
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
                if [[ "$ID" == "almalinux" ]]; then
                    echo "Detected AlmaLinux $DISTRO_VER (ManyLinux)"
                fi
                echo "Disable makeself tar options for EL8."
                MAKESELF_OPT_HEADER="--header ./rocm-makeself-header-pre.sh --help-header ./rocm-installer/VERSION"
                MAKESELF_OPT_TAR=""
            else
                echo "WARNING: Unsupported EL version: $DISTRO_VER, using el8 as default"
                DISTRO_TAG="el8"
                BUILD_OS=el8
            fi
            ;;
        sles)
            BUILD_DISTRO_PACKAGE_TYPE=rpm
            # Determine SLE tag based on major version (15.x -> sle15, 16.x -> sle16)
            if [[ "$DISTRO_MAJOR_VER" == "16" ]]; then
                DISTRO_TAG="sle16"
                BUILD_OS=sle16
            elif [[ "$DISTRO_MAJOR_VER" == "15" ]]; then
                DISTRO_TAG="sle15"
                BUILD_OS=sle15
            else
                echo "WARNING: Unsupported SLES version: $DISTRO_VER, using sle15 as default"
                DISTRO_TAG="sle15"
                BUILD_OS=sle15
            fi
            ;;
        amzn)
            BUILD_DISTRO_PACKAGE_TYPE=rpm
            DISTRO_TAG="amzn23"
            BUILD_OS=amzn23
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

    echo "Build running on $DISTRO_NAME $DISTRO_VER (tag: $DISTRO_TAG)."
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

setup_version() {
    echo -------------------------------------------------------------
    echo Setting version and build info...
    
    BUILD_INFO=$BUILD_NUMBER-$ROCK_RELEASE_TAG

    get_version
    
     # set the runfile installer name
    BUILD_INSTALLER_NAME="rocm-installer_$INSTALLER_VERSION.$ROCM_VER-$BUILD_INFO"

    # get the amdgpu-dkms build/version info
    if [ -f "$AMDGPU_DKMS_FILE" ]; then
        AMDGPU_DKMS_BUILD_NUM=$(cat "$AMDGPU_DKMS_FILE")
    fi

    echo "INSTALLER_VERSION        = $INSTALLER_VERSION"
    echo "ROCM_VER                 = $ROCM_VER"
    echo "BUILD_NUMBER             = $BUILD_NUMBER"
    echo "ROCK_RELEASE_TAG         = $ROCK_RELEASE_TAG"
    echo "AMDGPU_DKMS_BUILD_NUM    = $AMDGPU_DKMS_BUILD_NUM"
    echo "BUILD_INSTALLER_NAME     = $BUILD_INSTALLER_NAME"

    # Update the version file
    echo "$INSTALLER_VERSION" > "$VERSION_FILE"
    echo "$ROCM_VER" >> "$VERSION_FILE"
    echo "$ROCK_RELEASE_TAG" >> "$VERSION_FILE"
    echo "$AMDGPU_DKMS_BUILD_NUM" >> "$VERSION_FILE"
    echo "$BUILD_INSTALLER_NAME" >> "$VERSION_FILE"
}

print_directory_size() {
    local dir_path="$1"
    local dir_name="${2:-$(basename "$dir_path")}"

    if [ -d "$dir_path" ]; then
        local size=$(du -sh "$dir_path" 2>/dev/null | awk '{print $1}')
        if [ -n "$size" ]; then
            echo "  Directory size: $size ($dir_name)"
        fi
    fi
}

install_makeself() {
    echo ----------------------
    echo -e "\e[32mInstalling makeself...\e[0m"
    
    local makeself_ver="2.4.5"
    local makeself_url="https://github.com/megastep/makeself/releases/download/release-$makeself_ver/makeself-$makeself_ver.run"
    
    # Download the makeself package
    echo "Downloading makeself package from github..."
    wget -q "$makeself_url"
    
    if [[ $? -ne 0 ]]; then
        echo -e "\e[31mmakeself package not found: $makeself_url.\e[0m"
        exit 1
    fi
    
    $SUDO chmod +x "makeself-$makeself_ver.run"

    # Install the makeself package
    echo -e "\e[32mInstalling makeself package...\e[0m"
    bash "makeself-$makeself_ver.run"

    # Clean up
    echo "Cleaning up..."
    rm -f makeself-$makeself_ver.run

    # Add makeself to PATH
    echo "Adding makeself to PATH..."
    $SUDO ln -sf "$PWD/makeself-$makeself_ver/makeself.sh" /usr/local/bin/makeself

    echo Installing makeself...Complete
}

install_ncurses_deb() {
    echo Installing ncurses libraries...

    # Install ncurses development libraries
    $SUDO apt-get install -y libncurses5-dev libncurses-dev

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

    # Install ncurses development libraries
    $SUDO dnf install -y ncurses-devel

    # For AlmaLinux 8, install ncurses-static from devel repo
    if [[ $DISTRO_NAME == "almalinux" ]] && [[ $DISTRO_VER == 8* ]]; then
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
    else
        # For other EL distros, try standard install
        echo "Installing ncurses-static..."
        $SUDO dnf install -y ncurses-static || echo "WARNING: ncurses-static not available"
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

    # Install wget for downloading packages
    $SUDO apt-get install -y wget

    # Install tools for UI (including ncurses libraries)
    $SUDO apt-get install -y cmake
    $SUDO apt-get install -y gcc g++

    # Install ncurses libraries
    install_ncurses_deb

    # Install binutils for ar command
    $SUDO apt-get install -y binutils

    # Install makeself for .run creation
    $SUDO apt-get install -y makeself > /dev/null 2>&1
    if [[ $? -ne 0 ]]; then
        install_makeself
    fi

    # Check the version of makeself and enable cleanup script support if >= 2.4.2
    makeself_version_min=2.4.2
    makeself_version=$(makeself --version)
    makeself_version=${makeself_version#Makeself version }

    if [[ "$(printf '%s\n' "$makeself_version_min" "$makeself_version" | sort -V | head -n1)" = "$makeself_version_min" ]]; then
        MAKESELF_OPT_CLEANUP+="--cleanup ./cleanup-install.sh"
        echo Enabling cleanup script support.
    fi

    echo Installing DEB tools...Complete
}

install_tools_el(){
    echo Installing EL tools...

    # Install wget for downloading packages
    $SUDO dnf install -y wget binutils tar rpm-build cpio dpkg
    
    # Install tools for UI (including ncurses libraries)
    $SUDO dnf install -y cmake 2>/dev/null || echo "cmake already installed or using custom build"
    $SUDO dnf install -y gcc gcc-c++

    # Install ncurses libraries
    install_ncurses_el

    if [[ $DISTRO_NAME == "amzn" ]]; then
        $SUDO dnf install -y tar bzip2
    fi
    
    # Install makself for .run creation either from repos or directly from github
    $SUDO dnf install -y makeself > /dev/null 2>&1
    if [[ $? -ne 0 ]]; then
        install_makeself
    fi
    
    # Check the version of makself and enable cleanup script support if >= 2.4.2
    makeself_version_min=2.4.2
    makeself_version=$(makeself --version)
    makeself_version=${makeself_version#Makeself version }

    if [[ "$(printf '%s\n' "$makeself_version_min" "$makeself_version" | sort -V | head -n1)" = "$makeself_version_min" ]]; then
        MAKESELF_OPT_CLEANUP+="--cleanup ./cleanup-install.sh"
        echo Enabling cleanup script support.
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

extract_rocm_packages_deb() {
    echo "Extracting ROCm DEB packages for $BUILD_OS..."

    # Extract all ROCm DEB packages (common and gfx-specific)
    # The extractor script will auto-detect all packages-rocm*-deb directories
    echo "Extracting ROCm DEB packages (common and gfx-specific)..."

    if [ $BUILD_DISTRO_PACKAGE_TYPE == "rpm" ]; then
        # On RPM-based systems, use nodpkg extractor and extract to separate directory
        echo "Using nodpkg extractor for RPM-based system (nocontent mode)"
        PACKAGE_ROCM_DIR="$PWD/packages-rocm-deb" EXTRACT_FORMAT=deb ./package-extractor-debs-nodpkg.sh rocm ext-rocm="../rocm-installer/component-rocm-deb" nocontent
    else
        # On DEB-based systems, use standard extractor
        PACKAGE_ROCM_DIR="$PWD/packages-rocm-deb" EXTRACT_FORMAT=deb ./package-extractor-debs.sh rocm ext-rocm="../rocm-installer/component-rocm" $EXTRACT_TYPE
    fi

    if [[ $? -ne 0 ]]; then
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

    # AMDGPU packages are stored in subdirectories: packages-amdgpu/<distro_tag>
    AMDGPU_PKG_DIR="packages-amdgpu/${DISTRO_TAG}"

    # Verify AMDGPU package directory exists
    if [ ! -d "$AMDGPU_PKG_DIR" ]; then
        echo -e "\e[31mERROR: $AMDGPU_PKG_DIR directory not found!\e[0m"
        echo "Please run setup-installer.sh first to download AMDGPU packages."
        exit 1
    fi

    # Extract the AMDGPU packages to component-amdgpu/<distro_tag>
    PACKAGE_AMDGPU_DIR="$PWD/$AMDGPU_PKG_DIR" EXTRACT_FORMAT=deb ./package-extractor-debs.sh amdgpu ext-amdgpu="${EXTRACT_DIR}/component-amdgpu/${DISTRO_TAG}"
    if [[ $? -ne 0 ]]; then
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
    PACKAGE_ROCM_DIR="$PWD/packages-rocm-rpm" EXTRACT_FORMAT=rpm ./package-extractor-rpms.sh rocm ext-rocm="../rocm-installer" $EXTRACT_TYPE
    if [[ $? -ne 0 ]]; then
        echo -e "\e[31mFailed extraction of ROCm RPM packages.\e[0m"
        exit 1
    fi

    echo "ROCm RPM package extraction complete."
    print_directory_size "../rocm-installer/component-rocm" "component-rocm"
}

extract_amdgpu_packages_rpm() {
    echo "Extracting AMDGPU RPM packages for $BUILD_OS (tag: $DISTRO_TAG)..."

    # AMDGPU packages are stored in subdirectories: packages-amdgpu/<distro_tag>
    AMDGPU_PKG_DIR="packages-amdgpu/${DISTRO_TAG}"

    # Verify AMDGPU package directory exists
    if [ ! -d "$AMDGPU_PKG_DIR" ]; then
        echo -e "\e[31mERROR: $AMDGPU_PKG_DIR directory not found!\e[0m"
        echo "Please run setup-installer.sh first to download AMDGPU packages."
        exit 1
    fi

    # Extract the AMDGPU packages to component-amdgpu/<distro_tag>
    PACKAGE_AMDGPU_DIR="$PWD/$AMDGPU_PKG_DIR" EXTRACT_FORMAT=rpm ./package-extractor-rpms.sh amdgpu ext-amdgpu="${EXTRACT_DIR}/component-amdgpu/${DISTRO_TAG}"
    if [[ $? -ne 0 ]]; then
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
            local DISTRO_TAG="$(basename "$amdgpu_dir")"

            echo "Extracting AMDGPU packages from $amdgpu_dir (tag: $DISTRO_TAG)..."

            # Extract the AMDGPU packages to component-amdgpu/<distro_tag>
            PACKAGE_AMDGPU_DIR="$PWD/$amdgpu_dir" ./package-extractor-all.sh amdgpu pkgs-amdgpu="$PWD/$amdgpu_dir" ext-amdgpu="${EXTRACT_DIR}/component-amdgpu/${DISTRO_TAG}"
            if [[ $? -ne 0 ]]; then
                echo -e "\e[31mFailed extraction of AMDGPU packages from $amdgpu_dir.\e[0m"
                exit 1
            fi

            print_directory_size "${EXTRACT_DIR}/component-amdgpu/${DISTRO_TAG}" "component-amdgpu/${DISTRO_TAG}"
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
        pushd package-extractor

        # Extract ROCm packages
        extract_packages_rocm

        # Extract AMDGPU packages
        extract_packages_amdgpu

        popd
    else
        echo Extract Packages disabled.
    fi

    echo Running Package Extractor...Complete
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

        pushd $BUILD_DIR_UI
            # UI now reads VERSION file at runtime - no version parameters needed
            cmake ..
            make
            if [[ $? -ne 0 ]]; then
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
        popd
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
        
        echo "MAKESELF_OPT_HEADER  = $MAKESELF_OPT_HEADER"
        echo "MAKESELF_OPT         = $MAKESELF_OPT"
        echo "MAKESELF_OPT_CLEANUP = $MAKESELF_OPT_CLEANUP"
        echo "MAKESELF_OPT_TAR     = $MAKESELF_OPT_TAR"
        
        makeself $MAKESELF_OPT_HEADER $MAKESELF_OPT $MAKESELF_OPT_CLEANUP $MAKESELF_OPT_TAR ./rocm-installer "./$BUILD_DIR/$BUILD_INSTALLER_NAME.run" "ROCm Runfile Installer" ./install-init.sh
        if [[ $? -ne 0 ]]; then
            echo -e "\e[31mFailed makeself build.\e[0m"
            exit 1
        fi

        echo Building installer runfile...Complete

        # Display the built runfile name and size
        RUNFILE_PATH="./$BUILD_DIR/$BUILD_INSTALLER_NAME.run"
        if [ -f "$RUNFILE_PATH" ]; then
            RUNFILE_SIZE=$(du -h "$RUNFILE_PATH" | awk '{print $1}')
            RUNFILE_SIZE_BYTES=$(stat -c%s "$RUNFILE_PATH" 2>/dev/null || stat -f%z "$RUNFILE_PATH" 2>/dev/null)
            echo ""
            echo -e "\e[32m========================================\e[0m"
            echo -e "\e[32mBuilt runfile: $BUILD_INSTALLER_NAME.run\e[0m"
            echo -e "\e[32mSize: $RUNFILE_SIZE ($RUNFILE_SIZE_BYTES bytes)\e[0m"
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
echo ROCM RUNFILE INSTALLER BUILDER
echo ==============================

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
    *)
        echo "Unknown option: $1"
        shift
        ;;
    esac
done

# Install any required tools for the build
install_tools

# Extract all ROCm/AMDGPU packages
extract_packages

# Setup version/build info
setup_version

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
echo -e "\e[32mTotal build time: ${BUILD_HOURS}h ${BUILD_MINUTES}m ${BUILD_SECONDS}s (${BUILD_ELAPSED} seconds)\e[0m"
echo ==============================
echo ""
