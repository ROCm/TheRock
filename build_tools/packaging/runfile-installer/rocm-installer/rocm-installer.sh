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

# Logs
RUN_INSTALLER_LOG_DIR="$PWD/logs"
RUN_INSTALLER_CURRENT_LOG="$RUN_INSTALLER_LOG_DIR/install_$(date +%s).log"

# Source extract directories
EXTRACT_ROCM_DIR="$PWD/component-rocm"
EXTRACT_AMDGPU_DIR="$PWD/component-amdgpu"

# Target install directories
TARGET_ROCM_DEFAULT_DIR="/opt"
TARGET_ROCM_DIR="$TARGET_ROCM_DEFAULT_DIR"
TARGET_AMDGPU_DIR="/"

# Component Configuration
COMPO_ROCM_FILE="$EXTRACT_ROCM_DIR/base/rocm-packages.config"
COMPO_ROCM_LIST="$EXTRACT_ROCM_DIR/base/components.txt"
COMPO_AMDGPU_FILE=""  # Set dynamically after os_release() determines distro
COMPO_INSTALL="core"  # Default component: core, core-dev, dev-tools, core-sdk (comma-separated)
COMPO_META_DIR="$EXTRACT_ROCM_DIR/meta"
USER_SPECIFIED_COMPO=0  # Track if user explicitly specified compo= argument
USER_SPECIFIED_GFX=0    # Track if user explicitly specified gfx= argument
COMPONENTS=
COMPONENTS_GFX=
INSTALL_GFX=

# Install Configuration
RSYNC_OPTS_ROCM="--keep-dirlinks -rlp "
RSYNC_OPTS_AMDGPU="-a --keep-dirlinks --no-perms --no-owner --no-group --omit-dir-times "
ROCM_INSTALL=0
AMDGPU_INSTALL=0
AMDGPU_START=0
NCURSES_BAR=1

COMPONENT_COUNT=0
POSTINST_COUNT=0
PRERM_COUNT=0
POSTRM_COUNT=0
PROMPT_USER=0
POST_ROCM_INSTALL=1
VERBOSE=0

# Installer preqreqs
INSTALLER_DEPS=(rsync)

# Uninstall data
INSTALLED_AMDGPU_DKMS_BUILD_NUM=0
FORCE_UNINSTALL_AMDGPU=0

###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: bash $PROG [options]

[options]:
    help    = Displays this help information.
    version = Display version information.
    
    Dependencies:
    -------------
        deps=<arg> <compo>
        
        <arg>
            list <compo>         = Lists required dependencies for install <compo>.
            validate <compo>     = Validates installed and not installed required dependencies for install <compo>.
            install-only <compo> = Install dependencies only for <compo>.
            install <compo>      = Install with dependencies for <compo>.
            
            file <file_path>      = Install with dependencies from a dependencies configuration file with path <file_path>.
            file-only <file_path> = Install with dependencies from a dependencies configuration file only with path <file_path>.
            
        <compo> = install component (rocm/amdgpu/rocm amdgpu)
    
    Install:
    --------
        rocm   = Enable ROCm components install.
        amdgpu = Enable amdgpu driver install.

        force  = Force ROCm/amdgpu driver install.

        target=<directory>
               <directory> = Target directory path for ROCm component install.

        gfx=<arch>
            <arch> = GPU architecture to install (e.g., gfx94x, gfx950, etc.)
                     Installs base components plus architecture-specific components.
                     If not specified, only base components are installed.
                     Available architectures: gfx94x, gfx950

        compo=<component_list>
            <component_list> = Comma-separated list of ROCm components to install.
                               If not specified, defaults to 'core'.
                               Available components:
                                   core      = Core ROCm components (amdrocm-core)
                                   core-dev  = Core development components (amdrocm-core-devel)
                                   dev-tools = Developer tools (amdrocm-developer-tools)
                                   core-sdk  = Core SDK components (amdrocm-core-sdk)
                               Examples:
                                   compo=core
                                   compo=core,dev-tools
                                   compo=core-sdk,dev-tools

    Post Install:
    -------------
        Post-install configuration runs by default after ROCm installation (scripts, symlink create, etc.)

        postrocm     = Run post ROCm installation configuration.
                       Use this to run post-install after installing with 'nopostrocm'.
                       Default target: /opt (can override with target=<path>)
                       Optional: compo= and gfx= to match original installation args
                       If compo=/gfx= not specified, auto-detects installed components

        nopostrocm   = Disable post ROCm installation configuration.
                       By default, post-install runs automatically with 'rocm'.
                       Use this flag to skip post-install processing.

        amdgpu-start = Start the amdgpu driver after install.

        gpu-access=<access_type>
        
                   <access_type>
                       user = Add current user to render,video group for GPU access
                       all  = Grant GPU access to all users on the system via udev rules.
                 
    Uninstall:
    ----------
        uninstall-rocm target=<directory/rocm-ver> = Uninstall ROCm version at target directory path.
                              <directory/rocm-ver> = ROCm version target directory path for uninstall.
                                         rocm-ver  = ROCm version directory: rocm-x.y.z (x=major, y=minor, patch number)

                             * If target=<directory/rocm-ver> is not provided, uninstall will be from /opt/rocm-x.y.z
                             * Optional: Use compo= and gfx= for selective uninstall (same args as install)
                             * If compo=/gfx= not specified, auto-detects and uninstalls all installed components
      
        uninstall-amdgpu = Uninstall amdgpu driver.
                       
    
    Information/Debug:
    ------------------
    findrocm = Search for an install of ROCm.
    complist = List the version of ROCm components included in the installer.
    prompt   = Run the installer with user prompts.
    verbose  = Run installer with verbose logging
    
+++++++++++++++++++++++++++++++
Usage examples:
+++++++++++++++++++++++++++++++
    
# ROCm installation (no Dependency install)

    * ROCm install location (default) => /opt/rocm-x.y.z
        bash $PROG gfx=gfx94x rocm

# ROCm + Dependency installation

    * ROCm install location (default) => /opt/rocm-x.y.z
        bash $PROG deps=install gfx=gfx94x rocm

# ROCm + Dependency installation + ROCm target location

    * ROCm install location => $HOME/myrocm/rocm-x.y.z
        bash $PROG deps=install target="$HOME/myrocm" gfx=gfx94x rocm

# ROCm + Dependency installation + gpu access (post-install runs by default)

    bash $PROG deps=install gfx=gfx94x rocm gpu-access=all
    bash $PROG deps=install target="$HOME/myrocm" gfx=gfx94x rocm gpu-access=all

# ROCm + Component selection (post-install runs by default)

    * Install core + dev-tools
        bash $PROG compo=core,dev-tools gfx=gfx94x rocm

    * Install core-sdk with gfx support
        bash $PROG compo=core-sdk gfx=gfx94x rocm

    * Install without post-install processing
        bash $PROG gfx=gfx94x rocm nopostrocm

    * Run post-install later (after installing with nopostrocm)
        bash $PROG postrocm                                      # Auto-detect, uses default /opt
        bash $PROG target=/custom/path postrocm                  # Auto-detect, custom location
        bash $PROG gfx=gfx94x postrocm                           # Explicit gfx, uses default /opt
        bash $PROG compo=core-sdk gfx=gfx94x postrocm            # With component selection
        bash $PROG target=/custom/path gfx=gfx94x postrocm       # Custom location with gfx

# ROCm + GFX architecture-specific installation

    * Install base ROCm + gfx94x components to default location
        bash $PROG gfx=gfx94x rocm

    * Install base ROCm + gfx950 components with dependencies
        bash $PROG deps=install gfx=gfx950 rocm

    * Install base ROCm + gfx94x components to custom location
        bash $PROG deps=install target="$HOME/myrocm" gfx=gfx94x rocm

# ROCm + Component selection

    * Install core components only (default)
        bash $PROG gfx=gfx94x rocm
        bash $PROG compo=core gfx=gfx94x rocm

    * Install core + developer tools
        bash $PROG compo=core,dev-tools gfx=gfx94x rocm

    * Install core SDK with gfx94x support
        bash $PROG compo=core-sdk gfx=gfx94x rocm

    * Install core development components with dependencies
        bash $PROG deps=install compo=core-dev gfx=gfx950 rocm

    ** Recommended ***

# amdgpu Driver installation (no Dependency install)
    
    bash $PROG amdgpu

# amdgpu Driver + Dependency installation
    
    bash $PROG deps=install amdgpu
    
# Combined Installation

    bash $PROG deps=install gfx=gfx94x rocm amdgpu gpu-access=all
    bash $PROG deps=install target="$HOME/myrocm" gfx=gfx94x rocm amdgpu gpu-access=all
    
# Uninstall

    * Complete uninstall (auto-detects all installed components)
        bash $PROG uninstall-rocm
        bash $PROG target="$HOME/myrocm/rocm-x.y.z" uninstall-rocm

    * Selective uninstall (specify same compo=/gfx= as install)
        bash $PROG target=/opt/rocm-7.11 compo=core gfx=gfx950 uninstall-rocm
        bash $PROG target=/opt/rocm-7.11 compo=core-sdk gfx=gfx94x uninstall-rocm

    * Uninstall amdgpu driver
        bash $PROG uninstall-amdgpu

    * Uninstall combined
        bash $PROG uninstall-amdgpu uninstall-rocm
    
END_USAGE
}

os_release() {
    if [[ -r  /etc/os-release ]]; then
        . /etc/os-release

        DISTRO_NAME=$ID

        case "$ID" in
        ubuntu|debian)
            DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')

            INSTALL_SCRIPTLET_ARG="configure"
            UNINSTALL_SCRIPTLET_ARG="remove"
            PKG_INSTALLED_CMD="apt list --installed"
            UPDATE_INITRAMFS_CMD="update-initramfs -u -k all"
            PACKAGE_TYPE="deb"
            ;;
        rhel|centos|ol|rocky|almalinux|amzn)
            DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')

            if [[ $DISTRO_VER != 9* ]] && [[ "$DISTRO_NAME" = "rocky" ]]; then
                echo "$DISTRO_NAME $DISTRO_VER is not a supported OS"
                exit 1
            fi

            if [[ $DISTRO_VER != 9* ]] && [[ "$DISTRO_NAME" = "centos" ]]; then
                echo "$DISTRO_NAME $DISTRO_VER is not a supported OS"
                exit 1
            fi

            if [[ $DISTRO_VER != 8* ]] && [[ "$DISTRO_NAME" = "almalinux" ]]; then
                echo "WARNING: This installer was built for AlmaLinux 8.x"
                echo "Detected AlmaLinux $DISTRO_VER - installation may not work correctly"
            fi

            INSTALL_SCRIPTLET_ARG="1"
            UNINSTALL_SCRIPTLET_ARG="0"
            PKG_INSTALLED_CMD="rpm -qa"
            UPDATE_INITRAMFS_CMD="dracut -f --regenerate-all"
            PACKAGE_TYPE="rpm"

            if ! rpm -qa | grep -qE "ncurses-[0-9]"; then
                NCURSES_BAR=0
            fi

            ;;
        sles)
            if rpm -qa | grep -q "awk"; then
                DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
            else
                DISTRO_VER=$(grep '^VERSION_ID=' /etc/os-release | cut -d'=' -f2 | tr -d '"')
            fi

            INSTALL_SCRIPTLET_ARG="1"
            UNINSTALL_SCRIPTLET_ARG="0"
            PKG_INSTALLED_CMD="rpm -qa"
            UPDATE_INITRAMFS_CMD="dracut -f --regenerate-all"
            PACKAGE_TYPE="rpm"
            ;;
        *)
            echo "$ID is not a supported OS"
            exit 1
            ;;
        esac
    else
        echo "Unsupported OS"
        exit 1
    fi

    # Extract major version from DISTRO_VER (e.g., 22.04 -> 22, 9.3 -> 9, 15.6 -> 15)
    DISTRO_MAJOR_VER=${DISTRO_VER%%.*}

    echo "Installing for $DISTRO_NAME $DISTRO_VER."
}

get_version() {
    local i=0
    local version_file=""

    # Find VERSION file - check both current directory and parent directory
    if [ -f "./VERSION" ]; then
        version_file="./VERSION"
    elif [ -f "../VERSION" ]; then
        version_file="../VERSION"
    fi

    # Read VERSION file (written by package-extractor and build-installer)
    # Line 1: Installer version (from package-extractor)
    # Line 2: ROCm version (from package-extractor)
    # Line 3: Rock release tag (from build-installer)
    # Line 4: AMDGPU DKMS build number (from build-installer)
    # Line 5: Build installer name (from build-installer)
    if [ -n "$version_file" ]; then
        while IFS= read -r line; do
            case $i in
                0) INSTALLER_VERSION="$line" ;;
                1) ROCM_VERSION="$line" ;;
                2) ROCK_RELEASE_TAG="$line" ;;
                3) AMDGPU_DKMS_BUILD_NUM="$line" ;;
                4) BUILD_INSTALLER_NAME="$line" ;;
            esac
            i=$((i+1))
        done < "$version_file"
    fi

    # If AMDGPU build number wasn't in VERSION file (pre-build state),
    # try reading from component-amdgpu directory
    if [[ -z "$AMDGPU_DKMS_BUILD_NUM" || "$AMDGPU_DKMS_BUILD_NUM" == "" ]]; then
        local amdgpu_ver_file="./component-amdgpu/amdgpu-dkms-ver.txt"
        if [ -f "$amdgpu_ver_file" ]; then
            AMDGPU_DKMS_BUILD_NUM=$(cat "$amdgpu_ver_file" | tr -d '[:space:]')
        else
            AMDGPU_DKMS_BUILD_NUM="N/A"
        fi
    fi

    echo "Installer Version: $INSTALLER_VERSION"
    echo "ROCm Version     : $ROCM_VERSION"
    echo "Rock Release     : $ROCK_RELEASE_TAG"
    echo "AMDGPU Build     : $AMDGPU_DKMS_BUILD_NUM"
    echo "Installer Name   : $BUILD_INSTALLER_NAME"

    # Update PROG if running from runfile
    if [[ $RUNFILE_INSTALL == 1 && -n "$BUILD_INSTALLER_NAME" ]]; then
        PROG="$BUILD_INSTALLER_NAME"
    fi
}

setup_rocm_version_info() {
    echo "Setting up ROCm version information from base directory..."

    # Find the ROCm version directory in the base directory
    ROCM_CORE_VER_DIR=$(find "$EXTRACT_ROCM_DIR/base/amdrocm-base"*/content/rocm -type d -name "core*" -print -quit)

    if [[ -n "$ROCM_CORE_VER_DIR" ]]; then
        # Extract the relative path from content/ (e.g., rocm/core-7.11)
        # This represents the actual installation structure
        ROCM_CONTENT_BASE=$(dirname "$ROCM_CORE_VER_DIR")
        ROCM_CONTENT_BASE=${ROCM_CONTENT_BASE%/content*}/content
        INSTALLER_ROCM_VERSION_NAME=${ROCM_CORE_VER_DIR#$ROCM_CONTENT_BASE/}

        # Extract base version for version string (e.g., core-7.11 -> 7.11)
        ROCM_VER_BASE=$(basename "$ROCM_CORE_VER_DIR")
        ROCM_VER_BASE=${ROCM_VER_BASE#core-}
        INSTALLER_ROCM_VERSION="$ROCM_VER_BASE"

        # Extract short version (major.minor only, e.g., 7.11)
        ROCM_VER=$(echo "$ROCM_VER_BASE" | cut -d '.' -f 1-2)

        echo "  ROCM_CORE_VER_DIR          : $ROCM_CORE_VER_DIR"
        echo "  INSTALLER_ROCM_VERSION     : $INSTALLER_ROCM_VERSION"
        echo "  INSTALLER_ROCM_VERSION_NAME: $INSTALLER_ROCM_VERSION_NAME"
        echo "  ROCM_VER                   : $ROCM_VER"
    else
        print_err "Could not find ROCm core version directory in base"
        return 1
    fi

    return 0
}

get_amdgpu_distro_tag() {
    # Map the current distro to the AMDGPU component directory name
    local distro_tag=""

    case "$DISTRO_NAME" in
        ubuntu)
            # Ubuntu: 24.04 -> ub24, 22.04 -> ub22
            distro_tag="ub${DISTRO_MAJOR_VER}"
            ;;
        debian)
            # Debian: 12 -> ub22, 13 -> ub24
            case "$DISTRO_MAJOR_VER" in
                12)
                    distro_tag="ub22"
                    ;;
                13)
                    distro_tag="ub24"
                    ;;
                *)
                    distro_tag="ub${DISTRO_MAJOR_VER}"
                    ;;
            esac
            ;;
        rhel)
            # RHEL: 9 -> el9, 8 -> el8
            distro_tag="el${DISTRO_MAJOR_VER}"
            ;;
        centos)
            # CentOS Stream: 9 -> el9, 8 -> el8
            distro_tag="el${DISTRO_MAJOR_VER}"
            ;;
        ol)
            # Oracle Linux: 9 -> el9, 8 -> el8
            distro_tag="el${DISTRO_MAJOR_VER}"
            ;;
        rocky)
            # Rocky: 9 -> el9, 8 -> el8
            distro_tag="el${DISTRO_MAJOR_VER}"
            ;;
        almalinux)
            # AlmaLinux: 9 -> el9, 8 -> el8
            distro_tag="el${DISTRO_MAJOR_VER}"
            ;;
        amzn)
            # Amazon Linux: 2023 -> amzn2023
            distro_tag="amzn${DISTRO_VER}"
            ;;
        sles)
            # SLES: 15.x -> sle15, 16.x -> sle16
            distro_tag="sle${DISTRO_MAJOR_VER}"
            ;;
        *)
            echo "Unknown distro: $DISTRO_NAME"
            return 1
            ;;
    esac

    echo "$distro_tag"
    return 0
}

print_no_err() {
    local msg=$1
    echo -e "\e[32m++++++++++++++++++++++++++++++++++++\e[0m"
    echo -e "\e[32m$msg\e[0m"
    echo -e "\e[32m++++++++++++++++++++++++++++++++++++\e[0m"
}

print_err() {
    local msg=$1
    echo -e "\e[31m++++++++++++++++++++++++++++++++++++\e[0m"
    echo -e "\e[31mError: $msg\e[0m"
    echo -e "\e[31m++++++++++++++++++++++++++++++++++++\e[0m"
}

print_warning() {
    local msg=$1
    echo -e "\e[93m++++++++++++++++++++++++++++++++++++\e[0m"
    echo -e "\e[93mWarning: $msg\e[0m"
    echo -e "\e[93m++++++++++++++++++++++++++++++++++++\e[0m"
}

print_str() {
    local str=$1
    local clr=$2
    
    if [[ $VERBOSE == 1 ]]; then
        if [[ $clr == 1 ]]; then
            echo -e "\e[93m$str\e[0m"   # yellow
        elif [[ $clr == 2 ]]; then
            echo -e "\e[32m$str\e[0m"   # green  
        elif [[ $clr == 3 ]]; then
            echo -e "\e[31m$str\e[0m"   # red
        elif [[ $clr == 4 ]]; then
            echo -e "\e[35m$str\e[0m"   # purple
        elif [[ $clr == 5 ]]; then
            echo -e "\e[36m$str\e[0m"   # cyan
        else
            echo "$str"                 # white
        fi
    fi
}

prompt_user() {
    if [[ $PROMPT_USER == 1 ]]; then
        read -p "$1" option
    else
        option=y
    fi
}

dump_rocm_state() {
    echo ============================
    echo ROCm Install Summary
    echo ============================
    
    local rocm_install_loc=$TARGET_DIR
    local ls_opt=
    
    for dir in "$rocm_install_loc"/*; do
        if [[ -d "$dir" && $(basename "$dir") == "$INSTALLER_ROCM_VERSION_NAME" ]]; then
            rocm_directory=$dir
            break
        fi
    done
    
    echo -e "\e[32mROCm Installed to: $rocm_directory\e[0m"

    if [[ $VERBOSE == 1 ]]; then
        ls_opt="-la"
        
        echo ----------------------------
        echo -e "\e[95m$rocm_install_loc\e[0m"
        ls $ls_opt $rocm_install_loc
        
        echo ----------------------------
        echo -e "\e[95m$rocm_directory\e[0m"
        ls $ls_opt $rocm_directory
        
        echo ----------------------------
        echo -e "\e[95m/etc/ld.so.conf.d\e[0m"
        ls /etc/ld.so.conf.d
        
        echo ----------------------------
        echo -e "\e[95m/etc/alternatives\e[0m"
        ls $ls_opt /etc/alternatives
        
        echo ----------------------------
        echo -e "\e[95m$rocm_directory/include\e[0m"
        ls $ls_opt $rocm_directory/include
        
        echo ----------------------------
        echo -e "\e[95m$rocm_directory/bin\e[0m"
        ls $ls_opt $rocm_directory/bin
        
        echo ----------------------------
        echo -e "\e[95m$rocm_directory/lib\e[0m"
        ls $ls_opt $rocm_directory/lib
    fi
    
    echo ----------------------------
    echo -e "\e[95mInstalled Components:\e[0m"
    echo "Base: $COMPONENTS"
    if [[ -n $COMPONENTS_GFX ]]; then
        echo "GFX (${INSTALL_GFX}): $COMPONENTS_GFX"
    fi
}

get_available_gfx_archs() {
    # Scan for available GFX architectures in the installer
    local archs=()
    for gfx_dir in "$EXTRACT_ROCM_DIR"/gfx*/; do
        if [ -d "$gfx_dir" ]; then
            gfx_name=$(basename "$gfx_dir")
            archs+=("$gfx_name")
        fi
    done
    echo "${archs[@]}"
}

validate_gfx_arg() {
    # Validate gfx= argument for ROCm installation

    # Check if gfx= is required but not provided
    # Only required when actually installing ROCm components (not deps-only)
    if [[ $ROCM_INSTALL == 1 && "$DEPS_ARG" != "install-only" ]]; then
        if [[ -z "$INSTALL_GFX" ]]; then
            # Get available architectures from installer
            local available_archs=($(get_available_gfx_archs))
            print_err "The gfx= argument is required when installing ROCm."
            echo "Example: $PROG gfx=gfx94x rocm"
            if [ ${#available_archs[@]} -gt 0 ]; then
                echo "Available architectures: ${available_archs[*]}"
            else
                echo "Available architectures: gfx94x, gfx950"
            fi
            exit 1
        fi
    fi

    # Validate gfx= format and value if provided
    if [[ -n "$INSTALL_GFX" ]]; then
        # Check for multiple architectures (commas or spaces)
        if [[ "$INSTALL_GFX" == *,* ]] || [[ "$INSTALL_GFX" == *" "* ]]; then
            print_err "The gfx= argument must specify only ONE architecture."
            echo "Invalid: gfx=$INSTALL_GFX"
            echo "Example: gfx=gfx94x (not gfx=gfx94x,gfx950)"
            exit 1
        fi

        # Validate format: must start with "gfx" followed by alphanumeric
        if [[ ! "$INSTALL_GFX" =~ ^gfx[0-9a-z]+$ ]]; then
            local available_archs=($(get_available_gfx_archs))
            print_err "Invalid gfx= format: $INSTALL_GFX"
            echo "The architecture must be in format: gfx<arch>"
            if [ ${#available_archs[@]} -gt 0 ]; then
                echo "Available architectures: ${available_archs[*]}"
            else
                echo "Examples: gfx94x, gfx950, gfx103x"
            fi
            exit 1
        fi

        # Validate against available architectures in installer
        local available_archs=($(get_available_gfx_archs))
        if [ ${#available_archs[@]} -gt 0 ]; then
            local valid_arch=0
            for arch in "${available_archs[@]}"; do
                if [[ "$INSTALL_GFX" == "$arch" ]]; then
                    valid_arch=1
                    break
                fi
            done

            if [ $valid_arch -eq 0 ]; then
                print_err "Architecture '$INSTALL_GFX' is not available in this installer."
                echo "Available architectures: ${available_archs[*]}"
                exit 1
            fi
        fi
    fi
}

dump_stats() {
    echo ----------------------------
    echo STATS
    echo -----
    
    echo "COMPONENT_COUNT = $COMPONENT_COUNT"
    echo "POSTINST_COUNT  = $POSTINST_COUNT"
    
    echo -----
    local stat_dir="$1/$INSTALLER_ROCM_VERSION_NAME"
    
    if [[ "$stat_dir" == //* ]]; then
        stat_dir="/${stat_dir#//}"
    fi

    echo $stat_dir:
    echo ----------------------------
    echo "size:" 
    echo "-----"
    echo "$(du -sh $stat_dir | awk '{print $1}')"
    echo "$(du -sb $stat_dir | awk '{print $1}')" bytes
    echo "------"
    echo "types:"
    echo "------"
    echo "files = $(find $stat_dir -type f | wc -l)"
    echo "dirs  = $(find $stat_dir -type d | wc -l)"
    echo "links = $(find $stat_dir -type l | wc -l)"
    echo "        ------"
    echo "        $(find $stat_dir | wc -l)"
    echo ----------------------------
}

install_deps() {
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    echo -e "\e[96mINSTALL Dependencies : $DISTRO_NAME $DISTRO_VER\e[0m"
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
    echo Installing Dependencies...
    
    echo "DEPS_ARG = $DEPS_ARG"
    
    local status=0
    
    # select switch deps to be install (rocm/amdgpu)
    if [[ $ROCM_INSTALL == 1 ]]; then
        deps_rocm="rocm"
    fi
    
    if [[ $AMDGPU_INSTALL == 1 ]]; then
        deps_amdgpu="amdgpu"
    fi
    
    # check for verbose logging enable
    if [[ $VERBOSE == 1 ]]; then
        depOp="verbose "
    fi
    
    if [[ $PROMPT_USER == 1 ]]; then
        depOp+="prompt"
    fi
    
    # parse the dependency args
    if [[ $DEPS_ARG == "list" ]]; then
        
        echo Listing required dependencies
        
        ./deps-installer.sh $deps_rocm $deps_amdgpu $depOp list
        status=$?
        
         if [[ status -ne 0 ]]; then
            print_err "Failed Dependencies list."
            exit 1
        fi
        
        exit 0
        
    elif [[ $DEPS_ARG == "validate" ]]; then

        echo Validating required dependencies

        ./deps-installer.sh $deps_rocm $deps_amdgpu $depOp
        status=$?
        
        if [[ status -ne 0 ]]; then
            print_err "Failed Dependencies validation."
            exit 1
        fi
        
        exit 0
        
    elif [[ $DEPS_ARG == "install" ]] || [[ $DEPS_ARG == "install-only" ]]; then
    
       ./deps-installer.sh $deps_rocm $deps_amdgpu $depOp install
        status=$?
        
        if [[ status -ne 0 ]]; then
            print_err "Failed Dependencies Install."
            exit 1
        fi
        
        if [[ $DEPS_ARG == "install-only" ]]; then
            echo Only dependencies installed.  Exiting.
            exit 0
        fi
    
    elif [[ $DEPS_ARG == "file" ]] || [[ $DEPS_ARG == "file-only" ]]; then
    
        ./deps-installer.sh install-file "$DEPS_ARG2" $depOp
        status=$?
        
        if [[ status -ne 0 ]]; then
            print_err "Failed Dependencies Install."
            exit 1
        fi
        
        if [[ $DEPS_ARG == "file-only" ]]; then
            echo Only dependencies installed.  Exiting.
            exit 0
        fi

    else
        print_err "Invalid dependencies argument."
        exit 1
    fi

    echo Installing Dependencies...Complete.
}

read_components() {
    local component_var=${1:-"base"}  # "base" or "gfx"

    echo --------------------------------
    echo "Read Component Configuration: $COMPO_FILE ..."

    if [ ! -f $COMPO_FILE ]; then
        print_err "Component config file $COMPO_FILE does not exist."
        exit 1
    fi

    if [[ "$component_var" == "gfx" ]]; then
        # Append to COMPONENTS_GFX
        while IFS= read -r compo; do
            COMPONENTS_GFX="$COMPONENTS_GFX $compo"
        done < "$COMPO_FILE"
        echo "COMPONENTS_GFX = $COMPONENTS_GFX"
    else
        # Append to COMPONENTS (base)
        while IFS= read -r compo; do
            COMPONENTS="$COMPONENTS $compo"
        done < "$COMPO_FILE"
        echo "COMPONENTS = $COMPONENTS"
    fi

    echo "Read Component Configuration...Complete."
}

list_components() {
    echo --------------------------------
    
    if [ -f $COMPO_ROCM_LIST ]; then    
        while IFS= read -r compo; do
            echo "$compo"
        done < "$COMPO_ROCM_LIST"
    else
        print_err "Components list $COMPO_ROCM_LIST does not exist."
    fi
    
    exit 0
}

set_prefix_scriptlet() {
    local rocm_ver_dir=$1

    # Set the PREFIX variable for extracted scriptlets (both RPM and DEB)
    # The scriptlets use RPM_INSTALL_PREFIX0 regardless of package format
    if [[ -n $rocm_ver_dir ]]; then
        echo "Setting PREFIX0 = $rocm_ver_dir"
        export RPM_INSTALL_PREFIX0="$rocm_ver_dir"

        # Ensure sudo preserves environment variables for both RPM and DEB systems
        if [[ -n $SUDO_OPTS ]]; then
            SUDO_OPTS="$SUDO -E"
        fi
        echo SUDO_OPTS = $SUDO_OPTS
    fi
}

configure_scriptlet() {
    print_str "Configuring scriptlet."
    
    local scriptlet=$(cat $1)
    
    local rocm_default="/opt"
    local rocm_reloc="$TARGET_DIR"
    local postinst_reloc="$1-reloc"
    
    echo "config: $rocm_reloc"
    
    if echo "$scriptlet" | grep -q '/opt'; then
         print_str "/opt detected -> $rocm_reloc"
    fi
    
    print_str "Using scriptlet: $1"
    
    sed "s|$rocm_default|$rocm_reloc|g" $1 > "$postinst_reloc"
    $SUDO chmod +x "$postinst_reloc"
}

install_postinst_scriptlet() {
    local component=$1
    local extract_dir=${2:-"$EXTRACT_DIR"}
    local postinst_scriptlet="$extract_dir/$component/scriptlets/postinst"

    # execute post install with arg "configure" or "1"
    if [[ -s "$postinst_scriptlet" ]]; then
        echo --------------------------------
        echo -e "\e[92mExecuting post install script for $component...\e[0m"

        if [[ $VERBOSE == 1 ]]; then
            cat "$postinst_scriptlet"
        fi

        if [[ ! $TARGET_DIR == "/" ]]; then
            print_str "Running Reloc."
            configure_scriptlet "$postinst_scriptlet"
            $SUDO_OPTS "$postinst_scriptlet-reloc" "$INSTALL_SCRIPTLET_ARG"
        else
            $SUDO_OPTS "$postinst_scriptlet" "$INSTALL_SCRIPTLET_ARG"
        fi

        echo -e "\e[92mComplete: $?\e[0m"

        POSTINST_COUNT=$((POSTINST_COUNT+1))
    fi
}

patch_scriptlet_version() {
    local scriptlet_file="$1"
    local search_string="$2"
    local replace_string="$3"

    if [[ $VERBOSE == 1 ]]; then
        echo "Replacing '$search_string' with '$replace_string'"
        echo "Processing $scriptlet_file"
    fi

    # Create a backup of the original file
    cp "$scriptlet_file" "$scriptlet_file.bak"

    # Perform the replacement
    sed -i "s/$search_string/$replace_string/g" "$scriptlet_file"

    # Check if replacement was successful
    if [ $? -eq 0 ]; then
        echo "Successfully updated $scriptlet_file"
    else
        echo "Error processing $file"
        # Restore backup if there was an error
        mv "$scriptlet_file.bak" "$scriptlet_file"
    fi

}

uninstall_prerm_scriptlet() {
    local component=$1
    local extract_dir=${2:-"$EXTRACT_DIR"}
    local prerm_scriptlet="$extract_dir/$component/scriptlets/prerm"

    # execute pre-install with arg "remove" or "0"
    if [[ -s "$prerm_scriptlet" ]]; then
        echo --------------------------------
        echo -e "\e[92mExecuting prerm script for $component...\e[0m"

        if [[ $VERBOSE == 1 ]]; then
            cat "$prerm_scriptlet"
        fi

        if [[ ! $TARGET_DIR == "/" ]]; then
            print_str "echo Running Reloc."
            configure_scriptlet "$prerm_scriptlet"
            $SUDO_OPTS "$prerm_scriptlet-reloc" "$UNINSTALL_SCRIPTLET_ARG"
        else
            $SUDO_OPTS "$prerm_scriptlet" "$UNINSTALL_SCRIPTLET_ARG"
        fi

        echo -e "\e[92mComplete: $?\e[0m"

        PRERM_COUNT=$((PRERM_COUNT+1))
    fi
}

uninstall_prerm_scriptlet_amdgpu() {
    local component=$1
    local prerm_scriptlet="$EXTRACT_DIR/$component/scriptlets/prerm"

    # execute pre-install with arg "remove" or "0"
    if [[ -s "$prerm_scriptlet" ]]; then
        echo --------------------------------
        echo -e "\e[92mExecuting prerm script for $component...\e[0m"

        if [[ $FORCE_UNINSTALL_AMDGPU == 1 ]]; then
            echo "Patching prerm scriptlet $prerm_scriptlet"
            patch_scriptlet_version $prerm_scriptlet $AMDGPU_DKMS_BUILD_NUM $INSTALLED_AMDGPU_DKMS_BUILD_NUM
        fi
        
        if [[ $VERBOSE == 1 ]]; then
            cat "$prerm_scriptlet"
        fi

        if [[ ! $TARGET_DIR == "/" ]]; then
            print_str "echo Running Reloc."
            configure_scriptlet "$prerm_scriptlet"
            $SUDO_OPTS "$prerm_scriptlet-reloc" "$UNINSTALL_SCRIPTLET_ARG"
        else
            $SUDO_OPTS "$prerm_scriptlet" "$UNINSTALL_SCRIPTLET_ARG"
        fi
        
        echo -e "\e[92mComplete: $?\e[0m"
        
        PRERM_COUNT=$((PRERM_COUNT+1))

        if [[ $FORCE_UNINSTALL_AMDGPU == 1 ]]; then
            echo "Restoring prerm scriptlet $prerm_scriptlet"
            mv "$prerm_scriptlet.bak" "$prerm_scriptlet"
        fi
    fi
}

uninstall_postrm_scriptlet() {
    local component=$1
    local extract_dir=${2:-"$EXTRACT_DIR"}
    local postrm_scriptlet="$extract_dir/$component/scriptlets/postrm"

    # execute post uninstall with arg "remove" or "0"
    if [[ -s "$postrm_scriptlet" ]]; then
        echo --------------------------------
        echo -e "\e[92mExecuting postrm script for $component...\e[0m"

        if [[ $VERBOSE == 1 ]]; then
            cat "$postrm_scriptlet"
        fi

        if [[ ! $TARGET_DIR == "/" ]]; then
            print_str "echo Running Reloc."
            configure_scriptlet "$postrm_scriptlet"
            $SUDO_OPTS "$postrm_scriptlet-reloc" "$UNINSTALL_SCRIPTLET_ARG"
        else
            $SUDO_OPTS "$postrm_scriptlet" "$UNINSTALL_SCRIPTLET_ARG"
        fi

        echo -e "\e[92mComplete: $?\e[0m"

        POSTRM_COUNT=$((POSTRM_COUNT+1))
    fi
}

detect_installed_base_components() {
    # Auto-detect which base ROCm components are actually installed at the target
    # Only identifies components that have files in the installation
    #
    # Args:
    #   $1 - ROCm installation directory to check (e.g., /opt/rocm/core-7.11)
    #
    # Returns:
    #   Sets global COMPONENTS variable with space-separated list of installed packages

    local rocm_install_dir="$1"

    COMPONENTS=""

    echo "--------------------------------"
    echo "Auto-detecting installed base components..."

    # Check each base package to see if its files are actually installed
    for pkg_dir in "$EXTRACT_ROCM_DIR/base"/*; do
        if [ -d "$pkg_dir" ]; then
            local pkg_name=$(basename "$pkg_dir")
            local content_dir="$pkg_dir/content"

            # Check if this package's files are actually installed
            if [ -d "$content_dir" ]; then
                local pkg_installed=0
                local file_count=0

                # Look for sample files from this package's content directory
                # Check up to 20 files to determine if package is installed
                # (checking more files makes detection more robust for packages where
                # early files may have been overwritten by other packages)
                while IFS= read -r -d '' file && [ $file_count -lt 20 ]; do
                    # Get the relative path from content directory
                    # Files are at: content/rocm/core-7.11/bin/..., need to strip to: bin/...
                    local rel_path="${file#$content_dir/}"
                    # Strip the rocm/core-X.XX/ prefix to get actual install path
                    rel_path="${rel_path#${INSTALLER_ROCM_VERSION_NAME}/}"
                    # Check if this file exists in the installation
                    if [ -f "$rocm_install_dir/$rel_path" ] || [ -L "$rocm_install_dir/$rel_path" ]; then
                        pkg_installed=1
                        break
                    fi
                    file_count=$((file_count + 1))
                done < <(find "$content_dir" -type f -print0 2>/dev/null)

                if [ $pkg_installed -eq 1 ]; then
                    echo "Detected installed component: $pkg_name"
                    COMPONENTS="$COMPONENTS $pkg_name"
                fi
            fi
        fi
    done

    if [ -n "$COMPONENTS" ]; then
        echo "Found $(echo $COMPONENTS | wc -w) installed base component(s)"
    else
        echo "No installed base components detected"
    fi
    echo "--------------------------------"
}

add_meta_packages_from_detected_components() {
    # Add meta packages that exist in the search directory and have scriptlets
    # Meta packages typically have no content files but have pre/post install/remove scripts
    #
    # Args:
    #   $1 - Component list variable name (COMPONENTS or COMPONENTS_GFX)
    #   $2 - Directory to search for meta packages (base or gfx directory)
    #
    # Returns:
    #   Updates the specified component list variable with meta package names

    local component_list_var="$1"
    local search_dir="$2"
    local current_components="${!component_list_var}"

    # Look for packages in search_dir that have scriptlets but were not detected by file checking
    # (These are typically meta packages with no content)
    for pkg_dir in "$search_dir"/*; do
        if [ ! -d "$pkg_dir" ]; then
            continue
        fi

        local pkg_name=$(basename "$pkg_dir")

        # Skip if already in the component list
        if echo " $current_components " | grep -q " $pkg_name "; then
            continue
        fi

        # Check if this package has removal scriptlets (prerm or postrm)
        if [ -d "$pkg_dir/scriptlets" ]; then
            if [ -s "$pkg_dir/scriptlets/prerm" ] || [ -s "$pkg_dir/scriptlets/postrm" ]; then
                # Package has removal scriptlets, add it
                echo "Adding meta package: $pkg_name (has scriptlets)"
                eval "$component_list_var=\"\$$component_list_var $pkg_name\""
            fi
        fi
    done
}

detect_installed_gfx_architectures() {
    # Auto-detect which GFX architectures are actually installed at the target
    # Only identifies GFX architectures that have files in the installation
    #
    # Args:
    #   $1 - ROCm installation directory to check (e.g., /opt/rocm/core-7.11)
    #
    # Returns:
    #   Sets global array: gfx_dirs - array of extraction directories for detected GFX architectures

    local rocm_install_dir="$1"

    gfx_dirs=()

    echo "--------------------------------"
    echo "Auto-detecting installed GFX architectures..."

    for gfx_dir in "$EXTRACT_ROCM_DIR"/gfx*/; do
        if [ -d "$gfx_dir" ]; then
            local gfx_arch=$(basename "$gfx_dir")
            gfx_arch=${gfx_arch#gfx}

            # Check if this GFX architecture is actually installed at the target
            # Look for GFX-specific files (libraries/kernels with gfx identifier)
            # Extract the numeric part (e.g., "94" from "94x", "1150" from "1150")
            local gfx_pattern="${gfx_arch%x}"  # Remove trailing 'x' if present

            # Search for files with this GFX pattern in the target installation
            local gfx_files_found=0
            if [ -d "$rocm_install_dir/lib" ]; then
                # Look for GFX-specific files (typically .co, .dat, .so files with gfx in name)
                if find "$rocm_install_dir/lib" -type f \( -name "*gfx${gfx_pattern}*" -o -name "*gfx_${gfx_pattern}*" \) 2>/dev/null | head -1 | grep -q .; then
                    gfx_files_found=1
                fi
            fi

            if [ $gfx_files_found -eq 1 ]; then
                echo "Detected installed GFX architecture: gfx${gfx_arch}"
                gfx_dirs+=("$gfx_dir")
            fi
        fi
    done

    if [ ${#gfx_dirs[@]} -gt 0 ]; then
        echo "Found ${#gfx_dirs[@]} installed GFX architecture(s)"
    else
        echo "No installed GFX architectures detected"
    fi
    echo "--------------------------------"
}

detect_installed_gfx_components() {
    # Detect which GFX components are installed for each detected GFX architecture
    #
    # Args:
    #   $1 - ROCm installation directory to check (e.g., /opt/rocm/core-7.11)
    #
    # Returns:
    #   Sets global array: gfx_components - array of "components|extract_dir" entries

    local rocm_install_dir="$1"

    gfx_components=()

    echo "--------------------------------"
    echo "Auto-detecting installed GFX components..."

    for gfx_dir in "${gfx_dirs[@]}"; do
        local gfx_arch=$(basename "$gfx_dir")
        echo "Checking components for $gfx_arch..."

        # Detect which GFX packages are actually installed
        local gfx_comps=""
        for pkg_dir in "$gfx_dir"/*; do
            if [ -d "$pkg_dir" ]; then
                local pkg_name=$(basename "$pkg_dir")
                local content_dir="$pkg_dir/content"

                # Check if this package's files are actually installed
                if [ -d "$content_dir" ]; then
                    local pkg_installed=0
                    local file_count=0

                    # Check sample files to determine if package is installed
                    # Check up to 20 files for more robust detection
                    while IFS= read -r -d '' file && [ $file_count -lt 20 ]; do
                        local rel_path="${file#$content_dir/}"
                        # Strip the rocm/core-X.XX/ prefix to get actual install path
                        rel_path="${rel_path#${INSTALLER_ROCM_VERSION_NAME}/}"
                        if [ -f "$rocm_install_dir/$rel_path" ] || [ -L "$rocm_install_dir/$rel_path" ]; then
                            pkg_installed=1
                            break
                        fi
                        file_count=$((file_count + 1))
                    done < <(find "$content_dir" -type f -print0 2>/dev/null)

                    if [ $pkg_installed -eq 1 ]; then
                        echo "  Detected: $pkg_name"
                        gfx_comps="$gfx_comps $pkg_name"
                    fi
                fi
            fi
        done

        if [ -n "$gfx_comps" ]; then
            gfx_components+=("$gfx_comps|$gfx_dir")
        fi
    done

    if [ ${#gfx_components[@]} -gt 0 ]; then
        echo "Found GFX components in ${#gfx_components[@]} architecture(s)"
    else
        echo "No GFX components detected"
    fi
    echo "--------------------------------"
}

detect_meta_packages() {
    # Detect and add meta packages that have scriptlets but no content files
    # Meta packages are packages with prerm/postrm scriptlets but were not detected by file checking
    #
    # Returns:
    #   Updates COMPONENTS (for base) and gfx_components array (for GFX)

    echo "--------------------------------"
    echo "Detecting meta packages with scriptlets..."

    # Check base directory for meta packages
    for pkg_dir in "$EXTRACT_ROCM_DIR/base"/*; do
        if [ ! -d "$pkg_dir" ]; then
            continue
        fi

        local pkg_name=$(basename "$pkg_dir")

        # Skip if already in the component list
        if echo " $COMPONENTS " | grep -q " $pkg_name "; then
            continue
        fi

        # Check if this package has removal scriptlets (prerm or postrm)
        if [ -d "$pkg_dir/scriptlets" ]; then
            if [ -s "$pkg_dir/scriptlets/prerm" ] || [ -s "$pkg_dir/scriptlets/postrm" ]; then
                # Package has removal scriptlets, add it
                echo "  Adding base meta package: $pkg_name"
                COMPONENTS="$COMPONENTS $pkg_name"
            fi
        fi
    done

    # Check GFX directories for meta packages
    local updated_gfx_components=()
    for entry in "${gfx_components[@]}"; do
        local comps="${entry%|*}"
        local gfx_extract_dir="${entry#*|}"
        local gfx_arch=$(basename "$gfx_extract_dir")

        # Add meta packages with scriptlets to this GFX directory's component list
        for pkg_dir in "$gfx_extract_dir"/*; do
            if [ ! -d "$pkg_dir" ]; then
                continue
            fi

            local pkg_name=$(basename "$pkg_dir")

            # Skip if already in the component list
            if echo " $comps " | grep -q " $pkg_name "; then
                continue
            fi

            # Check if this package has removal scriptlets (prerm or postrm)
            if [ -d "$pkg_dir/scriptlets" ]; then
                if [ -s "$pkg_dir/scriptlets/prerm" ] || [ -s "$pkg_dir/scriptlets/postrm" ]; then
                    # Package has removal scriptlets, add it
                    echo "  Adding $gfx_arch meta package: $pkg_name"
                    comps="$comps $pkg_name"
                fi
            fi
        done

        # Update the array entry with the new component list
        updated_gfx_components+=("$comps|$gfx_extract_dir")
    done
    gfx_components=("${updated_gfx_components[@]}")

    echo "Meta package detection complete"
    echo "--------------------------------"
}

install_rocm_component() {
    echo --------------------------------

    local component=$1
    local extract_dir=${2:-"$EXTRACT_DIR"}
    local content_dir="$extract_dir/$component/content"
    local script_dir="$extract_dir/$component/scriptlets"

    echo Copying content component: $component...
    
    if [[ -n $INSTALLED_PKGS ]]; then
        local matches=$(echo "$INSTALLED_PKGS" | grep -E "^($component)/")
        if [[ -n $matches ]]; then
            print_warning "Package installation of ROCm package: $component"
            echo $matches
            read -p "Overwrite package install of $compo (y/n): " option
            if [[ $option == "Y" || $option == "y" ]]; then
                echo "Proceeding with install..."
                # Copy the component content/data to the target location
                $SUDO rsync $RSYNC_OPTS_ROCM "$content_dir"/* "$TARGET_DIR"
                if [ $? -ne 0 ]; then
                    print_err "rsync error."
                    exit 1
                fi
                COMPONENT_COUNT=$((COMPONENT_COUNT+1))
            else
                echo "Skipping $component install."
            fi
        fi
    else
        # Copy the component content/data to the target location
        $SUDO rsync $RSYNC_OPTS_ROCM "$content_dir"/* "$TARGET_DIR"
        if [ $? -ne 0 ]; then
            print_err "rsync error."
            exit 1
        fi
        COMPONENT_COUNT=$((COMPONENT_COUNT+1))
    fi
        
    echo Copying content component: $component...Complete.
}

draw_progress_bar() {
    local progress=$1
    local width=50
    local filled=$((progress * width / 100))
    local empty=$((width - filled + 1))
    if [[ $NCURSES_BAR = 1 ]]; then
        tput sc
        tput el
        printf "%0.s▇" $(seq 1 $filled)
        printf "%0.s " $(seq 1 $empty)
        tput rc
    else
        printf "\r "
        printf "%0.s#" $(seq 1 $filled)
        printf "%0.s " $(seq 1 $empty)
    fi
}

check_rocm_package_install() {
    #echo Checking for package installation: $1...
    
    local rocm_loc=$1
    local ret=0

    # Package install only for /opt installs
    if [[ "$TARGET_DIR" == "/opt" || "$rocm_loc" == "/opt/rocm"* ]]; then
        # check for a rocm-core package and if it matches the version of rocm being installed
        local rocm_core_pkg=$($PKG_INSTALLED_CMD 2>&1 | grep "rocm-core")

        local rocm_ver_name=$(basename "$rocm_loc")
        # Strip both "rocm-" and "core-" prefixes to get version number
        local rocm_ver=${rocm_ver_name#rocm-}
        rocm_ver=${rocm_ver#core-}

        IFS='.' read -r x y z <<< "$rocm_ver"
        local rocm_core_ver=$(printf "%d%02d%02d" "$x" "$y" "$z")
        
        if [[ -n $rocm_core_pkg ]] && [[ "$rocm_core_pkg" == *"$rocm_core_ver"* ]]; then
            echo rocm-core package detected : $rocm_core_ver
        
            # cached the installed packages
            INSTALLED_PKGS=$($PKG_INSTALLED_CMD 2>&1)
            ret=1
        fi
    
        if [[ $FORCE_INSTALL == 1 ]]; then
            echo Force install for package-based install.
            INSTALLED_PKGS=
        fi
    fi
    
    return $ret
}

find_rocm_with_progress() {
    ROCM_DIR=
    ROCM_TARGET_ROOT=0
    
    local rocm_find_base=
    local rocm_version_dir=""
    local rocm_depth=
    
    local find_opt=$1
    local found=1
    local progress=0
    local temp_file=$(mktemp)
    
    # optimize the search based on if the target arg is set or "all" option
    if [[ "$find_opt" == "all" ]]; then
        echo Using no target.
        rocm_find_base="/"
    else
        echo Using target argument.
        rocm_depth="-maxdepth 4"
        rocm_find_base="$TARGET_DIR"
    fi

    # Look for the rocm install directory
    echo "Looking for ROCm at: $rocm_find_base"

    # Start the find command in the background
    # Use regex to match only paths ending in /rocm/core-* (not subdirectories)
    find "$rocm_find_base" $rocm_depth -type d -regex '.*/rocm/core-[^/]*$' ! -path '*/rocm-installer/component-rocm/*' ! -path '*/component-rocm/base/*/rocm/core-*' ! -path '*/component-rocm/gfx*/*/rocm/core-*' -print 2>/dev/null > "$temp_file" &
    local find_pid=$!
    
    # Update the progress bar while the find command is running
    while kill -0 "$find_pid" 2>/dev/null; do
        draw_progress_bar $progress
        progress=$((progress + 10))
        if [[ $progress -ge 100 ]]; then
            progress=0
        fi
        sleep 0.1
    done

    # Wait for the find command to complete
    wait "$find_pid"

    # Draw the final progress bar
    draw_progress_bar 100
    echo
    
    # Read the output from the temporary file into the variable
    rocm_version_dir=$(cat "$temp_file")
    
    # Check if the version directory was found and set the rocm directory path
    if [ -n "$rocm_version_dir" ]; then
        echo "ROCm detected in target $rocm_find_base"

        ROCM_DIR="$rocm_version_dir"
        echo "ROCm Install Directory found."

        # check if the path is root at the default /opt/rocm*
        if [[ "$ROCM_DIR" == /opt/rocm* ]]; then
            echo ROCm Default Root path.
            ROCM_TARGET_ROOT=1
        fi

        # list any rocm install paths
        ROCM_INSTALLS=
        echo "ROCm Installation/s:"
        while IFS= read -r rocm_inst; do
            echo "    $rocm_inst"
            ROCM_INSTALLS+="${rocm_inst},"
        done < <(sort -V "$temp_file")
        
        found=0
    fi
    
    if [ -f "$temp_file" ]; then
        rm "$temp_file"
    fi

    return $found
}

prereq_installer_check() {
    local not_install=""
    
    # Check if the require packages are installed on the system for installer to function
    for pkg in ${INSTALLER_DEPS[@]}; do
        # Check if this a package install of rocm
        if ! $PKG_INSTALLED_CMD 2>&1 | grep "$pkg" > /dev/null 2>&1; then
            echo "Package $pkg not installed."
            not_install+="$pkg "
        else
            echo $pkg is installed
        fi
    done  
    
    if [[ -n $not_install ]]; then
        print_err "ROCm Runfile installer requires installation of the following packages: $not_install"
        exit 1
    fi
}

query_prev_rocm() {
    local inst=$1
    local pkg_install=0

    # Check for a package manager install of the install version
    check_rocm_package_install "$inst"
    pkg_install=$?
    
    if [[ $pkg_install -eq 0 ]]; then
        print_warning "Runfile Installation of ROCm detected : $inst"
    else
        print_warning "Package manager Installation of ROCm detected : $inst"
    fi

    echo -e "Overwriting an existing installation may result in a loss of functionality.\n"
    read -p "Do you wish to continue with a new Runfile ROCm installation (y/n): " option
    if [[ $option == "Y" || $option == "y" ]]; then
        echo "Proceeding with install..."
    else
        echo -e "Exiting Installer.\n"
        
        if [[ $pkg_install -eq 0 ]]; then
            echo -e "Please uninstall previous version/s of ROCm using the Runfile installer.\n"
            echo "Usage:"
            echo "------"
            echo "bash $PROG target=${inst%/} uninstall-rocm"
        else
            print_err "Package installation of ROCm: $inst"
            echo "Please uninstall previous version of ROCm using the package manager."
        fi
        exit 1
    fi
}

process_prev_rocm() {
    local prev_install=0
    
    # Check if rocm install is being forced, if not, prompt the user
    if [[ $FORCE_INSTALL == 1 ]]; then
        print_warning "Forcing ROCm install."
    else
        # Check the list of rocm installs for the current target
        IFS=',' read -ra rocm_install <<< "$ROCM_INSTALLS"
        for inst in "${rocm_install[@]}"; do
            # Check if the same version is being installed
            if [[ "$inst" == *"$INSTALLER_ROCM_VERSION"* ]]; then
                echo Version match: $INSTALLER_ROCM_VERSION
                prev_install=1
                break
            fi
        done
        
        # Query the user for processing of the previous install of rocm
        if [[ $prev_install -eq 1 ]]; then
            query_prev_rocm "$inst"
        fi  
    fi
}

preinstall_rocm() {
    echo --------------------------------
    echo Preinstall ROCm...
    
    # Check for installer prerequisites
    prereq_installer_check
    
    # Check for any previous installs of ROCm for the current target
    find_rocm_with_progress "$TARGET_DIR"

    if [[ $? -eq 0 ]]; then
        process_prev_rocm
    else
        print_no_err "ROCm Install not found."
    fi
    
    echo Preinstall ROCm...Complete.
    echo --------------------------------
}

set_rocm_target() {
    EXTRACT_DIR="$EXTRACT_ROCM_DIR"
    TARGET_DIR="$TARGET_ROCM_DIR"
    
    if [[ $TARGET_DIR == "/" ]]; then
        TARGET_DIR=/opt
    fi
    
    echo "EXTRACT_DIR: $EXTRACT_DIR"
    echo "TARGET_DIR : $TARGET_DIR"
}

configure_rocm_install() {
    echo "Configuring ROCm installation using component list: $COMPO_INSTALL"

    # ROCm version info should already be set by setup_rocm_version_info() called earlier
    if [[ -z "$ROCM_VER" || -z "$ROCM_CORE_VER_DIR" ]]; then
        print_err "ROCm version information not available. setup_rocm_version_info() should be called first."
        exit 1
    fi

    # Reset component arrays
    COMPONENTS=
    COMPONENTS_GFX=

    # Split comma-separated component list
    IFS=',' read -ra COMPO_LIST <<< "$COMPO_INSTALL"

    # Process each requested component
    for compo_name in "${COMPO_LIST[@]}"; do
        # Trim whitespace
        compo_name=$(echo "$compo_name" | xargs)

        echo "Processing component: $compo_name"

        case "$compo_name" in
            core)
                local meta_base="amdrocm-core${ROCM_VER}"
                ;;
            core-dev)
                local meta_base="amdrocm-core-devel${ROCM_VER}"
                ;;
            dev-tools)
                local meta_base="amdrocm-developer-tools${ROCM_VER}"
                ;;
            core-sdk)
                local meta_base="amdrocm-core-sdk${ROCM_VER}"
                ;;
            *)
                print_err "Unknown component: $compo_name"
                echo "Valid components: core, core-dev, dev-tools, core-sdk"
                exit 1
                ;;
        esac

        # Read base meta config (for dev-tools and base components)
        if [[ "$compo_name" == "dev-tools" ]]; then
            # dev-tools is base-only (no gfx variant)
            local base_meta_file="$COMPO_META_DIR/${meta_base}-meta.config"
            if [ -f "$base_meta_file" ]; then
                echo "  Reading base meta config: $base_meta_file"
                while IFS= read -r pkg; do
                    # Check if package is in base directory
                    if [ -d "$EXTRACT_ROCM_DIR/base/$pkg" ]; then
                        COMPONENTS="$COMPONENTS $pkg"
                    fi
                done < "$base_meta_file"
            else
                print_err "Meta config file not found: $base_meta_file"
                exit 1
            fi
        else
            # For core, core-dev, core-sdk: read both base packages and gfx packages
            # First, try to read from base meta config (if gfx not specified)
            # or from gfx-specific meta config (if gfx specified)

            if [[ -n $INSTALL_GFX ]]; then
                # GFX-specific installation
                local gfx_meta_file="$COMPO_META_DIR/${meta_base}-${INSTALL_GFX}-meta.config"

                if [ -f "$gfx_meta_file" ]; then
                    echo "  Reading GFX meta config: $gfx_meta_file"
                    while IFS= read -r pkg; do
                        # Split packages between base and gfx directories
                        if [ -d "$EXTRACT_ROCM_DIR/base/$pkg" ]; then
                            # Package is in base directory
                            if [[ ! " $COMPONENTS " =~ " $pkg " ]]; then
                                COMPONENTS="$COMPONENTS $pkg"
                            fi
                        elif [ -d "$EXTRACT_ROCM_DIR/${INSTALL_GFX}/$pkg" ]; then
                            # Package is in gfx directory
                            if [[ ! " $COMPONENTS_GFX " =~ " $pkg " ]]; then
                                COMPONENTS_GFX="$COMPONENTS_GFX $pkg"
                            fi
                        fi
                    done < "$gfx_meta_file"
                else
                    print_err "GFX meta config file not found: $gfx_meta_file"
                    echo "Available GFX architectures:"
                    for gfx_dir in "$EXTRACT_ROCM_DIR"/gfx*/; do
                        if [ -d "$gfx_dir" ]; then
                            gfx_name=$(basename "$gfx_dir")
                            echo "  - $gfx_name"
                        fi
                    done
                    exit 1
                fi
            else
                # Base-only installation (no gfx specified)
                # For components that have base variants, install only base packages
                local base_meta_file="$COMPO_META_DIR/${meta_base}-meta.config"

                # Try base-specific meta file first (for base-only installs)
                if [ -f "$base_meta_file" ]; then
                    echo "  Reading base meta config: $base_meta_file"
                    while IFS= read -r pkg; do
                        # Only add packages that are in base directory
                        if [ -d "$EXTRACT_ROCM_DIR/base/$pkg" ]; then
                            if [[ ! " $COMPONENTS " =~ " $pkg " ]]; then
                                COMPONENTS="$COMPONENTS $pkg"
                            fi
                        fi
                    done < "$base_meta_file"
                else
                    # If no base-specific meta file, try to find any gfx variant and extract base packages
                    local any_gfx_meta_file=$(find "$COMPO_META_DIR" -name "${meta_base}-gfx*-meta.config" -print -quit)
                    if [ -f "$any_gfx_meta_file" ]; then
                        echo "  Reading base packages from: $any_gfx_meta_file"
                        while IFS= read -r pkg; do
                            # Only add packages that are in base directory
                            if [ -d "$EXTRACT_ROCM_DIR/base/$pkg" ]; then
                                if [[ ! " $COMPONENTS " =~ " $pkg " ]]; then
                                    COMPONENTS="$COMPONENTS $pkg"
                                fi
                            fi
                        done < "$any_gfx_meta_file"
                    else
                        print_err "No meta config file found for component: $compo_name"
                        exit 1
                    fi
                fi
            fi
        fi
    done

    # Trim leading/trailing spaces
    COMPONENTS=$(echo "$COMPONENTS" | xargs)
    COMPONENTS_GFX=$(echo "$COMPONENTS_GFX" | xargs)

    echo "Base components to install: $COMPONENTS"
    if [[ -n $INSTALL_GFX ]]; then
        echo "GFX components to install: $COMPONENTS_GFX"
    fi

    if [[ -z "$COMPONENTS" && -z "$COMPONENTS_GFX" ]]; then
        print_err "No components found for installation"
        exit 1
    fi
}

install_rocm() {
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    echo -e "\e[96mINSTALL ROCm\e[0m"
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
    set_rocm_target
    
    # If using a target, check that the target directory for install exists
    if [[ -n "$INSTALL_TARGET" && ! -d "$TARGET_DIR" ]]; then
        print_err "Target directory $TARGET_DIR for install does not exist."
        exit 1
    fi

    # ROCm version info should already be set by setup_rocm_version_info() called earlier
    if [[ -z "$INSTALLER_ROCM_VERSION_NAME" ]]; then
        print_err "ROCm version information not available. setup_rocm_version_info() should be called first."
        exit 1
    fi

    # Check if rocm is installable
    preinstall_rocm
    
    prompt_user "Install ROCm (y/n): "
    if [[ $option == "N" || $option == "n" ]]; then
        echo "Exiting Installer."
        exit 1
    fi
    
    # configure the rocm components for install
    configure_rocm_install

    # Install base ROCm components from component-rocm/base
    for compo in ${COMPONENTS[@]}; do
        echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        echo -e "\e[32mInstalling base component: $compo\e[0m"
        install_rocm_component $compo "$EXTRACT_ROCM_DIR/base"
        echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    done

    # Install GFX-specific components from component-rocm/gfxXYZ if specified
    if [[ -n $INSTALL_GFX ]]; then
        local gfx_extract_dir="$EXTRACT_ROCM_DIR/${INSTALL_GFX}"
        for compo in ${COMPONENTS_GFX[@]}; do
            echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            echo -e "\e[32mInstalling GFX component (${INSTALL_GFX}): $compo\e[0m"
            install_rocm_component $compo "$gfx_extract_dir"
            echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        done
    fi

    dump_rocm_state
    dump_stats "$TARGET_DIR"
}

uninstall_rocm_target() {
    local inst=$1

    # set the version directory
    local rocm_ver_dir="${inst%/}"
    local rocm_rm_dir="${inst%/\rocm*}"
    
    echo "ROCM Version Directory : $rocm_ver_dir/"
    echo "ROCm Removal Directory : $rocm_rm_dir"
    
    # Start the uninstall
    prompt_user "Uninstall ROCm (y/n): "
    if [[ $option == "N" || $option == "n" ]]; then
        echo "Exiting Installer."
        exit 1
    fi
    
    echo -e "\e[95mUninstalling ROCm target: $rocm_ver_dir\e[0m"

    # if the directory for removal exists, then remove the components and delete it
    if [[ -d "$rocm_rm_dir" && "rocm_rm_dir" != "/" ]]; then
        echo Uninstalling components from config.

        # Reset COMPONENTS to avoid accumulation across multiple uninstall calls
        COMPONENTS=
        COMPONENTS_GFX=

        # Check if user explicitly specified compo= or gfx= for selective uninstall
        if [[ $USER_SPECIFIED_COMPO -eq 1 || $USER_SPECIFIED_GFX -eq 1 ]]; then
            # Selective uninstall: use compo= and gfx= arguments (same as install)
            echo "Selective uninstall using compo=$COMPO_INSTALL gfx=${INSTALL_GFX:-none}"
            configure_rocm_install
        else
            # Auto-detect uninstall: remove everything that's actually installed
            echo "Auto-detecting installed components for complete uninstall..."

            # ROCm version info should already be set by setup_rocm_version_info() called earlier
            # If not set, that's okay for uninstall - we can still try based on directory structure

            # For auto-detection, we need to use component-rocm which has content/ directories.
            # component-rocm-deb only has scriptlets but no content, so detection cannot match files.
            # Save the scriptlet directory and temporarily switch to component-rocm for detection.
            local saved_extract_rocm_dir="$EXTRACT_ROCM_DIR"
            if [ -d "$PWD/component-rocm" ]; then
                EXTRACT_ROCM_DIR="$PWD/component-rocm"
            fi

            # Step 1: Detect which base components are actually installed at the target
            detect_installed_base_components "$rocm_ver_dir"

            # Step 2: Detect which GFX architectures are actually installed at the target
            detect_installed_gfx_architectures "$rocm_ver_dir"

            # Step 3: Detect which GFX components are installed for each architecture
            detect_installed_gfx_components "$rocm_ver_dir"

            # Restore the scriptlet directory for running removal scripts
            EXTRACT_ROCM_DIR="$saved_extract_rocm_dir"

            # Step 4: Detect meta packages with scriptlets
            detect_meta_packages

            # Build COMPONENTS_GFX string from gfx_components array for display
            COMPONENTS_GFX=""
            for entry in "${gfx_components[@]}"; do
                local comps="${entry%|*}"
                COMPONENTS_GFX="$COMPONENTS_GFX $comps"
            done
        fi

        echo "Base components for uninstall: $COMPONENTS"
        if [[ -n "$COMPONENTS_GFX" ]]; then
            echo "GFX components for uninstall: $COMPONENTS_GFX"
        fi

        # Set the PREFIX variable for rpm-based extracted scriptlets if required
        set_prefix_scriptlet "$rocm_ver_dir"

        # Run the pre-remove scripts for base components
        echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if [[ -n "$COMPONENTS" ]]; then
            echo "prerm executing for base components...."
            for compo in ${COMPONENTS[@]}; do
                uninstall_prerm_scriptlet $compo "$EXTRACT_ROCM_DIR/base"
            done
        fi

        # Run the pre-remove scripts for all detected GFX components
        if [ ${#gfx_components[@]} -gt 0 ]; then
            echo "prerm executing for GFX components...."
            for entry in "${gfx_components[@]}"; do
                local comps="${entry%|*}"
                local gfx_extract_dir="${entry#*|}"
                gfx_extract_dir="${gfx_extract_dir%/}"  # Remove trailing slash

                for compo in $comps; do
                    uninstall_prerm_scriptlet "$compo" "$gfx_extract_dir"
                done
            done
        fi
        echo "prerm executing....Complete."

        # Run the post-remove scripts for base components
        echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if [[ -n "$COMPONENTS" ]]; then
            echo "postrm executing for base components...."
            for compo in ${COMPONENTS[@]}; do
                uninstall_postrm_scriptlet $compo "$EXTRACT_ROCM_DIR/base"
            done
        fi

        # Run the post-remove scripts for all detected GFX components
        if [ ${#gfx_components[@]} -gt 0 ]; then
            echo "postrm executing for GFX components...."
            for entry in "${gfx_components[@]}"; do
                local comps="${entry%|*}"
                local gfx_extract_dir="${entry#*|}"
                gfx_extract_dir="${gfx_extract_dir%/}"  # Remove trailing slash

                for compo in $comps; do
                    uninstall_postrm_scriptlet "$compo" "$gfx_extract_dir"
                done
            done
        fi
        echo "postrm executing....Complete."
        
        if [ -d "$rocm_ver_dir" ]; then
            echo -e "\e[93mRemoving ROCm version directory: $rocm_ver_dir\e[0m"
            $SUDO rm -r "$rocm_ver_dir"
        fi

        # Check if the "rocm" symlink exists
        if [[ -L "$rocm_rm_dir/rocm" ]]; then
            echo "Found symlink 'rocm': $rocm_rm_dir/rocm"

            local item_count=$(find "$rocm_rm_dir" -mindepth 1 -maxdepth 1 | wc -l)

            # If the directory contains only the "rocm" symlink, delete it
            if [[ $item_count -eq 1 ]]; then
                $SUDO rm "$rocm_rm_dir/rocm"
                echo "Removing symlink 'rocm'."
            fi
        fi

        # Check if /opt/rocm (or equivalent parent) directory is empty and remove it
        local rocm_parent_dir="$rocm_rm_dir/rocm"
        if [ -d "$rocm_parent_dir" ]; then
            local parent_item_count=$(find "$rocm_parent_dir" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)

            if [[ $parent_item_count -eq 0 ]]; then
                echo -e "\e[93mRemoving empty ROCm parent directory: $rocm_parent_dir\e[0m"
                $SUDO rmdir "$rocm_parent_dir"
            else
                echo "ROCm parent directory is not empty ($parent_item_count items), keeping it."
            fi
        fi
    else
        print_err "ROCm remove target: $rocm_rm_dir does not exist."
        exit 1
    fi
    
    echo "PRERM_COUNT  = $PRERM_COUNT"
    echo "POSTRM_COUNT = $POSTRM_COUNT"
    echo -e "\e[95mUNINSTALL ROCm Components. Complete.\e[0m"   
}

uninstall_rocm() {
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    echo -e "\e[95mUNINSTALL ROCm\e[0m"
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
    set_rocm_target
    
    # Check for any previous installs of ROCm
    find_rocm_with_progress "$TARGET_DIR"
    
    if [[ $? -eq 0 ]]; then
    
        # Update the target for scriptlet hanndling
        if [[ "$TARGET_DIR" == *"rocm"* ]]; then
            TARGET_DIR="${TARGET_ROCM_DIR%/\rocm*}"
            echo "TARGET_DIR : $TARGET_DIR"
        fi

        # Check the list of rocm installs for the current target
        IFS=',' read -ra rocm_install <<< "$ROCM_INSTALLS"
        
        print_no_err "ROCm Installs found: ${#rocm_install[@]}"
        
        # Check if multiple rocm installs at current target
        if [[ ${#rocm_install[@]} > 1 ]]; then
            echo "Multiple ROCm installs for target=$INSTALL_TARGET"
            read -p "Do you wish to uninstall all ROCm installations at target (y/n): " option
            if [[ $option == "Y" || $option == "y" ]]; then
                echo "Proceeding with uninstall..."
            else
                echo "Exiting uninstall."
                exit 1
            fi
        fi
        
        # Uninstall each rocm install at target
        for inst in "${rocm_install[@]}"; do
        
            # Check for a package manager install of the install version
            check_rocm_package_install "$inst"
            if [[ $? -eq 1 ]]; then
                print_err "Package installation of ROCm: $inst"
                echo "Please uninstall previous version of ROCm using the package manager."
                exit 1
            fi
            
            uninstall_rocm_target "$inst"
            
        done
    else
        print_err "ROCm Install Directory not found."
        exit 1
    fi  
}

install_amdgpu_component() {
    echo --------------------------------

    local component=$1
    local content_dir="$EXTRACT_DIR/$component/content"
    local script_dir="$EXTRACT_DIR/$component/scriptlets"

    echo Copying content component: $component...

    # Copy the component content/data to the target location 

    if [[ $component == "amdgpu-dkms" ]]; then

        $SUDO rsync $RSYNC_OPTS_AMDGPU "$content_dir/"* "$TARGET_DIR"
        if [ $? -ne 0 ]; then
            print_err "rsync error."
            exit 1
        fi

        if [ -f "$script_dir/amdgpu_firmware" ]; then
            # workaround amdgpu_firmware being called via amdgpu-dkms.amdgpu_firmware
            $SUDO cp -p $script_dir/amdgpu_firmware $script_dir/amdgpu-dkms.amdgpu_firmware
        fi
    else
        $SUDO rsync $RSYNC_OPTS_AMDGPU "$content_dir/"* "$TARGET_DIR"

        if [ $? -ne 0 ]; then
            print_err "rsync error."
            exit 1
        fi
    fi

    COMPONENT_COUNT=$((COMPONENT_COUNT+1))

    echo Copying content component: $component...Complete.

    # Process any scriptlets
    for scriptlet in $script_dir/*; do
        if [[ -f $scriptlet ]]; then
            print_str "Detected: $scriptlet."
        fi
    done

    # Execute any postinst scriptlets
    install_postinst_scriptlet $component
}

query_prev_driver_version() {
    # Get DKMS status output
    # SUSE require sudo for dkms
    dkms_output=$($SUDO dkms status)

    while read -r line; do
        # Extract driver name and version (format is "module, version, kernel/arch/...")
        if [[ $line =~ ^([^\/]+)\/([^,]+),\ ([^,]+),\ (.+)$ ]]; then
            driver=${BASH_REMATCH[1]}
            if [ $driver = "amdgpu" ]; then
                INSTALLED_AMDGPU_DKMS_BUILD_NUM="${BASH_REMATCH[2]}"
                kernel_version=${BASH_REMATCH[3]}
                if [[ $VERBOSE == 1 ]]; then
                    echo "Driver: $driver"
                    echo "Version: $INSTALLED_AMDGPU_DKMS_BUILD_NUM"
                    echo "Kernel Version: $kernel_version"
                fi
            fi
        fi
    done < <(echo $dkms_output)
}

preinstall_amdgpu() {
    echo --------------------------------
    echo Preinstall amdgpu...

    # Check for installer prerequisites
    prereq_installer_check

    # Check for a previous amdgpu install via the package manager
    if $PKG_INSTALLED_CMD 2>&1 | grep "amdgpu-dkms"; then
        print_err "Package installation of amdgpu"
        echo "Please uninstall previous version of amdgpu using the package manager."
        exit 1
    else
        echo "No package-based amdgpu install found."
    fi

    # Check if deps are install ie. dkms
    if $PKG_INSTALLED_CMD 2>&1 | grep "dkms" > /dev/null 2>&1; then
        echo "dkms package installed."
        # Check if driver already present in dkms
        query_prev_driver_version

        if [ ! $INSTALLED_AMDGPU_DKMS_BUILD_NUM = 0 ] ; then
            print_warning "The amdgpu driver installed, version $INSTALLED_AMDGPU_DKMS_BUILD_NUM"
            echo "Consider uninstalling previous versions of amdgpu using the Runfile installer."
            echo "Usage: bash $PROG uninstall-amdgpu"
        fi
    else
        print_err "dkms package not installed."
        exit 1
    fi

    echo Preinstall amdgpu...Complete.
    echo --------------------------------
}

install_amdgpu() {
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    echo -e "\e[96mINSTALL AMDGPU\e[0m"
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Check if amdgpu is installable
    preinstall_amdgpu
    
    echo "AMDGPU_DISTRO_TAG  = $AMDGPU_DISTRO_TAG"
    echo "EXTRACT_AMDGPU_DIR = $EXTRACT_AMDGPU_DIR"
    echo "TARGET_AMDGPU_DIR  = $TARGET_AMDGPU_DIR"

    EXTRACT_DIR="$EXTRACT_AMDGPU_DIR"
    TARGET_DIR="$TARGET_AMDGPU_DIR"
    COMPO_FILE="$COMPO_AMDGPU_FILE"

    # Reset COMPONENTS to avoid including ROCm components
    COMPONENTS=

    read_components
    
    # Install each component in the component list for amdgpu
    for compo in ${COMPONENTS[@]}; do
        echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        echo -e "\e[32mInstalling $compo\e[0m"
        install_amdgpu_component $compo
        echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    done
    
    # Start amdgpu after install if required
    if [[ $AMDGPU_START == 1 ]]; then
        echo Starting amdgpu driver...
        $SUDO modprobe amdgpu
        echo Starting amdgpu driver...Complete.
    fi
}

find_and_delete() {
    local file_path=$1
    local type=$2
    local force_remove_dir=0

    find $file_path -type $type -print0 | while IFS= read -r -d '' filename; do
        remove_filename=$(echo $filename|sed -e "s%$path_to_files%%g")

        if [[ $FORCE_UNINSTALL_AMDGPU == 1 ]]; then
            if [[ "$remove_filename" == *"$AMDGPU_DKMS_BUILD_NUM"* ]]; then
                force_remove_filename=${remove_filename//$AMDGPU_DKMS_BUILD_NUM/$INSTALLED_AMDGPU_DKMS_BUILD_NUM}
                # Workaround to delete all folders in /usr/src/amdgpu because of diffrent versions numbers in directory name
                # delete all files and subfolders
                force_remove_dir=$(dirname "$force_remove_filename")
                if [[ $VERBOSE == 1 ]]; then
                    echo "remove: $force_remove_dir"
                fi
                $SUDO rm -rf "$force_remove_dir" 2>/dev/null
            fi
            if [[ "$remove_filename" == *"/lib/firmware/updates"* ]]; then
                force_remove_dir=$(dirname "$remove_filename")
                remove_filename="$force_remove_dir/*"
                if [[ $VERBOSE == 1 ]]; then
                    echo "remove: $remove_filename"
                fi
                # Here delete only files, folder deleted as in normal uninstall
                $SUDO rm -f $remove_filename 2>/dev/null
            fi
        fi

        if [ -e "$remove_filename" ] || [ -L "$remove_filename" ]; then
            if [[ $VERBOSE == 1 ]]; then
                echo "remove: $remove_filename"
            fi
            # workaround for files with spaces, ex /usr/src/amdgpu-6.10.5-2084815.24.04/amd/dkms/m4/drm_vblank_crtc_config .m4
            $SUDO rm "$remove_filename" 2>/dev/null
        fi
    done
}

delete_empty_dirs() {
    local files_dir=$1
    local installed_dir=$2

    # Recursively check all subdirs under files starting from the deepest level

    find "$files_dir" -depth -type d | while read -r subdir_files_dir; do
        # Remove the base path of installation to get the relative path
        relativePath="${subdir_files_dir#$files_dir}"

        # Construct the corresponding path in installed dir
        subdir_installed="$relativePath"

        # Check if the subdir exists in installation and is empty
        if [ -d "$subdir_installed" ] && [ -z "$(ls -A "$subdir_installed")" ]; then
            if [[ $VERBOSE == 1 ]]; then
                echo "Deleting empty directory: $subdir_installed"
            fi
            $SUDO rmdir "$subdir_installed" 2>/dev/null
        fi
    done
}

uninstall_amdgpu() {
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    echo -e "\e[95mUNINSTALL amdgpu\e[0m"
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    echo "EXTRACT_AMDGPU_DIR = $EXTRACT_AMDGPU_DIR"
    echo "TARGET_AMDGPU_DIR  = $TARGET_AMDGPU_DIR"

    EXTRACT_DIR="$EXTRACT_AMDGPU_DIR"
    TARGET_DIR="$TARGET_AMDGPU_DIR"

    query_prev_driver_version

    if [ $INSTALLED_AMDGPU_DKMS_BUILD_NUM == 0 ] ; then
        print_err "amdgpu driver not installed."
        echo "Please install amdgpu using the Runfile installer."
        echo "Usage: bash $PROG amdgpu"
        exit 1
    fi

    echo "Installed amdgpu version $INSTALLED_AMDGPU_DKMS_BUILD_NUM"
    echo "Runfile amdgpu version $AMDGPU_DKMS_BUILD_NUM"

    if [ ! $INSTALLED_AMDGPU_DKMS_BUILD_NUM == $AMDGPU_DKMS_BUILD_NUM ] ; then
        print_err "amdgpu driver installed version does not match runfile version."
        prompt_user "Force uninstall (y/n): "
        if [[ $option == "Y" || $option == "y" ]]; then
            FORCE_UNINSTALL_AMDGPU=1;
        fi
        if [[ $option == "N" || $option == "n" ]]; then
            exit 1
        fi
    fi

    echo Uninstalling components from config.

    COMPO_FILE="$COMPO_AMDGPU_FILE"

    # Reset COMPONENTS to avoid including ROCm components
    COMPONENTS=

    read_components

    # Run the pre-remove scripts for each component
    # Workaround for amdgpu packages order
    #for compo in ${COMPONENTS[@]}; do
    remove_arr=($COMPONENTS)
    for(( i=0; i<${#remove_arr[@]}; i++ )) do
        compo=${remove_arr[i]}

        uninstall_prerm_scriptlet_amdgpu $compo

        # remove files
        path_to_files="$EXTRACT_AMDGPU_DIR/$compo/content"

        echo "Removing amdgpu files..."
        find_and_delete $path_to_files "l"
        find_and_delete $path_to_files "f"

        delete_empty_dirs $path_to_files "/"

        uninstall_postrm_scriptlet $compo
    done

    # Update initramfs for all kernels
    $SUDO $UPDATE_INITRAMFS_CMD

    echo "PRERM_COUNT  = $PRERM_COUNT"
    echo "POSTRM_COUNT = $POSTRM_COUNT"
    echo -e "\e[95mUNINSTALL amdgpu Components. Complete.\e[0m"
}

install_postint_scriptlets() {
    echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    echo Running post install scripts...

    # Run post-install scripts for base components
    if [[ -n "$COMPONENTS" ]]; then
        echo "postinst executing for base components...."
        for compo in ${COMPONENTS[@]}; do
            install_postinst_scriptlet $compo "$EXTRACT_ROCM_DIR/base"
        done
    fi

    # Run post-install scripts for GFX-specific components
    # Handle both explicit gfx= specification and auto-detected GFX components
    if [[ -n $INSTALL_GFX ]]; then
        # Explicit gfx= was specified
        local gfx_extract_dir="$EXTRACT_ROCM_DIR/${INSTALL_GFX}"
        echo "postinst executing for GFX components ($INSTALL_GFX)...."
        for compo in ${COMPONENTS_GFX[@]}; do
            install_postinst_scriptlet $compo "$gfx_extract_dir"
        done
    elif [ ${#gfx_components[@]} -gt 0 ]; then
        # Auto-detected GFX components (from uninstall-like detection)
        echo "postinst executing for auto-detected GFX components...."
        for entry in "${gfx_components[@]}"; do
            local comps="${entry%|*}"
            local gfx_extract_dir="${entry#*|}"
            gfx_extract_dir="${gfx_extract_dir%/}"  # Remove trailing slash

            for compo in $comps; do
                install_postinst_scriptlet "$compo" "$gfx_extract_dir"
            done
        done
    fi

    echo Running post install scripts...Complete.
}

install_post_rocm_etc() {
    local component=$1
    local content_etc_dir="$2"
    
    if [[ "$component" == "rocm-opencl" ]]; then
        echo -e "\e[32mProcessing for rocm-opencl.\e[0m"
        if [ -d "$content_etc_dir/OpenCL/vendors/" ]; then
            echo /etc/OpenCL/vendors/
            $SUDO mkdir -p /etc/OpenCL/vendors
            $SUDO chmod 755 /etc/OpenCL /etc/OpenCL/vendors
            $SUDO rsync $RSYNC_OPTS_ROCM "$content_etc_dir/OpenCL/vendors/"* "/etc/OpenCL/vendors/"
            if [ $? -ne 0 ]; then
                print_err "rsync error."
                exit 1
            fi
            $SUDO ldconfig
        fi
    fi
}

install_post_rocm_extras() {
    echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    echo Setting up extra ROCm post install...

    local content_etc_dir=

    # Process base components
    if [[ -n "$COMPONENTS" ]]; then
        for compo in ${COMPONENTS[@]}; do
            content_etc_dir="$EXTRACT_ROCM_DIR/base/$compo/content-etc"
            if [ -d "$content_etc_dir" ]; then
                echo "/etc content for component: $compo"
                install_post_rocm_etc "$compo" "$content_etc_dir"
            fi
        done
    fi

    # Process GFX-specific components
    # Handle both explicit gfx= specification and auto-detected GFX components
    if [[ -n $INSTALL_GFX ]]; then
        # Explicit gfx= was specified
        local gfx_extract_dir="$EXTRACT_ROCM_DIR/${INSTALL_GFX}"
        for compo in ${COMPONENTS_GFX[@]}; do
            content_etc_dir="$gfx_extract_dir/$compo/content-etc"
            if [ -d "$content_etc_dir" ]; then
                echo "/etc content for GFX component: $compo"
                install_post_rocm_etc "$compo" "$content_etc_dir"
            fi
        done
    elif [ ${#gfx_components[@]} -gt 0 ]; then
        # Auto-detected GFX components (from uninstall-like detection)
        for entry in "${gfx_components[@]}"; do
            local comps="${entry%|*}"
            local gfx_extract_dir="${entry#*|}"
            gfx_extract_dir="${gfx_extract_dir%/}"  # Remove trailing slash

            for compo in $comps; do
                content_etc_dir="$gfx_extract_dir/$compo/content-etc"
                if [ -d "$content_etc_dir" ]; then
                    echo "/etc content for auto-detected GFX component: $compo"
                    install_post_rocm_etc "$compo" "$content_etc_dir"
                fi
            done
        done
    fi

    echo Setting up extra ROCm post install...Complete.
}

install_post_rocm() {
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    echo -e "\e[96mINSTALL ROCm post-install config\e[0m"
    echo =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    local rocm_ver_dir=

    # Select component directory based on OS type
    if [ "$PACKAGE_TYPE" == "deb" ]; then
        if [ -d "$PWD/component-rocm-deb" ]; then
            EXTRACT_ROCM_DIR="$PWD/component-rocm-deb"
            echo "Using DEB component directory: $EXTRACT_ROCM_DIR"
        else
            print_warning "component-rocm-deb not found, falling back to component-rocm"
            EXTRACT_ROCM_DIR="$PWD/component-rocm"
        fi
    else
        EXTRACT_ROCM_DIR="$PWD/component-rocm"
        echo "Using RPM component directory: $EXTRACT_ROCM_DIR"
    fi

    # check if postrocm is part of a rocm install
    if [[ $ROCM_INSTALL == 1 ]]; then
        echo ROCm post-install...
        if [[ $TARGET_DIR == "/" ]]; then
            rocm_ver_dir="/opt/$INSTALLER_ROCM_VERSION_NAME"
        else
            rocm_ver_dir="$TARGET_DIR/$INSTALLER_ROCM_VERSION_NAME"
        fi

        # Components are already configured from the rocm install
        # COMPONENTS and COMPONENTS_GFX are already populated by configure_rocm_install

    else
        echo ROCm post-install for target...

        # Use default target if not specified
        if [[ -z "$INSTALL_TARGET" ]]; then
            echo "No target specified, using default: $TARGET_ROCM_DEFAULT_DIR"
            TARGET_ROCM_DIR="$TARGET_ROCM_DEFAULT_DIR"
        fi

        set_rocm_target

        # check if target has a rocm install
        find_rocm_with_progress "$TARGET_DIR"
        if [[ $? -ne 0 ]]; then
            print_err "ROCm runfile install at target $TARGET_DIR not found."
            exit 1
        fi

        IFS=',' read -ra rocm_install <<< "$ROCM_INSTALLS"
        print_no_err "ROCm Installs found: ${#rocm_install[@]}"

        # Only allow for single post-rocm install
        if [[ ${#rocm_install[@]} > 1 ]]; then
            print_err "Multiple ROCm installation found.  Please select a single target for post install."
            exit 1
        fi

        rocm_ver_dir="$ROCM_DIR"
        TARGET_DIR="${TARGET_DIR%/\rocm-[0-9]*}"

        # Check if user explicitly specified compo= or gfx= for selective post-install
        if [[ $USER_SPECIFIED_COMPO -eq 1 || $USER_SPECIFIED_GFX -eq 1 ]]; then
            # Selective post-install: use compo= and gfx= arguments (same as install)
            echo "Selective post-install using compo=$COMPO_INSTALL gfx=${INSTALL_GFX:-none}"
            configure_rocm_install
        else
            # Auto-detect post-install: configure everything that's actually installed
            echo "Auto-detecting installed components for post-install..."

            # ROCm version info should already be set by setup_rocm_version_info() called earlier
            # If not set, that's okay for post-install - we can still try based on directory structure

            # For auto-detection, we need to use component-rocm which has content/ directories.
            # component-rocm-deb only has scriptlets but no content, so detection cannot match files.
            # Save the scriptlet directory and temporarily switch to component-rocm for detection.
            local saved_extract_rocm_dir="$EXTRACT_ROCM_DIR"
            if [ -d "$PWD/component-rocm" ]; then
                EXTRACT_ROCM_DIR="$PWD/component-rocm"
                echo "Using component-rocm for auto-detection (has content/ directories)"
            fi

            # Step 1: Detect which base components are actually installed at the target
            detect_installed_base_components "$rocm_ver_dir"

            # Step 2: Detect which GFX architectures are actually installed at the target
            detect_installed_gfx_architectures "$rocm_ver_dir"

            # Step 3: Detect which GFX components are installed for each architecture
            detect_installed_gfx_components "$rocm_ver_dir"

            # Restore the scriptlet directory for running post-install scripts
            EXTRACT_ROCM_DIR="$saved_extract_rocm_dir"

            # Step 4: Detect meta packages with scriptlets
            detect_meta_packages

            # Build COMPONENTS_GFX string from gfx_components array for display
            COMPONENTS_GFX=""
            for entry in "${gfx_components[@]}"; do
                local comps="${entry%|*}"
                COMPONENTS_GFX="$COMPONENTS_GFX $comps"
            done
        fi

        # Now check if the target rocm version matches the installer rocm version
        local rocm_ver_name=$(basename "${rocm_install[0]}")
        # Remove both "rocm-" and "core-" prefixes and extract version
        local rocm_ver_extracted="${rocm_ver_name#rocm-}"  # Remove rocm- prefix if present
        rocm_ver_extracted="${rocm_ver_extracted#core-}"   # Remove core- prefix if present
        local rocm_ver_short=$(echo "$rocm_ver_extracted" | cut -d '.' -f 1-2)

        echo "Installer ROCm version: $ROCM_VER"
        echo "Target ROCm version   : $rocm_ver_short"

        if [[ "$rocm_ver_short" != "$ROCM_VER" ]]; then
            print_err "ROCm version mismatch between installer ($ROCM_VER) and target ($rocm_ver_short)."
            exit 1
        fi
    fi

    echo "rocm_ver_dir: $rocm_ver_dir"
    echo "TARGET_DIR  : $TARGET_DIR"

    # Verify that components are configured
    if [[ -z "$COMPONENTS" && -z "$COMPONENTS_GFX" ]]; then
        print_err "No components configured for post-install. Use compo= to specify components."
        exit 1
    fi

    echo "Post-install will process the following components:"
    echo "  Base components: $COMPONENTS"
    if [[ -n "$COMPONENTS_GFX" ]]; then
        echo "  GFX components: $COMPONENTS_GFX"
    fi

    # Set the PREFIX variable for rpm-based extracted scriptlets if required
    set_prefix_scriptlet "$rocm_ver_dir"

    # Run all postinstall scripts for the components
    install_postint_scriptlets

    # Execute any extra non-scriptlet post install
    install_post_rocm_extras
}

set_gpu_access() {
    echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    echo Setting GPU Access...

    if [[ $GPU_ACCESS == "user" ]]; then
        echo Adding current user: $USER to render,video group.

        $SUDO usermod -aG render,video $USER

        echo -e "\e[31m< System reboot may be required >\e[0m"

    elif [[ $GPU_ACCESS == "all" ]]; then
        echo Enabling GPU access for all users.

        if [ ! -d "/etc/udev/rules.d" ]; then
            $SUDO mkdir -p /etc/udev/rules.d
        fi

        if [ ! -f "/etc/udev/rules.d/70-amdgpu.rules" ]; then
            $SUDO touch "/etc/udev/rules.d/70-amdgpu.rules"
        fi

        if ! grep -q "drm" "/etc/udev/rules.d/70-amdgpu.rules"; then
            # add udev rule
            echo Adding udev rule.
            $SUDO tee -a /etc/udev/rules.d/70-amdgpu.rules <<EOF
KERNEL=="kfd", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", MODE="0666"
EOF
            $SUDO udevadm control --reload-rules && $SUDO udevadm trigger
        fi
    else
        print_err "Invalid GPU Access option."
    fi

    echo Setting GPU Access...Complete.
}

####### Main script ###############################################################

# Create the installer log directory
if [ ! -d $RUN_INSTALLER_LOG_DIR ]; then
    mkdir -p $RUN_INSTALLER_LOG_DIR
fi

exec > >(tee -a "$RUN_INSTALLER_CURRENT_LOG") 2>&1

echo =================================
echo ROCm INSTALLER
echo =================================

SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)
SUDO_OPTS="$SUDO"
PROG=${0##*/}

os_release

# Set up distro-specific AMDGPU paths after distro is determined
AMDGPU_DISTRO_TAG=$(get_amdgpu_distro_tag)
if [ $? -ne 0 ]; then
    print_err "Failed to determine AMDGPU distro tag"
    exit 1
fi

# Update AMDGPU paths to use distro-specific subdirectory
EXTRACT_AMDGPU_DIR="$EXTRACT_AMDGPU_DIR/$AMDGPU_DISTRO_TAG"
COMPO_AMDGPU_FILE="$EXTRACT_AMDGPU_DIR/amdgpu-packages.config"

echo "args: $@"
echo --------------------------------

# parse args
while (($#))
do
    case "$1" in
    help)
        usage
        exit 0
        ;;
    version)
        get_version
        exit 0
        ;;
    deps=*)
        DEPS_ARG="${1#*=}"
        DEPS_ARG2="$2"
        echo "Using Dependency args : $DEPS_ARG"
        if [[ $DEPS_ARG == "file" ]]; then
            echo "Using Dependency args2: $DEPS_ARG2"
        fi
        shift
        ;;
    amdgpu)
        AMDGPU_INSTALL=1
        AMDGPU_ARG="dkms"
        echo "Using amdgpu args : $AMDGPU_ARG"
        shift
        ;;
    amdgpu=*)
        AMDGPU_INSTALL=1
        AMDGPU_ARG="${1#*=}"
        echo "Using amdgpu args : $AMDGPU_ARG"
        shift
        ;;
    amdgpu-start)
        AMDGPU_START=1
        echo "Start amdgpu on install."
        shift
        ;;
    rocm)
        ROCM_INSTALL=1
        ROCM_ARG="all"
        echo "Using ROCm args : $ROCM_ARG"
        shift
        ;;
    rocm=*)
        ROCM_INSTALL=1
        ROCM_ARG="${1#*=}"
        echo "Using ROCm args : $ROCM_ARG"
        shift
        ;;
    target=*)
        INSTALL_TARGET="${1#*=}"
        echo Using install target location: $INSTALL_TARGET
        TARGET_ROCM_DIR="$INSTALL_TARGET"
        shift
        ;;
    gfx=*)
        INSTALL_GFX="${1#*=}"
        USER_SPECIFIED_GFX=1
        echo Using GFX architecture: $INSTALL_GFX
        shift
        ;;
    compo=*)
        COMPO_INSTALL="${1#*=}"
        USER_SPECIFIED_COMPO=1
        echo Using component installation: $COMPO_INSTALL
        shift
        ;;
    force)
        FORCE_INSTALL=1
        echo "Forcing install."
        shift
        ;;
    runfile)
        RUNFILE_INSTALL=1
        echo "install from runfile."
        shift
        ;;
    postrocm)
        echo "Enabling Post ROCm install."
        POST_ROCM_INSTALL=1
        shift
        ;;
    nopostrocm)
        echo "Disabling Post ROCm install."
        POST_ROCM_INSTALL=0
        shift
        ;;
    gpu-access=*)
        GPU_ACCESS="${1#*=}"
        echo Setting GPU access: $GPU_ACCESS
        shift
        ;;
    findrocm)
        echo "Finding ROCm install."
        
        find_rocm_with_progress "all"

        if [[ $? -eq 1 ]]; then
            echo "ROCm Install not found."
            exit 1
        fi
        
        IFS=',' read -ra rocm_install <<< "$ROCM_INSTALLS"
        echo
        print_no_err "ROCm Installs found: ${#rocm_install[@]}"
        
        echo -e "\nChecking rocm installation type...\n"
        for inst in "${rocm_install[@]}"; do
            check_rocm_package_install "$inst"
            if [[ $? -eq 0 ]]; then
                echo "Runfile        : $inst"
            else
                echo "Package manager: $inst"
            fi
            
        done
        
        exit 0
        ;;
    complist)
        echo Component List
        list_components
        shift
        ;;
    prompt)
        echo "Enabling user prompts."
        PROMPT_USER=1
        shift
        ;;
    verbose)
        echo "Enabling verbose logging."
        VERBOSE=1
        RSYNC_OPTS_ROCM+="--itemize-changes -v "
        RSYNC_OPTS_AMDGPU+="--itemize-changes -v "
        shift
        ;;
    uninstall-rocm)
        echo "Enabling Uninstall ROCm"
        UNINSTALL_ROCM=1

        DEPS_ARG=
        ROCM_INSTALL=0
        AMDGPU_INSTALL=0
        POST_ROCM_INSTALL=0
        GPU_ACCESS=
        shift
        ;;
    uninstall-amdgpu)
        echo "Enabling Uninstall amdgpu"
        UNINSTALL_AMDGPU=1
        
        DEPS_ARG=
        ROCM_INSTALL=0
        AMDGPU_INSTALL=0
        GPU_ACCESS=
        shift
        ;;
    *)
        print_err "Invalid argument: $1"
        echo "For usage: $PROG help"
        exit 1
        ;;
    esac
done

# Validate gfx= argument
validate_gfx_arg

get_version

# Set up ROCm version information early if any ROCm operations are requested
if [[ $ROCM_INSTALL == 1 || $POST_ROCM_INSTALL == 1 || $UNINSTALL_ROCM == 1 ]]; then
    setup_rocm_version_info
fi

prompt_user "Start installation (y/n): "
if [[ $option == "N" || $option == "n" ]]; then
    echo "Exiting Installer."
    exit 1
fi

# Install/Check any dependencies
if [[ -n $DEPS_ARG ]]; then
    install_deps
fi

# Install ROCm components if required
if [[ $ROCM_INSTALL == 1 ]]; then
    install_rocm
fi

# Install/set any post install configuration
if [[ $POST_ROCM_INSTALL == 1 ]]; then
    install_post_rocm
fi

# Install AMDGPU components if required
if [[ $AMDGPU_INSTALL == 1 ]]; then
    install_amdgpu
fi

# Apply any GPU access requirements
if [[ -n $GPU_ACCESS ]]; then
    set_gpu_access
fi

# Uninstall if required
if [[ $UNINSTALL_ROCM == 1 ]]; then
    uninstall_rocm
fi

# Uninstall amdgpu if required
if [[ $UNINSTALL_AMDGPU == 1 ]]; then
    uninstall_amdgpu
fi

prompt_user "Exit installer (y/n): "
if [[ $option == "Y" || $option == "y" ]]; then
    echo "Exiting Installer."
fi

echo "Installer log stored in: $RUN_INSTALLER_CURRENT_LOG"

