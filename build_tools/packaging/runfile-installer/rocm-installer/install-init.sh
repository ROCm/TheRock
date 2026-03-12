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


###### Functions ###############################################################

os_release() {
    if [[ -r  /etc/os-release ]]; then
        # shellcheck source=/dev/null
        . /etc/os-release

        DISTRO_NAME=$ID

        case "$ID" in
        ubuntu)
            DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
            echo "Installer running on Ubuntu $DISTRO_VER."
            ;;
        debian)
            DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
            echo "Installer running on Debian $DISTRO_VER."
            ;;
        rhel)
            DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
            echo "Installer running on RHEL $DISTRO_VER."
            ;;
        centos)
            DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
            echo "Installer running on CentOS Stream $DISTRO_VER."

            # CentOS Stream 9 support only
            if [[ $DISTRO_VER != 9* ]]; then
                echo "$DISTRO_NAME $DISTRO_VER is not a supported OS"
                exit 1
            fi

            ;;
        ol)
            DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
            echo "Installer running on Oracle Linux $DISTRO_VER."
            ;;
        rocky)
            DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
            echo "Installer running on Rocky Linux $DISTRO_VER."

            # Rocky 9 support only
            if [[ $DISTRO_VER != 9* ]]; then
                echo "$DISTRO_NAME $DISTRO_VER is not a supported OS"
                exit 1
            fi

            ;;
        almalinux)
            DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
            echo "Installer running on AlmaLinux $DISTRO_VER."

            # AlmaLinux 8.x support (ManyLinux)
            if [[ $DISTRO_VER != 8* ]]; then
                echo "WARNING: This installer was built for AlmaLinux 8.x"
                echo "Detected AlmaLinux $DISTRO_VER - installation may not work correctly"
            fi

            ;;
        sles)
            if rpm -qa | grep -q "awk"; then
                DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
            else
                DISTRO_VER=$(grep '^VERSION_ID=' /etc/os-release | cut -d'=' -f2 | tr -d '"')
            fi

            echo "Installer running on SLES $DISTRO_VER."

            ;;
        amzn)
            DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')
            echo "Installer running on Amazon Linux $DISTRO_VER."
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
}

run_with_progress() {
    local message="$1"
    shift

    # Start progress indicator in background
    (
        local i=0
        local spin='-\|/'
        while true; do
            i=$(( (i+1) %4 ))
            printf "\r[%c] %s " "${spin:$i:1}" "$message"
            sleep 0.1
        done
    ) &
    local progress_pid=$!

    # Run the command and capture exit status
    "$@" 2>/dev/null
    local exit_status=$?

    # Cleanup progress indicator
    kill $progress_pid 2>/dev/null
    wait $progress_pid 2>/dev/null
    printf "\r"

    return $exit_status
}

extract_components() {
    echo "-------------------------------------------------------------"
    echo "Extracting components..."
    echo "-------------------------------------------------------------"

    cd "$INSTALLER_DIR" || exit 1

    local extract_start
    extract_start=$(date +%s)

    # Extract components archive using appropriate method
    if [[ "$ARCHIVE_TYPE" == "xz" ]]; then
        echo "Extracting compressed archive: $COMPONENTS_ARCHIVE"
        # Use embedded xz-static to avoid system dependencies
        XZ_STATIC="$INSTALLER_DIR/bin/xz-static"
        if [[ ! -f "$XZ_STATIC" ]]; then
            echo -e "\e[31mERROR: xz-static binary not found at $XZ_STATIC\e[0m"
            exit 1
        fi

        if run_with_progress "Extracting..." bash -c "\"$XZ_STATIC\" -dc \"$COMPONENTS_ARCHIVE\" | tar -xf -"; then
            local extract_end
            extract_end=$(date +%s)
            local extract_duration=$((extract_end - extract_start))
            echo -e "\e[32mExtracted components successfully ($extract_duration seconds).\e[0m"
            # Keep the archive for now, cleanup script will remove it
        else
            echo -e "\e[31mERROR: Failed to extract compressed components archive\e[0m"
            exit 1
        fi
    elif [[ "$ARCHIVE_TYPE" == "gzip" ]]; then
        echo "Extracting compressed archive: $COMPONENTS_ARCHIVE"

        if run_with_progress "Extracting..." tar -xzf "$COMPONENTS_ARCHIVE"; then
            local extract_end
            extract_end=$(date +%s)
            local extract_duration=$((extract_end - extract_start))
            echo -e "\e[32mExtracted components successfully ($extract_duration seconds).\e[0m"
            # Keep the archive for now, cleanup script will remove it
        else
            echo -e "\e[31mERROR: Failed to extract compressed components archive\e[0m"
            exit 1
        fi
    fi

    cd - >/dev/null || exit
    echo ""
}


####### Main script ###############################################################

os_release

# parse args
while (($#))
do
    case "$1" in
    *)
        ARGS+="$1 "
        shift
        ;;
    esac
done

# Check if hybrid compression mode is used
INSTALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for compressed components (supports both old and new archive names)
COMPONENTS_ARCHIVE=""
ARCHIVE_TYPE=""

# Check for xz-compressed archives first (hybrid mode)
if [[ -f "$INSTALLER_DIR/components.tar.xz" ]]; then
    COMPONENTS_ARCHIVE="components.tar.xz"
    ARCHIVE_TYPE="xz"
# Then check for gzip-compressed archives (hybrid mode)
elif [[ -f "$INSTALLER_DIR/components.tar.gz" ]]; then
    COMPONENTS_ARCHIVE="components.tar.gz"
    ARCHIVE_TYPE="gzip"
fi

if [[ -n "$COMPONENTS_ARCHIVE" ]]; then
    # Check if components are already extracted (to avoid duplicate extraction)
    if [[ -d "$INSTALLER_DIR/component-rocm" ]] || [[ -d "$INSTALLER_DIR/component-amdgpu" ]]; then
        echo "Components already extracted, skipping..."
    else
        extract_components
    fi
fi

# Check if noexec was requested - exit before running installer/UI
for arg in $ARGS; do
    if [[ "$arg" == "noexec" || "$arg" == "--noexec" ]]; then
        echo "noexec mode: extraction complete, exiting."
        exit 0
    fi
done

# Check if tests are compressed (hybrid compression mode)
if [[ -f "$INSTALLER_DIR/tests.tar.xz" ]]; then
    export TESTS_COMPRESSED="yes"
    export TESTS_ARCHIVE="$INSTALLER_DIR/tests.tar.xz"
    echo "Test components archive present. Tests will be extracted automatically when needed."
    echo ""
else
    export TESTS_COMPRESSED="no"
fi

if [ -z "$ARGS" ]; then
    # Set TERMINFO path for static ncurses compatibility across distros
    # Static ncurses built on AlmaLinux 8.10 looks for terminfo in /usr/share/terminfo
    # Ubuntu 22.04 has it in /lib/terminfo, Ubuntu 24.04 has it in /usr/share/terminfo
    if [ -z "$TERMINFO" ]; then
        if [ -d "/lib/terminfo" ]; then
            export TERMINFO=/lib/terminfo
        elif [ -d "/usr/share/terminfo" ]; then
            export TERMINFO=/usr/share/terminfo
        fi
    fi

    # Set TERM fallback for better compatibility
    if [ -z "$TERM" ]; then
        export TERM=linux
    fi

    echo Using ROCm Installer UI.
    ./rocm_ui
else
    echo Using ROCm Installer script with args: "$ARGS"
    # shellcheck disable=SC2086  # ARGS is intentionally unquoted to split arguments
    ./rocm-installer.sh $ARGS
fi

