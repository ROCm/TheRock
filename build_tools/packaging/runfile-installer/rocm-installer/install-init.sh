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

get_version() {
    i=0
    
    while IFS= read -r line; do
        case $i in
            0) INSTALLER_VERSION="$line" ;;
            1) ROCM_VERSION="$line" ;;
            2) DISTRO_BUILD_VERSION="$line" ;;
            3) ROCM_BUILD_NUM="$line" ;;
            4) AMDGPU_DKMS_BUILD_NUM="$line" ;;
            5) BUILD_INSTALLER_NAME="$line" ;;
        esac
        
        i=$((i+1))
    done < "./VERSION"
}

print_version() {
    echo ROCm Runfile Installer Version : $INSTALLER_VERSION-$ROCM_VERSION
    echo ROCm Runfile Installer Package : $BUILD_INSTALLER_NAME
}

os_release() {
    get_version
    
    if [[ -r  /etc/os-release ]]; then
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


####### Main script ###############################################################

SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)

os_release

# parse args
while (($#))
do
    case "$1" in
    version)
        print_version
        exit 0
        ;;
    *)
        ARGS+="$1 "
        shift
        ;;
    esac
done

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
    ./rocm-installer.sh runfile $ARGS
fi

