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
        . /etc/os-release

        DISTRO_NAME=$ID
        DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')

        case "$ID" in
        ubuntu)
	    echo "Running test on Ubuntu $DISTRO_VER."
	    DISTRO_PACKAGE_MGR="apt"
	    PACKAGE_TYPE="deb"
	    ;;
	rhel)
	    echo "Running test on RHEL $DISTRO_VER."
	    DISTRO_PACKAGE_MGR="dnf"
	    PACKAGE_TYPE="rpm"
            ;;
        sles)
	    echo "Running test on SUSE $DISTRO_VER."
	    DISTRO_PACKAGE_MGR="zypper"
	    PACKAGE_TYPE="rpm"
            ;;
        *)
            echo "$ID is not a Unsupported OS"
            exit 1
            ;;
        esac
    else
        echo "Unsupported OS"
        exit 1
    fi
}

setup_rocm() {
    echo ------------------------------------------------------
    echo Setting up ROCm paths...
    
    # Look for the rocm directory
    ROCM_VER_DIR=$(find / -type f -path '*/opt/rocm-*/.info/version' ! -path '*/rocm-installer/component-rocm/*' -print -quit 2>/dev/null)

    if [ -n "$ROCM_VER_DIR" ]; then
        echo "ROCm Install Directory found at: $ROCM_VER_DIR"
    
        ROCM_DIR=${ROCM_VER_DIR%%.info*}
        echo ROCM_DIR = $ROCM_DIR
    else
        echo "ROCm Install Directory not found"
        exit 1
    fi

    # Set the ROCm paths
    export ROCM_PATH="$ROCM_DIR"

    echo Setting up ROCm paths...Complete.
}

test-rocprof-sys() {
    echo ------------------------------------------------------
    echo TESTING rocprof-sys...
    echo ------------------------------------------------------
    
    source $ROCM_PATH/share/rocprofiler-systems/setup-env.sh

    #rocprof-sys-instrument --help
    #rocprof-sys-avail --help

    rocprof-sys-instrument --version
    rocprof-sys-avail --version
}

test-rocprof-compute() {
    echo ------------------------------------------------------
    echo TESTING rocprof-compute...
    echo ------------------------------------------------------
    
    python3 -m pip install -r $ROCM_PATH/libexec/rocprofiler-compute/requirements.txt
    which rocprof-compute
    rocprof-compute --version
}


####### Main script ###############################################################

echo ===============================
echo ROCPROF TOOLS TESTER
echo ===============================

SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)
echo SUDO: $SUDO

os_release

setup_rocm

test-rocprof-sys

test-rocprof-compute

