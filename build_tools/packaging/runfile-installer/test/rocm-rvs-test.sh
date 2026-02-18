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

usage() {
cat <<END_USAGE
Usage: $PROG [options]

[options}:
    help = Display this help information.

    build = Build RVS tool source and use this version

    rel=<release number>
        Set the RVS release sourced for the build ie.rel=6.3.1
END_USAGE
}

os_release() {
    if [[ -r  /etc/os-release ]]; then
        . /etc/os-release

        DISTRO_NAME=$ID
        DISTRO_VER=$(awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release | tr -d '"')

        case "$ID" in
        ubuntu|debian)
	    DISTRO_PACKAGE_MGR="apt"
	    PACKAGE_TYPE="deb"
	    ;;
	rhel|ol|rocky)
	    DISTRO_PACKAGE_MGR="dnf"
	    PACKAGE_TYPE="rpm"
            ;;
        sles)
	    DISTRO_PACKAGE_MGR="zypper"
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

    echo "Running test on $DISTRO_NAME $DISTRO_VER."
}


install_deps() {
    echo ------------------------------------------------------
    echo Install dependencies...

    # install any dependencies for rocm-examples
    if [ $DISTRO_PACKAGE_MGR == "apt" ]; then
        $SUDO apt-get install -y libpci3 libpci-dev doxygen unzip cmake git libyaml-cpp-dev

    elif [ $DISTRO_PACKAGE_MGR == "dnf" ]; then
        $SUDO dnf install -y cmake3 doxygen git gcc-c++ yaml-cpp-devel pciutils-devel

#  Temporary leave old code here until tests on RHEL8 and Rocky performed,
#  maybe we will need separate packages installation for RHEL8/9/Rocky
#        if [[ $DISTRO_VER == 8* ]]; then
#            echo Installing deps for ${DISTRO_NAME}8...
#            $SUDO dnf install -y gcc-c++ git cmake glfw-devel vulkan-headers vulkan-loader vulkan-validation-layers mesa-libGL-devel
#            $SUDO dnf install -y gcc-toolset-11
#            install_glslang
#
#        elif [[ $DISTRO_VER == 9* ]]; then
#	        echo Installing deps for ${DISTRO_NAME}9...
#            if [ $DISTRO_NAME = "rocky" ]; then
#                $SUDO dnf install -y gcc-c++ git cmake glfw-devel glslang-devel vulkan-loader-devel libshaderc-devel glslc
#            else
#                $SUDO dnf install -y cmake3 doxygen git gcc-c++ yaml-cpp-devel pciutils-devel
#            fi
#        else
#            echo "Unsupported version for EL."
#            exit 1
#        fi

    elif [ $DISTRO_PACKAGE_MGR == "zypper" ]; then
        $SUDO zypper install -y cmake doxygen pciutils-devel libpci3 git gcc-c++ yaml-cpp-devel

#  Temporary leave old code here until tests on different versions of SLES performed,
#  maybe we will need separate packages installation
#        if [[ $DISTRO_VER == 15.5 ]]; then
#            $SUDO pip install cmake
#        else
#            $SUDO zypper install -y cmake
#        fi

    else
        echo Unsupported Distro.
        exit 1
    fi

    echo Install dependencies...Complete.
}

setup_rocm() {
    echo ------------------------------------------------------
    echo Setting up ROCm paths...

    # Look for the rocm directory
    ROCM_VER_DIR=$(find / -type f -path '*/rocm-*/.info/version' ! -path '*/rocm-installer/component-rocm/*' -print -quit 2>/dev/null)

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

    # Set compiler paths
    export CXX=$ROCM_PATH/llvm/bin/amdclang++
    export CC=$ROCM_PATH/llvm/bin/amdclang

    echo Setting up ROCm paths...Complete.
}

get_rocm_rvs() {
    echo ------------------------------------------------------
    echo Downloading rocm-rvs : $RVS_REL ...

    if [ -d ROCmValidationSuite ]; then
        $SUDO rm -r ROCmValidationSuite
    fi

    # Download the rocm-rvs source (use release if present)
    if [[ -n $RVS_REL ]]; then
        git clone https://github.com/ROCm/ROCmValidationSuite.git -b "release/rocm-rel-$RVS_REL"
    else
        git clone https://github.com/ROCm/ROCmValidationSuite.git
    fi

    echo Downloading rocm-rvs : $RVS_REL ...Complete
}

build_rocm_rvs() {
    echo ------------------------------------------------------
    echo Building rocm-rvs...

    cd ROCmValidationSuite

    # Build rocm-rvs
    mkdir build

    cmake -B ./build -DROCM_PATH=$ROCM_PATH -DCMAKE_INSTALL_PREFIX=./build -DCPACK_PACKAGING_INSTALL_PREFIX=./build -DCMAKE_PREFIX_PATH=$ROCM_PATH
    make -C ./build -j$(nproc)

    echo Building rocm-rvs...Complete.
}

test_rocm_rvs() {
    echo ------------------------------------------------------
    echo Testing rocm-rvs...

    $RVS_PATH/rvs -g
# Commented out RCQT tests, it does not work for our installation
#    $RVS_PATH/rvs -c $RVS_CONFIG_PATH/rcqt_single.conf
    $RVS_PATH/rvs -c $RVS_CONFIG_PATH/gst_selfcheck.conf
    $RVS_PATH/rvs -c $RVS_CONFIG_PATH/gst_single.conf

    echo Testing rocm-rvs...Complete.
}

####### Main script ##############################################################

echo ===============================
echo ROCM-RVS TESTER
echo ===============================

PROG=${0##*/}
RVS_BUILD=0
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
    build)
        RVS_BUILD=1
        echo "Perform RVS buid"
        shift
        ;;
    rel=*)
        RVS_REL="${1#*=}"
        echo "Using RVS release : $RVS_REL"
        shift
        ;;
    *)
        shift
        ;;
    esac
done

# Look for the rocm directory
ROCM_VER_DIR=$(find / -type f -path '*/rocm-*/.info/version' ! -path '*/rocm-installer/component-rocm/*' -print -quit 2>/dev/null)

if [ -n "$ROCM_VER_DIR" ]; then
    echo ------------------------------------------------------
    echo "ROCm Install Directory found at: $ROCM_VER_DIR"

    ROCM_DIR=${ROCM_VER_DIR%%.info*}
    echo ROCM_DIR = $ROCM_DIR
else
    echo "ROCm Install Directory not found"
    exit 1
fi

setup_rocm

if [[ $RVS_BUILD == 1 ]]; then
    install_deps

    get_rocm_rvs

    build_rocm_rvs

    RVS_PATH="./build/bin"
    RVS_CONFIG_PATH="./rvs/conf"
else
    RVS_PATH="$ROCM_DIR/bin"
    RVS_CONFIG_PATH="$ROCM_DIR/share/rocm-validation-suite/conf"
fi

test_rocm_rvs

