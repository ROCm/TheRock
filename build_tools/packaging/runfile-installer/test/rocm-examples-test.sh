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
ROCM_EXAMPLES_LOGS_DIR="$PWD/logs"
ROCM_EXAMPLES_CURRENT_LOG="$ROCM_EXAMPLES_LOGS_DIR/rocm-examples_$(date +%s).log"


###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $PROG [options]

[options}:
    help = Display this help information.
    
    rel=<release number> 
        Set the ROCm release sourced for the rocm-examples test ie.rel=6.3.1
        
    mainline
        Set rocm-examples sourced from amd-mainline branch.
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
	rhel|ol|rocky|amzn)
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

install_glslang() {
    echo ------------------------------------------------------
    echo Install glslang...

    if [ -d glslang ]; then
        $SUDO rm -r glslang
    fi

    # glslang is not available from repos, build from source
    git clone https://github.com/KhronosGroup/glslang.git
    cd glslang
    python3 ./update_glslang_sources.py
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$(pwd)/install"
    cd build
    make -j$(nproc) install

    cd ..
    export PATH="$PATH:$(pwd)/install/bin"
    cd ..

    echo Install glslang...Complete.
}

# Needed for test rocprofv3-advanced in amd-mainline
install_pyyaml() {
    echo ------------------------------------------------------
    echo "Installing YAML (pyyaml) for python3"
    python3 -m pip install pyyaml
}

install_shaderc() {
    echo ------------------------------------------------------
    echo Install shaderc...

    if [ -d shaderc ]; then
        $SUDO rm -r shaderc
    fi

    # shaderc is not available from repos, build from source
    git clone https://github.com/google/shaderc
    cd shaderc
    ./utils/git-sync-deps
    mkdir build
    cd build
    # workaround for gcc 8 in rhel 8.10
    cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD_LIBRARIES="-lstdc++fs" .. 
    ninja

    cd ..
    export PATH="$PATH:$(pwd)/build/glslc"
    cd ..

    echo Install shaderc...Complete.
}

install_deps() {
    echo ------------------------------------------------------
    echo Install dependencies...

    # install any dependencies for rocm-examples
    if [ $DISTRO_PACKAGE_MGR == "apt" ]; then
        # Ubuntu 22 or Debian 12
        if [[ $DISTRO_VER == 22* ]] || [[ $DISTRO_VER == 12* ]]; then
            echo Installing deps for ${DISTRO_NAME} 22...
            $SUDO apt-get install -y git cmake libglfw3-dev libsuitesparse-dev libtbb-dev glslang-tools libdw-dev
            install_pyyaml
        # Ubuntu 24 or Debian 13
        elif [[ $DISTRO_VER == 24* ]] || [[ $DISTRO_VER == 13* ]]; then
            echo Installing deps for ${DISTRO_NAME} 24...
            $SUDO apt-get install -y git cmake libglfw3-dev libsuitesparse-dev libtbb-dev glslang-tools glslc libdw-dev
        else
            echo Installing deps for ${DISTRO_NAME}...
            $SUDO apt-get install -y git cmake libglfw3-dev libsuitesparse-dev libtbb-dev glslang-tools glslc
        fi

    elif [ $DISTRO_PACKAGE_MGR == "dnf" ]; then

        if [[ $DISTRO_VER == 8* ]]; then
            echo Installing deps for ${DISTRO_NAME}8...
            $SUDO dnf install -y gcc-c++ git cmake glfw-devel vulkan-headers vulkan-loader-devel vulkan-validation-layers mesa-libGL-devel
            $SUDO dnf install -y gcc-toolset-11 ninja-build
            install_glslang
            install_shaderc

        elif [[ $DISTRO_VER == 9* ]]; then
            echo Installing deps for ${DISTRO_NAME}9...
            if [ $DISTRO_NAME = "rocky" ]; then
                $SUDO dnf install -y gcc-c++ git cmake glfw-devel glslang-devel vulkan-loader-devel libshaderc-devel glslc
            else
                $SUDO dnf install -y gcc-c++ git cmake glfw-devel glslang-devel vulkan-loader-devel libshaderc-devel glslc
            fi
            
        elif [[ $DISTRO_VER == 10* ]]; then
            echo Installing deps for ${DISTRO_NAME}10...
            $SUDO dnf install -y gcc-c++ git cmake glfw-devel glslang-devel vulkan-loader-devel libshaderc-devel glslc elfutils-devel
        
        elif [[ $DISTRO_VER == 2023 ]] && [[ $DISTRO_NAME = "amzn" ]] ; then
            echo Installing deps for $DISTRO_NAME $DISTRO_VER...
            $SUDO dnf install -y gcc-c++ git cmake glslang-devel vulkan-loader-devel libshaderc-devel glslc
            
        else
            echo "Unsupported version for EL."
            exit 1
        fi
        
    elif [ $DISTRO_PACKAGE_MGR == "zypper" ]; then			
        $SUDO zypper install -y git cmake gcc14-c++ vulkan-tools vulkan-devel vulkan-validationlayers shaderc libdw-devel
        
        install_glslang
        install_pyyaml
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

get_rocm_examples() {
    echo ------------------------------------------------------
    echo Downloading rocm-examples : $ROCM_REL ...
    
    if [ -d rocm-examples ]; then
        $SUDO rm -r rocm-examples
    fi

    # Download the rocm-example source (use release if present)
    if [[ -n $ROCM_REL ]]; then
        git clone https://github.com/ROCm/rocm-examples.git -b "release/rocm-rel-$ROCM_REL"
    elif [[ $IS_MAINLINE -eq 1 ]]; then
        git clone https://github.com/ROCm/rocm-examples.git -b "amd-mainline"
    else
        git clone https://github.com/ROCm/rocm-examples.git --depth=1
    fi
    
    echo Downloading rocm-examples : $ROCM_REL ...Complete
}

# Users must disable SELinux before running rocm systems profiler
# https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/install/install.html#post-installation-troubleshooting
disable_SELinux() {
    if [[ $DISTRO_PACKAGE_MGR == "dnf" ]]; then
        echo ------------------------------------------------------
        echo Disabling SELinux which is required by rocm systems profiler on OL/RHEL/Rocky
        sudo setenforce 0
    fi
}

enable_SELinux() {
    if [[ $DISTRO_PACKAGE_MGR == "dnf" ]]; then
        echo ------------------------------------------------------
        echo Re-enable SELinux after tests have finished running.
        sudo setenforce 1
    fi
}

build_rocm_examples() {
    echo ------------------------------------------------------
    echo Building rocm-examples...
    
    cd rocm-examples
    
    echo --------------------------
    git branch --show-current
    git rev-parse HEAD
    echo --------------------------
    
    # Build rocm-examples
    mkdir build && cd build

    # Using -Wno-dev to suppress lots of warnings during compilation.
    # To be removed after test update.
    cmake .. -DROCM_ROOT=$ROCM_PATH -Wno-dev
    cmake --build . -- -j$(nproc)
    
    echo Building rocm-examples...Complete.
}

test_rocm_examples() {
    echo ------------------------------------------------------
    echo Testing rocm-examples...
    
    ctest --output-on-failure
    
    echo Testing rocm-examples...Complete.
}

####### Main script ##############################################################

# Create the extraction log directory
if [ ! -d $ROCM_EXAMPLES_LOGS_DIR ]; then
    mkdir -p $ROCM_EXAMPLES_LOGS_DIR
fi

exec > >(tee -a "$ROCM_EXAMPLES_CURRENT_LOG") 2>&1

echo ===============================
echo ROCM-EXAMPLES TESTER
echo ===============================

IS_MAINLINE=0
PROG=${0##*/}
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
    rel=*)
        ROCM_REL="${1#*=}"
        echo "Using ROCm release : $ROCM_REL"
        shift
        ;;
    mainline)
        IS_MAINLINE=1
        echo "Using amd-mainline branch"
        shift
        ;;
    *)
        shift
        ;;
    esac
done

if [[ $IS_MAINLINE -eq 1 ]] && [[ -n $ROCM_REL ]]; then
    echo "You can't choose both amd-mainline and release branch $ROCM_REL!"
    exit 1
fi

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

install_deps

disable_SELinux

get_rocm_examples

build_rocm_examples

test_rocm_examples

enable_SELinux

if [[ -n $ROCM_EXAMPLES_CURRENT_LOG ]]; then
    echo -e "\e[32mrocm-examples log stored in: $ROCM_EXAMPLES_CURRENT_LOG\e[0m"
fi

