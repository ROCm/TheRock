#!/bin/bash

# NOTE: This test script uses old static config files that have been removed.
# It needs to be updated to use the new dynamic config generation system.
# See package-puller-repo.sh for the new config system usage.

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
            echo "$ID is not a Unsupported OS"
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
    echo ++++++++++++++++++++++++++++++++
    echo Installing deps...

    # install any dependencies for rocdecode
    if [ $DISTRO_PACKAGE_MGR == "apt" ]; then
        $SUDO apt-get install -y cmake pkg-config
        $SUDO apt-get install -y vainfo

        $SUDO apt-get install -y python3-pandas python3-tabulate

    elif [ $DISTRO_PACKAGE_MGR == "dnf" ]; then
        $SUDO dnf install pkg-config

        if [[ $DISTRO_VER == 8* ]]; then
            echo Installing deps for RHEL8...

            $SUDO dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm
            $SUDO dnf install -y https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-8.noarch.rpm
            $SUDO dnf install -y cmake gcc-c++ ffmpeg ffmpeg-devel mpg123-libs libva-utils
            $SUDO dnf install -y gcc-toolset-11
            
            #source ../package-puller/config/el/8/rocm-$ROCM_VER-el8.config
            source ../package-puller/config/el/8/rocm-7.2-pre-rel-el8.config
            
        elif [[ $DISTRO_VER == 9* ]]; then
	        echo Installing deps for RHEL9...
	        
            $SUDO dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-9.noarch.rpm
            $SUDO dnf install -y https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-9.noarch.rpm
            $SUDO dnf install -y cmake gcc-c++ ffmpeg ffmpeg-devel mpg123-libs libva-utils

            #source ../package-puller/config/el/9/rocm-$ROCM_VER-el9.config
            source ../package-puller/config/el/9/rocm-7.2-pre-rel-el9.config
	        
        elif [[ $DISTRO_VER == 10* ]]; then
	        echo Installing deps for RHEL10...
	        
            $SUDO dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-10.noarch.rpm
            $SUDO dnf install -y https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-10.noarch.rpm
            $SUDO dnf install -y cmake gcc-c++ ffmpeg ffmpeg-devel mpg123-libs

            #source ../package-puller/config/el/10/rocm-$ROCM_VER-el10.config
            source ../package-puller/config/el/10/rocm-7.2-pre-rel-el10.config
	    
        else
            echo "Unsupported version for EL."
            exit 1
        fi
        
        # Install Python packages for user because site package not writable for user in RHEL
        python3 -m pip install --user pandas tabulate
        
        echo "$AMDGPU_REPO" | $SUDO tee -a /etc/yum.repos.d/amdgpu-build.repo
        if [ -n "$GRAPHICS_REPO" ]; then
            echo "$GRAPHICS_REPO" | $SUDO tee -a /etc/yum.repos.d/amdgpu-graphics.repo
        fi

    	$SUDO dnf clean all
    	$SUDO rm -rf /var/cache/dnf/*
    	
    	$SUDO dnf install -y mesa-amdgpu-va-drivers
    	
        # check if the libva-amdgpu-devel package is available and install it
        dnf info libva-amdgpu-devel >/dev/null 2>&1
        if [[ $? -eq 0 ]]; then
            $SUDO dnf install -y libva-amdgpu-devel
    	fi
        
    elif [ $DISTRO_PACKAGE_MGR == "zypper" ]; then
        $SUDO zypper install -y cmake ffmpeg-4-libavcodec-devel ffmpeg-4-libavformat-devel ffmpeg-4-libavutil-devel libva-utils libstdc++-devel
       
        python3 -m pip install pandas tabulate
       
        if [[ $DISTRO_VER == 15.6 ]]; then
            source ../package-puller/config/sle/15.6/rocm-$ROCM_VER-sle-15.6.config
        elif [[ $DISTRO_VER == 15.7 ]]; then
            #source ../package-puller/config/sle/15.6/rocm-$ROCM_VER-sle-15.6.config
            source ../package-puller/config/sle/15.6/rocm-7.2-pre-rel-sle-15.6.config
        else
            echo SLES $DISTRO_VER is not supported.
            exit 1
        fi
       
        echo "$AMDGPU_REPO" | $SUDO tee -a /etc/zypp/repos.d/amdgpu-build.repo
        if [ -n "$GRAPHICS_REPO" ]; then
            echo "$GRAPHICS_REPO" | $SUDO tee -a /etc/zypp/repos.d/amdgpu-graphics.repo
        fi

        $SUDO zypper clean
        $SUDO zypper --gpg-auto-import-keys refresh
        
        $SUDO zypper install -y mesa-amdgpu-va-drivers
        
        # install libva-amdgpu-devel for rocm 7.0.1 or eariler
        if [[ $ROCM_VER_MAJOR -lt 7 ]]; then
            $SUDO zypper install libva-amdgpu-devel
        elif [[ $ROCM_VER_MAJOR -eq 7 ]] && [[ $ROCM_VER_MINOR -eq 0 ]] && [[ $ROCM_VER_PATCH -lt 2 ]]; then
            $SUDO zypper install libva-amdgpu-devel
        else
            echo Not installing libva-amdgpu-devel.
        fi
       
    else
        echo Unsupported Distro.
        exit 1
    fi
    
    echo Installing deps...Complete.
}

cleanup() {
    echo ++++++++++++++++++++++++++++++++
    echo Cleaning up...
    
    if [ $DISTRO_PACKAGE_MGR == "dnf" ]; then
        $SUDO rm /etc/yum.repos.d/amdgpu-build.repo
        if [ -e /etc/yum.repos.d/amdgpu-graphics.repo ]; then
            $SUDO rm /etc/yum.repos.d/amdgpu-graphics.repo
        fi
        
        # cleanup dnf cache
        $SUDO dnf clean all
        $SUDO rm -r /var/cache/dnf/*
        
    elif [ $DISTRO_PACKAGE_MGR == "zypper" ]; then
        $SUDO rm /etc/zypp/repos.d/amdgpu-build.repo
        if [ -e /etc/zypp/repos.d/amdgpu-graphics.repo ]; then
            $SUDO rm /etc/zypp/repos.d/amdgpu-graphics.repo
        fi
        $SUDO zypper clean
    fi

    echo Cleaning up...Complete.
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
    
    local rocm_ver_name=$(basename "$ROCM_DIR")
    ROCM_VER=${rocm_ver_name#rocm-}
    
    ROCM_VER_MAJOR=${ROCM_VER:0:1}
    ROCM_VER_MINOR=${ROCM_VER:2:1}
    ROCM_VER_PATCH=${ROCM_VER:4:1}

    if [[ "$ROCM_VER_PATCH" == "0" || "$ROCM_VER_PATCH" == "00" ]]; then
        ROCM_VER=$ROCM_VER_MAJOR.$ROCM_VER_MINOR
    fi
    
    echo "ROCM_VER = $ROCM_VER"

    # Set the ROCm paths
    export ROCM_PATH="$ROCM_DIR"
    
    # Set compiler paths
    if [[ $DISTRO_PACKAGE_MGR != "zypper" ]]; then
        export CXX=$ROCM_PATH/llvm/bin/amdclang++
        export CC=$ROCM_PATH/llvm/bin/amdclang
    fi
    
    echo Setting up ROCm paths...Complete.
}

test_va() {
    echo ------------------------------------------------------
    echo VAINFO..
    echo ------------------------------------------------------
    
    # libva info
    vainfo --display drm
}

test_samples() {
    echo ------------------------------------------------------
    echo TESTING rocDecodeSamples...
    echo ------------------------------------------------------
    
    mkdir -p rocdecode-test/samples/videoDecode/build
    cd rocdecode-test/samples/videoDecode/build
    cmake $ROCM_PATH/share/rocdecode/samples/videoDecode
    cmake --build . -- -j$(nproc)

    cp $ROCM_PATH/share/rocdecode/test/testScripts/run_rocDecodeSamples.py .
    
    echo ">>>>> running test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    python3 ./run_rocDecodeSamples.py --rocDecode_directory "../../.." --files_directory "$ROCM_PATH/share/rocdecode/video"
    
    echo TESTING rocDecodeSamples...Complete
}

test_perf() {
    echo ------------------------------------------------------
    echo TESTING videodecodeperf...
    echo ------------------------------------------------------
    
    mkdir -p rocdecode-test/samples/videoDecodePerf/build
    cd rocdecode-test/samples/videoDecodePerf/build
    cmake $ROCM_PATH/share/rocdecode/samples/videoDecodePerf
    cmake --build . -- -j$(nproc)

    echo ">>>>> running test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    ./videodecodeperf -i /$ROCM_PATH/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4 -t 4
    
    echo TESTING videodecodeperf...Complete
}

test_simple() {
    echo ------------------------------------------------------
    echo TESTING simple [ctests]...
    echo ------------------------------------------------------
    
    mkdir -p rocdecode-test/simple/build
    cd rocdecode-test/simple/build
    cmake $ROCM_PATH/share/rocdecode/test

    echo ">>>>> running test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    ctest --output-on-failure --stop-on-failure
    
    echo TESTING simple [ctests]...Complete
}

test_rocdecodenegativetest() {
    echo ------------------------------------------------------
    echo TESTING rocdecodenegativetest...
    echo ------------------------------------------------------
    
    cmake -B rocdecode-test/rocdecodenegativetest/build $ROCM_PATH/share/rocdecode/test/rocDecodeNegativeApiTests
    cmake --build rocdecode-test/rocdecodenegativetest/build

    echo ">>>>> running test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    ./rocdecode-test/rocdecodenegativetest/build/rocdecodenegativetest
    
    echo TESTING rocdecodenegativetest...Complete

}

####### Main script ###############################################################

echo ===============================
echo ROCDECODE TESTER
echo ===============================

SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)
echo SUDO: $SUDO

os_release

if [ -d rocdecode-test ]; then
    echo Removing old rocdecode-test
    rm -r rocdecode-test
fi

setup_rocm

install_deps

test_va

# Run the tests
test_samples

test_perf

if [ -f "$ROCM_PATH/share/rocdecode/test/CMakeLists.txt" ]; then
    test_simple
else
    echo "Unable to run test_simple as the file $ROCM_PATH/share/rocdecode/test/CMakeLists.txt doesn't exist"
fi

if [ -f "$ROCM_PATH/share/rocdecode/test/rocDecodeNegativeApiTests/CMakeLists.txt" ]; then
    test_rocdecodenegativetest
else
    echo "Unable to run test_rocdecodenegativetest as the file $ROCM_PATH/share/rocdecode/test/rocDecodeNegativeApiTests/CMakeLists.txt doesn't exist"
fi

cleanup

