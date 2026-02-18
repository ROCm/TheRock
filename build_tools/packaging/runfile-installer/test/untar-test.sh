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

TAR_DIR=~/runfile
RUNFILE=


###### Functions ###############################################################

usage() {
cat <<END_USAGE
Usage: $PROG [options]

[options}:
    help = Display this help information.
    
    rel=<release number> 
        Set the ROCm release sourced for the rocm-examples test ie.rel=6.3.1
END_USAGE
}

find_rocm_installer() {
    local root_loc=~
    local found_runfile=""
    
    # Check if root location of the search exists
    if [[ ! -d "$root_loc" ]]; then
        echo "Error: Search location '$root_loc' does not exist"
        return 1
    fi
    
    # Find the runfile installer
    found_runfile=$(find "$root_loc" -type f -name "*rocm-installer*.run*" | head -1)
    
    # Check if file was found
    if [[ -n "$found_runfile" ]]; then
        echo "Found ROCm installer: $found_runfile"
        RUNFILE="$found_runfile"
        return 0
    else
        echo "No ROCm installer file found in '$root_loc'"
        return 1
    fi
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
    
    # get the rocm version info
    local rocm_ver_name=$(basename "$ROCM_DIR")
    
    ROCM_VER=${rocm_ver_name#rocm-}
    echo ROCM_VER = $ROCM_VER

    # Set the ROCm paths
    export ROCM_PATH="$ROCM_DIR"
    
    # Set compiler paths
    export CXX=$ROCM_PATH/llvm/bin/amdclang++
    export CC=$ROCM_PATH/llvm/bin/amdclang
    
    echo Setting up ROCm paths...Complete.
}

####### Main script ###############################################################

echo ===============================
echo UNTAR TESTER
echo ===============================

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
    *)
        shift
        ;;
    esac
done

script_name=rocm-examples-test.sh
scripts_dir="$( cd "$( dirname "$0" )" && pwd )"

# Search for the runfile installer
find_rocm_installer
if [ $? -ne 0 ]; then
    exit 1
fi

sudo rm -rf $TAR_DIR
mkdir $TAR_DIR

# run the installer
bash $RUNFILE untar $TAR_DIR

setup_rocm

# setup the rocm module
cd $TAR_DIR/rocm-$ROCM_VER/
./setup-modules-$ROCM_VER.sh
source /etc/profile.d/modules.sh
module load rocm/$ROCM_VER

# install dependencies
cd ../
bash $RUNFILE deps=install-only rocm amdgpu
bash $RUNFILE gpu-access=all

amd-smi

if [[ ! -f $scripts_dir/$script_name ]]; then
    echo "$scripts_dir/$script_name script not found, exiting."
    exit 1
fi

bash $scripts_dir/$script_name "rel=$ROCM_REL"

