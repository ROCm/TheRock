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

ROCM_COMP_DIR=component-rocm
ROCM_COMP_DEB_DIR=component-rocm-deb
AMDGPU_COMP_DIR=component-amdgpu

CLEAN_SCRIPT_LIST=(install-init.sh rocm-installer.sh deps-installer.sh cleanup-install.sh)

###### Functions ###############################################################

print_msg() {
    local msg=$1
    echo -e "\e[93m$msg\e[0m"
}

cleanup_install() {
    if [ -d $ROCM_COMP_DIR ]; then
        print_msg "Cleaning up rocm components..."

        $SUDO rm -rf $ROCM_COMP_DIR

        print_msg "Cleaning up rocm components...Complete"
    fi

    if [ -d $ROCM_COMP_DEB_DIR ]; then
        print_msg "Cleaning up rocm deb components..."

        $SUDO rm -rf $ROCM_COMP_DEB_DIR

        print_msg "Cleaning up rocm deb components...Complete"
    fi

    if [ -d $AMDGPU_COMP_DIR ]; then
        print_msg "Cleaning up amdgpu components..."
        
        $SUDO rm -rf $AMDGPU_COMP_DIR
        
        print_msg "Cleaning up amdgpu components...Complete"
    fi
    
    # Remove any installer scripts
    for index in ${CLEAN_SCRIPT_LIST[@]}; do
        if [ -f $index ]; then
            print_msg "==== Removing $index ===="
            $SUDO rm $index
        fi
    done
    
    # Cleanup any configuration files
    if [ -d config ]; then
        print_msg "Cleaning up config..."
        
        $SUDO rm -rf config
        
        print_msg "Cleaning up config...Complete"
    fi
    
    # Remove the UI binary
    if [ -f rocm_ui ]; then
        $SUDO rm rocm_ui
    fi
}


####### Main script ###############################################################

SUDO=$([[ $(id -u) -ne 0 ]] && echo "sudo" ||:)

cleanup_install
