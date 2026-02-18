#!/bin/bash

# #############################################################################
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

ROCM_GDB_SETUP=0

###### Functions ###############################################################
os_release() {
    if [[ -r  /etc/os-release ]]; then
        . /etc/os-release

        DISTRO_NAME=$ID

        case "$ID" in
        ubuntu|debian)
            DISTRO_PACKAGE_MGR="apt"
            ROCM_GDB_SETUP=1
            ;;
        rhel|ol|rocky)
            DISTRO_PACKAGE_MGR="dnf"
            ;;
        sles)
            DISTRO_PACKAGE_MGR="zypper"   
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

    echo "Setting up on $DISTRO_NAME."
}

####### Main script ###############################################################

echo =================================
echo ROCm Setup
echo =================================

os_release

# Check if the current working directory is named rocm-x.y.z
current_dir=$(basename "$(pwd)")
if [[ "$current_dir" =~ ^rocm-[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    # Extract the version x.y.z from the directory name
    ROCM_VERSION=$(echo "$current_dir" | sed -E 's/rocm-([0-9]+\.[0-9]+\.[0-9]+)/\1/')

    # Save the current working directory path
    ROCM_INSTALL_PATH=$(pwd)
    
    # Get the full rocm version
    ROCM_BUILD_VERSION=$(<"$ROCM_INSTALL_PATH/.info/version")
    
     # Convert x.y.z to x0y0z format for ROCM_STR
    ROCM_VERSION_STR=$(echo "$ROCM_VERSION" | awk -F. '{printf "%d%02d%02d", $1, $2, $3}')
    
    # Extract the build number
    ROCM_BUILD_NUM=${ROCM_BUILD_VERSION##*-}
    
    echo "ROCM_VERSION      : $ROCM_VERSION"
    echo "ROCM_BUILD_VERSION: $ROCM_BUILD_VERSION"
    echo "ROCM_BUILD_NUM    : $ROCM_BUILD_NUM"
    echo "ROCM_VERSION_STR  : $ROCM_VERSION_STR"
    echo "ROCM_INSTALL_PATH : $ROCM_INSTALL_PATH"

    # Create the new script setup-modules-$ROCM_VERSION.sh
    SETUP_SCRIPT="setup-modules-$ROCM_VERSION.sh"
    cat << EOF > "$SETUP_SCRIPT"
#!/bin/bash

ROCM_INSTALL_PATH=$ROCM_INSTALL_PATH
ROCM_VERSION=$ROCM_VERSION
ROCM_GDB_SETUP=$ROCM_GDB_SETUP

setup_rocm_gdb() {
    echo Setting up rocm-gdb
    if [[ \$ROCM_GDB_SETUP == 1 ]]; then
        PYTHON_LIB_INSTALLED=\$(find /lib/ -name 'libpython3*.so' | head -n 1)
        echo "Installing rocm-gdb with \$PYTHON_LIB_INSTALLED."
        ln -s \$PYTHON_LIB_INSTALLED $ROCM_INSTALL_PATH/lib/amdpythonlib.so
    fi
    echo Setting up rocm-gdb. Complete.
}

setup_rocm_llvm() {
    echo Setting up rocm-llvm
    ln -s "$ROCM_INSTALL_PATH/lib/llvm/bin/amdclang" "$ROCM_INSTALL_PATH/bin/amdclang"
    ln -s "$ROCM_INSTALL_PATH/lib/llvm/bin/amdclang++" "$ROCM_INSTALL_PATH/bin/amdclang++"
    ln -s "$ROCM_INSTALL_PATH/lib/llvm/bin/amdclang-cl" "$ROCM_INSTALL_PATH/bin/amdclang-cl"
    ln -s "$ROCM_INSTALL_PATH/lib/llvm/bin/amdclang-cpp" "$ROCM_INSTALL_PATH/bin/amdclang-cpp"
    ln -s "$ROCM_INSTALL_PATH/lib/llvm/bin/amdflang" "$ROCM_INSTALL_PATH/bin/amdflang"
    if [[ -L "$ROCM_INSTALL_PATH/lib/llvm/bin/amdflang-new" ]]; then
        ln -s "$ROCM_INSTALL_PATH/lib/llvm/bin/amdflang-new" "$ROCM_INSTALL_PATH/bin/amdflang-new"
    fi
    if [[ -L "$ROCM_INSTALL_PATH/lib/llvm/bin/amdflang-classic" ]]; then
        ln -s "$ROCM_INSTALL_PATH/lib/llvm/bin/amdflang-classic" "$ROCM_INSTALL_PATH/bin/amdflang-classic"
    fi
    if [[ -f "$ROCM_INSTALL_PATH/lib/llvm/bin/offload-arch" ]]; then
        ln -s "$ROCM_INSTALL_PATH/lib/llvm/bin/offload-arch" "$ROCM_INSTALL_PATH/bin/offload-arch"
    fi
    ln -s "$ROCM_INSTALL_PATH/lib/llvm/bin/amdlld" "$ROCM_INSTALL_PATH/bin/amdlld"
    echo Setting up rocm-llvm. Complete.
}

setup_rocm_opencl() {
    echo Setting up rocm-opencl
    \$SUDO echo "$ROCM_INSTALL_PATH"/lib | \$SUDO tee /etc/ld.so.conf.d/10-rocm-opencl.conf > /dev/null
    \$SUDO chmod 644 /etc/ld.so.conf.d/10-rocm-opencl.conf
    \$SUDO ldconfig

    \$SUDO mkdir -p /etc/OpenCL/vendors
    \$SUDO chmod 755 /etc/OpenCL /etc/OpenCL/vendors
    \$SUDO echo "libamdocl64.so" | \$SUDO tee /etc/OpenCL/vendors/amdocl64_${ROCM_VERSION_STR}_${ROCM_BUILD_NUM}.icd > /dev/null
    \$SUDO chmod 644 /etc/OpenCL/vendors/amdocl64_${ROCM_VERSION_STR}_${ROCM_BUILD_NUM}.icd
    echo Setting up rocm-opencl. Complete.
}

setup_migraphx() {
    \$SUDO mkdir -p /usr/lib/python3/dist-packages
    \$SUDO echo "$ROCM_INSTALL_PATH/lib" | \$SUDO tee /usr/lib/python3/dist-packages/MIGraphX.pth > /dev/null
            
    \$SUDO mkdir -p /usr/lib/python2.7/dist-packages
    \$SUDO echo "$ROCM_INSTALL_PATH/lib" | \$SUDO tee /usr/lib/python2.7/dist-packages/MIGraphX.pth > /dev/null
}

echo =================================
echo ROCm $ROCM_VERSION Module Setup 
echo =================================

SUDO=\$([[ \$(id -u) -ne 0 ]] && echo "sudo" ||:)

echo Setting Paths
export ROCM_PATH=\$ROCM_INSTALL_PATH
export PATH=\$PATH:\$ROCM_PATH/bin:\$ROCM_PATH/llvm/bin
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$ROCM_PATH/lib:\$ROCM_PATH/llvm/lib
echo Setting Paths. Complete.

now=\$(date -u +%s)
altscore=\$((6 - 3))
altscore=\$((altscore * 14 + 4)) # Allow up to 14 minor
altscore=\$((altscore * 14 + 2)) # Allow up to 14 patch
altscore=\$((altscore*1000000+(\$now-1600000000)/60))

echo altscore = \$altscore

echo Setting up modules
\$SUDO $DISTRO_PACKAGE_MGR install -y environment-modules

for loc in "/usr/share/modules/modulefiles" "/usr/local/Modules/modulefiles" "/usr/share/Modules/modulefiles" "/usr/share/Modules/3.2.10/modulefiles"; do
    if [ -d "\$loc" ]; then
        check=\${loc%%\/modulefiles}
        if grep -q "\$check" "/etc/profile.d/modules.sh"; then
            \$SUDO mkdir -p "\$loc/rocm"
            \$SUDO update-alternatives --install "\$loc/rocm/\$ROCM_VERSION" "rocmmod\$ROCM_VERSION" "\$ROCM_PATH/lib/rocmmod" "\$altscore"
            break;
        fi
    fi
done

setup_rocm_gdb

setup_rocm_llvm

setup_rocm_opencl

setup_migraphx

EOF

    # Make the new script executable
    chmod +x "$SETUP_SCRIPT"
    echo "Created $SETUP_SCRIPT"
else
    echo "The current working directory is not named rocm-x.y.z."
    exit 1
fi
