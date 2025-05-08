#!/usr/bin/bash

# Causes bash process to die immediately after child process returns error
# to make sure that script does not continue logic if error has happened.
set -e
set -o pipefail

SCRIPT_DIR="$(cd $(dirname $0) && pwd)"
if ! source $SCRIPT_DIR/env_init.sh; then
        echo "Failed to find python virtual-env"
        echo "Make sure that TheRock has been build first"
        exit 1
fi

unset LDFLAGS
unset CFLAGS
unset CPPFLAGS
unset PKG_CONFIG_PATH
export CMAKE_PREFIX_PATH="$(realpath $SCRIPT_DIR/../../build/dist/rocm)"
echo ${CMAKE_PREFIX_PATH}
export CMAKE_C_COMPILER=${CMAKE_PREFIX_PATH}/bin/hipcc
export CMAKE_CXX_COMPILER=${CMAKE_PREFIX_PATH}/bin/hipcc
export HIP_DEVICE_LIB_PATH=${CMAKE_PREFIX_PATH}/lib/llvm/amdgcn/bitcode

cd $SCRIPT_DIR/pytorch_vision
ROCM_PATH=${CMAKE_PREFIX_PATH} FORCE_CUDA=1 TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_VIDEO_CODEC=0 CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} BUILD_VERSION=0.22.0 BUILD_NUMBER=1 VERSION_NAME=0.22.0 python setup.py bdist_wheel

# if there are multiple wheel files, find the newest one and install it
unset -v latest_wheel_file
for cur_file in dist/*.whl; do
    [[ $cur_file -nt "$latest_wheel_file" ]] && latest_wheel_file=$cur_file
done
if [ ! -z "$latest_wheel_file" ]; then
    echo "installing $latest_wheel_file"
    # do not use "pip install --force-reinstall because it can uninstall
    # own build other packages and then re-install incorrect onew from internet
    pip uninstall --yes "$latest_wheel_file"
    pip install "$latest_wheel_file"
else
    echo "Failed to build and find pytorch vision wheel for pip install in directory $SCRIPT_DIR/pytorch_vision/dist"
    exit 1
fi
cd $SCRIPT_DIR
