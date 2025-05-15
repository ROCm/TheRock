#!/bin/bash

# Windows equivalent to the Linux build_pytorch_torch.sh.
# TODO: merge into `ptbuild.py develop` or `ptbuild.py bdist_wheel` somehow?

set -eo pipefail

SCRIPT_DIR="$(cd $(dirname $0) && pwd)"

if [ ! -n "${ROCM_HOME}" ]; then
    export ROCM_HOME=$(realpath $SCRIPT_DIR/../../build/dist/rocm)
    echo "ROCM_HOME: $ROCM_HOME"
fi
if [ -d ${ROCM_HOME} ]; then
    export PATH=${ROCM_HOME}/bin:$PATH
else
    echo "Could not find ROCM_HOME: $ROCM_HOME"
    exit 1
fi

BUILD_DIR_ROOT=${ROCM_HOME?}/../..

# Environment variables recommended here:
# https://github.com/ROCm/TheRock/discussions/409#discussioncomment-13032345
export USE_ROCM=ON
export USE_KINETO=0
export BUILD_TEST=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export CMAKE_PREFIX_PATH="${ROCM_HOME?}"
# TODO(#410): Fix HIP_CLANG_PATH setting so hipcc finds the tools on its own
export HIP_CLANG_PATH="${ROCM_HOME?}/lib/llvm/bin"
export CC=${ROCM_HOME?}/lib/llvm/bin/clang-cl
export CXX=${ROCM_HOME?}/lib/llvm/bin/clang-cl
export DISTUTILS_USE_SDK=1

# Match this with `-DTHEROCK_AMDGPU_FAMILIES` used to bulid TheRock.
# TODO: pull from environment variable, option, etc.?
export PYTORCH_ROCM_ARCH=gfx1100
# export PYTORCH_ROCM_ARCH=gfx1151

# Alternate paths relative to this script: 'src/', 'src/pytorch/'.
PYTORCH_SRC_DIR="${SCRIPT_DIR?}/pytorch"
cd ${PYTORCH_SRC_DIR}

# TODO(#590): Fix/disable warning logs that flood the console buffer and output
# 			  to console instead, or match TheRock's build/logs/ setup.
LOGS_DIR="${HOME}/.therock"
mkdir -p ${LOGS_DIR}
printf -v DATE_STR '%(%Y-%m-%d_%H%M%S)T' -1
LOG_FILE_NAME="${LOGS_DIR}/logs_pytorch_windows_${DATE_STR}.txt"

echo "Running \"python setup.py bdist_wheel\", directing output to $LOG_FILE_NAME"
python setup.py bdist_wheel > ${LOG_FILE_NAME} 2>&1

DIST_DIR="${PYTORCH_SRC_DIR}/dist"
echo "Build completed! Wheels should be at ${DIST_DIR}:"
ls ${DIST_DIR}
