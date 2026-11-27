#!/bin/bash

set -e

rm -rf rocm-pyout

# 1. Build Triton's LLVM (one-time, cached)
./external-builds/pytorch/build_triton_llvm.sh

# 2. Apply patches
cd external-builds/pytorch/triton
for patch in ../../../patches/triton/*.patch; do
    if git apply --reverse --check "$patch" 2>/dev/null; then
        echo "Already applied: $(basename $patch)"
    else
        echo "Applying: $(basename $patch)"
        git apply "$patch"
    fi
done
cd ../pytorch
for patch in ../../../patches/pytorch/*.patch; do
    if git apply --reverse --check "$patch" 2>/dev/null; then
        echo "Already applied: $(basename $patch)"
    else
        echo "Applying: $(basename $patch)"
        git apply "$patch"
    fi
done
cd ../../..

# 3. Build codegen library (one-time, cached)
LLVM_SYSPATH=$(dirname $(dirname $(find ~/.triton -name "llvm-config" -path "*/llvm-ppc64le-*/install/bin/*" 2>/dev/null | head -1)))

if [[ -z "$LLVM_SYSPATH" || ! -d "$LLVM_SYSPATH" ]]; then
    echo "ERROR: Cannot find cached Triton LLVM. Run build_triton_llvm.sh first."
    exit 1
fi

if [[ ! -f /tmp/codegen-install/lib/libtriton_amd_codegen.so ]]; then
    echo "Building AMD codegen library..."
    cmake -G Ninja \
      -S external-builds/pytorch/triton/third_party/amd/backend/codegen \
      -B /tmp/codegen-build \
      -DLLVM_DIR=$LLVM_SYSPATH/lib/cmake/llvm \
      -DCMAKE_INSTALL_PREFIX=/tmp/codegen-install \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build /tmp/codegen-build --target triton_amd_codegen
    cmake --install /tmp/codegen-build
else
    echo "AMD codegen library already built at /tmp/codegen-install/lib"
fi

# 4. Build triton + pytorch
export LLVM_SYSPATH=$LLVM_SYSPATH
export TRITON_AMD_CODEGEN_PATH=/tmp/codegen-install/lib
export TRITON_BUILD_PROTON=OFF
export MAX_JOBS=32
export USE_ROCM_CK_GEMM=0
export AOTRITON_RUNTIME_FROM_SOURCE=1

external-builds/pytorch/build_prod_wheels.py build --clean --output-dir rocm-pyout
