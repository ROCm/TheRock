#!/usr/bin/env bash
# build-triton-llvm.sh — Build LLVM from source for Triton on platforms
# without pre-built binaries (ppc64le, riscv64, s390x, etc.).
#
# Triton downloads pre-built LLVM for x86_64 and aarch64.  On other
# platforms you must build LLVM yourself and set LLVM_SYSPATH.
#
# This script reads the expected LLVM commit from Triton's
# cmake/llvm-info.json (or cmake/llvm-build-info.json), builds it
# with the projects and targets Triton needs, and caches the result
# under ~/.triton/llvm-<arch>-<hash>/.
#
# Usage:
#   ./build-triton-llvm.sh [triton-dir]
#
#   triton-dir  Path to the Triton source tree (default: auto-detect)
#
# After building, set these before running build_prod_wheels.py:
#
#   export LLVM_SYSPATH=<printed install path>
#   export TRITON_AMD_CODEGEN_PATH=$ROCM_HOME/lib/llvm
#   export TRITON_BUILD_PROTON=OFF
#   export MAX_JOBS=32

set -euo pipefail

TRITON_DIR="${1:-}"

# Auto-detect triton dir
if [[ -z "$TRITON_DIR" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    for candidate in \
        "$SCRIPT_DIR/external-builds/pytorch/triton" \
        "$SCRIPT_DIR/triton" \
        "$SCRIPT_DIR"; do
        if [[ -f "$candidate/cmake/llvm-info.json" ]]; then
            TRITON_DIR="$candidate"
            break
        fi
    done
fi

if [[ ! -f "$TRITON_DIR/cmake/llvm-info.json" ]]; then
    echo "ERROR: Cannot find cmake/llvm-info.json in $TRITON_DIR"
    echo "Usage: $0 [path-to-triton-source]"
    exit 1
fi

# Read LLVM hash from llvm-info.json (or llvm-build-info.json)
LLVM_HASH=$(python3 -c "
import json, sys
for name in ['cmake/llvm-info.json', 'cmake/llvm-build-info.json']:
    try:
        print(json.load(open('$TRITON_DIR/' + name))['llvm_hash'])
        sys.exit(0)
    except (FileNotFoundError, KeyError):
        pass
print('ERROR', file=sys.stderr)
sys.exit(1)
")

SHORT_HASH="${LLVM_HASH:0:8}"
ARCH="$(uname -m)"
CACHE_DIR="$HOME/.triton/llvm-${ARCH}-${SHORT_HASH}"
INSTALL_DIR="$CACHE_DIR/install"
MAX_JOBS="${MAX_JOBS:-32}"

# Check cache
if [[ -f "$INSTALL_DIR/bin/llvm-config" ]]; then
    echo "LLVM for Triton ($ARCH) already built and cached."
    echo ""
    echo "  export LLVM_SYSPATH=$INSTALL_DIR"
    echo ""
    exit 0
fi

echo "============================================"
echo "Building LLVM for Triton"
echo "============================================"
echo "  Arch:       $ARCH"
echo "  LLVM hash:  $SHORT_HASH ($LLVM_HASH)"
echo "  Cache dir:  $CACHE_DIR"
echo "  Install to: $INSTALL_DIR"
echo "  Jobs:       $MAX_JOBS"
echo ""
echo "  This will take ~1 hour on the first run."
echo "============================================"

mkdir -p "$CACHE_DIR"
SOURCE_DIR="$CACHE_DIR/llvm-project"
BUILD_DIR="$CACHE_DIR/build"

# Clone
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    echo "Cloning llvm-project..."
    git clone --depth 1 https://github.com/llvm/llvm-project.git "$SOURCE_DIR"
fi

# Fetch the specific commit
cd "$SOURCE_DIR"
if ! git cat-file -t "$LLVM_HASH" &>/dev/null; then
    echo "Fetching commit $SHORT_HASH..."
    # Try shallow fetch first, fall back to unshallow if needed
    if ! git fetch origin "$LLVM_HASH" --depth 1 2>/dev/null; then
        echo "Shallow fetch failed, unshallowing..."
        git fetch --unshallow origin
        git fetch origin "$LLVM_HASH"
    fi
fi
git checkout "$LLVM_HASH"

# Configure
echo "Configuring..."
cmake -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir;lld;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;AMDGPU;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    llvm

# Build
echo "Building with $MAX_JOBS jobs..."
cmake --build "$BUILD_DIR" -j"$MAX_JOBS"

# Install
echo "Installing..."
cmake --install "$BUILD_DIR"

echo ""
echo "============================================"
echo "LLVM for Triton built successfully!"
echo ""
echo "Set this before building Triton:"
echo ""
echo "  export LLVM_SYSPATH=$INSTALL_DIR"
echo "============================================"
