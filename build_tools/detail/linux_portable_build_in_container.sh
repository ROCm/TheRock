#!/bin/bash
# See corresponding linux_build_portable.py which invokes this within a
# container.
set -e
set -o pipefail
trap 'kill -TERM 0' INT

OUTPUT_DIR="/therock/output"
mkdir -p "$OUTPUT_DIR/caches"

export CCACHE_DIR="$OUTPUT_DIR/caches/container/ccache"
export PIP_CACHE_DIR="$OUTPUT_DIR/caches/container/pip"
mkdir -p "$CCACHE_DIR"
mkdir -p "$PIP_CACHE_DIR"

pip install -r /therock/src/requirements.txt

export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

set -o xtrace
time cmake -GNinja -S /therock/src -B "$OUTPUT_DIR/build" \
  -DTHEROCK_BUNDLE_SYSDEPS=ON \
  "$@"
time cmake --build "$OUTPUT_DIR/build"
