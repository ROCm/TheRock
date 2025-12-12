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

# Build manylinux Python executables argument if MANYLINUX is set
PYTHON_EXECUTABLES_ARG=""
if [ "${MANYLINUX}" = "1" ] || [ "${MANYLINUX}" = "true" ]; then
  PYTHON_EXECUTABLES_ARG="-DTHEROCK_DIST_PYTHON_EXECUTABLES=/opt/python/cp38-cp38/bin/python;/opt/python/cp39-cp39/bin/python;/opt/python/cp310-cp310/bin/python;/opt/python/cp311-cp311/bin/python;/opt/python/cp312-cp312/bin/python;/opt/python/cp313-cp313/bin/python"
fi

set -o xtrace
time cmake -GNinja -S /therock/src -B "$OUTPUT_DIR/build" \
  -DTHEROCK_BUNDLE_SYSDEPS=ON \
  ${PYTHON_EXECUTABLES_ARG} \
  "$@"
time cmake --build "$OUTPUT_DIR/build" --target therock-archives therock-dist
