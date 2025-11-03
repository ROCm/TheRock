#!/usr/bin/bash
set -e

SOURCE_DIR="${1:?Source directory must be given}"
ZSTD_CMAKELIST="$SOURCE_DIR/build/cmake/lib/CMakeLists.txt"
echo "Patching sources..."

sed -i -E 's/(OUTPUT_NAME)[[:space:]]+zstd/\1 rocm_sysdeps_zstd/' "$ZSTD_CMAKELIST"
