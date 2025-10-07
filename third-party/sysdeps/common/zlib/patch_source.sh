#!/usr/bin/bash
set -e

SOURCE_DIR="${1:?Source directory must be given}"
ZLIB_CMAKELIST="$SOURCE_DIR/CMakeLists.txt"
echo "Patching sources..."

sed -i -E 's/(OUTPUT_NAME)[[:space:]]+z\)/\1 rocm_sysdeps_z)/' "$ZLIB_CMAKELIST"
