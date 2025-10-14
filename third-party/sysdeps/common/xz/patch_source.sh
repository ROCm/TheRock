#!/usr/bin/bash
set -e

SOURCE_DIR="${1:?Source directory must be given}"
XZ_CMAKELIST="$SOURCE_DIR/CMakeLists.txt"
echo "Patching sources..."

sed -i -E 's/(OUTPUT_NAME)[[:space:]]+z\)/\1 rocm_sysdeps_z)/' "$XZ_CMAKELIST"
