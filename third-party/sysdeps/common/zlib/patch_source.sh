#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
ZLIB_CMAKELIST="$SOURCE_DIR/CMakeLists.txt"
echo "Patching sources..."

# Rename OUTPUT_NAME to rocm_sysdeps_z. Handles two forms:
# - 1.3.1: both libs in one call: set_target_properties(zlib zlibstatic PROPERTIES OUTPUT_NAME z)
# - 1.3.2+: shared lib separately: set_target_properties(zlib PROPERTIES ... OUTPUT_NAME z)
sed -i -E 's/(OUTPUT_NAME)[[:space:]]+z\)/\1 rocm_sysdeps_z)/' "$ZLIB_CMAKELIST"
# 1.3.2+: static lib output name on its own line using a CMake variable:
#   zlibstatic PROPERTIES EXPORT_NAME ZLIBSTATIC OUTPUT_NAME
#                                                z${zlib_static_suffix})
sed -i -E 's/z\$\{zlib_static_suffix\}/rocm_sysdeps_z/' "$ZLIB_CMAKELIST"
