#!/usr/bin/bash
set -e

SOURCE_DIR="${1:?Source directory must be given}"
LIBLZMA_CMAKELIST="$SOURCE_DIR/CMakeLists.txt"
echo "Patching sources..."

# Look in the CMakeList.txt of the project to be build what the library "OUTPUT_NAME" is
sed -i -E 's/(OUTPUT_NAME)[[:space:]]+"lzma"/\1 "rocm_sysdeps_liblzma"/' "$LIBLZMA_CMAKELIST"
