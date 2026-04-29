#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Patches TBB CMakeLists files to rename library output names with the
# rocm_sysdeps_ prefix, so that installed libraries do not conflict with
# system-installed TBB. Must be run before the cmake configure step.
#
# Args: source_dir

set -e

SOURCE_DIR="${1:?Source directory must be given}"

echo "Patching TBB sources..."

# tbb: insert OUTPUT_NAME after the SOVERSION line in set_target_properties.
# TBB_BINARY_VERSION is resolved at cmake configure time (evaluates to "12").
sed -i 's/    SOVERSION \${TBB_BINARY_VERSION}$/    SOVERSION ${TBB_BINARY_VERSION}\n    OUTPUT_NAME "rocm_sysdeps_tbb"/' \
    "$SOURCE_DIR/src/tbb/CMakeLists.txt"

# tbbmalloc: same. TBBMALLOC_BINARY_VERSION evaluates to "2".
sed -i 's/    SOVERSION \${TBBMALLOC_BINARY_VERSION}$/    SOVERSION ${TBBMALLOC_BINARY_VERSION}\n    OUTPUT_NAME "rocm_sysdeps_tbbmalloc"/' \
    "$SOURCE_DIR/src/tbbmalloc/CMakeLists.txt"

# tbbmalloc_proxy: SOVERSION is followed by a closing paren on the same line.
sed -i 's/    SOVERSION \${TBBMALLOC_BINARY_VERSION})$/    SOVERSION ${TBBMALLOC_BINARY_VERSION}\n    OUTPUT_NAME "rocm_sysdeps_tbbmalloc_proxy")/' \
    "$SOURCE_DIR/src/tbbmalloc_proxy/CMakeLists.txt"

echo "Done patching TBB sources."
