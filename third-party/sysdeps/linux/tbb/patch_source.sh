#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Patches TBB CMakeLists files to rename library output names with the
# rocm_sysdeps_ prefix, so that installed libraries do not conflict with
# system-installed TBB. Must be run before the cmake configure step.
#
# The sed patterns below are pinned to oneTBB 2022.3.0 and must be
# revalidated when bumping the version.
#
# Args: source_dir

set -e

SOURCE_DIR="${1:?Source directory must be given}"
CMAKE_SOURCE_DIR="${2:?CMake Source directory must be given}"

verify_patch() {
    local file="$1" name="$2"
    if ! grep -q "OUTPUT_NAME.*rocm_sysdeps_${name}" "$file"; then
        echo "ERROR: OUTPUT_NAME patch for ${name} did not apply to ${file}" >&2
        echo "The sed patterns may need updating for this oneTBB version." >&2
        exit 1
    fi
}

echo "Patching TBB sources..."

# tbb: insert OUTPUT_NAME after the SOVERSION line in set_target_properties.
patch "$SOURCE_DIR/src/tbb/CMakeLists.txt" "$CMAKE_SOURCE_DIR/patches/third-party/sysdeps/linux/tbb/0001-Add-OUTPUT_NAME-to-tbb.patch"
verify_patch "$SOURCE_DIR/src/tbb/CMakeLists.txt" "tbb"

# tbbmalloc: same.
patch "$SOURCE_DIR/src/tbbmalloc/CMakeLists.txt" "$CMAKE_SOURCE_DIR/patches/third-party/sysdeps/linux/tbb/0001-Add-OUTPUT_NAME-to-tbbmalloc.patch"
verify_patch "$SOURCE_DIR/src/tbbmalloc/CMakeLists.txt" "tbbmalloc"

# tbbmalloc_proxy: SOVERSION is followed by a closing paren on the same line.
patch "$SOURCE_DIR/src/tbbmalloc_proxy/CMakeLists.txt" "$CMAKE_SOURCE_DIR/patches/third-party/sysdeps/linux/tbb/0001-Add-OUTPUT_NAME-to-tbbmalloc_proxy.patch"
verify_patch "$SOURCE_DIR/src/tbbmalloc_proxy/CMakeLists.txt" "tbbmalloc_proxy"

echo "Done patching TBB sources."
