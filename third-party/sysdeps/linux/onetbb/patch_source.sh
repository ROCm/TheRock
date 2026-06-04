#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TBB_CMAKE="$SOURCE_DIR/src/tbb/CMakeLists.txt"
TBBMALLOC_CMAKE="$SOURCE_DIR/src/tbbmalloc/CMakeLists.txt"
TBBMALLOC_PROXY_CMAKE="$SOURCE_DIR/src/tbbmalloc_proxy/CMakeLists.txt"

echo "Patching sources..."

# Inject an OUTPUT_NAME override on Linux after the ALIAS target line so the
# built shared libraries are named librocm_sysdeps_<name>.so.* . The upstream
# CMake only sets OUTPUT_NAME on Windows, so a Linux build otherwise produces
# the unprefixed lib<name>.so . Idempotent: only patch if the marker is absent.
inject_output_name() {
    local file="$1" alias_line="$2" output_name="$3"
    if grep -qF "OUTPUT_NAME \"${output_name}\"" "$file"; then
        return 0
    fi
    if ! grep -qF "$alias_line" "$file"; then
        echo "ERROR: alias line '$alias_line' not found in $file" >&2
        echo "       upstream layout may have changed" >&2
        exit 1
    fi
    local marker_escaped
    marker_escaped="$(printf '%s\n' "$alias_line" | sed -e 's/[\/&]/\\&/g')"
    sed -i "/^${marker_escaped}$/a set_target_properties(${output_name#rocm_sysdeps_} PROPERTIES OUTPUT_NAME \"${output_name}\")" "$file"
}

inject_output_name "$TBB_CMAKE"               "add_library(TBB::tbb ALIAS tbb)"                           "rocm_sysdeps_tbb"
inject_output_name "$TBBMALLOC_CMAKE"         "add_library(TBB::tbbmalloc ALIAS tbbmalloc)"               "rocm_sysdeps_tbbmalloc"
inject_output_name "$TBBMALLOC_PROXY_CMAKE"   "add_library(TBB::tbbmalloc_proxy ALIAS tbbmalloc_proxy)"   "rocm_sysdeps_tbbmalloc_proxy"

# Replace upstream symbol-version (.def) scripts with our broad
# AMDROCM_SYSDEPS_1.0 map. oneTBB wires these via
# -Wl,--version-script=<src>/def/lin<arch>-<lib>.def so overwriting them in
# place is sufficient (mirrors the libmount pattern).
echo "Updating version scripts..."
for def_file in \
    "$SOURCE_DIR/src/tbb/def/lin64-tbb.def" \
    "$SOURCE_DIR/src/tbb/def/lin32-tbb.def" \
    "$SOURCE_DIR/src/tbbmalloc/def/lin64-tbbmalloc.def" \
    "$SOURCE_DIR/src/tbbmalloc/def/lin32-tbbmalloc.def" \
    "$SOURCE_DIR/src/tbbmalloc_proxy/def/lin64-proxy.def" \
    "$SOURCE_DIR/src/tbbmalloc_proxy/def/lin32-proxy.def"; do
    if [ -f "$def_file" ]; then
        echo "Updating $def_file"
        cp "$SCRIPT_DIR/version.lds" "$def_file"
    fi
done
