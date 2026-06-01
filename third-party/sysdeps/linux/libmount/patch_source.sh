#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBMOUNT_MESON="$SOURCE_DIR/libmount/meson.build"
LIBBLKID_MESON="$SOURCE_DIR/libblkid/meson.build"

echo "Patching sources..."

# Rename the libmount shared library: 'mount' -> 'rocm_sysdeps_mount'.
# The library() call lives in libmount/meson.build with the name on its own line.
sed -i "s/^  'mount',\$/  'rocm_sysdeps_mount',/g" "$LIBMOUNT_MESON"

# Rename the libblkid shared library: 'blkid' -> 'rocm_sysdeps_blkid'.
# Declared via both_libraries() in libblkid/meson.build.
sed -i "s/^  'blkid',\$/  'rocm_sysdeps_blkid',/g" "$LIBBLKID_MESON"

# Replace upstream symbol-version scripts with our broad AMDROCM_SYSDEPS_1.0 map.
# Meson already wires these files in via link_args: -Wl,--version-script=...sym,
# so overwriting them here is sufficient (mirrors the libnl pattern).
echo "Updating version scripts..."
for sym_file in \
    "$SOURCE_DIR/libmount/src/libmount.sym" \
    "$SOURCE_DIR/libblkid/src/libblkid.sym"; do
    if [ -f "$sym_file" ]; then
        echo "Updating $sym_file"
        cp "$SCRIPT_DIR/version.lds" "$sym_file"
    fi
done
