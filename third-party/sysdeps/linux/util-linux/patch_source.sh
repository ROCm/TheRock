#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Patching sources..."

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
