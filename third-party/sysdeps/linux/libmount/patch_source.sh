#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBMOUNT_MESON="$SOURCE_DIR/libmount/meson.build"
LIBBLKID_MESON="$SOURCE_DIR/libblkid/meson.build"

echo "Patching sources..."

# Replace a single line in $2 with $3; idempotent. Fails only if neither the
# original pattern $1 nor the replacement $3 is present, which means upstream
# changed the meson.build layout.
replace_or_die() {
    local pattern="$1" file="$2" replacement="$3"
    if grep -qE "$pattern" "$file"; then
        sed -i -E "s/${pattern}/${replacement}/g" "$file"
    elif ! grep -qF "$replacement" "$file"; then
        echo "ERROR: neither pattern '$pattern' nor replacement '$replacement' found in $file" >&2
        echo "       upstream meson.build layout may have changed" >&2
        exit 1
    fi
}

# Rename the libmount shared library: 'mount' -> 'rocm_sysdeps_mount'.
# The library() call lives in libmount/meson.build with the name on its own line.
replace_or_die "^  'mount',$" "$LIBMOUNT_MESON" "  'rocm_sysdeps_mount',"

# Rename the libblkid shared library: 'blkid' -> 'rocm_sysdeps_blkid'.
# Declared via both_libraries() in libblkid/meson.build.
replace_or_die "^  'blkid',$" "$LIBBLKID_MESON" "  'rocm_sysdeps_blkid',"

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
