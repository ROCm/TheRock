#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="$1"
if [ -z "$SOURCE_DIR" ]; then
    echo "Usage: $0 <source_dir>"
    exit 1
fi

cd "$SOURCE_DIR"

# Update .pc file to reference the prefixed library name
if [ -f libffi.pc.in ]; then
    sed -i 's/-lffi/-lrocm_sysdeps_ffi/g' libffi.pc.in
fi

# Patch Makefile.in to use our version.lds instead of libffi.map
# The build directory will have a copy of version.lds, so we reference it locally
if [ -f Makefile.in ]; then
    # Replace libffi.map in version script variable assignments only
    sed -i 's/--version-script,libffi\.map/--version-script,version.lds/g' Makefile.in
    # Replace libffi.map in dependency variable
    sed -i 's/libffi_version_dep = libffi\.map/libffi_version_dep = version.lds/g' Makefile.in
    # Remove the rule that generates version.lds from version.lds.in (we provide it directly)
    sed -i '/^version\.lds:.*version\.lds\.in/,/^[^\t]/{ /^version\.lds:/d; /^\t/d; }' Makefile.in
fi

echo "libffi source patching complete"
