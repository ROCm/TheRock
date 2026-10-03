#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBFFI_MAKEFILE="$SOURCE_DIR/Makefile.in"
LIBFFI_MAPFILE="$SOURCE_DIR/libffi.map.in"

echo "Patching sources..."
# Prefix the installed libtool target (and its generated automake variables) so
# the shared library gets SONAME librocm_sysdeps_ffi.so and can coexist with a
# system libffi of the same SONAME.
sed -i 's/libffi\.la/librocm_sysdeps_ffi.la/g' "$LIBFFI_MAKEFILE"
sed -i 's/libffi_la_/librocm_sysdeps_ffi_la_/g' "$LIBFFI_MAKEFILE"

# libffi generates its linker version script by running the C preprocessor over
# libffi.map.in; add -P so cpp does not emit line markers that would corrupt the
# (now plain) version script.
sed -i 's/-E -x assembler-with-cpp/-E -P -x assembler-with-cpp/' "$LIBFFI_MAKEFILE"

# Replace the existing version symbols with our custom ones.
echo "Updating version script..."
cp "$SCRIPT_DIR/libffi.map" "$LIBFFI_MAPFILE"
