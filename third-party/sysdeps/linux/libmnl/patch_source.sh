#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBMNL_MAKEFILE="$SOURCE_DIR/src/Makefile.in"
LIBMNL_MAPFILE="$SOURCE_DIR/src/libmnl.map"

echo "Patching sources..."
sed -i 's/libmnl\.la/librocm_sysdeps_mnl.la/g' "$LIBMNL_MAKEFILE"
sed -i 's/libmnl_la_/librocm_sysdeps_mnl_la_/g' "$LIBMNL_MAKEFILE"
sed -i 's/am_libmnl_la_/am_librocm_sysdeps_mnl_la_/g' "$LIBMNL_MAKEFILE"

# Replace the existing version symbols with our custom ones
echo "Updating version script..."
cp "$SCRIPT_DIR/libmnl.map" "$LIBMNL_MAPFILE"
