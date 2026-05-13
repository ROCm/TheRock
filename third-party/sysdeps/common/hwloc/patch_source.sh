#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"

echo "Patching hwloc sources to rename main library..."

# Patch the src/Makefile.in to change library name from libhwloc to librocm_sysdeps_hwloc
HWLOC_MAKEFILE="$SOURCE_DIR/src/Makefile.in"

if [ -f "$HWLOC_MAKEFILE" ]; then
  echo "Patching $HWLOC_MAKEFILE..."
  sed -i 's/libhwloc\.la/librocm_sysdeps_hwloc.la/g' "$HWLOC_MAKEFILE"
  sed -i 's/libhwloc_la_SOURCES/librocm_sysdeps_hwloc_la_SOURCES/g' "$HWLOC_MAKEFILE"
  sed -i 's/libhwloc_la_LDFLAGS/librocm_sysdeps_hwloc_la_LDFLAGS/g' "$HWLOC_MAKEFILE"
  sed -i 's/libhwloc_la_LIBADD/librocm_sysdeps_hwloc_la_LIBADD/g' "$HWLOC_MAKEFILE"
  sed -i 's/libhwloc_la_DEPENDENCIES/librocm_sysdeps_hwloc_la_DEPENDENCIES/g' "$HWLOC_MAKEFILE"
fi

# Patch all Makefile.in files that reference libhwloc.la
echo "Patching all Makefile.in files that reference libhwloc.la..."
find "$SOURCE_DIR" -name "Makefile.in" -type f -exec grep -l "libhwloc\.la" {} \; | while read -r makefile; do
  echo "  Patching $makefile..."
  sed -i 's|/src/libhwloc\.la|/src/librocm_sysdeps_hwloc.la|g' "$makefile"
done

echo "hwloc patching completed."
