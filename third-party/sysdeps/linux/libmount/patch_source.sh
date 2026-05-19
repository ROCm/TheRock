#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"

echo "Patching libmount sources to rename main library..."

# Patch the src/meson.build to change library name from mount to rocm_sysdeps_mount
SRC_MESON_BUILD="$SOURCE_DIR/src/meson.build"

if [ -f "$SRC_MESON_BUILD" ]; then
  echo "Patching $SRC_MESON_BUILD..."
  # Change the variable name and the library name parameter
  # Original: libmount = library('mount', ...)
  # Result:   librocm_sysdeps_mount = library('rocm_sysdeps_mount', ...)
  sed -i "s/^libmount = library($/librocm_sysdeps_mount = library(/g" "$SRC_MESON_BUILD"
  sed -i "s/^  'mount',$/  'rocm_sysdeps_mount',/g" "$SRC_MESON_BUILD"
  # Update dependency declaration
  sed -i 's/link_with : libmount/link_with : librocm_sysdeps_mount/g' "$SRC_MESON_BUILD"
fi

# Patch the root meson.build pkg.generate reference
ROOT_MESON_BUILD="$SOURCE_DIR/meson.build"
if [ -f "$ROOT_MESON_BUILD" ]; then
  echo "Patching $ROOT_MESON_BUILD..."
  # Update pkg.generate to NOT pass the library object, use libraries parameter instead
  # This way meson won't auto-generate -lrocm_sysdeps_mount
  # Original:
  #   pkg.generate(
  #     libmount,
  #     description : '...',
  #   )
  # Result:
  #   pkg.generate(
  #     name : 'mount',
  #     libraries : ['-L${libdir}', '-lmount'],
  #     description : '...',
  #   )
  sed -i '/^pkg\.generate($/,/^)$/ {
    s/^  libmount,$/  name : '\''mount'\'',\n  libraries : ['\''-L${libdir}'\'', '\''-lmount'\''],/
  }' "$ROOT_MESON_BUILD"
fi

echo "libmount patching completed."
