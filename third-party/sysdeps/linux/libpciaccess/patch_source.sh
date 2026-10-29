#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_LDS="$SCRIPT_DIR/version.lds"

echo "Patching libpciaccess sources to rename main library..."

# Patch the src/meson.build to change library name from pciaccess to rocm_sysdeps_pciaccess
SRC_MESON_BUILD="$SOURCE_DIR/src/meson.build"

if [ -f "$SRC_MESON_BUILD" ]; then
  echo "Patching $SRC_MESON_BUILD..."
  # Change the variable name and the library name parameter
  # Original: libpciaccess = library('pciaccess', ...)
  # Result:   librocm_sysdeps_pciaccess = library('rocm_sysdeps_pciaccess', ...)
  sed -i "s/^libpciaccess = library($/librocm_sysdeps_pciaccess = library(/g" "$SRC_MESON_BUILD"
  sed -i "s/^  'pciaccess',$/  'rocm_sysdeps_pciaccess',/g" "$SRC_MESON_BUILD"
  # Update dependency declaration
  sed -i 's/link_with : libpciaccess/link_with : librocm_sysdeps_pciaccess/g' "$SRC_MESON_BUILD"
  # Apply symbol versioning to the library target instead of to the whole
  # project. Passing -Wl,--version-script via LDFLAGS applies it to every link in
  # the project, and meson duplicates such arguments in its compiler sanity check
  # (1.12.0), which ld rejects with "duplicate version tag". Per-target link_args
  # are not duplicated and are the correct scope for a version script anyway.
  # This script patches the source tree in place, so only add link_args once.
  if ! grep -q -- "--version-script" "$SRC_MESON_BUILD"; then
    sed -i "/^librocm_sysdeps_pciaccess = library($/,/^)$/ s|^\([[:space:]]*\)install : true,|\1link_args : ['-Wl,--version-script=$VERSION_LDS'],\n\1install : true,|" "$SRC_MESON_BUILD"
  fi
  if ! grep -q -- "--version-script" "$SRC_MESON_BUILD"; then
    echo "ERROR: Failed to patch $SRC_MESON_BUILD with --version-script" >&2
    exit 1
  fi
fi

# Patch the root meson.build pkg.generate reference
ROOT_MESON_BUILD="$SOURCE_DIR/meson.build"
if [ -f "$ROOT_MESON_BUILD" ]; then
  echo "Patching $ROOT_MESON_BUILD..."
  # Update pkg.generate to NOT pass the library object, use libraries parameter instead
  # This way meson won't auto-generate -lrocm_sysdeps_pciaccess
  # Original:
  #   pkg.generate(
  #     libpciaccess,
  #     description : '...',
  #   )
  # Result:
  #   pkg.generate(
  #     name : 'pciaccess',
  #     libraries : ['-L${libdir}', '-lpciaccess'],
  #     description : '...',
  #   )
  sed -i '/^pkg\.generate($/,/^)$/ {
    s/^  libpciaccess,$/  name : '\''pciaccess'\'',\n  libraries : ['\''-L${libdir}'\'', '\''-lpciaccess'\''],/
  }' "$ROOT_MESON_BUILD"
fi

echo "libpciaccess patching completed."
