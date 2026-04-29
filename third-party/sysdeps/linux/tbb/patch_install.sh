#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Patches installed TBB libraries for portable sysdeps distribution:
#   - Sets RPATH to $ORIGIN on all real (non-symlink) shared libraries.
#   - Makes tbb.pc relocatable by replacing the absolute prefix.
#   - Removes the upstream cmake config (TheRock provides its own wrapper).
#
# Args: install_dir
# Env:  PATCHELF

set -e

PREFIX="${1:?Expected install prefix argument}"
PATCHELF="${PATCHELF:?PATCHELF env var required}"

LIB_DIR="$PREFIX/lib"

echo "Patching TBB install..."

shopt -s nullglob

# Set RPATH to $ORIGIN on all real (non-symlink) rocm_sysdeps TBB libraries and
# create unprefixed libtbb*.so symlinks pointing at the SONAME symlink
# (e.g. libtbb.so -> librocm_sysdeps_tbb.so.12).
for lib in "$LIB_DIR"/librocm_sysdeps_tbb*.so.*; do
    [ -L "$lib" ] && continue
    "$PATCHELF" --set-rpath '$ORIGIN' "$lib"
    # Create unprefixed .so symlink pointing at the SONAME (e.g. libtbb.so -> librocm_sysdeps_tbb.so.12).
    # $lib is the real file (librocm_sysdeps_tbb.so.12.17); strip the minor version to get the SONAME.
    base=$(basename "$lib")                    # librocm_sysdeps_tbb.so.12.17
    soname="${base%.*}"                        # librocm_sysdeps_tbb.so.12
    unprefixed="${soname/rocm_sysdeps_/}"      # libtbb.so.12
    symlink="${unprefixed%.so.*}.so"           # libtbb.so
    ln -sf "$soname" "$LIB_DIR/$symlink"
done

# Make tbb.pc relocatable: replace the absolute build-time prefix with
# a pcfiledir-relative path. .pc files live at $PREFIX/lib/pkgconfig,
# so ${pcfiledir}/../.. resolves back to $PREFIX at runtime.
PC="$LIB_DIR/pkgconfig/tbb.pc"
if [ -f "$PC" ]; then
    abs_prefix=$(grep '^prefix=' "$PC" | cut -d= -f2-)
    sed -i "s|prefix=${abs_prefix}|prefix=\${pcfiledir}/../..|" "$PC"
    sed -i "s|${abs_prefix}/|\${prefix}/|g" "$PC"
fi

# Remove the upstream cmake config. It references the original (unprefixed)
# library names and would conflict with TheRock's wrapper config provided via
# therock_cmake_subproject_provide_package.
rm -rf "$PREFIX/lib/cmake/TBB"

echo "Done patching TBB install."
