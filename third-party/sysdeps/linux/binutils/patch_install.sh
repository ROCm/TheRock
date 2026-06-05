#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

PREFIX="${1:?Expected install prefix argument}"
PATCHELF="${PATCHELF:-patchelf}"

echo "Patching binutils install..."

# Copy lib64 to lib if it exists.
if [ -d "$PREFIX/lib64" ]; then
  cp -r $PREFIX/lib64/* $PREFIX/lib
  rm -rf $PREFIX/lib64
fi

# We don't want library descriptors or binaries.
rm -f $PREFIX/lib/*.la

# Create libiberty.so from the PIC-compiled libiberty.a.
# libiberty is GPL-licensed so it must be dynamically linked (not statically) in
# this MIT-licensed project. The SONAME must be set so that binaries linking
# against libiberty.so embed only the SONAME ("libiberty.so") in DT_NEEDED rather
# than the absolute build-tree path, allowing the runtime linker to resolve it via
# RPATH at the installed location.
if [ -f "$PREFIX/lib/libiberty.a" ]; then
  echo "Creating libiberty.so from libiberty.a..."
  cc -shared \
     -Wl,-soname,libiberty.so \
     -Wl,--whole-archive "$PREFIX/lib/libiberty.a" \
     -Wl,--no-whole-archive \
     -o "$PREFIX/lib/libiberty.so"
  "${PATCHELF}" --set-rpath '$ORIGIN' "$PREFIX/lib/libiberty.so"
  echo "Created $PREFIX/lib/libiberty.so"
fi

if [ -d "$PREFIX/lib/bfd-plugins" ]; then
  rm -rf $PREFIX/lib/bfd-plugins
fi
if [ -d "$PREFIX/lib/gprofng" ]; then
  rm -rf $PREFIX/lib/gprofng
fi
if [ -d "$PREFIX/bin" ]; then
  rm -rf $PREFIX/bin
fi
if [ -d "$PREFIX/share" ]; then
  rm -rf $PREFIX/share
fi
if [ -d "$PREFIX/etc" ]; then
  rm -rf $PREFIX/etc
fi
if [ -d "$PREFIX/x86_64-pc-linux-gnu" ]; then
  rm -rf $PREFIX/x86_64-pc-linux-gnu
fi

echo "Done patching binutils install."
