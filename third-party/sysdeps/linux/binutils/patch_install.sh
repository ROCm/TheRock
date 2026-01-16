#!/bin/bash
# Patches installed binaries from the external build system.
# Args: install_dir patchelf_binary
set -e

PREFIX="${1:?Expected install prefix argument}"
PATCHELF="${PATCHELF:-patchelf}"
THEROCK_SOURCE_DIR="${THEROCK_SOURCE_DIR:?THEROCK_SOURCE_DIR not defined}"
Python3_EXECUTABLE="${Python3_EXECUTABLE:?Python3_EXECUTABLE not defined}"

echo "binutils::patch_install.sh - prefix - $PREFIX"

# Copy lib64 to lib if it exists
if [ -d "$PREFIX/lib64" ]; then
  cp -r $PREFIX/lib64/* $PREFIX/lib/
  rm -rf $PREFIX/lib64
fi

# We don't want library descriptors or binaries.
rm -f $PREFIX/lib/*.la

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
