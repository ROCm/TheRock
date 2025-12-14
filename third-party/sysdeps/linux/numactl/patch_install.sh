#!/bin/bash
set -e

PREFIX="${1:?Expected install prefix argument}"
PATCHELF="${PATCHELF:-patchelf}"
THEROCK_SOURCE_DIR="${THEROCK_SOURCE_DIR:?THEROCK_SOURCE_DIR not defined}"
Python3_EXECUTABLE="${Python3_EXECUTABLE:-python3}"

"$Python3_EXECUTABLE" "$THEROCK_SOURCE_DIR/build_tools/patch_linux_so.py" \
  --patchelf "${PATCHELF}" --add-prefix rocm_sysdeps_ \
  $PREFIX/lib/libnuma.so

# pc files are not output with a relative prefix. Sed it to relative.
sed -i -E 's|^prefix=.+|prefix=${pcfiledir}/../..|' $PREFIX/lib/pkgconfig/*.pc
sed -i -E 's|^exec_prefix=.+|exec_prefix=${pcfiledir}/../..|' $PREFIX/lib/pkgconfig/*.pc
sed -i -E 's|^libdir=.+|libdir=${prefix}/lib|' $PREFIX/lib/pkgconfig/*.pc
sed -i -E 's|^includedir=.+|includedir=${prefix}/include|' $PREFIX/lib/pkgconfig/*.pc

# We don't want library descriptors or binaries.
rm $PREFIX/lib/*.la
rm -Rf $PREFIX/bin
