#!/bin/bash
set -e

PREFIX="${1:?Expected install prefix argument}"

# Rename librocm_sysdeps_zstd.so to libzstd.so
mv $PREFIX/lib/librocm_sysdeps_zstd.so $PREFIX/lib/libzstd.so
# pc files are not output with a relative prefix. Sed it to relative.
sed -i -E 's|^prefix=.+|prefix=${pcfiledir}/../..|' $PREFIX/lib/pkgconfig/*.pc
