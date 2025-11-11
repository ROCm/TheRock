#!/bin/bash
set -e

PREFIX="${1:?Expected install prefix argument}"
PATCHELF="${PATCHELF:-patchelf}"
THEROCK_SOURCE_DIR="${THEROCK_SOURCE_DIR:?THEROCK_SOURCE_DIR not defined}"
Python3_EXECUTABLE="${Python3_EXECUTABLE:?Python3_EXECUTABLE not defined}"

"$Python3_EXECUTABLE" "$THEROCK_SOURCE_DIR/build_tools/patch_linux_so.py" \
  --patchelf "${PATCHELF}" --add-prefix rocm_sysdeps_ \
  "$PREFIX/lib/libcap.so" \
  "$PREFIX/lib/libpsx.so"

# pc files are not output with a relative prefix. Sed it to relative.
if [ -d "$PREFIX/lib/pkgconfig" ]; then
  for pcfile in "$PREFIX/lib/pkgconfig"/*.pc; do
    if [ -f "$pcfile" ]; then
      sed -i -E 's|^prefix=.+|prefix=${pcfiledir}/../..|' "$pcfile"
      sed -i -E 's|^exec_prefix=.+|exec_prefix=${pcfiledir}/../..|' "$pcfile"
      sed -i -E 's|^libdir=.+|libdir=${prefix}/lib|' "$pcfile"
      sed -i -E 's|^includedir=.+|includedir=${prefix}/include|' "$pcfile"
    fi
  done
fi
