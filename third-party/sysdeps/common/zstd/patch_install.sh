#!/bin/bash
set -e

PREFIX="${1:?Expected install prefix argument}"
PATCHELF="${PATCHELF:-patchelf}"
THEROCK_SOURCE_DIR="${THEROCK_SOURCE_DIR:?THEROCK_SOURCE_DIR not defined}"
Python3_EXECUTABLE="${Python3_EXECUTABLE:?Python3_EXECUTABLE not defined}"

"$Python3_EXECUTABLE" "$THEROCK_SOURCE_DIR/build_tools/patch_linux_so.py" \
  --patchelf "${PATCHELF}" --add-prefix rocm_sysdeps_ \
  $PREFIX/lib/libzstd.so

# pc files are not output with a relative prefix. Sed it to relative.
sed -i -E 's|^prefix=.+|prefix=${pcfiledir}/../..|' $PREFIX/lib/pkgconfig/*.pc

# Rename -lzstd in the pc file.
sed -i -E 's|-lzstd|-lrocm_sysdeps_zstd|' $PREFIX/lib/pkgconfig/*.pc

# Rename the IMPORTED_LOCATION and SONAME in the CMake target files.
sed -i -E 's|lib/libzstd\.so\.[0-9\.]+|lib/librocm_sysdeps_zstd.so.1|' $PREFIX/lib/cmake/zstd/zstdTargets-*.cmake
sed -i -E 's|libzstd\.so\.1|librocm_sysdeps_zstd\.so\.1|' $PREFIX/lib/cmake/zstd/zstdTargets-*.cmake
