#!/bin/bash
set -e

PREFIX="${1:?Expected install prefix argument}"
THEROCK_SOURCE_DIR="${THEROCK_SOURCE_DIR:?THEROCK_SOURCE_DIR not defined}"

# NOTE: gRPC is built as static libraries (.a), not shared libraries (.so).
# SONAME rewriting is not needed for static libraries.
# Only .pc and .cmake file modifications are necessary for relocatability.

# Update .pc files to use relative paths
if [ -d "$PREFIX/lib/pkgconfig" ]; then
  echo "Updating pkgconfig files"
  for pcfile in "$PREFIX/lib/pkgconfig"/*.pc; do
    if [ -f "$pcfile" ]; then
      sed -i -E 's|^prefix=.+|prefix=${pcfiledir}/../..|' "$pcfile"
      sed -i -E 's|^exec_prefix=.+|exec_prefix=${prefix}|' "$pcfile"
      sed -i -E 's|^libdir=.+|libdir=${prefix}/lib|' "$pcfile"
      sed -i -E 's|^includedir=.+|includedir=${prefix}/include|' "$pcfile"
    fi
  done
fi

# Update CMake config files to be relocatable
echo "Updating CMake config files"

# Update gRPC CMake configs
if [ -d "$PREFIX/lib/cmake/grpc" ]; then
  for cmfile in "$PREFIX/lib/cmake/grpc"/*.cmake; do
    if [ -f "$cmfile" ]; then
      # Make paths relative by using CMAKE_CURRENT_LIST_DIR
      sed -i 's|INTERFACE_INCLUDE_DIRECTORIES "[^"]*include|INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include|g' "$cmfile"
      sed -i 's|IMPORTED_LOCATION "[^"]*lib/\([^"]*\)"|IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/\1"|g' "$cmfile"
    fi
  done
fi

# Update protobuf CMake configs
if [ -d "$PREFIX/lib/cmake/protobuf" ]; then
  for cmfile in "$PREFIX/lib/cmake/protobuf"/*.cmake; do
    if [ -f "$cmfile" ]; then
      sed -i 's|INTERFACE_INCLUDE_DIRECTORIES "[^"]*include|INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include|g' "$cmfile"
      sed -i 's|IMPORTED_LOCATION "[^"]*lib/\([^"]*\)"|IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/\1"|g' "$cmfile"
    fi
  done
fi

# Update absl CMake configs
if [ -d "$PREFIX/lib/cmake/absl" ]; then
  for cmfile in "$PREFIX/lib/cmake/absl"/*.cmake; do
    if [ -f "$cmfile" ]; then
      sed -i 's|INTERFACE_INCLUDE_DIRECTORIES "[^"]*include|INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include|g' "$cmfile"
      sed -i 's|IMPORTED_LOCATION "[^"]*lib/\([^"]*\)"|IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/\1"|g' "$cmfile"
    fi
  done
fi

echo "gRPC patching completed successfully"
