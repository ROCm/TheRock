#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
NUMACTL_MAKEFILE="$SOURCE_DIR/Makefile.am"

echo "Patching sources..."
sed -i 's/libnuma\.la/librocm_sysdeps_numa.la/g' "$NUMACTL_MAKEFILE"
sed -i 's/libnuma_la_SOURCES/librocm_sysdeps_numa_la_SOURCES/g' "$NUMACTL_MAKEFILE"
sed -i 's/libnuma_la_LDFLAGS/librocm_sysdeps_numa_la_LDFLAGS/g' "$NUMACTL_MAKEFILE"

# numactl's configure unconditionally links -latomic via
# AC_SEARCH_LIBS([__atomic_fetch_and_1], [atomic]). On the toolchains we build
# with, the compiler lowers libnuma's byte-sized atomics inline, so libatomic is
# never actually used and the probe only leaves an unused libatomic.so.1
# dependency.
#
# NOTE: this is unconditional, so it assumes the target toolchain lowers these
# atomics inline. On a toolchain that emits them out-of-line (the one case seen
# is RISC-V + GCC; RISC-V + LLVM lowers inline), libatomic must be linked again.
_libatomic_probe='^AC_SEARCH_LIBS(\[__atomic_fetch_and_1\],[[:space:]]*\[atomic\])$'
if ! grep -q "$_libatomic_probe" "$SOURCE_DIR/configure.ac"; then
  echo "Error: expected libatomic probe not found in numactl configure.ac; the" \
       "source may have changed and this patch needs revisiting" >&2
  exit 1
fi
sed -i "/$_libatomic_probe/d" "$SOURCE_DIR/configure.ac"

sed -i -E 's|\b(libnuma_)|AMDROCM_SYSDEPS_1.0_\1|' $SOURCE_DIR/versions.ldscript
sed -i -E 's|@(libnuma_)|@AMDROCM_SYSDEPS_1.0_\1|' $SOURCE_DIR/*.c
