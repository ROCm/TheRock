#!/usr/bin/bash
set -e

SOURCE_DIR="${1:?Source directory must be given}"
NUMACTL_MAKEFILE="$SOURCE_DIR/Makefile.am"

echo "Patching sources..."
sed -i 's/libnuma\.la/librocm_sysdeps_numa.la/g' "$NUMACTL_MAKEFILE"
sed -i 's/libnuma_la_SOURCES/librocm_sysdeps_numa_la_SOURCES/g' "$NUMACTL_MAKEFILE"
sed -i 's/libnuma_la_LDFLAGS/librocm_sysdeps_numa_la_LDFLAGS/g' "$NUMACTL_MAKEFILE"

sed -i -E 's|\b(libnuma_)|AMDROCM_SYSDEPS_1.0_\1|' $SOURCE_DIR/versions.ldscript
sed -i -E 's|@(libnuma_)|@AMDROCM_SYSDEPS_1.0_\1|' $SOURCE_DIR/*.c
