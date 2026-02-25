#!/usr/bin/bash
set -e

SOURCE_DIR="${1:?Source directory must be given}"
GMP_MAKEFILE="$SOURCE_DIR/Makefile.in"

echo "Patching sources..."
sed -i 's/libgmp\.la/librocm_sysdeps_gmp.la/g' "$GMP_MAKEFILE"
sed -i 's/libgmp_la_SOURCES/librocm_sysdeps_gmp_la_SOURCES/g' "$GMP_MAKEFILE"
sed -i 's/libgmp_la_LDFLAGS/librocm_sysdeps_gmp_la_LDFLAGS/g' "$GMP_MAKEFILE"
sed -i 's/libgmp_la_LIBADD/librocm_sysdeps_gmp_la_LIBADD/g' "$GMP_MAKEFILE"
sed -i 's/libgmp_la_DEPENDENCIES/librocm_sysdeps_gmp_la_DEPENDENCIES/g' "$GMP_MAKEFILE"
