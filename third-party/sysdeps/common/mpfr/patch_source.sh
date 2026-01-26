#!/usr/bin/bash
set -e

SOURCE_DIR="${1:?Source directory must be given}"
MPFR_MAKEFILE="$SOURCE_DIR/src/Makefile.am"

echo "Patching sources..."
sed -i 's/libmpfr\.la/librocm_sysdeps_mpfr.la/g' "$MPFR_MAKEFILE"
sed -i 's/libmpfr_la_SOURCES/librocm_sysdeps_mpfr_la_SOURCES/g' "$MPFR_MAKEFILE"
sed -i 's/libmpfr_la_LDFLAGS/librocm_sysdeps_mpfr_la_LDFLAGS/g' "$MPFR_MAKEFILE"
sed -i 's/libmpfr_la_LIBADD/librocm_sysdeps_mpfr_la_LIBADD/g' "$MPFR_MAKEFILE"
