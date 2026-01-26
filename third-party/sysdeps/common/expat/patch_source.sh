#!/usr/bin/bash
set -e

SOURCE_DIR="${1:?Source directory must be given}"

echo "Patching sources..."

EXPAT_LIB_MAKEFILE="$SOURCE_DIR/lib/Makefile.am"
sed -i 's/libexpat\.la/librocm_sysdeps_expat.la/g' "$EXPAT_LIB_MAKEFILE"
sed -i 's/libexpat_la_SOURCES/librocm_sysdeps_expat_la_SOURCES/g' "$EXPAT_LIB_MAKEFILE"
sed -i 's/libexpat_la_LDFLAGS/librocm_sysdeps_expat_la_LDFLAGS/g' "$EXPAT_LIB_MAKEFILE"

# Expat's Makefile.am still tries to build the 'doc' directory even when
# --without-docbook or --disable-docs is passed. This patch removes 'doc'
# from SUBDIRS to prevent xmlwf.1 and other documentation targets from
# being generated during the build.
EXPAT_ROOT_MAKEFILE="$SOURCE_DIR/Makefile.am"
sed -i 's/SUBDIRS += xmlwf doc/SUBDIRS += xmlwf/g' "$EXPAT_ROOT_MAKEFILE"

EXPAT_EXAMPLES_MAKEFILE="$SOURCE_DIR/examples/Makefile.am"
EXPAT_TESTS_MAKEFILE="$SOURCE_DIR/tests/benchmark/Makefile.am"
EXPAT_XMLWF_MAKEFILE="$SOURCE_DIR/xmlwf/Makefile.am"

sed -i 's/libexpat\.la/librocm_sysdeps_expat.la/g' "$EXPAT_EXAMPLES_MAKEFILE"
sed -i 's/libexpat\.la/librocm_sysdeps_expat.la/g' "$EXPAT_TESTS_MAKEFILE"
sed -i 's/libexpat\.la/librocm_sysdeps_expat.la/g' "$EXPAT_XMLWF_MAKEFILE"
