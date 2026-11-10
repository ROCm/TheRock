#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Rename libpopt to the rocm_sysdeps_ prefix so the bundled sysdep can coexist with a
# system libpopt, and prefix its symbol version (LIBPOPT_0) the same way. The install
# step (patch_install.py) restores the libpopt.so dev symlink to the prefixed SONAME.

set -e

SOURCE_DIR="${1:?Source directory must be given}"

echo "Patching popt sources..."

# The library and its tests link libpopt.la by name. The shipped configure is used without
# autoreconf, so rewrite the generated Makefile.in files only: editing a Makefile.am would
# make automake's rebuild rules regenerate the build system.
while IFS= read -r -d '' makefile; do
  sed -i -e 's/\blibpopt\.la\b/librocm_sysdeps_popt.la/g' \
         -e 's/\blibpopt_la_/librocm_sysdeps_popt_la_/g' "$makefile"
done < <(find "$SOURCE_DIR" -name Makefile.in -print0)

sed -i -E 's/\bLIBPOPT_0\b/AMDROCM_SYSDEPS_1.0_LIBPOPT_0/g' "$SOURCE_DIR/src/libpopt.vers"

echo "popt source patch complete."
