#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Rename libxml2 to the rocm_sysdeps_ prefix so the bundled sysdep can coexist with a
# system libxml2, and prefix its symbol versions (LIBXML2_<version>) the same way. The
# install step (patch_install.py) restores the libxml2.so dev symlink to the prefixed
# SONAME.

set -e

SOURCE_DIR="${1:?Source directory must be given}"

echo "Patching libxml2 sources..."

# The library, its programs and tests link libxml2.la by name: rewrite every Makefile.am.
while IFS= read -r -d '' makefile; do
  sed -i -e 's/\blibxml2\.la\b/librocm_sysdeps_xml2.la/g' \
         -e 's/\blibxml2_la_/librocm_sysdeps_xml2_la_/g' "$makefile"
done < <(find "$SOURCE_DIR" -name Makefile.am -print0)

sed -i -E 's/\bLIBXML2_([0-9])/AMDROCM_SYSDEPS_1.0_LIBXML2_\1/g' "$SOURCE_DIR/libxml2.syms"

echo "libxml2 source patch complete."
