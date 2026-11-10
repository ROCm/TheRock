#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Rename liblttng-ctl to the rocm_sysdeps_ prefix so the bundled sysdep can coexist with
# a system lttng-tools. The install step (patch_install.py) restores the liblttng-ctl.so
# dev symlink to the prefixed SONAME. Also skip building the tests and examples, which
# are not shipped.

set -e

SOURCE_DIR="${1:?Source directory must be given}"

echo "Patching lttng-tools sources..."

# The daemons, the lttng command, the tests and the examples link liblttng-ctl.la by
# name: rewrite every Makefile.am.
while IFS= read -r -d '' makefile; do
  sed -i -e 's/\bliblttng-ctl\.la\b/librocm_sysdeps_lttng_ctl.la/g' \
         -e 's/\bliblttng_ctl_la_/librocm_sysdeps_lttng_ctl_la_/g' "$makefile"
done < <(find "$SOURCE_DIR" -name Makefile.am -print0)

sed -i -E 's/^SUBDIRS = include doc src tests doc\/examples$/SUBDIRS = include doc src/' \
  "$SOURCE_DIR/Makefile.am"
grep -q '^SUBDIRS = include doc src$' "$SOURCE_DIR/Makefile.am" || {
  echo "error: top-level SUBDIRS not as expected" >&2
  exit 1
}

echo "lttng-tools source patch complete."
