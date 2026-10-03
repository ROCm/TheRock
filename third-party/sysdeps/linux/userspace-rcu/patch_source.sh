#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Rename the userspace-rcu libtool libraries to the rocm_sysdeps_ prefix so the
# bundled sysdep can coexist with a system liburcu. The install step
# (patch_install.py) restores lib<name>.so dev symlinks pointing at the
# prefixed SONAMEs.

set -e

SOURCE_DIR="${1:?Source directory must be given}"
MAKEFILE_AM="$SOURCE_DIR/src/Makefile.am"

echo "Patching userspace-rcu sources..."

# All eight libraries produced by src/Makefile.am. Order longest-first so that
# e.g. "liburcu" does not partially match "liburcu-common" during sed.
LIBS=(
  liburcu-common
  liburcu-signal
  liburcu-memb
  liburcu-qsbr
  liburcu-cds
  liburcu-bp
  liburcu-mb
  liburcu
)

for lib in "${LIBS[@]}"; do
  # librocm_sysdeps_urcu_common  (from liburcu-common)
  suffix="${lib#liburcu}"                # "", "-common", "-bp", ...
  suffix="${suffix//-/_}"                # "", "_common", "_bp", ...
  newname="librocm_sysdeps_urcu${suffix}"
  # .la target filename references
  sed -i "s/${lib}\.la/${newname}.la/g" "$MAKEFILE_AM"
  # Automake-canonicalized variable prefix (dashes -> underscores).
  canon_old="${lib//-/_}_la_"
  sed -i "s/${canon_old}/${newname}_la_/g" "$MAKEFILE_AM"
done

echo "userspace-rcu source patch complete."
