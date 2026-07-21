#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Rename the lttng-ust libtool libraries to the rocm_sysdeps_ prefix so the
# bundled sysdep can coexist with a system liblttng-ust. The install step
# (patch_install.py) restores lib<name>.so dev symlinks pointing at the
# prefixed SONAMEs.
#
# Only the libraries built by the default (C/C++) configure are renamed. The
# optional Java (JNI) and Python agents are not enabled in our configure and so
# are not built; their own agent libraries are deliberately NOT renamed here.
# (References to the renamed base liblttng-ust from any such Makefile.am are
# still rewritten by the loop below, but the agents themselves are unsupported
# by this sysdep -- enabling them would require extending the rename set.)

set -e

SOURCE_DIR="${1:?Source directory must be given}"

echo "Patching lttng-ust sources..."

# All default-built libraries, longest names first to avoid partial-substring
# clobbering (e.g. "liblttng-ust" within "liblttng-ust-common").
LIBS=(
  liblttng-ust-cyg-profile-fast
  liblttng-ust-pthread-wrapper
  liblttng-ust-libc-wrapper
  liblttng-ust-cyg-profile
  liblttng-ust-tracepoint
  liblttng-ust-common
  liblttng-ust-fork
  liblttng-ust-ctl
  liblttng-ust-dl
  liblttng-ust-fd
  liblttng-ust
)

# Every per-library Makefile.am under src/lib may reference any sibling .la
# (cross-library LIBADD/DEPENDENCIES), so rewrite them all.
for makefile in "$SOURCE_DIR"/src/lib/*/Makefile.am; do
  [ -f "$makefile" ] || continue
  for lib in "${LIBS[@]}"; do
    suffix="${lib#liblttng-ust}"          # "", "-common", "-cyg-profile-fast", ...
    suffix="${suffix//-/_}"               # "", "_common", "_cyg_profile_fast", ...
    newname="librocm_sysdeps_lttng_ust${suffix}"
    sed -i "s/${lib}\.la/${newname}.la/g" "$makefile"
    canon_old="${lib//-/_}_la_"
    sed -i "s/${canon_old}/${newname}_la_/g" "$makefile"
  done
done

echo "lttng-ust source patch complete."
