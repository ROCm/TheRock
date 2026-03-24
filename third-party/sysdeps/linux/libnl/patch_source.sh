#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBNL_MAKEFILE="$SOURCE_DIR/Makefile.in"

echo "Patching sources..."

# Patch library names in main Makefile
sed -i 's/lib\/libnl-3\.la/lib\/librocm_sysdeps_nl_3.la/g' "$LIBNL_MAKEFILE"
sed -i 's/libnl-genl-3\.la/librocm_sysdeps_nl_genl_3.la/g' "$LIBNL_MAKEFILE"
sed -i 's/libnl-route-3\.la/librocm_sysdeps_nl_route_3.la/g' "$LIBNL_MAKEFILE"
sed -i 's/libnl-idiag-3\.la/librocm_sysdeps_nl_idiag_3.la/g' "$LIBNL_MAKEFILE"
sed -i 's/libnl-nf-3\.la/librocm_sysdeps_nl_nf_3.la/g' "$LIBNL_MAKEFILE"
sed -i 's/libnl-xfrm-3\.la/librocm_sysdeps_nl_xfrm_3.la/g' "$LIBNL_MAKEFILE"
sed -i 's/libnl-cli-3\.la/librocm_sysdeps_nl_cli_3.la/g' "$LIBNL_MAKEFILE"

# Patch variable names for each library
# Core library (libnl-3)
sed -i 's/lib_libnl_3_la_/lib_librocm_sysdeps_nl_3_la_/g' "$LIBNL_MAKEFILE"
sed -i 's/am_lib_libnl_3_la_/am_lib_librocm_sysdeps_nl_3_la_/g' "$LIBNL_MAKEFILE"

# Generic netlink library (libnl-genl-3)
sed -i 's/lib_libnl_genl_3_la_/lib_librocm_sysdeps_nl_genl_3_la_/g' "$LIBNL_MAKEFILE"
sed -i 's/am_lib_libnl_genl_3_la_/am_lib_librocm_sysdeps_nl_genl_3_la_/g' "$LIBNL_MAKEFILE"

# Route library (libnl-route-3)
sed -i 's/lib_libnl_route_3_la_/lib_librocm_sysdeps_nl_route_3_la_/g' "$LIBNL_MAKEFILE"
sed -i 's/am_lib_libnl_route_3_la_/am_lib_librocm_sysdeps_nl_route_3_la_/g' "$LIBNL_MAKEFILE"

# Idiag library (libnl-idiag-3)
sed -i 's/lib_libnl_idiag_3_la_/lib_librocm_sysdeps_nl_idiag_3_la_/g' "$LIBNL_MAKEFILE"
sed -i 's/am_lib_libnl_idiag_3_la_/am_lib_librocm_sysdeps_nl_idiag_3_la_/g' "$LIBNL_MAKEFILE"

# Netfilter library (libnl-nf-3)
sed -i 's/lib_libnl_nf_3_la_/lib_librocm_sysdeps_nl_nf_3_la_/g' "$LIBNL_MAKEFILE"
sed -i 's/am_lib_libnl_nf_3_la_/am_lib_librocm_sysdeps_nl_nf_3_la_/g' "$LIBNL_MAKEFILE"

# XFRM library (libnl-xfrm-3)
sed -i 's/lib_libnl_xfrm_3_la_/lib_librocm_sysdeps_nl_xfrm_3_la_/g' "$LIBNL_MAKEFILE"
sed -i 's/am_lib_libnl_xfrm_3_la_/am_lib_librocm_sysdeps_nl_xfrm_3_la_/g' "$LIBNL_MAKEFILE"

# CLI library (libnl-cli-3) - optional
sed -i 's/lib_libnl_cli_3_la_/lib_librocm_sysdeps_nl_cli_3_la_/g' "$LIBNL_MAKEFILE"
sed -i 's/am_lib_libnl_cli_3_la_/am_lib_librocm_sysdeps_nl_cli_3_la_/g' "$LIBNL_MAKEFILE"

# Replace all version scripts with our custom ROCM_SYSDEPS versioning
echo "Updating version scripts..."
for sym_file in "$SOURCE_DIR"/libnl*.sym; do
    if [ -f "$sym_file" ]; then
        echo "Updating $sym_file"
        cp "$SCRIPT_DIR/libnl.map" "$sym_file"
    fi
done
