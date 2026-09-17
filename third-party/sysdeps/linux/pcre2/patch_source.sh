#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"
PCRE2_MAKEFILE="$SOURCE_DIR/Makefile.in"

echo "Patching sources..."
# Prefix the 8-bit libtool target (and its generated automake variables) so the
# shared library gets SONAME librocm_sysdeps_pcre2-8.so and can coexist with a
# system libpcre2-8. The symbol version script is injected via LDFLAGS.
sed -i 's/libpcre2-8\.la/librocm_sysdeps_pcre2-8.la/g' "$PCRE2_MAKEFILE"
sed -i 's/libpcre2_8_la_/librocm_sysdeps_pcre2_8_la_/g' "$PCRE2_MAKEFILE"
