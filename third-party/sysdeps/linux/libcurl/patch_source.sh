#!/usr/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

SOURCE_DIR="${1:?Source directory must be given}"

echo "Patching sources..."
# Prefix the libcurl libtool target across every Makefile.in (lib/ defines it,
# src/ links it) so the shared library gets SONAME librocm_sysdeps_curl.so and
# can coexist with a system libcurl. The symbol version script is injected via
# LDFLAGS by the parent CMakeLists.
find "$SOURCE_DIR" -name Makefile.in -exec sed -i \
  -e 's/libcurl\.la/librocm_sysdeps_curl.la/g' \
  -e 's/libcurl_la_/librocm_sysdeps_curl_la_/g' {} +
