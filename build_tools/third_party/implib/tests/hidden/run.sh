#!/bin/sh

# Copyright 2021-2025 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

# This test checks that IMPLIB_EXPORT_SHIMS works as expected.

set -eu

cd $(dirname $0)

if test -n "${1:-}"; then
  ARCH="$1"
fi

. ../common.sh

export LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-}

$CC $CFLAGS -shared -fPIC interposed.c -o libinterposed.so
${PYTHON:-} ../../implib-gen.py -q --target $TARGET libinterposed.so

$CC $CFLAGS -shared -fPIC user.c libinterposed.so.tramp.S libinterposed.so.init.c -o libuser.so
$CC $CFLAGS -shared -fPIC user.c libinterposed.so.tramp.S libinterposed.so.init.c -DIMPLIB_EXPORT_SHIMS -o libuser_export_shims.so

if test $(readelf -D -sW libuser_export_shims.so | grep foo | wc -l) -eq 0; then
  echo "Shim symbol NOT exported by default" >&2
  exit 1
fi

if test $(readelf -D -sW libuser.so | grep foo | wc -l) -gt 0; then
  echo "Hidden shim symbol exported" >&2
  exit 1
fi

echo SUCCESS
