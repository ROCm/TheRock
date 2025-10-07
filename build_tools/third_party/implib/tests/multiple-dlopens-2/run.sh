#!/bin/sh

# Copyright 2022-2025 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

set -eu

cd $(dirname $0)

if test -n "${1:-}"; then
  ARCH="$1"
fi

. ../common.sh

CFLAGS="-g -O2 $CFLAGS"

# Build shlib to test against
$CC $CFLAGS -shared -fPIC interposed.c -o libinterposed.so

# Prepare implib
${PYTHON:-} ../../implib-gen.py -q --target $TARGET --dlopen-callback=my_load_library libinterposed.so

# Build app
$CC $CFLAGS -fPIE main.c libinterposed.so.tramp.S libinterposed.so.init.c $LIBS

LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-} $INTERP ./a.out > a.out.log
diff test.ref a.out.log

echo SUCCESS
