#!/bin/sh

# Copyright 2024 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

# This is a simple test that verifies that parameters are correctly passed on stack.
# Run it like
#   ./run.sh ARCH

set -eu

cd $(dirname $0)

if test -n "${1:-}"; then
  ARCH="$1"
fi

. ../common.sh

CFLAGS="-g -O2 $CFLAGS"

# Build shlib to test against
$CC $CFLAGS -shared -fPIC interposed.c dummy.c -o libinterposed.so

# Prepare implib
${PYTHON:-} ../../implib-gen.py -q --target $TARGET libinterposed.so

# Build app
$CC $CFLAGS main.c libinterposed.so.tramp.S libinterposed.so.init.c $LIBS

LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-} $INTERP ./a.out int > a.out.log
diff test.ref a.out.log

LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-} $INTERP ./a.out float > a.out.log
diff test.ref a.out.log

echo SUCCESS
