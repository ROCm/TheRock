#!/bin/sh

# Copyright 2025 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

# This is a test for checking thread safety of Implib's shims.
#
# There is also a separate driver for Deterministic Simulation Testing
# in run_unthread.sh.

set -eu

cd $(dirname $0)

if test -n "${1:-}"; then
  ARCH="$1"
fi

. ../common.sh

CFLAGS="-g -O2 $CFLAGS"

$CC $CFLAGS -shared -fPIC interposed.c -o libinterposed.so

${PYTHON:-} ../../implib-gen.py -q --target $TARGET interposed.def

$CC $CFLAGS -fPIE main.c interposed.tramp.S interposed.init.c $LIBS

LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-} $INTERP ./a.out > a.out.log
diff test.ref a.out.log
