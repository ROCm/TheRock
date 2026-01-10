#!/bin/sh

# Copyright 2025 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

# This is a simple benchmark of Implib's overhead.
#
# On my machine results are very close, Implib being 0.5-1% slower.

set -eu

cd $(dirname $0)

if test -n "${1:-}"; then
  ARCH="$1"
fi

. ../common.sh

CFLAGS="-g -O2 $CFLAGS"
N=10

# Need sudo for nice...
RUN='nice -n -20 taskset 1'

$CC $CFLAGS -shared -fPIC interposed.c -o libinterposed.so
${PYTHON:-} ../../implib-gen.py -q --target $TARGET libinterposed.so

export LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-}

# Baseline

$CC $CFLAGS -DBASELINE main.c

echo "Baseline:"
$RUN time ./a.out

# Normal

$CC $CFLAGS main.c -L. -linterposed $LIBS

echo "Normal:"
$RUN time ./a.out

# Implib

$CC $CFLAGS main.c libinterposed.so.tramp.S libinterposed.so.init.c $LIBS

echo "Implib:"
$RUN time ./a.out

# Implib (IMPLIB_EXPORT_SHIMS)

$CC $CFLAGS -DIMPLIB_EXPORT_SHIMS -rdynamic main.c libinterposed.so.tramp.S libinterposed.so.init.c $LIBS

echo "Implib (IMPLIB_EXPORT_SHIMS):"
$RUN time ./a.out
