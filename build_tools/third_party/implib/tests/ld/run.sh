#!/bin/sh

# Copyright 2020-2024 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

# This test checks that linker wrapper works.

set -eu

cd $(dirname $0)

if test -n "${1:-}"; then
  ARCH="$1"
fi

. ../common.sh

export LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-}

$CC $CFLAGS -shared -fPIC interposed.c -o libinterposed.so
$CC $CFLAGS main.c -L. -linterposed
$INTERP ./a.out 2>&1 | tee ref.log

${PYTHON:-} ../../implib-gen.py -q --target $TARGET libinterposed.so
ln -sf ../../scripts/ld ${PREFIX}ld
trap "rm -f $PWD/${PREFIX}ld" EXIT
if $CC --version | grep -qE 'clang|^lcc'; then
  # Some compilers do not allow overriding ld via playing with PATH
  CFLAGS="$CFLAGS -B."
fi
PATH=.:../..:$PATH $CC $CFLAGS -Wno-deprecated main.c -L. -linterposed
if readelf -d a.out | grep -q libinterposed; then
  echo "Linker wrapper failed to wrap library"
  exit 1
fi
$INTERP ./a.out 2>&1 | tee new.log

if ! diff ref.log new.log; then
  echo "Linker wrapper failed to intercept functions"
  exit 1
fi

echo SUCCESS
