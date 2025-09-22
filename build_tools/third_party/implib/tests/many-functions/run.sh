#!/bin/sh

# Copyright 2023-2024 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

set -eu
#set -x

cd $(dirname $0)

if test -n "${1:-}"; then
  ARCH="$1"
fi

. ../common.sh

for i in `seq 0 8192`; do
  echo "int foo$i() { return $i; }"
done > test.c

$CC $CFLAGS -shared -fPIC test.c -o libtest.so

${PYTHON:-} ../../implib-gen.py -q --target $TARGET libtest.so

$CC $CFLAGS libtest.so.* main.c $LIBS

LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-} $INTERP ./a.out
