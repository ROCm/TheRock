#!/bin/sh

# Copyright 2022 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

# This test checks that verbose mode does not cause a crash.

set -eu

cd $(dirname $0)

if test -n "${1:-}"; then
  ARCH="$1"
fi

. ../common.sh

export LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-}

$CXX $CFLAGS -shared -frtti -fPIC interposed.cpp -o libinterposed.so
${PYTHON:-} ../../implib-gen.py -vvv -q --target $TARGET --vtables libinterposed.so >/dev/null 2>&1

echo SUCCESS
