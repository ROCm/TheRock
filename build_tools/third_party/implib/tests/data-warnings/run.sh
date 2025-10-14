#!/bin/sh

# Copyright 2019-2025 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

# This test checks that Implib.so emits warnings on unsupported constructs.

set -eu

cd $(dirname $0)

if test -n "${1:-}"; then
  ARCH="$1"
fi

. ../common.sh

export LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-}

$CC $CFLAGS -shared -fPIC interposed.c
${PYTHON:-} ../../implib-gen.py -q --target $TARGET a.out 2>err.log

if ! diff err.ref err.log; then
  echo "Warnings are not emitted for data symbols"
  exit 1
fi

echo SUCCESS
