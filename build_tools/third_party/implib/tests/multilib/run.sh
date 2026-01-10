#!/bin/sh

# Copyright 2020-2024 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

# This is an example for https://stackoverflow.com/questions/64489128/automated-function-redirection-via-library-wrappers-in-c/64495628#64495628
# It generates a single libC.so which redirects calls to libA.so and libB.so

set -eu

cd $(dirname $0)

if test -n "${1:-}"; then
  ARCH="$1"
fi

. ../common.sh

export LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-}

LIB_CFLAGS='-shared -fPIC -fvisibility=hidden'

# Compile wrapped libs

$CC $CFLAGS $LIB_CFLAGS a.c -o libA.so
$CC $CFLAGS $LIB_CFLAGS b.c -o libB.so

# Generate and compile wrapper

${PYTHON:-} ../../implib-gen.py -q --target $TARGET --symbol-list libA_syms.txt libA.so
${PYTHON:-} ../../implib-gen.py -q --target $TARGET --symbol-list libB_syms.txt libB.so

$CC $CFLAGS $LIB_CFLAGS -DIMPLIB_EXPORT_SHIMS libA.so.* libB.so.* -o libC.so

# Use it in final app

$CC $CFLAGS main.c -L. -lC $LIBS
$INTERP ./a.out >out.log 2>&1

if ! diff out.ref out.log; then
  echo 'Creating unified wrapper for several libraries does not work'
  exit 1
fi

echo SUCCESS
