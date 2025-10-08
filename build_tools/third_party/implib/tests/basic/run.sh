#!/bin/sh

# Copyright 2017-2022 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

# This is a simple test for Implib.so functionality.
# Run it like
#   ./run.sh ARCH
# where ARCH stands for any supported arch (arm, x86_64, etc., see `implib-gen -h' for full list).
# Note that you may need to install qemu-user for respective platform
# (i386 also needs gcc-multilib).

set -eu

cd $(dirname $0)

if test -n "${1:-}"; then
  ARCH="$1"
fi

. ../common.sh

CFLAGS="-g -O2 $CFLAGS"
CFLAGS="-Wno-unused-command-line-argument $CFLAGS"  # For -no-pie on BSD

# Build shlib to test against
$CC $CFLAGS -shared -fPIC interposed.c -o libinterposed.so

# Standalone executables

for ADD_CFLAGS in '-no-pie' '-fPIE'; do
  # Check for older compilers
  case "$ADD_CFLAGS" in
  -no-pie)
    (strings $(which $CC) | grep -q no-pie) || continue
    ;;
  esac

  for ADD_GFLAGS in '' '--no-lazy-load'; do
    echo "Standalone executable: GFLAGS += '$ADD_GFLAGS', CFLAGS += '$ADD_CFLAGS'"

    # Prepare implib
    ${PYTHON:-} ../../implib-gen.py -q --target $TARGET $ADD_GFLAGS libinterposed.so

    # Build app
    $CC $CFLAGS $ADD_CFLAGS main.c test.c libinterposed.so.tramp.S libinterposed.so.init.c $LIBS

    LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-} $INTERP ./a.out > a.out.log
    diff test.ref a.out.log
  done
done

# Shlibs

for ADD_GFLAGS in '' '--no-lazy-load'; do
  echo "Shared library: GFLAGS += '$ADD_GFLAGS'"

  # Prepare implib
  ${PYTHON:-} ../../implib-gen.py -q --target $TARGET $ADD_GFLAGS libinterposed.so

  # Build shlib
  $CC $CFLAGS -shared -fPIC shlib.c test.c libinterposed.so.tramp.S libinterposed.so.init.c $LIBS -o shlib.so

  # Build app
  $CC $CFLAGS $ADD_CFLAGS main.c shlib.so

  LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-} $INTERP ./a.out > a.out.log
  diff test.ref a.out.log
done

echo SUCCESS
