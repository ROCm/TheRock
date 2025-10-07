#!/bin/sh

# Copyright 2025 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

# This is a Deterministic Simulation test via unthread library
# (https://github.com/mpdn/unthread)

set -eu

cd $(dirname $0)

CC=clang

CFLAGS="-g -O2 -fsanitize=thread"

UNTHREAD=$HOME/src/unthread
CFLAGS="-I$UNTHREAD/include $CFLAGS"
LIBS="$UNTHREAD/bin/unthread.o -ldl"

# ASLR keeps breaking Tsan mmaps
if test $(cat /proc/sys/kernel/randomize_va_space) != 0; then
  echo 0 | sudo tee /proc/sys/kernel/randomize_va_space
fi

# Build shlib to test against
$CC $CFLAGS -shared -fPIC interposed.c -o libinterposed.so

# Prepare implib
${PYTHON:-} ../../implib-gen.py -q libinterposed.so

# Test without export shims

$CC $CFLAGS main.c libinterposed.so.tramp.S libinterposed.so.init.c $LIBS

for i in $(seq 1 1000); do
  export UNTHREAD_SEED=$(printf '%032x' $i)
  LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-} ./a.out > a.out.log
  diff test.ref a.out.log
done

# Test with export shims

$CC $CFLAGS -DIMPLIB_EXPORT_SHIMS main.c libinterposed.so.tramp.S libinterposed.so.init.c $LIBS

for i in $(seq 1 1000); do
  export UNTHREAD_SEED=$(printf '%032x' $i)
  LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH:-} ./a.out > a.out.log
  diff test.ref a.out.log
done

echo SUCCESS
