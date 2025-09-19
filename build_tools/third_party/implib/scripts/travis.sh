#!/bin/sh

# Copyright 2019-2025 Yury Gribov
#
# The MIT License (MIT)
# 
# Use of this source code is governed by MIT license that can be
# found in the LICENSE.txt file.

set -eu
# TODO: use pipefail here and in test scripts

if test -n "${TRAVIS:-}" -o -n "${GITHUB_ACTIONS:-}"; then
  set -x
fi

cd $(dirname $0)/..

ARCH=${ARCH:-}
export PYTHON="${PYTHON:-python3}"

tests/basic/run.sh $ARCH
tests/exceptions/run.sh $ARCH
tests/data-warnings/run.sh $ARCH
tests/vtables/run.sh $ARCH
if test -z "$ARCH" && ! echo "${CC:-}" | grep -q musl-gcc; then
  # TODO: enable for other targets
  tests/ld/run.sh
fi
if ! echo "$ARCH" | grep -q 'i[0-9]86'; then
  # TODO: symtab on x86 seems to be corrupted
  tests/multilib/run.sh $ARCH
fi
tests/hidden/run.sh $ARCH
tests/verbose/run.sh $ARCH
tests/no_dlopen/run.sh $ARCH
if ! echo "${CC:-}" | grep -q musl-gcc; then  # Musl does not implement dlclose
  tests/multiple-dlopens/run.sh $ARCH
  tests/multiple-dlopens-2/run.sh $ARCH
  tests/multiple-dlopens-3/run.sh $ARCH
fi
if ! echo "$ARCH" | grep -q powerpc; then
  tests/many-functions/run.sh $ARCH
fi
tests/stack-args/run.sh $ARCH
if ! echo "$ARCH" | grep -q 'powerpc\|mips\|riscv'; then
  # TODO: support vector types for remaining platforms
  tests/vector-args/run.sh $ARCH
fi
tests/thread/run.sh $ARCH
tests/thread-2/run.sh $ARCH
tests/def/run.sh $ARCH

echo 'All tests passed'
