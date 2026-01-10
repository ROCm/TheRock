#!/bin/sh

set -eu
#set -x

rm -rf states
if ! java -jar ./tla2tools.jar -workers `nproc` -coverage 0 Init.tla \
    | grep -q 'Model checking completed. No error has been found'; then
  echo >&2 'TLA verification failed'
  exit 1
fi

for inv in never_0 Prop; do
  rm -f Init.pml.trail
  spin -run -ltl $inv Init.pml 2>/dev/null
  if test -f Init.pml.trail; then
    echo >&2 'SPIN verification failed'
    exit 1
  fi
done

echo SUCCESS
