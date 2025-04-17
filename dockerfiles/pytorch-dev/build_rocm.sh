#!/bin/bash
set -e

AMDGPU_TARGETS="${1:?Must define --build-arg AMDGPU_TARGETS=gfxZZZZ}"

time cmake -GNinja -S/therock/src -B /therock/build \
  -DTHEROCK_BUNDLE_SYSDEPS=ON \
  -DTHEROCK_ENABLE_ALL=ON \
  -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=custom \
  "-DTHEROCK_AMDGPU_TARGETS=$AMDGPU_TARGETS"
cmake --build /therock/build
ctest --test-dir /therock/build
