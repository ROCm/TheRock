#!/bin/bash
# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

GOOGLE_TEST_VERSION="$1"

curl --silent --fail --show-error --location \
    "https://github.com/google/googletest/releases/download/v${GOOGLE_TEST_VERSION}/googletest-${GOOGLE_TEST_VERSION}.tar.gz" \
    --output gtest.tar.xz

tar xf gtest.tar.xz
cd "googletest-${GOOGLE_TEST_VERSION}" && mkdir build && cd build

cmake -GNinja .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
ninja
ninja install
