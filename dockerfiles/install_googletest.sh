#!/bin/bash
# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

GOOGLE_TEST_VERSION="$1"
case "${GOOGLE_TEST_VERSION}" in
    1.16.0)
        GOOGLE_TEST_SHA256="78c676fc63881529bf97bf9d45948d905a66833fbfa5318ea2cd7478cb98f399"
        ;;
    *)
        echo "Unsupported googletest version: ${GOOGLE_TEST_VERSION}" >&2
        exit 1
        ;;
esac

GOOGLE_TEST_ARCHIVE="googletest-${GOOGLE_TEST_VERSION}.tar.gz"
curl --silent --fail --show-error --location \
    "https://github.com/google/googletest/releases/download/v${GOOGLE_TEST_VERSION}/${GOOGLE_TEST_ARCHIVE}" \
    --output "${GOOGLE_TEST_ARCHIVE}"

printf '%s  %s\n' "${GOOGLE_TEST_SHA256}" "${GOOGLE_TEST_ARCHIVE}" | sha256sum --check --strict

tar xzf "${GOOGLE_TEST_ARCHIVE}"
cd "googletest-${GOOGLE_TEST_VERSION}" && mkdir build && cd build

cmake -GNinja .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
ninja
ninja install
