#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

CCACHE_VERSION="$1"

ARCH="$(uname -m)"
case "${CCACHE_VERSION}:${ARCH}" in
    4.12.1:x86_64)
        CCACHE_SHA256="742e6a6e17c0a060046874eece2949b221c228e1119698a4c6e0b096cbc87152"
        ;;
    *)
        echo "Unsupported ccache version/architecture: ${CCACHE_VERSION}/${ARCH}" >&2
        exit 1
        ;;
esac

CCACHE_ARCHIVE="ccache-${CCACHE_VERSION}-linux-${ARCH}.tar.xz"
curl --silent --fail --show-error --location \
    "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/${CCACHE_ARCHIVE}" \
    --output "${CCACHE_ARCHIVE}"

printf '%s  %s\n' "${CCACHE_SHA256}" "${CCACHE_ARCHIVE}" | sha256sum --check --strict

tar xf "${CCACHE_ARCHIVE}"
cp ccache-${CCACHE_VERSION}-linux-${ARCH}/ccache /usr/local/bin
