#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

CMAKE_VERSION="$1"

ARCH="$(uname -m)"
INSTALL_PREFIX="/usr/local/therock-tools"
case "${CMAKE_VERSION}:${ARCH}" in
    3.27.9:x86_64)
        CMAKE_SHA256="341c415b98abeebc0a31903dc65d4ad2eba1e897fea7ed723a8d6066cc7a21ae"
        ;;
    3.27.9:aarch64)
        CMAKE_SHA256="f6628eee0dc3ca849e662bdfe9b7ca52324ad41e2acd87462e0d782fba5cc5d9"
        ;;
    3.31.12:x86_64)
        CMAKE_SHA256="04c85c742484a490ee9deabab70c0af61b2592f99f768e87743f078366c1bbe2"
        ;;
    3.31.12:aarch64)
        CMAKE_SHA256="846017cab38b185fd5050e1c09e081d71e92e517898521dd5b5483c987720074"
        ;;
    *)
        echo "Unsupported CMake version/architecture: ${CMAKE_VERSION}/${ARCH}" >&2
        exit 1
        ;;
esac

mkdir -p "${INSTALL_PREFIX}"

INSTALLER="cmake-installer.sh"
curl --silent --fail --show-error --location \
    "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${ARCH}.sh" \
    --output "${INSTALLER}"

printf '%s  %s\n' "${CMAKE_SHA256}" "${INSTALLER}" | sha256sum --check --strict

chmod +x "${INSTALLER}"
"./${INSTALLER}" --skip-license --prefix="${INSTALL_PREFIX}"
