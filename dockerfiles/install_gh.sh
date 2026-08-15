#!/bin/bash
# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

GH_VERSION="${1:-2.97.0}"
ARCH="$(uname -m)"
case "${GH_VERSION}:${ARCH}" in
    2.97.0:x86_64)
        GH_SHA256="a2c9b8497e1f85b1ad0dfcb78b5a622e098801b8e461e459e88e1ee12f018112"
        ;;
    *)
        echo "Unsupported gh CLI version/architecture: ${GH_VERSION}/${ARCH}" >&2
        exit 1
        ;;
esac

GH_ARCHIVE="gh_${GH_VERSION}_linux_amd64.tar.gz"

curl --silent --fail --show-error --location \
    "https://github.com/cli/cli/releases/download/v${GH_VERSION}/${GH_ARCHIVE}" \
    --output "${GH_ARCHIVE}"

printf '%s  %s\n' "${GH_SHA256}" "${GH_ARCHIVE}" | sha256sum --check --strict

tar -xzf "${GH_ARCHIVE}"
install -m 0755 "gh_${GH_VERSION}_linux_amd64/bin/gh" /usr/local/bin/gh

gh --version
